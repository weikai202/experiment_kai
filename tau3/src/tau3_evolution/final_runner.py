"""Crash-resumable fixed-order Vanilla/G000/G003 final evaluation and reporting."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from . import SYSTEM_ORDER
from .canonical import atomic_write_json, canonical_sha256, read_json
from .failure_analysis import (
    EvaluationRecord,
    TrustedEpisodeTrace,
    trusted_trace_from_native_receipt,
)
from .final_evaluation import (
    ExecutionBinding,
    FinalEvaluationPlan,
    FinalTestOnceGuard,
    TaskTrialCluster,
)
from .ledger import AccountingSnapshot, DurableAccountingLedger


@dataclass(frozen=True)
class FinalUnitMaterial:
    evaluation: EvaluationRecord
    accounting: AccountingSnapshot
    artifact_sha256: str
    accounting_ledger_path: str
    execution_binding_sha256: str
    native_trace_receipt: dict


@dataclass(frozen=True)
class FinalUnitResult:
    evaluation: EvaluationRecord
    accounting: AccountingSnapshot
    direct_latency_seconds: float
    latency_clock: str
    material_completed_at_unix_ns: int
    artifact_sha256: str
    accounting_ledger_path: str
    execution_binding_sha256: str
    native_trace_receipt: dict
    trusted_trace_sha256: str


class FinalSystemExecutor(Protocol):
    def execute(
        self,
        *,
        system_id: str,
        generation_id: str | None,
        cluster: TaskTrialCluster,
        simulator_seed: int,
        manifest_position: int,
        accounting_scope_id: str,
        plan_sha256: str,
        execution_binding_sha256: str,
        unit_dispatch_id: str,
    ) -> FinalUnitMaterial: ...

    def recover(
        self,
        *,
        unit_dispatch_id: str,
        plan_sha256: str,
        execution_binding_sha256: str,
    ) -> FinalUnitMaterial | None: ...


@dataclass(frozen=True)
class SystemReport:
    system_id: str
    task_trial_count: int
    mean_native_reward: float
    full_success_count: int
    full_success_rate: float
    total_running_time_seconds: float
    total_tokens: int | None
    usage_complete: bool
    total_cost: int | None
    cost_unit: str
    cost_complete: bool


@dataclass(frozen=True)
class FinalReport:
    plan_sha256: str
    systems: tuple[SystemReport, ...]
    unit_result_hashes: tuple[str, ...]
    report_sha256: str


def _generation(system_id: str) -> str | None:
    return {"vanilla": None, "generation_0": "g000", "updated": "g003"}[system_id]


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text(encoding="utf-8").strip() if path.exists() else "process-local"


class ResumableFinalRunner:
    def __init__(
        self,
        root: str | Path,
        executor: FinalSystemExecutor,
        once_guard: FinalTestOnceGuard,
        *,
        time_ns=time.time_ns,
        monotonic_ns=time.monotonic_ns,
        boot_id: str | None = None,
    ):
        self.root = Path(root)
        self.executor = executor
        self.once_guard = once_guard
        self.time_ns = time_ns
        self.monotonic_ns = monotonic_ns
        self.boot_id = boot_id or _boot_id()
        self.execution_owner_sha256 = canonical_sha256(
            {"final_output_root": str(self.root.resolve())}
        )
        self.state_path = self.root / "execution.json"
        self.report_path = self.root / "report.json"

    def _units(self, plan: FinalEvaluationPlan):
        position = 0
        for system_id in SYSTEM_ORDER:
            for cluster in plan.task_clusters:
                for seed in cluster.simulator_seeds:
                    yield position, system_id, cluster, seed
                    position += 1

    def _unit_path(self, position: int) -> Path:
        return self.root / "units" / f"{position:06d}.json"

    def _material_path(self, position: int) -> Path:
        return self.root / "units" / f"{position:06d}.material.json"

    def _load_or_claim(self, plan: FinalEvaluationPlan) -> dict:
        if self.state_path.exists():
            state = read_json(self.state_path)
            if state.get("plan_sha256") != plan.plan_sha256:
                raise RuntimeError("final runner already belongs to a different plan")
            if state.get("status") not in {"claimed", "running", "complete"}:
                raise RuntimeError("invalid final runner state")
            state.setdefault("active_dispatch", None)
            state.setdefault("system_starts", {})
            state.setdefault("system_timings", {})
            return state
        state = {
            "plan_sha256": plan.plan_sha256,
            "system_order": list(SYSTEM_ORDER),
            "completed_positions": [],
            "active_dispatch": None,
            "system_starts": {},
            "system_timings": {},
            "status": "claimed",
        }
        atomic_write_json(self.state_path, state)
        return state

    def _material_for_unit(
        self, plan, state, position, system_id, cluster, seed, per_system
    ) -> tuple[FinalUnitMaterial, dict, bool]:
        material_path = self._material_path(position)
        dispatch_id = canonical_sha256(
            {
                "plan_sha256": plan.plan_sha256,
                "execution_binding_sha256": plan.execution_binding.binding_sha256,
                "position": position,
                "system_id": system_id,
                "domain": cluster.domain,
                "task_id": cluster.task_id,
                "simulator_seed": seed,
            }
        )
        active = state.get("active_dispatch")
        expected_identity = {
            "position": position,
            "unit_dispatch_id": dispatch_id,
            "system_id": system_id,
            "domain": cluster.domain,
            "task_id": cluster.task_id,
            "simulator_seed": seed,
        }
        fresh = False
        if material_path.exists():
            if active is None or any(active.get(k) != v for k, v in expected_identity.items()):
                raise RuntimeError("durable final material has no matching claimed dispatch")
            material = _material_from_json(read_json(material_path))
        elif active is not None:
            if any(active.get(k) != v for k, v in expected_identity.items()):
                raise RuntimeError("claimed final dispatch does not match next unit")
            material = self.executor.recover(
                unit_dispatch_id=dispatch_id,
                plan_sha256=plan.plan_sha256,
                execution_binding_sha256=plan.execution_binding.binding_sha256,
            )
            if material is None:
                raise RuntimeError(
                    "claimed final unit has unknown outcome; refusing sealed-test redispatch"
                )
            atomic_write_json(material_path, _material_to_json(material))
        else:
            active = {
                **expected_identity,
                "started_at_unix_ns": self.time_ns(),
                "started_monotonic_ns": self.monotonic_ns(),
                "boot_id": self.boot_id,
            }
            state["active_dispatch"] = active
            atomic_write_json(self.state_path, state)
            fresh = True
            material = self.executor.execute(
                system_id=system_id,
                generation_id=_generation(system_id),
                cluster=cluster,
                simulator_seed=seed,
                manifest_position=position % per_system,
                accounting_scope_id=f"final:{system_id}:{cluster.domain}:{cluster.task_id}:{seed}",
                plan_sha256=plan.plan_sha256,
                execution_binding_sha256=plan.execution_binding.binding_sha256,
                unit_dispatch_id=dispatch_id,
            )
            atomic_write_json(material_path, _material_to_json(material))
        return material, active, fresh

    def run(self, plan: FinalEvaluationPlan) -> FinalReport:
        self.once_guard.claim_or_resume(plan, self.execution_owner_sha256)
        state = self._load_or_claim(plan)
        if state["status"] == "complete":
            return self._read_report(plan, state)
        state["status"] = "running"
        atomic_write_json(self.state_path, state)
        self.once_guard.mark_running(plan, self.execution_owner_sha256)
        per_system = sum(len(cluster.simulator_seeds) for cluster in plan.task_clusters)
        units = tuple(self._units(plan))
        all_results: list[FinalUnitResult] = []
        hashes: list[str] = []
        for position, system_id, cluster, seed in units:
            if system_id not in state["system_starts"]:
                state["system_starts"][system_id] = {
                    "started_at_unix_ns": self.time_ns(),
                    "started_monotonic_ns": self.monotonic_ns(),
                    "boot_id": self.boot_id,
                }
                atomic_write_json(self.state_path, state)
            path = self._unit_path(position)
            if path.exists():
                payload = read_json(path)
                result = _unit_from_json(payload)
            else:
                material, active, _ = self._material_for_unit(
                    plan, state, position, system_id, cluster, seed, per_system
                )
                trace = _validate_material(
                    material,
                    position,
                    system_id,
                    cluster,
                    seed,
                    per_system,
                    plan.execution_binding,
                )
                completed_at = self._material_path(position).stat().st_mtime_ns
                if active["boot_id"] == self.boot_id:
                    latency = max(
                        0.0,
                        (self.monotonic_ns() - active["started_monotonic_ns"]) / 1_000_000_000,
                    )
                    latency_clock = "monotonic_same_boot"
                else:
                    latency = max(
                        0.0,
                        (completed_at - active["started_at_unix_ns"]) / 1_000_000_000,
                    )
                    latency_clock = "wall_recovery_cross_boot"
                result = FinalUnitResult(
                    material.evaluation,
                    material.accounting,
                    float(latency),
                    latency_clock,
                    completed_at,
                    material.artifact_sha256,
                    material.accounting_ledger_path,
                    material.execution_binding_sha256,
                    material.native_trace_receipt,
                    trace.trajectory_sha256,
                )
                payload = _unit_to_json(result)
                atomic_write_json(path, payload)
            _validate_unit(
                result,
                position,
                system_id,
                cluster,
                seed,
                per_system,
                plan.execution_binding,
            )
            digest = canonical_sha256(payload)
            all_results.append(result)
            hashes.append(digest)
            if position not in state["completed_positions"]:
                state["completed_positions"].append(position)
            state["active_dispatch"] = None
            atomic_write_json(self.state_path, state)
            if (position + 1) % per_system == 0 and system_id not in state["system_timings"]:
                start = state["system_starts"][system_id]
                if start["boot_id"] == self.boot_id:
                    duration = (self.monotonic_ns() - start["started_monotonic_ns"]) / 1_000_000_000
                    clock = "monotonic_same_boot"
                else:
                    duration = (self.time_ns() - start["started_at_unix_ns"]) / 1_000_000_000
                    clock = "wall_recovery_cross_boot"
                state["system_timings"][system_id] = {
                    "direct_latency_seconds": float(max(0.0, duration)),
                    "latency_clock": clock,
                    "completed_after_position": position,
                }
                atomic_write_json(self.state_path, state)
        if state["completed_positions"] != list(range(len(units))):
            raise RuntimeError("final result positions are incomplete or out of order")
        reports = tuple(
            _aggregate_system(
                system_id,
                [result for result in all_results if result.evaluation.system_id == system_id],
                state["system_timings"][system_id]["direct_latency_seconds"],
            )
            for system_id in SYSTEM_ORDER
        )
        report_payload = {
            "plan_sha256": plan.plan_sha256,
            "systems": [asdict(row) for row in reports],
            "unit_result_hashes": hashes,
        }
        report = FinalReport(
            plan.plan_sha256,
            reports,
            tuple(hashes),
            canonical_sha256(report_payload),
        )
        atomic_write_json(self.report_path, asdict(report))
        self.once_guard.complete_resumable(plan, self.execution_owner_sha256, report.report_sha256)
        state["status"] = "complete"
        state["report_sha256"] = report.report_sha256
        atomic_write_json(self.state_path, state)
        return report

    def _read_report(self, plan: FinalEvaluationPlan, state: dict) -> FinalReport:
        payload = read_json(self.report_path)
        report = FinalReport(
            payload["plan_sha256"],
            tuple(SystemReport(**row) for row in payload["systems"]),
            tuple(payload["unit_result_hashes"]),
            payload["report_sha256"],
        )
        base = {
            "plan_sha256": report.plan_sha256,
            "systems": [asdict(row) for row in report.systems],
            "unit_result_hashes": list(report.unit_result_hashes),
        }
        if report.plan_sha256 != plan.plan_sha256 or report.report_sha256 != canonical_sha256(base):
            raise ValueError("final report hash or plan binding mismatch")
        if state.get("report_sha256") != report.report_sha256:
            raise ValueError("final execution state/report hash mismatch")
        self.once_guard.validate_completed(plan, self.execution_owner_sha256, report.report_sha256)
        units = tuple(self._units(plan))
        expected_paths = tuple(self._unit_path(position) for position, *_ in units)
        unit_root = self.root / "units"
        actual_paths = tuple(
            sorted(
                path
                for path in unit_root.glob("*.json")
                if not path.name.endswith(".material.json")
            )
        )
        if actual_paths != expected_paths or len(report.unit_result_hashes) != len(units):
            raise ValueError("completed final unit result set is incomplete or out of order")
        per_system = sum(len(cluster.simulator_seeds) for cluster in plan.task_clusters)
        for expected_hash, path, unit in zip(
            report.unit_result_hashes, expected_paths, units, strict=True
        ):
            position, system_id, cluster, seed = unit
            unit_payload = read_json(path)
            if canonical_sha256(unit_payload) != expected_hash:
                raise ValueError("completed final unit content hash mismatch")
            result = _unit_from_json(unit_payload)
            _validate_unit(
                result,
                position,
                system_id,
                cluster,
                seed,
                per_system,
                plan.execution_binding,
            )
        return report


def _validate_material(
    material, position, system_id, cluster, seed, per_system, execution_binding: ExecutionBinding
) -> TrustedEpisodeTrace:
    row = material.evaluation
    if (
        row.system_id != system_id
        or row.domain != cluster.domain
        or row.task_id != cluster.task_id
        or row.simulator_seed != seed
        or row.manifest_position != position % per_system
    ):
        raise ValueError("final executor returned wrong task-trial identity")
    if material.execution_binding_sha256 != execution_binding.binding_sha256:
        raise ValueError("final unit did not use the frozen execution binding")
    if material.accounting.checkpoint_material_sha256 != material.artifact_sha256:
        raise ValueError("unit accounting is not checkpoint-bound")
    ledger = DurableAccountingLedger(material.accounting_ledger_path)
    if ledger.ledger_sha256 != material.accounting.ledger_sha256:
        raise ValueError("unit accounting ledger hash mismatch")
    recomputed = ledger.snapshot(
        scope_id=material.accounting.scope_id,
        checkpoint_material_sha256=material.artifact_sha256,
    )
    if recomputed != material.accounting:
        raise ValueError("unit accounting snapshot is not authoritative")
    trace = trusted_trace_from_native_receipt(material.native_trace_receipt)
    expected_generation_sha256 = {
        "vanilla": execution_binding.vanilla_generation_sha256,
        "generation_0": execution_binding.generation_0_sha256,
        "updated": execution_binding.updated_generation_sha256,
    }[system_id]
    if trace.generation_state_document.get("state_sha256") != expected_generation_sha256:
        raise ValueError("trusted trace generation state does not match execution binding")
    if (
        trace.system_id != system_id
        or trace.generation_id != _generation(system_id)
        or trace.domain != cluster.domain
        or trace.task_id != cluster.task_id
        or trace.simulator_seed != seed
        or trace.reward != row.reward
        or trace.evaluator_record_sha256 != row.evaluator_record_sha256
        or trace.trajectory_sha256 != row.trajectory_sha256
        or trace.accounting_ledger_document != ledger.document
    ):
        raise ValueError("native trusted trace does not match the final unit")
    return trace


def _validate_unit(
    result, position, system_id, cluster, seed, per_system, execution_binding: ExecutionBinding
):
    material = FinalUnitMaterial(
        result.evaluation,
        result.accounting,
        result.artifact_sha256,
        result.accounting_ledger_path,
        result.execution_binding_sha256,
        result.native_trace_receipt,
    )
    trace = _validate_material(
        material,
        position,
        system_id,
        cluster,
        seed,
        per_system,
        execution_binding,
    )
    if result.trusted_trace_sha256 != trace.trajectory_sha256:
        raise ValueError("persisted trusted trace hash mismatch")
    if (
        type(result.direct_latency_seconds) is not float
        or not math.isfinite(result.direct_latency_seconds)
        or result.direct_latency_seconds < 0
        or result.latency_clock not in {"monotonic_same_boot", "wall_recovery_cross_boot"}
        or type(result.material_completed_at_unix_ns) is not int
    ):
        raise ValueError("invalid runner-owned direct unit latency")


def _aggregate_system(
    system_id: str, results: list[FinalUnitResult], direct_system_latency_seconds: float
) -> SystemReport:
    if not results:
        raise ValueError("system has no final results")
    usage_complete = all(row.accounting.usage_complete for row in results)
    cost_complete = all(row.accounting.cost_complete for row in results)
    rewards = [row.evaluation.reward for row in results]
    return SystemReport(
        system_id,
        len(results),
        math.fsum(rewards) / len(rewards),
        sum(value == 1.0 for value in rewards),
        sum(value == 1.0 for value in rewards) / len(rewards),
        direct_system_latency_seconds,
        sum(row.accounting.total_tokens or 0 for row in results) if usage_complete else None,
        usage_complete,
        sum(row.accounting.total_cost or 0 for row in results) if cost_complete else None,
        "qwen_effective_output_tokens",
        cost_complete,
    )


def _material_to_json(material: FinalUnitMaterial) -> dict:
    return asdict(material)


def _material_from_json(payload: dict) -> FinalUnitMaterial:
    return FinalUnitMaterial(
        EvaluationRecord(**payload["evaluation"]),
        AccountingSnapshot(**payload["accounting"]),
        payload["artifact_sha256"],
        payload["accounting_ledger_path"],
        payload["execution_binding_sha256"],
        payload["native_trace_receipt"],
    )


def _unit_to_json(result: FinalUnitResult) -> dict:
    return asdict(result)


def _unit_from_json(payload: dict) -> FinalUnitResult:
    return FinalUnitResult(
        EvaluationRecord(**payload["evaluation"]),
        AccountingSnapshot(**payload["accounting"]),
        payload["direct_latency_seconds"],
        payload["latency_clock"],
        payload["material_completed_at_unix_ns"],
        payload["artifact_sha256"],
        payload["accounting_ledger_path"],
        payload["execution_binding_sha256"],
        payload["native_trace_receipt"],
        payload["trusted_trace_sha256"],
    )
