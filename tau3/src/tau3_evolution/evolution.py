"""Exactly-three-round generation evolution with durable accounting recovery."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

from .access import TaskRef
from .accounting import RoundAccounting, UsageAttempt
from .canonical import canonical_sha256
from .checkpoint import RoundCheckpointStore, RoundMaterial
from .ledger import AccountingSnapshot, DurableAccountingLedger
from .manifests import EVOLUTION_PROTOCOL


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text(encoding="utf-8").strip() if path.exists() else "process-local"


@dataclass(frozen=True)
class RoundExecution:
    completed_task_keys: tuple[str, ...]
    committed_substantive_effect_ids: tuple[str, ...]
    usage_attempts: tuple[UsageAttempt, ...]
    artifact_hashes: tuple[str, ...]
    accepted_skill_versions: tuple[tuple[str, int], ...]
    failure_lineage_hashes: tuple[str, ...]
    accounting_ledger_path: str
    accounting_scope_id: str
    started_at_unix_ns: int
    started_monotonic_ns: int | None = None
    started_boot_id: str | None = None


class RoundEngine(Protocol):
    started_monotonic_ns: int | None = None
    started_boot_id: str | None = None

    def execute_round(
        self,
        *,
        round_index: int,
        input_generation_id: str,
        output_generation_id: str,
        train_tasks: tuple[TaskRef, ...],
        manifest_sha256: str,
    ) -> RoundExecution: ...


@dataclass(frozen=True)
class CompletedRound:
    round_index: int
    input_generation_id: str
    output_generation_id: str
    accounting: RoundAccounting
    accounting_snapshot_sha256: str
    material_sha256: str
    recovered: bool


def _task_refs(manifest: dict, round_index: int) -> tuple[TaskRef, ...]:
    refs = tuple(
        TaskRef(domain, task_id)
        for domain, split in manifest["domains"].items()
        for task_id in split["rounds"][round_index]
    )
    counts = {domain: sum(ref.domain == domain for ref in refs) for domain in manifest["domains"]}
    if counts != {"airline": 8, "retail": 20, "telecom": 20} or len(refs) != 48:
        raise ValueError("every evolution round requires 8/20/20 domain tasks")
    return refs


def _round_accounting(round_index: int, snapshot: AccountingSnapshot) -> RoundAccounting:
    return RoundAccounting(
        round_index=round_index,
        total_running_time_seconds=None,
        total_input_tokens=snapshot.total_input_tokens,
        total_output_tokens=snapshot.total_output_tokens,
        total_tokens=snapshot.total_tokens,
        usage_complete=snapshot.usage_complete,
        total_cost=snapshot.total_cost,
        cost_unit=snapshot.cost_unit,
        cost_complete=snapshot.cost_complete,
        physical_attempt_count=snapshot.physical_attempt_count,
    )


class EvolutionRunner:
    def __init__(
        self,
        *,
        manifest: dict,
        checkpoints: RoundCheckpointStore,
        engine: RoundEngine,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time_ns: Callable[[], int] = time.time_ns,
        boot_id: str | None = None,
    ):
        if manifest.get("protocol") != EVOLUTION_PROTOCOL:
            raise ValueError("three-generation evolution requires the 144/34 manifest")
        self.manifest = manifest
        self.checkpoints = checkpoints
        self.engine = engine
        self.monotonic = monotonic
        self.wall_time_ns = wall_time_ns
        self.boot_id = boot_id or _boot_id()

    def run(self, run_id: str) -> tuple[CompletedRound, CompletedRound, CompletedRound]:
        if not run_id:
            raise ValueError("run_id required")
        rows = tuple(self._run_round(run_id, index) for index in range(3))
        return rows  # type: ignore[return-value]

    def _from_receipt(self, round_index: int, receipt: dict) -> CompletedRound:
        if receipt.get("status") != "complete":
            raise RuntimeError("round timing has not been durably finalized")
        accounting_payload = dict(receipt["accounting"])
        snapshot_sha = accounting_payload.pop("accounting_snapshot_sha256")
        accounting = RoundAccounting(**accounting_payload)
        return CompletedRound(
            round_index,
            f"g{round_index:03d}",
            f"g{round_index + 1:03d}",
            accounting,
            snapshot_sha,
            receipt["material_sha256"],
            True,
        )

    def _run_round(self, run_id: str, round_index: int) -> CompletedRound:
        input_generation = f"g{round_index:03d}"
        output_generation = f"g{round_index + 1:03d}"
        material_doc, receipt = self.checkpoints.load(run_id, round_index)
        if receipt is not None and receipt["status"] == "complete":
            return self._from_receipt(round_index, receipt)
        if material_doc is not None:
            material = RoundMaterial(**material_doc)
            ledger = DurableAccountingLedger(material.accounting_ledger_path)
            if ledger.ledger_sha256 != material.accounting_ledger_sha256:
                raise ValueError("checkpoint/accounting ledger hash mismatch")
            material_sha = canonical_sha256(material_doc)
            snapshot = ledger.snapshot(
                scope_id=f"{run_id}:round:{round_index}",
                checkpoint_material_sha256=material_sha,
            )
            accounting = _round_accounting(round_index, snapshot)
            if receipt is None:
                self._mark(material, material_sha, snapshot, accounting)
                _, receipt = self.checkpoints.load(run_id, round_index)
                assert receipt is not None
            latency = (
                self.checkpoints.material_complete_boundary_ns(run_id, round_index)
                - material.started_at_unix_ns
            ) / 1_000_000_000
            accounting = replace(
                accounting,
                total_running_time_seconds=float(latency),
                latency_clock="wall_recovery_to_material_marker",
            )
            self.checkpoints.bind_timing(
                material,
                material_sha256=material_sha,
                accounting={
                    **asdict(accounting),
                    "accounting_snapshot_sha256": canonical_sha256(asdict(snapshot)),
                },
            )
            _, recovered_receipt = self.checkpoints.load(run_id, round_index)
            assert recovered_receipt is not None
            return self._from_receipt(round_index, recovered_receipt)

        refs = _task_refs(self.manifest, round_index)
        execution = self.engine.execute_round(
            round_index=round_index,
            input_generation_id=input_generation,
            output_generation_id=output_generation,
            train_tasks=refs,
            manifest_sha256=self.manifest["manifest_sha256"],
        )
        expected = tuple(f"{ref.domain}:{ref.task_id}" for ref in refs)
        if execution.completed_task_keys != expected:
            raise ValueError("round engine did not complete exact manifest-ordered tasks")
        expected_scope = f"{run_id}:round:{round_index}"
        if execution.accounting_scope_id != expected_scope:
            raise ValueError("round accounting scope mismatch")
        ledger = DurableAccountingLedger(execution.accounting_ledger_path)
        material = RoundMaterial(
            run_id,
            round_index,
            input_generation,
            output_generation,
            execution.completed_task_keys,
            execution.committed_substantive_effect_ids,
            execution.accounting_ledger_path,
            ledger.ledger_sha256,
            execution.artifact_hashes
            + execution.failure_lineage_hashes
            + (canonical_sha256(list(execution.accepted_skill_versions)),),
            execution.started_at_unix_ns,
            execution.started_monotonic_ns,
            execution.started_boot_id,
        )
        material_sha = self.checkpoints.stage_material(material)
        snapshot = ledger.snapshot(scope_id=expected_scope, checkpoint_material_sha256=material_sha)
        accounting = _round_accounting(round_index, snapshot)
        self._mark(material, material_sha, snapshot, accounting)
        # The durable wall timestamps span mid-round crash/recovery and close at material receipt.
        _, material_receipt = self.checkpoints.load(run_id, round_index)
        assert material_receipt is not None
        if execution.started_monotonic_ns is not None and execution.started_boot_id == self.boot_id:
            latency = (
                int(self.monotonic() * 1_000_000_000) - execution.started_monotonic_ns
            ) / 1_000_000_000
            clock = "monotonic_same_boot"
        else:
            latency = (
                self.checkpoints.material_complete_boundary_ns(run_id, round_index)
                - execution.started_at_unix_ns
            ) / 1_000_000_000
            clock = "wall_recovery_to_material_marker"
        accounting = replace(
            accounting,
            total_running_time_seconds=float(max(0.0, latency)),
            latency_clock=clock,
        )
        self.checkpoints.bind_timing(
            material,
            material_sha256=material_sha,
            accounting={
                **asdict(accounting),
                "accounting_snapshot_sha256": canonical_sha256(asdict(snapshot)),
            },
        )
        return CompletedRound(
            round_index,
            input_generation,
            output_generation,
            accounting,
            canonical_sha256(asdict(snapshot)),
            material_sha,
            False,
        )

    def _mark(
        self,
        material: RoundMaterial,
        material_sha: str,
        snapshot: AccountingSnapshot,
        accounting: RoundAccounting,
    ) -> None:
        self.checkpoints.mark_material_complete(
            material,
            material_sha256=material_sha,
            accounting={
                **asdict(accounting),
                "accounting_snapshot_sha256": canonical_sha256(asdict(snapshot)),
            },
            material_completed_at_unix_ns=self.wall_time_ns(),
        )
