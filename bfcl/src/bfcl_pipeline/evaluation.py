from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .access import FinalTestAuthority
from .accounting import AccountingSnapshot, UsageLedger
from .canonical import canonical_bytes, sha256_json
from .evolution import Generation
from .providers import QWEN_MODEL, MaxTokenCalibrationReceipt, QwenRequestConfig
from .resources import EMBEDDING_MODEL
from .splits import SplitManifest
from .trajectory import TrustedTrajectory

SYSTEM_ORDER = ("vanilla", "g000", "g003")
BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"


def _system_boot_id() -> str | None:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


@dataclass(frozen=True)
class FinalEvaluationPlan:
    protocol: str
    systems: tuple[str, ...]
    generation_ids: tuple[str | None, ...]
    sealed_test_family_ids: tuple[str, ...]
    dataset_manifest_sha256: str
    process_count: int
    qwen_model: str
    calibration_receipt_sha256: str
    calibrated_max_tokens: int
    calibration_evidence_source: str
    qwen_request_config_sha256: str
    embedding_model: str
    embedding_client_config_sha256: str
    g000_generation_sha256: str
    g000_skill_library_sha256: str
    g003_generation_sha256: str
    bfcl_revision: str
    environment_sha256: str
    evaluator_sha256: str
    resource_manifest_sha256: str
    experiment_config_sha256: str
    plan_sha256: str


def _validate_generation(generation: Generation) -> None:
    core = asdict(generation)
    supplied = core.pop("generation_sha256")
    if supplied != sha256_json(core):
        raise ValueError("Generation identity is not canonical")


def build_final_plan(
    manifest: SplitManifest,
    g000: Generation,
    g003: Generation,
    *,
    qwen_config: QwenRequestConfig,
    calibration_receipt: MaxTokenCalibrationReceipt,
    embedding_client_config_sha256: str,
    bfcl_revision: str,
    environment_sha256: str,
    evaluator_sha256: str,
    resource_manifest_sha256: str,
    experiment_config_sha256: str,
) -> FinalEvaluationPlan:
    _validate_generation(g000)
    _validate_generation(g003)
    if g000.generation_id != "g000" or g003.generation_id != "g003":
        raise ValueError("Updated is fixed to g003; a diagnostic best checkpoint cannot replace it")
    if g000.parent_generation_id is not None or g003.parent_generation_id != "g002":
        raise ValueError("Final generations do not satisfy the frozen G000 root and G003 lineage")
    calibration_receipt.validate(
        dataset_manifest_sha256=manifest.manifest_sha256,
        allowed_train_case_ids={case_id for shard in manifest.rounds for case_id in shard.case_ids},
        require_live=True,
    )
    qwen_kwargs = qwen_config.as_openai_kwargs(calibration_receipt)
    frozen_hashes = (
        embedding_client_config_sha256,
        environment_sha256,
        evaluator_sha256,
        resource_manifest_sha256,
        experiment_config_sha256,
    )
    if any(len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in frozen_hashes):
        raise ValueError("Final plan provenance hashes must be lowercase SHA-256 digests")
    if bfcl_revision != BFCL_REVISION:
        raise ValueError("Final plan BFCL revision is not the pinned source")
    if resource_manifest_sha256 != g003.skill_library_sha256:
        raise ValueError("G003 does not bind the final generation resource manifest")
    core = {
        "protocol": "bfcl_pipeline_aligned_final_v1",
        "systems": SYSTEM_ORDER,
        "generation_ids": (None, "g000", "g003"),
        "sealed_test_family_ids": manifest.sealed_test_family_ids,
        "dataset_manifest_sha256": manifest.manifest_sha256,
        "process_count": 1,
        "qwen_model": QWEN_MODEL,
        "qwen_request_config_sha256": sha256_json(qwen_kwargs),
        "embedding_model": EMBEDDING_MODEL,
        "calibration_receipt_sha256": calibration_receipt.receipt_sha256,
        "calibrated_max_tokens": calibration_receipt.chosen_max_tokens,
        "calibration_evidence_source": calibration_receipt.evidence_source,
        "embedding_client_config_sha256": embedding_client_config_sha256,
        "g000_generation_sha256": g000.generation_sha256,
        "g000_skill_library_sha256": g000.skill_library_sha256,
        "g003_generation_sha256": g003.generation_sha256,
        "bfcl_revision": bfcl_revision,
        "environment_sha256": environment_sha256,
        "evaluator_sha256": evaluator_sha256,
        "resource_manifest_sha256": resource_manifest_sha256,
        "experiment_config_sha256": experiment_config_sha256,
    }
    return FinalEvaluationPlan(**core, plan_sha256=sha256_json(core))


def _validate_plan(plan: FinalEvaluationPlan) -> None:
    core = asdict(plan)
    supplied = core.pop("plan_sha256")
    if (
        plan.systems != SYSTEM_ORDER
        or plan.generation_ids != (None, "g000", "g003")
        or plan.process_count != 1
        or supplied != sha256_json(core)
    ):
        raise ValueError("Final evaluation plan identity or hash is invalid")
    if plan.qwen_model != QWEN_MODEL or plan.embedding_model != EMBEDDING_MODEL or plan.bfcl_revision != BFCL_REVISION:
        raise ValueError("Final evaluation model identity is invalid")


@dataclass(frozen=True)
class FinalUnitDispatch:
    dispatch_id: str
    plan_sha256: str
    owner_id: str
    unit_index: int
    system: str
    generation_id: str | None
    generation_sha256: str | None
    resource_manifest_sha256: str | None
    case_id: str

    @classmethod
    def build(cls, plan: FinalEvaluationPlan, owner_id: str, unit_index: int, system: str, generation_id: str | None, case_id: str) -> "FinalUnitDispatch":
        if system not in plan.systems:
            raise ValueError("Final unit system is not in the frozen plan")
        core = {
            "plan_sha256": plan.plan_sha256,
            "owner_id": owner_id,
            "unit_index": unit_index,
            "system": system,
            "generation_id": generation_id,
            "generation_sha256": {
                "vanilla": None,
                "g000": plan.g000_generation_sha256,
                "g003": plan.g003_generation_sha256,
            }[system],
            "resource_manifest_sha256": {
                "vanilla": None,
                "g000": plan.g000_skill_library_sha256,
                "g003": plan.resource_manifest_sha256,
            }[system],
            "case_id": case_id,
        }
        return cls(sha256_json(core), **core)

    def validate(self) -> None:
        core = asdict(self)
        supplied = core.pop("dispatch_id")
        if supplied != sha256_json(core):
            raise ValueError("Final unit dispatch identity is invalid")


@dataclass(frozen=True)
class NativeExecutionReceipt:
    dispatch_id: str
    plan_sha256: str
    generation_sha256: str
    resource_manifest_sha256: str
    native_state_sha256: str
    official_result_sha256: str
    receipt_sha256: str

    @staticmethod
    def _is_sha256(value: str) -> bool:
        return len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    @classmethod
    def build(
        cls,
        dispatch: FinalUnitDispatch,
        native_state_sha256: str,
        official_result_sha256: str,
    ) -> "NativeExecutionReceipt":
        if dispatch.generation_sha256 is None or dispatch.resource_manifest_sha256 is None:
            raise ValueError("Native execution provenance is only defined for generation-backed systems")
        core = {
            "dispatch_id": dispatch.dispatch_id,
            "plan_sha256": dispatch.plan_sha256,
            "generation_sha256": dispatch.generation_sha256,
            "resource_manifest_sha256": dispatch.resource_manifest_sha256,
            "native_state_sha256": native_state_sha256,
            "official_result_sha256": official_result_sha256,
        }
        if not all(cls._is_sha256(value) for value in (native_state_sha256, official_result_sha256)):
            raise ValueError("Native execution provenance hashes are invalid")
        return cls(**core, receipt_sha256=sha256_json(core))

    def validate(self, dispatch: FinalUnitDispatch) -> None:
        core = asdict(self)
        supplied = core.pop("receipt_sha256")
        if supplied != sha256_json(core):
            raise ValueError("Native execution receipt hash is invalid")
        if not all(self._is_sha256(value) for value in (
            self.dispatch_id,
            self.plan_sha256,
            self.generation_sha256,
            self.resource_manifest_sha256,
            self.native_state_sha256,
            self.official_result_sha256,
            self.receipt_sha256,
        )):
            raise ValueError("Native execution receipt contains a non-canonical SHA-256 digest")
        if (
            self.dispatch_id,
            self.plan_sha256,
            self.generation_sha256,
            self.resource_manifest_sha256,
        ) != (
            dispatch.dispatch_id,
            dispatch.plan_sha256,
            dispatch.generation_sha256,
            dispatch.resource_manifest_sha256,
        ):
            raise ValueError("Native execution receipt does not bind the frozen generation")


@dataclass(frozen=True)
class FinalExecutionOutcome:
    official_valid: bool
    failure_signature_sha256: str | None
    sanitized_result: dict
    ledger: UsageLedger
    trusted_trajectory: TrustedTrajectory | None = None
    native_execution_receipt: NativeExecutionReceipt | None = None


@dataclass(frozen=True)
class FinalCaseReceipt:
    system: str
    generation_id: str | None
    generation_sha256: str | None
    resource_manifest_sha256: str | None
    case_id: str
    plan_sha256: str
    environment_sha256: str
    evaluator_sha256: str
    official_valid: bool
    artifact_sha256: str
    accounting_snapshot_sha256: str
    trajectory_sha256: str | None
    native_execution_receipt_sha256: str | None
    receipt_sha256: str


@dataclass(frozen=True)
class SystemRunReceipt:
    system: str
    generation_id: str | None
    plan_sha256: str
    owner_id: str
    unit_count: int
    first_unit_index: int
    last_unit_index: int
    artifact_sha256: tuple[str, ...]
    completion_payload_sha256: str
    direct_latency_seconds: float
    receipt_sha256: str

    def validate(
        self,
        plan: FinalEvaluationPlan,
        owner_id: str,
        system_index: int,
        artifacts: tuple["FinalUnitArtifact", ...],
    ) -> None:
        core = asdict(self)
        supplied = core.pop("receipt_sha256")
        if supplied != sha256_json(core):
            raise ValueError("System-run receipt hash is invalid")
        expected_first = system_index * len(artifacts)
        if (
            self.system,
            self.generation_id,
            self.plan_sha256,
            self.owner_id,
            self.unit_count,
            self.first_unit_index,
            self.last_unit_index,
            self.artifact_sha256,
        ) != (
            plan.systems[system_index],
            plan.generation_ids[system_index],
            plan.plan_sha256,
            owner_id,
            len(artifacts),
            expected_first,
            expected_first + len(artifacts) - 1,
            tuple(row.artifact_sha256 for row in artifacts),
        ):
            raise ValueError("System-run receipt does not match the frozen run")
        if not math.isfinite(self.direct_latency_seconds) or self.direct_latency_seconds < 0:
            raise ValueError("System-run direct latency is invalid")


@dataclass(frozen=True)
class FinalEvaluationResult:
    artifacts_by_system: tuple[tuple["FinalUnitArtifact", ...], ...]
    system_run_receipts: tuple[SystemRunReceipt, ...]

    def __iter__(self):
        return iter(self.artifacts_by_system)

    def __len__(self) -> int:
        return len(self.artifacts_by_system)

    def __getitem__(self, index):
        return self.artifacts_by_system[index]


@dataclass(frozen=True)
class FinalUnitArtifact:
    unit_index: int
    dispatch_id: str
    system: str
    generation_id: str | None
    generation_sha256: str | None
    resource_manifest_sha256: str | None
    case_id: str
    plan_sha256: str
    environment_sha256: str
    evaluator_sha256: str
    official_valid: bool
    failure_signature_sha256: str | None
    sanitized_result: dict
    sanitized_result_sha256: str
    trajectory_sha256: str | None
    native_execution_receipt: dict | None
    proven_skill_origins: tuple[tuple[str, str], ...]
    accounting: dict
    accounting_snapshot_sha256: str
    receipt: FinalCaseReceipt
    artifact_sha256: str

    @classmethod
    def build(
        cls,
        plan: FinalEvaluationPlan,
        dispatch: FinalUnitDispatch,
        outcome: FinalExecutionOutcome,
        direct_latency_seconds: float,
    ) -> "FinalUnitArtifact":
        dispatch.validate()
        if type(outcome.official_valid) is not bool or not isinstance(outcome.sanitized_result, dict):
            raise ValueError("Final outcome must contain sanitized official validity and result data")
        if not math.isfinite(direct_latency_seconds) or direct_latency_seconds < 0:
            raise ValueError("Final unit direct latency is invalid")
        if outcome.failure_signature_sha256 is not None and not outcome.failure_signature_sha256:
            raise ValueError("Failure signature cannot be empty")
        accounting = outcome.ledger.snapshot(direct_latency_seconds)
        accounting_dict = asdict(accounting)
        trajectory_sha256 = None
        native_receipt = outcome.native_execution_receipt
        if native_receipt is not None:
            native_receipt.validate(dispatch)
            if native_receipt.official_result_sha256 != sha256_json(outcome.sanitized_result):
                raise ValueError("Native execution receipt does not bind the official result")
        if dispatch.system != "vanilla" and outcome.trusted_trajectory is None and native_receipt is None:
            raise ValueError("Generation-backed final units require trusted execution provenance")
        proven: tuple[tuple[str, str], ...] = ()
        if outcome.trusted_trajectory is not None:
            trajectory = outcome.trusted_trajectory
            trajectory.validate()
            if trajectory.case_id != dispatch.case_id or trajectory.generation_id != dispatch.generation_id:
                raise ValueError("Trusted trajectory does not match the final unit")
            if (
                trajectory.generation.generation_sha256 != dispatch.generation_sha256
                or trajectory.resource_manifest.manifest_sha256 != dispatch.resource_manifest_sha256
            ):
                raise ValueError("Trusted trajectory does not bind the exact frozen generation and resources")
            trajectory_sha256 = trajectory.trajectory_sha256
            proven = tuple(sorted({
                (selection.skill_version, selection.creation_effect_id)
                for selection in trajectory.selections
                if trajectory.proves_skill_execution(selection.skill_version, selection.creation_effect_id)
            }))
        core = {
            "unit_index": dispatch.unit_index,
            "dispatch_id": dispatch.dispatch_id,
            "system": dispatch.system,
            "generation_id": dispatch.generation_id,
            "generation_sha256": dispatch.generation_sha256,
            "resource_manifest_sha256": dispatch.resource_manifest_sha256,
            "case_id": dispatch.case_id,
            "plan_sha256": plan.plan_sha256,
            "environment_sha256": plan.environment_sha256,
            "evaluator_sha256": plan.evaluator_sha256,
            "official_valid": outcome.official_valid,
            "failure_signature_sha256": outcome.failure_signature_sha256,
            "sanitized_result": outcome.sanitized_result,
            "sanitized_result_sha256": sha256_json(outcome.sanitized_result),
            "trajectory_sha256": trajectory_sha256,
            "native_execution_receipt": None if native_receipt is None else asdict(native_receipt),
            "proven_skill_origins": proven,
            "accounting": accounting_dict,
            "accounting_snapshot_sha256": accounting.snapshot_sha256,
        }
        artifact_sha256 = sha256_json(core)
        receipt_core = {
            "system": dispatch.system,
            "generation_id": dispatch.generation_id,
            "generation_sha256": dispatch.generation_sha256,
            "resource_manifest_sha256": dispatch.resource_manifest_sha256,
            "case_id": dispatch.case_id,
            "plan_sha256": plan.plan_sha256,
            "environment_sha256": plan.environment_sha256,
            "evaluator_sha256": plan.evaluator_sha256,
            "official_valid": outcome.official_valid,
            "artifact_sha256": artifact_sha256,
            "accounting_snapshot_sha256": accounting.snapshot_sha256,
            "trajectory_sha256": trajectory_sha256,
            "native_execution_receipt_sha256": None if native_receipt is None else native_receipt.receipt_sha256,
        }
        receipt = FinalCaseReceipt(**receipt_core, receipt_sha256=sha256_json(receipt_core))
        return cls(**core, receipt=receipt, artifact_sha256=artifact_sha256)

    def validate(self, plan: FinalEvaluationPlan, dispatch: FinalUnitDispatch) -> None:
        dispatch.validate()
        core = asdict(self)
        supplied_artifact = core.pop("artifact_sha256")
        receipt_dict = core.pop("receipt")
        if supplied_artifact != sha256_json(core):
            raise ValueError("Final unit artifact hash is invalid")
        accounting_core = dict(self.accounting)
        supplied_snapshot = accounting_core.pop("snapshot_sha256", None)
        if supplied_snapshot != self.accounting_snapshot_sha256 or supplied_snapshot != sha256_json(accounting_core):
            raise ValueError("Final unit accounting snapshot is invalid")
        if self.sanitized_result_sha256 != sha256_json(self.sanitized_result):
            raise ValueError("Final unit sanitized result hash is invalid")
        native_receipt = None if self.native_execution_receipt is None else NativeExecutionReceipt(**self.native_execution_receipt)
        if native_receipt is not None:
            native_receipt.validate(dispatch)
            if native_receipt.official_result_sha256 != self.sanitized_result_sha256:
                raise ValueError("Native execution receipt does not bind the persisted official result")
        if self.system != "vanilla" and self.trajectory_sha256 is None and native_receipt is None:
            raise ValueError("Generation-backed final artifact lacks trusted execution provenance")
        receipt_core = asdict(self.receipt)
        supplied_receipt = receipt_core.pop("receipt_sha256")
        if supplied_receipt != sha256_json(receipt_core):
            raise ValueError("Final case receipt hash is invalid")
        expected = (
            dispatch.unit_index, dispatch.dispatch_id, dispatch.system, dispatch.generation_id,
            dispatch.generation_sha256, dispatch.resource_manifest_sha256,
            dispatch.case_id, plan.plan_sha256, plan.environment_sha256, plan.evaluator_sha256,
        )
        actual = (
            self.unit_index, self.dispatch_id, self.system, self.generation_id,
            self.generation_sha256, self.resource_manifest_sha256,
            self.case_id, self.plan_sha256, self.environment_sha256, self.evaluator_sha256,
        )
        if actual != expected:
            raise ValueError("Final unit artifact does not match the frozen plan dispatch")
        if (
            self.receipt.system, self.receipt.generation_id,
            self.receipt.generation_sha256, self.receipt.resource_manifest_sha256,
            self.receipt.case_id,
            self.receipt.plan_sha256, self.receipt.environment_sha256,
            self.receipt.evaluator_sha256, self.receipt.official_valid,
            self.receipt.artifact_sha256, self.receipt.accounting_snapshot_sha256,
            self.receipt.trajectory_sha256, self.receipt.native_execution_receipt_sha256,
        ) != (
            self.system, self.generation_id,
            self.generation_sha256, self.resource_manifest_sha256,
            self.case_id, self.plan_sha256,
            self.environment_sha256, self.evaluator_sha256, self.official_valid,
            self.artifact_sha256, self.accounting_snapshot_sha256, self.trajectory_sha256,
            None if native_receipt is None else native_receipt.receipt_sha256,
        ):
            raise ValueError("Final receipt does not bind its authoritative artifact")


class FinalUnitExecutor(Protocol):
    def execute(self, dispatch: FinalUnitDispatch) -> FinalExecutionOutcome: ...
    def recover(self, dispatch: FinalUnitDispatch) -> FinalExecutionOutcome: ...


def _write_atomic(path: Path, payload: dict) -> str:
    digest = sha256_json(payload)
    envelope = {"payload": payload, "payload_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_bytes(envelope) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return digest


def _read_hashed(path: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != sha256_json(payload):
        raise ValueError(f"Durable final artifact hash mismatch: {path.name}")
    return payload


def _artifact_from_payload(payload: dict) -> FinalUnitArtifact:
    row = dict(payload)
    row["proven_skill_origins"] = tuple(tuple(x) for x in row["proven_skill_origins"])
    row["receipt"] = FinalCaseReceipt(**row["receipt"])
    return FinalUnitArtifact(**row)


def _system_receipt_from_payload(payload: dict) -> SystemRunReceipt:
    row = dict(payload)
    row["artifact_sha256"] = tuple(row["artifact_sha256"])
    return SystemRunReceipt(**row)


def _elapsed_from_anchor(
    anchor: dict,
    *,
    clock: Callable[[], float],
    wall_clock: Callable[[], float],
    current_boot_id: str | None,
) -> float:
    if current_boot_id is not None and current_boot_id == anchor.get("boot_id"):
        value = clock() - float(anchor["started_monotonic"])
    else:
        value = wall_clock() - float(anchor["started_wall_time"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Runner clock anchor produced invalid direct latency")
    return value


def _ensure_system_start(
    plan: FinalEvaluationPlan,
    owner_id: str,
    state_root: Path,
    system_index: int,
    *,
    clock: Callable[[], float],
    wall_clock: Callable[[], float],
    current_boot_id: str | None,
) -> dict:
    path = state_root / "system_runs" / f"{system_index:02d}.start.json"
    if path.exists():
        payload = _read_hashed(path)
    else:
        started_monotonic = clock()
        started_wall_time = wall_clock()
        if not math.isfinite(started_monotonic) or not math.isfinite(started_wall_time):
            raise ValueError("System-run clock anchor is invalid")
        payload = {
            "system": plan.systems[system_index],
            "generation_id": plan.generation_ids[system_index],
            "plan_sha256": plan.plan_sha256,
            "owner_id": owner_id,
            "started_monotonic": started_monotonic,
            "started_wall_time": started_wall_time,
            "boot_id": current_boot_id,
        }
        _write_atomic(path, payload)
    if (
        payload.get("system"),
        payload.get("generation_id"),
        payload.get("plan_sha256"),
        payload.get("owner_id"),
    ) != (plan.systems[system_index], plan.generation_ids[system_index], plan.plan_sha256, owner_id):
        raise ValueError("System-run start anchor does not match the frozen run")
    return payload


def _finalize_system_run(
    plan: FinalEvaluationPlan,
    owner_id: str,
    state_root: Path,
    system_index: int,
    artifacts: tuple[FinalUnitArtifact, ...],
    *,
    clock: Callable[[], float],
    wall_clock: Callable[[], float],
    current_boot_id: str | None,
) -> SystemRunReceipt:
    directory = state_root / "system_runs"
    receipt_path = directory / f"{system_index:02d}.receipt.json"
    if receipt_path.exists():
        receipt = _system_receipt_from_payload(_read_hashed(receipt_path))
        receipt.validate(plan, owner_id, system_index, artifacts)
        completion = _read_hashed(directory / f"{system_index:02d}.complete.json")
        if receipt.completion_payload_sha256 != sha256_json(completion):
            raise ValueError("System-run receipt does not bind its completion checkpoint")
        return receipt
    anchor = _ensure_system_start(
        plan,
        owner_id,
        state_root,
        system_index,
        clock=clock,
        wall_clock=wall_clock,
        current_boot_id=current_boot_id,
    )
    completion = {
        "system": plan.systems[system_index],
        "generation_id": plan.generation_ids[system_index],
        "plan_sha256": plan.plan_sha256,
        "owner_id": owner_id,
        "artifact_sha256": tuple(row.artifact_sha256 for row in artifacts),
    }
    completion_sha256 = _write_atomic(directory / f"{system_index:02d}.complete.json", completion)
    direct_latency_seconds = _elapsed_from_anchor(
        anchor,
        clock=clock,
        wall_clock=wall_clock,
        current_boot_id=current_boot_id,
    )
    core = {
        "system": plan.systems[system_index],
        "generation_id": plan.generation_ids[system_index],
        "plan_sha256": plan.plan_sha256,
        "owner_id": owner_id,
        "unit_count": len(artifacts),
        "first_unit_index": system_index * len(artifacts),
        "last_unit_index": (system_index + 1) * len(artifacts) - 1,
        "artifact_sha256": tuple(row.artifact_sha256 for row in artifacts),
        "completion_payload_sha256": completion_sha256,
        "direct_latency_seconds": direct_latency_seconds,
    }
    receipt = SystemRunReceipt(**core, receipt_sha256=sha256_json(core))
    receipt.validate(plan, owner_id, system_index, artifacts)
    _write_atomic(receipt_path, asdict(receipt))
    return receipt


def _expected_cases(plan: FinalEvaluationPlan) -> tuple[str, ...]:
    return tuple(
        f"multi_turn_{variant}_{int(family_id.rsplit('_', 1)[1])}"
        for family_id in plan.sealed_test_family_ids
        for variant in ("base", "miss_func", "miss_param", "long_context")
    )


def execute_final_plan(
    plan: FinalEvaluationPlan,
    case_ids: tuple[str, ...],
    executor: FinalUnitExecutor,
    *,
    state_root: Path,
    authority: FinalTestAuthority,
    after_unit: Callable[[int], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    boot_id: str | None = None,
) -> FinalEvaluationResult:
    _validate_plan(plan)
    expected_cases = _expected_cases(plan)
    if case_ids != expected_cases or len(case_ids) != 160:
        raise ValueError("Final executor requires the exact sealed manifest order")
    status = authority.status(plan.plan_sha256, plan.dataset_manifest_sha256)
    units = tuple(
        (system, generation_id, case_id)
        for system, generation_id in zip(plan.systems, plan.generation_ids)
        for case_id in case_ids
    )
    unit_dir = state_root / "units"
    artifacts: list[FinalUnitArtifact] = []
    first_missing = len(units)
    for index, (system, generation_id, case_id) in enumerate(units):
        path = unit_dir / f"{index:04d}.json"
        if not path.exists():
            first_missing = index
            break
        dispatch = FinalUnitDispatch.build(plan, authority.owner_id, index, system, generation_id, case_id)
        artifact = _artifact_from_payload(_read_hashed(path))
        artifact.validate(plan, dispatch)
        artifacts.append(artifact)
    if unit_dir.exists() and any(path.name.endswith(".json") and int(path.stem) >= first_missing for path in unit_dir.iterdir() if path.stem.isdigit()):
        later = [path for path in unit_dir.iterdir() if path.stem.isdigit() and int(path.stem) > first_missing]
        if later:
            raise ValueError("Durable final artifacts are not a contiguous manifest prefix")

    report_path = state_root / "final_report.json"
    if status == "complete":
        if len(artifacts) != len(units) or not report_path.exists():
            raise ValueError("Completed authority does not have a complete durable report")
    else:
        authority.assert_active(plan.plan_sha256, plan.dataset_manifest_sha256)
        current_boot_id = boot_id if boot_id is not None else _system_boot_id()
        active_path = state_root / "active_dispatch.json"
        active_payload = _read_hashed(active_path) if active_path.exists() else None
        for completed_system_index in range(first_missing // len(case_ids)):
            _finalize_system_run(
                plan,
                authority.owner_id,
                state_root,
                completed_system_index,
                tuple(artifacts[completed_system_index * len(case_ids):(completed_system_index + 1) * len(case_ids)]),
                clock=clock,
                wall_clock=wall_clock,
                current_boot_id=current_boot_id,
            )
        for index in range(first_missing, len(units)):
            system, generation_id, case_id = units[index]
            system_index = index // len(case_ids)
            _ensure_system_start(
                plan,
                authority.owner_id,
                state_root,
                system_index,
                clock=clock,
                wall_clock=wall_clock,
                current_boot_id=current_boot_id,
            )
            dispatch = FinalUnitDispatch.build(plan, authority.owner_id, index, system, generation_id, case_id)
            recovering = active_payload is not None and active_payload.get("dispatch") == asdict(dispatch)
            if recovering:
                started_monotonic = float(active_payload["started_monotonic"])
                started_wall_time = float(active_payload["started_wall_time"])
                started_boot_id = active_payload.get("boot_id")
                outcome = executor.recover(dispatch)
            else:
                started_monotonic = clock()
                started_wall_time = wall_clock()
                started_boot_id = current_boot_id
                if not math.isfinite(started_monotonic) or not math.isfinite(started_wall_time):
                    raise ValueError("Final unit runner clock anchor is invalid")
                _write_atomic(active_path, {
                    "dispatch": asdict(dispatch),
                    "started_monotonic": started_monotonic,
                    "started_wall_time": started_wall_time,
                    "boot_id": started_boot_id,
                })
                outcome = executor.execute(dispatch)
            prepared = FinalUnitArtifact.build(plan, dispatch, outcome, 0.0)
            _write_atomic(state_root / "unit_completion" / f"{index:04d}.json", asdict(prepared))
            direct_latency_seconds = _elapsed_from_anchor(
                {
                    "started_monotonic": started_monotonic,
                    "started_wall_time": started_wall_time,
                    "boot_id": started_boot_id,
                },
                clock=clock,
                wall_clock=wall_clock,
                current_boot_id=current_boot_id,
            )
            artifact = FinalUnitArtifact.build(plan, dispatch, outcome, direct_latency_seconds)
            artifact.validate(plan, dispatch)
            _write_atomic(unit_dir / f"{index:04d}.json", asdict(artifact))
            artifacts.append(artifact)
            active_payload = None
            if after_unit is not None:
                after_unit(index)
            if (index + 1) % len(case_ids) == 0:
                _finalize_system_run(
                    plan,
                    authority.owner_id,
                    state_root,
                    system_index,
                    tuple(artifacts[system_index * len(case_ids):(system_index + 1) * len(case_ids)]),
                    clock=clock,
                    wall_clock=wall_clock,
                    current_boot_id=current_boot_id,
                )
        system_receipts = tuple(
            _finalize_system_run(
                plan,
                authority.owner_id,
                state_root,
                system_index,
                tuple(artifacts[system_index * len(case_ids):(system_index + 1) * len(case_ids)]),
                clock=clock,
                wall_clock=wall_clock,
                current_boot_id=current_boot_id,
            )
            for system_index in range(len(plan.systems))
        )
        report_core = {
            "plan_sha256": plan.plan_sha256,
            "owner_id": authority.owner_id,
            "unit_count": len(artifacts),
            "artifact_sha256": tuple(x.artifact_sha256 for x in artifacts),
            "system_run_receipt_sha256": tuple(x.receipt_sha256 for x in system_receipts),
        }
        report_sha256 = _write_atomic(report_path, report_core)
        authority.complete(report_sha256)

    system_receipts = tuple(
        _system_receipt_from_payload(_read_hashed(state_root / "system_runs" / f"{system_index:02d}.receipt.json"))
        for system_index in range(len(plan.systems))
    )
    for system_index, receipt in enumerate(system_receipts):
        receipt.validate(
            plan,
            authority.owner_id,
            system_index,
            tuple(artifacts[system_index * len(case_ids):(system_index + 1) * len(case_ids)]),
        )
        completion = _read_hashed(state_root / "system_runs" / f"{system_index:02d}.complete.json")
        if receipt.completion_payload_sha256 != sha256_json(completion):
            raise ValueError("System-run receipt does not bind its completion checkpoint")
    report = _read_hashed(report_path)
    if authority.completed_report_sha256(plan.plan_sha256, plan.dataset_manifest_sha256) != sha256_json(report):
        raise ValueError("Authority marker does not bind the durable final report")
    expected_report = {
        "plan_sha256": plan.plan_sha256,
        "owner_id": authority.owner_id,
        "unit_count": len(units),
        "artifact_sha256": [x.artifact_sha256 for x in artifacts],
        "system_run_receipt_sha256": [x.receipt_sha256 for x in system_receipts],
    }
    if report != expected_report:
        raise ValueError("Durable final report does not match authoritative unit artifacts")
    return FinalEvaluationResult(
        tuple(
            tuple(artifacts[offset:offset + len(case_ids)])
            for offset in range(0, len(artifacts), len(case_ids))
        ),
        system_receipts,
    )
