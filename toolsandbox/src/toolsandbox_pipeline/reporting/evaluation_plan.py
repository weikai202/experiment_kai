"""Frozen final-plan validation and durable test-once authorization ledger."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_SH, LOCK_UN, flock
from pathlib import Path
from typing import Iterator, Literal

from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.reporting import FinalEvaluationPlan, SYSTEM_ORDER, SystemId

TestLedgerState = Literal[
    "authorized_not_started",
    "running",
    "completed",
    "terminal_failure",
    "reconciliation_required",
]


class EvaluationPlanError(RuntimeError):
    """Sanitized plan or one-time authorization failure."""


class SystemCompletionReceipt(StrictModel):
    """Restricted durable result pointer staged before the completed marker."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    result_reference: BlobReference
    result_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    timing_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_reference(self) -> "SystemCompletionReceipt":
        if (
            self.result_reference.sha256 != self.result_sha256
            or self.result_reference.schema_name != "PersistedSystemResult"
            or self.result_reference.schema_version != 1
        ):
            raise ValueError("system completion result reference mismatch")
        return self



class CoordinatorRegistryAuthority:
    """Launcher-created capability for the single coordinator registry root."""

    __slots__ = ("authority_id", "authority_sha256", "registry_root")
    _TOKEN = object()
    _AUTHORITY_FILE = ".coordinator-registry-authority.json"
    _LOCK_FILE = ".coordinator-registry-authority.lock"

    def __init__(
        self,
        token: object,
        registry_root: Path,
        authority_id: str,
        authority_sha256: str,
    ) -> None:
        if token is not self._TOKEN:
            raise TypeError("registry authority must be created or opened")
        self.registry_root = registry_root
        self.authority_id = authority_id
        self.authority_sha256 = authority_sha256

    @classmethod
    def create(
        cls, registry_root: Path, *, authority_id: str
    ) -> "CoordinatorRegistryAuthority":
        root = cls._validate_root(registry_root, create=True)
        if not authority_id or any(character.isspace() for character in authority_id):
            raise EvaluationPlanError("invalid coordinator authority identity")
        authority = cls(cls._TOKEN, root, authority_id, "sha256:" + "0" * 64)
        with authority.locked(exclusive=True, verify=False):
            payload = {
                "schema_version": 1,
                "protocol": "coordinator-test-registry-authority-v1",
                "authority_id": authority_id,
                "registry_root": str(root),
            }
            payload["authority_sha256"] = canonical_sha256(payload)
            path = root / cls._AUTHORITY_FILE
            try:
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except FileExistsError as error:
                raise EvaluationPlanError("coordinator registry authority exists") from error
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            cls._fsync_directory(root)
        return cls(cls._TOKEN, root, authority_id, payload["authority_sha256"])

    @classmethod
    def open(
        cls, registry_root: Path, *, authority_id: str
    ) -> "CoordinatorRegistryAuthority":
        root = cls._validate_root(registry_root, create=False)
        provisional = cls(cls._TOKEN, root, authority_id, "sha256:" + "0" * 64)
        with provisional.locked(exclusive=False, verify=False):
            payload = provisional._read_authority()
        if payload["authority_id"] != authority_id:
            raise EvaluationPlanError("coordinator authority identity mismatch")
        return cls(cls._TOKEN, root, authority_id, payload["authority_sha256"])

    @classmethod
    def _validate_root(cls, registry_root: Path, *, create: bool) -> Path:
        if not isinstance(registry_root, Path) or not registry_root.is_absolute():
            raise EvaluationPlanError("coordinator registry root must be absolute")
        _reject_symlink_ancestry(registry_root)
        if create:
            registry_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(registry_root, 0o700)
        if not registry_root.is_dir() or registry_root.is_symlink():
            raise EvaluationPlanError("invalid coordinator registry root")
        return registry_root.resolve(strict=True)

    @contextmanager
    def locked(
        self, *, exclusive: bool, verify: bool = True
    ) -> Iterator[None]:
        descriptor = os.open(
            self.registry_root / self._LOCK_FILE,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            flock(descriptor, LOCK_EX if exclusive else LOCK_SH)
            if verify:
                payload = self._read_authority()
                if (
                    payload["authority_id"] != self.authority_id
                    or payload["authority_sha256"] != self.authority_sha256
                ):
                    raise EvaluationPlanError("coordinator registry authority changed")
            yield
        finally:
            flock(descriptor, LOCK_UN)
            os.close(descriptor)

    def _read_authority(self) -> dict:
        payload = _load_canonical_object(
            self.registry_root / self._AUTHORITY_FILE,
            "coordinator registry authority",
        )
        if (
            set(payload)
            != {
                "schema_version",
                "protocol",
                "authority_id",
                "registry_root",
                "authority_sha256",
            }
            or payload["schema_version"] != 1
            or payload["protocol"] != "coordinator-test-registry-authority-v1"
            or payload["registry_root"] != str(self.registry_root)
            or payload["authority_sha256"]
            != canonical_sha256(
                {key: value for key, value in payload.items() if key != "authority_sha256"}
            )
        ):
            raise EvaluationPlanError("invalid coordinator registry authority")
        return payload

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _load_canonical_object(path: Path, label: str) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationPlanError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_uid != os.getuid()
            ):
                raise EvaluationPlanError(f"unsafe {label} file")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                raw = handle.read()
        finally:
            os.close(descriptor)
        payload = json.loads(raw, object_pairs_hook=unique_object)
    except Exception as error:
        raise EvaluationPlanError(f"{label} unavailable or invalid") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise EvaluationPlanError(f"{label} is not canonical")
    return payload


def load_final_plan(path: Path) -> FinalEvaluationPlan:
    """Strictly load a canonical plan without opening any dataset or secret."""

    if not path.is_absolute():
        raise EvaluationPlanError("final plan path must be absolute")
    _reject_symlink_ancestry(path)
    raw = path.read_bytes()
    try:
        plan = FinalEvaluationPlan.model_validate_json(raw, strict=True)
    except Exception as error:
        raise EvaluationPlanError("invalid final evaluation plan") from error
    if raw != canonical_json_bytes(plan.model_dump(mode="json")):
        raise EvaluationPlanError("final evaluation plan is not canonical JSON")
    validate_output_root(plan)
    return plan


def validate_output_root(plan: FinalEvaluationPlan) -> None:
    path = Path(plan.output_root)
    normalized = Path(os.path.normpath(str(path)))
    if path != normalized:
        raise EvaluationPlanError("output root is not normalized")
    _reject_symlink_ancestry(path)


def _reject_symlink_ancestry(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise EvaluationPlanError("path ancestry cannot contain a symlink")


def write_frozen_plan(path: Path, plan: FinalEvaluationPlan) -> None:
    """Create, never overwrite, the canonical approved plan with mode 0600."""

    validate_output_root(plan)
    _reject_symlink_ancestry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as error:
        if path.read_bytes() == canonical_json_bytes(plan.model_dump(mode="json")):
            return
        raise EvaluationPlanError("refusing to overwrite a different frozen plan") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(plan.model_dump(mode="json")))
        handle.flush()
        os.fsync(handle.fileno())


class TestOnceGuard:
    """Registry-scoped guard preventing completed test systems from rerunning."""

    def __init__(
        self, authority: CoordinatorRegistryAuthority, plan: FinalEvaluationPlan
    ) -> None:
        if type(authority) is not CoordinatorRegistryAuthority:
            raise TypeError("CoordinatorRegistryAuthority required")
        self.authority = authority
        self.registry_root = authority.registry_root
        self.plan = plan
        registry_key = canonical_sha256({
            "protocol": "test-once-registry-key-v1",
            "test_dataset_manifest_sha256": plan.test_dataset_manifest_sha256,
            "ordered_test_ids": [item.scenario_id for item in plan.scenarios],
            "ordered_system_ids": list(plan.ordered_system_ids),
        })[7:]
        self.ledger_path = self.registry_root / f"{registry_key}.json"

    @classmethod
    def create(
        cls, authority: CoordinatorRegistryAuthority, plan: FinalEvaluationPlan
    ) -> "TestOnceGuard":
        guard = cls(authority, plan)
        payload = guard._initial_payload()
        with authority.locked(exclusive=True):
            try:
                descriptor = os.open(
                    guard.ledger_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except FileExistsError as error:
                raise EvaluationPlanError("test authorization already exists") from error
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            CoordinatorRegistryAuthority._fsync_directory(guard.registry_root)
        return guard

    @classmethod
    def open(
        cls, authority: CoordinatorRegistryAuthority, plan: FinalEvaluationPlan
    ) -> "TestOnceGuard":
        guard = cls(authority, plan)
        payload = guard._read()
        if payload.get("plan_sha256") != plan.plan_sha256:
            raise EvaluationPlanError("resume plan identity mismatch")
        return guard

    def _initial_payload(self) -> dict:
        return {
            "schema_version": 1,
            "protocol": "test-once-ledger-v1",
            "plan_sha256": self.plan.plan_sha256,
            "test_dataset_manifest_sha256": self.plan.test_dataset_manifest_sha256,
            "systems": {
                system_id: {
                    "state": "authorized_not_started",
                    "episodes": {},
                    "material": None,
                    "completion": None,
                }
                for system_id in SYSTEM_ORDER
            },
        }

    def _read(self) -> dict:
        with self.authority.locked(exclusive=False):
            return self._read_unlocked()

    def _read_unlocked(self) -> dict:
        payload = _load_canonical_object(self.ledger_path, "test-once ledger")
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "protocol", "plan_sha256", "test_dataset_manifest_sha256", "systems"}
            or payload.get("schema_version") != 1
            or payload.get("protocol") != "test-once-ledger-v1"
            or payload.get("plan_sha256") != self.plan.plan_sha256
            or payload.get("test_dataset_manifest_sha256") != self.plan.test_dataset_manifest_sha256
        ):
            raise EvaluationPlanError("invalid test-once ledger")
        systems = payload.get("systems")
        if not isinstance(systems, dict) or set(systems) != set(SYSTEM_ORDER):
            raise EvaluationPlanError("invalid test-once system ledger")
        valid_states = {"authorized_not_started", "running", "completed", "terminal_failure", "reconciliation_required"}
        planned_ids = {item.scenario_id for item in self.plan.scenarios}
        all_episode_ids: set[str] = set()
        for system in systems.values():
            if (
                not isinstance(system, dict)
                or set(system) != {"state", "episodes", "material", "completion"}
            ):
                raise EvaluationPlanError("invalid test-once system record")
            if system["state"] not in valid_states or not isinstance(system["episodes"], dict):
                raise EvaluationPlanError("invalid test-once state or episodes")
            if not set(system["episodes"]) <= planned_ids:
                raise EvaluationPlanError("unplanned scenario in test-once ledger")
            episode_ids = tuple(system["episodes"].values())
            if any(not isinstance(item, str) or not item or any(character.isspace() for character in item) for item in episode_ids):
                raise EvaluationPlanError("invalid episode identity in test-once ledger")
            if len(set(episode_ids)) != len(episode_ids) or all_episode_ids.intersection(episode_ids):
                raise EvaluationPlanError("episode identities must be globally unique")
            all_episode_ids.update(episode_ids)
            completion = system["completion"]
            material = system["material"]
            if material is not None:
                try:
                    material_reference = BlobReference.model_validate(
                        material, strict=True
                    )
                except Exception as error:
                    raise EvaluationPlanError(
                        "invalid staged system material reference"
                    ) from error
                if (
                    material_reference.schema_name != "StagedSystemMaterial"
                    or material_reference.schema_version != 1
                    or material_reference.media_type
                    != "application/vnd.toolsandbox.canonical+json"
                    or material_reference.model_dump(mode="json") != material
                ):
                    raise EvaluationPlanError(
                        "invalid staged system material reference"
                    )
            if completion is not None:
                try:
                    receipt = SystemCompletionReceipt.model_validate(
                        completion, strict=True
                    )
                except Exception as error:
                    raise EvaluationPlanError(
                        "invalid system completion receipt"
                    ) from error
                if receipt.model_dump(mode="json") != completion:
                    raise EvaluationPlanError("noncanonical system completion receipt")
            if (system["state"] == "completed") != (completion is not None):
                if not (system["state"] == "running" and completion is not None):
                    raise EvaluationPlanError("system completion state mismatch")
            if completion is not None and material is None:
                raise EvaluationPlanError("completion requires staged system material")
            if system["state"] not in {"running", "completed"} and material is not None:
                raise EvaluationPlanError("terminal or unstarted system has material")
        return payload

    def _write_unlocked(self, payload: dict) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".test-once-", dir=self.registry_root
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.ledger_path)
            CoordinatorRegistryAuthority._fsync_directory(self.registry_root)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def state(self, system_id: SystemId) -> TestLedgerState:
        return self._read()["systems"][system_id]["state"]

    def start_system(self, system_id: SystemId) -> None:
        with self.authority.locked(exclusive=True):
            payload = self._read_unlocked()
            index = SYSTEM_ORDER.index(system_id)
            if any(payload["systems"][prior]["state"] != "completed" for prior in SYSTEM_ORDER[:index]):
                raise EvaluationPlanError("systems must start in fixed order")
            if payload["systems"][system_id]["state"] != "authorized_not_started":
                raise EvaluationPlanError("system already started; use identical resume")
            payload["systems"][system_id]["state"] = "running"
            self._write_unlocked(payload)

    def authorize_episode(self, system_id: SystemId, scenario_id: str, episode_id: str) -> None:
        with self.authority.locked(exclusive=True):
            payload = self._read_unlocked()
            system = payload["systems"][system_id]
            if system["state"] != "running":
                raise EvaluationPlanError("episode dispatch requires running system")
            if scenario_id not in {item.scenario_id for item in self.plan.scenarios}:
                raise EvaluationPlanError("scenario is outside frozen test plan")
            existing = system["episodes"].get(scenario_id)
            if existing is not None and existing != episode_id:
                raise EvaluationPlanError("scenario already has a distinct episode identity")
            if existing is None:
                system["episodes"][scenario_id] = episode_id
                self._write_unlocked(payload)

    def assert_identical_resume(self, system_id: SystemId, plan_sha256: str) -> None:
        if plan_sha256 != self.plan.plan_sha256:
            raise EvaluationPlanError("changed plan cannot resume")
        if self.state(system_id) not in {"running", "reconciliation_required"}:
            raise EvaluationPlanError("system is not resumable")

    def completion_receipt(
        self, system_id: SystemId
    ) -> SystemCompletionReceipt | None:
        payload = self._read()["systems"][system_id]["completion"]
        return (
            None
            if payload is None
            else SystemCompletionReceipt.model_validate(payload, strict=True)
        )

    def staged_material_reference(self, system_id: SystemId) -> BlobReference | None:
        payload = self._read()["systems"][system_id]["material"]
        return (
            None
            if payload is None
            else BlobReference.model_validate(payload, strict=True)
        )

    def stage_system_material(
        self, system_id: SystemId, reference: BlobReference
    ) -> None:
        if (
            type(reference) is not BlobReference
            or reference.schema_name != "StagedSystemMaterial"
            or reference.schema_version != 1
            or reference.media_type
            != "application/vnd.toolsandbox.canonical+json"
        ):
            raise TypeError("StagedSystemMaterial BlobReference required")
        with self.authority.locked(exclusive=True):
            payload = self._read_unlocked()
            system = payload["systems"][system_id]
            if system["state"] != "running" or system["completion"] is not None:
                raise EvaluationPlanError("system material cannot be staged")
            if set(system["episodes"]) != {
                item.scenario_id for item in self.plan.scenarios
            }:
                raise EvaluationPlanError(
                    "cannot stage material with missing scenario identities"
                )
            encoded = reference.model_dump(mode="json")
            if system["material"] is None:
                system["material"] = encoded
                self._write_unlocked(payload)
            elif system["material"] != encoded:
                raise EvaluationPlanError("staged system material identity conflict")

    def stage_system_completion(
        self, system_id: SystemId, receipt: SystemCompletionReceipt
    ) -> None:
        if type(receipt) is not SystemCompletionReceipt:
            raise TypeError("SystemCompletionReceipt required")
        with self.authority.locked(exclusive=True):
            payload = self._read_unlocked()
            system = payload["systems"][system_id]
            if system["state"] != "running":
                raise EvaluationPlanError("only a running system can stage completion")
            if set(system["episodes"]) != {
                item.scenario_id for item in self.plan.scenarios
            }:
                raise EvaluationPlanError(
                    "cannot stage completion with missing scenario identities"
                )
            if system["material"] is None:
                raise EvaluationPlanError("system material was not durably staged")
            encoded = receipt.model_dump(mode="json")
            if system["completion"] is None:
                system["completion"] = encoded
                self._write_unlocked(payload)
            elif system["completion"] != encoded:
                raise EvaluationPlanError("system completion identity conflict")

    def complete_system(
        self, system_id: SystemId, receipt: SystemCompletionReceipt
    ) -> None:
        if type(receipt) is not SystemCompletionReceipt:
            raise TypeError("SystemCompletionReceipt required")
        with self.authority.locked(exclusive=True):
            payload = self._read_unlocked()
            system = payload["systems"][system_id]
            if system["state"] != "running":
                raise EvaluationPlanError("only a running system can complete")
            if set(system["episodes"]) != {item.scenario_id for item in self.plan.scenarios}:
                raise EvaluationPlanError("cannot complete with missing scenario identities")
            if system["completion"] != receipt.model_dump(mode="json"):
                raise EvaluationPlanError("system completion was not durably staged")
            system["state"] = "completed"
            self._write_unlocked(payload)

    def mark_terminal_failure(self, system_id: SystemId) -> None:
        self._transition_from_running(system_id, "terminal_failure")

    def mark_reconciliation_required(self, system_id: SystemId) -> None:
        if self.plan.profile != "official_live":
            raise EvaluationPlanError("reconciliation state is official-live only")
        self._transition_from_running(system_id, "reconciliation_required")

    def _transition_from_running(self, system_id: SystemId, target: TestLedgerState) -> None:
        with self.authority.locked(exclusive=True):
            payload = self._read_unlocked()
            if payload["systems"][system_id]["state"] != "running":
                raise EvaluationPlanError("invalid test-once state transition")
            if (
                payload["systems"][system_id]["completion"] is not None
                or payload["systems"][system_id]["material"] is not None
            ):
                raise EvaluationPlanError("sealed system material must resume")
            payload["systems"][system_id]["state"] = target
            self._write_unlocked(payload)


__all__ = [
    "CoordinatorRegistryAuthority", "EvaluationPlanError", "SystemCompletionReceipt",
    "TestLedgerState", "TestOnceGuard",
    "load_final_plan", "validate_output_root", "write_frozen_plan",
]
