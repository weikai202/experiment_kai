from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_bytes, sha256_json
from .dataset import BFCL_VARIANTS
from .splits import SplitManifest

TRAIN_PURPOSES = frozenset({"evolution", "train_smoke", "calibration"})
DEV_PURPOSE = "skill_ab_validation"
FINAL_PURPOSE = "frozen_final_evaluation"


@dataclass(frozen=True)
class AccessReceipt:
    split: str
    purpose: str
    manifest_sha256: str
    case_ids: tuple[str, ...]
    receipt_sha256: str


class FinalTestAuthority:
    """Plan-bound lease that only the original owner may resume or complete."""

    def __init__(
        self,
        marker: Path,
        approved_plan_sha256: str,
        approved_manifest_sha256: str,
        owner_id: str,
    ):
        if not owner_id:
            raise ValueError("Final-test authority requires a stable owner ID")
        self.marker = marker
        self.approved_plan_sha256 = approved_plan_sha256
        self.approved_manifest_sha256 = approved_manifest_sha256
        self.owner_id = owner_id

    def _read(self) -> dict[str, Any]:
        envelope = json.loads(self.marker.read_text(encoding="utf-8"))
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or envelope.get("payload_sha256") != sha256_json(payload):
            raise PermissionError("Final-test authority marker is corrupted")
        return payload

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        envelope = {"payload": payload, "payload_sha256": sha256_json(payload)}
        temporary = self.marker.with_name(f".{self.marker.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            stream.write(canonical_bytes(envelope) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.marker)
        directory_fd = os.open(self.marker.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def acquire(self, plan_sha256: str, manifest_sha256: str) -> None:
        if plan_sha256 != self.approved_plan_sha256 or manifest_sha256 != self.approved_manifest_sha256:
            raise PermissionError("Final-test plan or dataset manifest is not approved")
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "owner_id": self.owner_id,
            "status": "active",
            "final_report_sha256": None,
        }
        envelope = {"payload": payload, "payload_sha256": sha256_json(payload)}
        try:
            fd = os.open(self.marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError:
            existing = self._read()
            identity = (existing.get("plan_sha256"), existing.get("manifest_sha256"), existing.get("owner_id"))
            if identity != (plan_sha256, manifest_sha256, self.owner_id):
                raise PermissionError("Final-test authority belongs to another plan or owner")
            if existing.get("status") != "active":
                raise PermissionError("Final-test authority is already completed")
            return
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_bytes(envelope) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(self.marker.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def assert_active(self, plan_sha256: str, manifest_sha256: str) -> None:
        if not self.marker.exists():
            raise PermissionError("Final-test authority has not been acquired")
        payload = self._read()
        if (
            payload.get("plan_sha256"),
            payload.get("manifest_sha256"),
            payload.get("owner_id"),
            payload.get("status"),
        ) != (plan_sha256, manifest_sha256, self.owner_id, "active"):
            raise PermissionError("Final-test authority is not active for this owner and plan")

    def status(self, plan_sha256: str, manifest_sha256: str) -> str:
        if not self.marker.exists():
            raise PermissionError("Final-test authority has not been acquired")
        payload = self._read()
        if (
            payload.get("plan_sha256"),
            payload.get("manifest_sha256"),
            payload.get("owner_id"),
        ) != (plan_sha256, manifest_sha256, self.owner_id):
            raise PermissionError("Final-test authority belongs to another plan or owner")
        status = payload.get("status")
        if status not in {"active", "complete"}:
            raise PermissionError("Final-test authority status is invalid")
        return status

    def completed_report_sha256(self, plan_sha256: str, manifest_sha256: str) -> str:
        if self.status(plan_sha256, manifest_sha256) != "complete":
            raise PermissionError("Final-test authority is not complete")
        payload = self._read()
        digest = payload.get("final_report_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise PermissionError("Final-test completion report hash is invalid")
        return digest

    def complete(self, final_report_sha256: str) -> None:
        self.assert_active(self.approved_plan_sha256, self.approved_manifest_sha256)
        if len(final_report_sha256) != 64:
            raise ValueError("Final report hash is invalid")
        self._write_atomic({
            "plan_sha256": self.approved_plan_sha256,
            "manifest_sha256": self.approved_manifest_sha256,
            "owner_id": self.owner_id,
            "status": "complete",
            "final_report_sha256": final_report_sha256,
        })


class DatasetGate:
    def __init__(self, manifest: SplitManifest, case_lookup: Mapping[str, Any]):
        self.manifest = manifest
        self.case_lookup = case_lookup

    def _ids(self, split: str) -> tuple[str, ...]:
        family_ids = {
            "train": self.manifest.train_family_ids,
            "dev": self.manifest.dev_family_ids,
            "test": self.manifest.sealed_test_family_ids,
        }.get(split)
        if family_ids is None:
            raise ValueError("Unknown split")
        return tuple(
            f"multi_turn_{variant}_{int(family_id.rsplit('_', 1)[1])}"
            for family_id in family_ids
            for variant in BFCL_VARIANTS
        )

    def open(self, split: str, purpose: str, *, authority: FinalTestAuthority | None = None, plan_sha256: str | None = None) -> tuple[tuple[Any, ...], AccessReceipt]:
        if split == "train" and purpose not in TRAIN_PURPOSES:
            raise PermissionError("Train access purpose is not allowed")
        if split == "train" and purpose == "evolution":
            raise PermissionError("Evolution must open exactly one train round")
        if split == "dev" and purpose != DEV_PURPOSE:
            raise PermissionError("Dev is restricted to Skill A/B validation")
        ids = self._ids(split)
        missing = [case_id for case_id in ids if case_id not in self.case_lookup]
        if missing:
            raise ValueError("Dataset lookup is incomplete")
        if split == "test":
            if purpose != FINAL_PURPOSE or authority is None or plan_sha256 is None:
                raise PermissionError("Test is sealed behind the final-test authority")
            authority.acquire(plan_sha256, self.manifest.manifest_sha256)
        core = {"split": split, "purpose": purpose, "manifest_sha256": self.manifest.manifest_sha256, "case_ids": ids}
        receipt = AccessReceipt(**core, receipt_sha256=sha256_json(core))
        return tuple(self.case_lookup[x] for x in ids), receipt

    def open_train_round(self, round_index: int, purpose: str = "evolution") -> tuple[tuple[Any, ...], AccessReceipt]:
        if purpose != "evolution" or round_index not in (0, 1, 2):
            raise PermissionError("Train round access requires evolution and round 0..2")
        ids = self.manifest.rounds[round_index].case_ids
        if any(case_id not in self.case_lookup for case_id in ids):
            raise ValueError("Dataset lookup is incomplete")
        core = {"split": f"train_round_{round_index}", "purpose": purpose, "manifest_sha256": self.manifest.manifest_sha256, "case_ids": ids}
        receipt = AccessReceipt(**core, receipt_sha256=sha256_json(core))
        return tuple(self.case_lookup[x] for x in ids), receipt


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        case_id = row.get("id")
        if not isinstance(case_id, str) or case_id in rows:
            raise ValueError("Dataset contains an invalid or duplicate ID")
        rows[case_id] = row
    return rows
