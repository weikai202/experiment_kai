"""Atomic round material checkpoints and two-stage durable completion receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .canonical import atomic_write_json, canonical_sha256, read_json


@dataclass(frozen=True)
class RoundMaterial:
    run_id: str
    round_index: int
    input_generation_id: str
    output_generation_id: str
    completed_task_keys: tuple[str, ...]
    committed_effect_ids: tuple[str, ...]
    accounting_ledger_path: str
    accounting_ledger_sha256: str
    artifact_hashes: tuple[str, ...]
    started_at_unix_ns: int

    started_monotonic_ns: int | None = None
    started_boot_id: str | None = None


class RoundCheckpointStore:
    def __init__(
        self,
        root: str | Path,
        *,
        receipt_boundary_ns: Callable[[Path], int] | None = None,
    ):
        self.root = Path(root)
        self.receipt_boundary_ns = receipt_boundary_ns or (lambda path: path.stat().st_mtime_ns)

    def _material_path(self, run_id: str, round_index: int) -> Path:
        return self.root / run_id / f"round_{round_index}.material.json"

    def _receipt_path(self, run_id: str, round_index: int) -> Path:
        return self.root / run_id / f"round_{round_index}.receipt.json"

    def stage_material(self, material: RoundMaterial) -> str:
        if material.round_index not in (0, 1, 2):
            raise ValueError("invalid round index")
        if type(material.started_at_unix_ns) is not int or material.started_at_unix_ns <= 0:
            raise ValueError("round material requires a positive durable start timestamp")
        payload = asdict(material)
        digest = canonical_sha256(payload)
        path = self._material_path(material.run_id, material.round_index)
        if path.exists():
            if canonical_sha256(read_json(path)) != digest:
                raise ValueError("conflicting round material checkpoint")
            return digest
        atomic_write_json(path, payload)
        return digest

    def mark_material_complete(
        self,
        material: RoundMaterial,
        *,
        material_sha256: str,
        accounting: dict,
        material_completed_at_unix_ns: int,
    ) -> None:
        if (
            type(material_completed_at_unix_ns) is not int
            or material_completed_at_unix_ns < material.started_at_unix_ns
        ):
            raise ValueError("invalid durable material completion timestamp")
        if self.stage_material(material) != material_sha256:
            raise ValueError("material hash mismatch")
        receipt = {
            "run_id": material.run_id,
            "round_index": material.round_index,
            "material_sha256": material_sha256,
            "accounting": accounting,
            "material_completed_at_unix_ns": material_completed_at_unix_ns,
            "status": "material_complete",
        }
        path = self._receipt_path(material.run_id, material.round_index)
        if path.exists():
            existing = read_json(path)
            if existing.get("status") == "complete":
                return
            if existing != receipt:
                raise ValueError("conflicting round completion marker")
            return
        atomic_write_json(path, receipt)

    def material_complete_boundary_ns(self, run_id: str, round_index: int) -> int:
        path = self._receipt_path(run_id, round_index)
        receipt = read_json(path)
        if receipt.get("status") not in {"material_complete", "complete"}:
            raise ValueError("durable material receipt required for its commit boundary")
        boundary = self.receipt_boundary_ns(path)
        if type(boundary) is not int or boundary <= 0:
            raise ValueError("invalid post-write material receipt boundary")
        return boundary

    def bind_timing(
        self, material: RoundMaterial, *, material_sha256: str, accounting: dict
    ) -> None:
        path = self._receipt_path(material.run_id, material.round_index)
        existing = read_json(path)
        if existing.get("material_sha256") != material_sha256 or existing.get("status") not in {
            "material_complete",
            "complete",
        }:
            raise ValueError("durable material completion marker required before timing")
        receipt = {
            "run_id": material.run_id,
            "round_index": material.round_index,
            "material_sha256": material_sha256,
            "accounting": accounting,
            "material_completed_at_unix_ns": existing["material_completed_at_unix_ns"],
            "status": "complete",
        }
        if existing.get("status") == "complete" and existing != receipt:
            raise ValueError("conflicting timing-bound receipt")
        atomic_write_json(path, receipt)

    def load(self, run_id: str, round_index: int) -> tuple[dict | None, dict | None]:
        material_path = self._material_path(run_id, round_index)
        receipt_path = self._receipt_path(run_id, round_index)
        material = read_json(material_path) if material_path.exists() else None
        receipt = read_json(receipt_path) if receipt_path.exists() else None
        if receipt is not None:
            if material is None or receipt["material_sha256"] != canonical_sha256(material):
                raise ValueError("orphaned or mismatched completion receipt")
            if receipt.get("status") not in {"material_complete", "complete"}:
                raise ValueError("unknown completion receipt status")
            completed_ns = receipt.get("material_completed_at_unix_ns")
            if type(completed_ns) is not int or completed_ns < material["started_at_unix_ns"]:
                raise ValueError("invalid durable timing receipt")
        return material, receipt
