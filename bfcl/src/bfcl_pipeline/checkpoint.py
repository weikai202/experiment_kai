from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, sha256_json


@dataclass(frozen=True)
class RoundCheckpoint:
    run_id: str
    round_index: int
    stage: str
    generation_id: str
    completed_case_ids: tuple[str, ...]
    payload_sha256: str


class CheckpointStore:
    """Atomic, hash-verified round checkpoints; callers resume, never redispatch."""

    def __init__(self, root: Path):
        self.root = root

    def persist(self, run_id: str, round_index: int, stage: str, generation_id: str, completed_case_ids: tuple[str, ...], material: dict[str, Any]) -> RoundCheckpoint:
        payload = {
            "run_id": run_id,
            "round_index": round_index,
            "stage": stage,
            "generation_id": generation_id,
            "completed_case_ids": completed_case_ids,
            "material": material,
        }
        digest = sha256_json(payload)
        envelope = {"payload": payload, "payload_sha256": digest}
        directory = self.root / run_id / f"round-{round_index}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{stage}.json"
        temporary = directory / f".{stage}.{os.getpid()}.tmp"
        with temporary.open("wb") as stream:
            stream.write(canonical_bytes(envelope) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return RoundCheckpoint(run_id, round_index, stage, generation_id, completed_case_ids, digest)

    def load(self, run_id: str, round_index: int, stage: str) -> dict[str, Any] | None:
        import json

        target = self.root / run_id / f"round-{round_index}" / f"{stage}.json"
        if not target.exists():
            return None
        envelope = json.loads(target.read_text(encoding="utf-8"))
        if sha256_json(envelope["payload"]) != envelope.get("payload_sha256"):
            raise ValueError("Checkpoint hash mismatch")
        return {**envelope["payload"], "payload_sha256": envelope["payload_sha256"]}

    def recovery_stage(self, run_id: str, round_index: int) -> str:
        for stage in ("accounting_finalized", "complete", "durable_completion", "generation_published", "effects_committed", "episodes_progress", "case_dispatch", "round_started"):
            if self.load(run_id, round_index, stage) is not None:
                return stage
        return "not_started"
