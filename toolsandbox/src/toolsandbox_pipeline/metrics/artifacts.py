"""Atomic deterministic materialization of immutable accounting views."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.accounting import (
    RequestMetricRecord,
    RoundMetricRecord,
    RunMetricRecord,
    TaskMetricRecord,
)


_FILES = {
    "request": "request_metrics.jsonl",
    "task": "task_metrics.jsonl",
    "round": "round_metrics.jsonl",
    "run": "run_metrics.json",
    "manifest": "manifest.json",
}


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _jsonl(records: Sequence[Any]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        for record in records
    )


class MetricsArtifactWriter:
    """Write views under one pre-authorized absolute run root."""

    def __init__(
        self,
        run_root: Path | str,
        *,
        crash_hook: Callable[[str, Path], None] | None = None,
    ) -> None:
        root = Path(run_root)
        if not root.is_absolute():
            raise ValueError("run root must be absolute")
        if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
            raise ValueError("symlinked run root is forbidden")
        if not root.exists() or not root.is_dir():
            raise ValueError("authorized run root must already exist")
        self.run_root = root.resolve(strict=True)
        self.metrics_root = self.run_root / "metrics"
        self._crash_hook = crash_hook or (lambda _point, _path: None)

    def materialize(
        self,
        *,
        run_id: str,
        request_records: Sequence[RequestMetricRecord],
        task_records: Sequence[TaskMetricRecord],
        round_records: Sequence[RoundMetricRecord],
        run_record: RunMetricRecord | None,
        ledger_high_water_marks: Mapping[str, int],
        pinned_qwen_provider: str,
        pinned_qwen_model: str,
    ) -> Mapping[str, str]:
        if not run_id or not pinned_qwen_provider or not pinned_qwen_model:
            raise ValueError("run and pinned Qwen identities are required")
        if any(type(value) is not int or value < 0 for value in ledger_high_water_marks.values()):
            raise ValueError("invalid ledger high-water mark")
        requests = tuple(sorted(request_records, key=lambda row: row.attempt.attempt_id))
        tasks = tuple(task_records)
        rounds = tuple(sorted(round_records, key=lambda row: row.round.round_index))
        self._validate_records(run_id, requests, tasks, rounds, run_record)
        self._prepare_directory()

        payloads: dict[str, bytes] = {
            _FILES["request"]: _jsonl(requests),
            _FILES["task"]: _jsonl(tasks),
            _FILES["round"]: _jsonl(rounds),
        }
        if run_record is not None:
            payloads[_FILES["run"]] = canonical_json_bytes(run_record.model_dump(mode="json"))
        elif (self.metrics_root / _FILES["run"]).exists():
            raise ValueError("closed run metrics cannot be withdrawn")
        for filename, payload in payloads.items():
            self._verify_prefix(self.metrics_root / filename, payload, filename.endswith(".jsonl"))
            self._atomic_write(self.metrics_root / filename, payload)

        file_entries = {
            name: {
                "sha256": _sha256_bytes(payload),
                "record_count": payload.count(b"\n") if name.endswith(".jsonl") else 1,
            }
            for name, payload in sorted(payloads.items())
        }
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "files": file_entries,
            "ledger_high_water_marks": dict(sorted(ledger_high_water_marks.items())),
            "record_schema_versions": {
                "request": 1, "task": 1, "round": 1, "run": 1,
            },
            "cost_unit": "qwen_effective_output_tokens",
            "pinned_qwen_provider": pinned_qwen_provider,
            "pinned_qwen_model": pinned_qwen_model,
        }
        manifest_payload = canonical_json_bytes(manifest)
        manifest_path = self.metrics_root / _FILES["manifest"]
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_bytes())
            if previous.get("run_id") != run_id:
                raise ValueError("manifest run identity conflict")
            old_marks = previous.get("ledger_high_water_marks", {})
            if any(
                key not in ledger_high_water_marks or ledger_high_water_marks[key] < value
                for key, value in old_marks.items()
            ):
                raise ValueError("decreasing ledger high-water mark")
            if (
                previous.get("pinned_qwen_provider") != pinned_qwen_provider
                or previous.get("pinned_qwen_model") != pinned_qwen_model
            ):
                raise ValueError("pinned Qwen identity conflict")
        if manifest_path.exists() and manifest_path.read_bytes() == manifest_payload:
            pass
        else:
            self._atomic_write(manifest_path, manifest_payload)
        return {name: entry["sha256"] for name, entry in file_entries.items()}

    def _prepare_directory(self) -> None:
        if self.metrics_root.exists():
            if self.metrics_root.is_symlink() or not self.metrics_root.is_dir():
                raise ValueError("invalid metrics directory")
            allowed = set(_FILES.values())
            unexpected = [path.name for path in self.metrics_root.iterdir() if path.name not in allowed]
            if unexpected:
                raise ValueError("unexpected metrics file")
        else:
            self.metrics_root.mkdir(mode=0o700)
            os.chmod(self.metrics_root, 0o700)

    @staticmethod
    def _validate_records(run_id, requests, tasks, rounds, run_record) -> None:
        all_records = (*requests, *tasks, *rounds, *((run_record,) if run_record else ()))
        if any(record.run_id != run_id for record in all_records):
            raise ValueError("wrong run identity")
        for rows, name in ((requests, "request"), (tasks, "task"), (rounds, "round")):
            ids = [row.record_id for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {name} record ID")
        indices = [row.round.round_index for row in rounds]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate round index")
        if run_record is not None:
            if not run_record.run.timing.timing_complete:
                raise ValueError("run metrics unavailable until scope close")
            if run_record.run.run_kind == "training" and indices != [0, 1, 2]:
                raise ValueError("formal training requires round records 0, 1, and 2")

    @staticmethod
    def _verify_prefix(path: Path, new_payload: bytes, jsonl: bool) -> None:
        if not path.exists():
            return
        if path.is_symlink() or not path.is_file():
            raise ValueError("invalid existing metric view")
        old = path.read_bytes()
        if jsonl:
            if old and not old.endswith(b"\n"):
                raise ValueError("invalid existing JSONL")
            if not new_payload.startswith(old):
                raise ValueError("existing metric view is not an exact prefix")
            for line in old.splitlines():
                parsed = json.loads(line)
                if canonical_json_bytes(parsed) != line:
                    raise ValueError("existing metric record is not canonical")
        elif old != new_payload:
            raise ValueError("immutable JSON record conflict")

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        if path.exists() and path.read_bytes() == payload:
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                raise ValueError("metric file mode mismatch")
            return
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._crash_hook("before_replace", path)
            os.replace(temp, path)
            self._crash_hook("after_replace", path)
            directory_fd = os.open(self.metrics_root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp.exists():
                temp.unlink()


__all__ = ["MetricsArtifactWriter"]
