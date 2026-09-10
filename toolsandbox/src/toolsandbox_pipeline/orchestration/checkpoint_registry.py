"""Build, serialize, and query the non-selecting checkpoint registry."""

from __future__ import annotations

from pathlib import Path

from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.run import (
    CheckpointObservationRegistry,
    DiagnosticCheckpointPointer,
    GenerationObservation,
)


def build_checkpoint_registry(
    run_id: str,
    entries: tuple[
        GenerationObservation, GenerationObservation,
        GenerationObservation, GenerationObservation,
    ],
) -> CheckpointObservationRegistry:
    produced = entries[1:]

    def pointer(metric: str) -> DiagnosticCheckpointPointer:
        selected = max(
            produced,
            key=lambda row: (getattr(row, metric), -int(row.generation_id[1:])),
        )
        return DiagnosticCheckpointPointer(
            metric=metric,
            generation_id=selected.generation_id,
            observed_value=getattr(selected, metric),
        )

    values = {
        "schema_version": 1,
        "run_id": run_id,
        "entries": entries,
        "updated_generation_id": "g003",
        "highest_observed_train_mean_similarity": pointer("mean_native_similarity"),
        "highest_observed_train_fully_successful_rate": pointer("fully_successful_rate"),
        "selection_allowed": False,
        "comparability": "different_train_shards_not_directly_comparable",
    }
    payload = {
        key: value.model_dump(mode="json") if hasattr(value, "model_dump") else (
            [item.model_dump(mode="json") for item in value]
            if key == "entries" else value
        )
        for key, value in values.items()
    }
    digest = canonical_sha256(["checkpoint-observation-registry-v1", payload])
    return CheckpointObservationRegistry(**values, registry_sha256=digest)


def registry_bytes(registry: CheckpointObservationRegistry) -> bytes:
    return canonical_json_bytes(registry.model_dump(mode="json"))


def query_checkpoint_registry(path: Path | str) -> CheckpointObservationRegistry:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise ValueError("registry path must be an absolute non-symlink file")
    raw = source.read_bytes()
    registry = CheckpointObservationRegistry.model_validate_json(raw, strict=True)
    if raw != registry_bytes(registry):
        raise ValueError("registry must use canonical JSON bytes")
    return registry


__all__ = ["build_checkpoint_registry", "query_checkpoint_registry", "registry_bytes"]
