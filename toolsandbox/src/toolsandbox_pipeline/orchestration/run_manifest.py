"""Canonical, non-secret Task 017 run-manifest validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest, TrainSmokeConfig


class RunManifestError(ValueError):
    """The requested run is not exactly the reviewed immutable protocol."""


def schema_bytes() -> bytes:
    return canonical_json_bytes(ResolvedRunManifest.model_json_schema(mode="validation"))


def manifest_bytes(manifest: ResolvedRunManifest) -> bytes:
    return canonical_json_bytes(manifest.model_dump(mode="json"))


def load_run_manifest(
    path: Path | str,
    *,
    run_root_state: str = "new",
) -> ResolvedRunManifest:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise RunManifestError("manifest must be an absolute non-symlink file")
    raw = source.read_bytes()
    try:
        manifest = ResolvedRunManifest.model_validate_json(raw, strict=True)
    except Exception as error:
        raise RunManifestError("invalid resolved run manifest") from error
    if raw != manifest_bytes(manifest):
        raise RunManifestError("resolved run manifest must use canonical JSON bytes")
    if run_root_state not in {"new", "existing", "ignore"}:
        raise RunManifestError("unknown run-root state")
    exists = Path(manifest.run_root).exists()
    if run_root_state == "new" and exists:
        raise RunManifestError("new run root must not already exist")
    if run_root_state == "existing" and not exists:
        raise RunManifestError("existing run root is required")
    return manifest


def verify_schema_file(path: Path | str) -> None:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise RunManifestError("schema must be an absolute non-symlink file")
    try:
        value: Any = json.loads(source.read_bytes())
    except Exception as error:
        raise RunManifestError("invalid JSON Schema file") from error
    if canonical_json_bytes(value) != schema_bytes():
        raise RunManifestError("JSON Schema does not match ResolvedRunManifest")


def validate_component_files(manifest: ResolvedRunManifest) -> None:
    """Verify pinned local bytes without opening datasets or looking up credentials."""

    from hashlib import sha256

    pairs = (
        (manifest.dataset_manifest_path, manifest.dataset_manifest_sha256),
        (manifest.seeds.policy_memory_path, manifest.seeds.policy_memory_sha256),
        (manifest.seeds.world_memory_path, manifest.seeds.world_memory_sha256),
        (manifest.seeds.skills_path, manifest.seeds.skills_sha256),
        (manifest.seeds.source_manifest_path, manifest.seeds.source_manifest_sha256),
    )
    for raw_path, expected in pairs:
        path = Path(raw_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise RunManifestError("pinned component must be an absolute regular file")
        actual = "sha256:" + sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RunManifestError("pinned component hash mismatch")


def load_train_smoke_config(path: Path | str) -> tuple[TrainSmokeConfig, str]:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise RunManifestError("smoke config must be an absolute non-symlink file")
    raw = source.read_bytes()
    try:
        config = TrainSmokeConfig.model_validate_json(raw, strict=True)
    except Exception as error:
        raise RunManifestError("invalid immutable train-smoke config") from error
    from hashlib import sha256

    return config, "sha256:" + sha256(raw).hexdigest()


def validate_train_smoke(manifest: ResolvedRunManifest, config: TrainSmokeConfig) -> None:
    if type(manifest) is not ResolvedRunManifest or type(config) is not TrainSmokeConfig:
        raise TypeError("strict manifest and smoke config required")
    if (
        manifest.purpose != "train_smoke"
        or manifest.profile != config.profile
        or manifest.fixture.mode != config.external_read_mode
        or manifest.qwen.model != config.qwen_model
        or manifest.qwen.enable_thinking != config.qwen_enable_thinking
        or manifest.embedding.model != config.embedding_model
        or manifest.user_simulator.model != config.user_simulator_model
        or not Path(manifest.run_root).as_posix().endswith(
            "/" + config.artifact_namespace + "/" + manifest.run_id
        )
    ):
        raise RunManifestError("train-smoke manifest/config mismatch")


__all__ = [
    "RunManifestError", "load_run_manifest", "load_train_smoke_config",
    "manifest_bytes", "schema_bytes", "validate_component_files",
    "validate_train_smoke", "verify_schema_file",
]
