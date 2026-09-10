import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.orchestration.run_manifest import (
    RunManifestError,
    load_run_manifest,
    manifest_bytes,
    schema_bytes,
    validate_component_files,
    verify_schema_file,
)
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest


HASH = "sha256:" + "a" * 64


def _sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_payload(tmp_path, *, purpose="train_smoke"):
    component = tmp_path / "component.json"
    component.write_bytes(b"{}")
    value_hash = _sha(component)
    run_root = tmp_path / "runs" / ("smoke/case" if purpose == "train_smoke" else "formal-case")
    return {
        "run_id": "case",
        "purpose": purpose,
        "profile": "strict_replay",
        "dataset_manifest_path": str(component),
        "dataset_manifest_sha256": value_hash,
        "ordered_train_shard_ids": ("train-shard-0", "train-shard-1", "train-shard-2"),
        "upstream_source_sha256": HASH,
        "dependency_lock_sha256": HASH,
        "container_image_digest": HASH,
        "environment_sha256": HASH,
        "python_patch_version": "3.10.16",
        "scenario_registry_sha256": HASH,
        "tool_inventory_sha256": HASH,
        "evaluator_registry_sha256": HASH,
        "schema_bundle_sha256": HASH,
        "qwen": {
            "endpoint_identity_sha256": HASH,
            "container_digest": HASH,
            "server_configuration_sha256": HASH,
            "decoding_configuration_sha256": HASH,
            "structured_output_wire_mode": "guided_json",
        },
        "embedding": {"client_configuration_sha256": HASH, "expected_dimension": 1536},
        "user_simulator": {
            "profile_sha256": HASH,
            "prompt_sha256": HASH,
            "few_shot_sha256": HASH,
            "tool_schema_sha256": HASH,
            "stop_configuration_sha256": HASH,
        },
        "online_prompt_manifest_sha256": HASH,
        "offline_prompt_manifest_sha256": HASH,
        "online_token_limits_sha256": HASH,
        "offline_token_limits_sha256": HASH,
        "calibrated_token_limits": True,
        "seeds": {
            "policy_memory_path": str(component),
            "policy_memory_sha256": value_hash,
            "world_memory_path": str(component),
            "world_memory_sha256": value_hash,
            "skills_path": str(component),
            "skills_sha256": value_hash,
            "source_manifest_path": str(component),
            "source_manifest_sha256": value_hash,
            "public_tool_schema_inventory_sha256": HASH,
            "skill_generator_version": "public-schema-skill-v1",
            "skill_generator_sha256": HASH,
            "skill_provenance_sha256": HASH,
        },
        "fixture": {
            "mode": "fixture",
            "manifest_sha256": HASH,
            "backend_configuration_sha256": HASH,
        },
        "checkpoint": {
            "schema_version": 1,
            "checkpoint_protocol_version": "checkpoint-v1",
            "hash_algorithm": "sha256",
            "database": "sqlite3",
            "journal_mode": "DELETE",
            "synchronous": "FULL",
            "foreign_keys": True,
            "busy_timeout_ms": 0,
            "single_writer": True,
            "file_mode": "0600",
            "directory_mode": "0700",
            "max_blob_bytes": 268435456,
            "max_checkpoint_bytes": 1073741824,
        },
        "checkpoint_registry_schema_sha256": HASH,
        "metrics_schema_sha256": HASH,
        "run_root": str(run_root),
    }


def test_manifest_is_frozen_strict_canonical_and_hash_stable(tmp_path):
    payload = manifest_payload(tmp_path)
    manifest = ResolvedRunManifest.model_validate(payload, strict=True)
    assert manifest.manifest_sha256 == manifest.manifest_sha256
    path = tmp_path / "manifest.json"
    path.write_bytes(manifest_bytes(manifest))
    assert load_run_manifest(path) == manifest
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RunManifestError, match="canonical"):
        load_run_manifest(path)
    with pytest.raises(ValidationError):
        ResolvedRunManifest.model_validate({**payload, "test_score": 1}, strict=True)


def test_manifest_denies_raw_endpoint_secret_provisional_and_protocol_drift(tmp_path):
    payload = manifest_payload(tmp_path)
    payload["qwen"]["endpoint_identity_sha256"] = "https://example.invalid/v1"
    with pytest.raises(ValidationError):
        ResolvedRunManifest.model_validate(payload, strict=True)
    payload = manifest_payload(tmp_path)
    payload["calibrated_token_limits"] = False
    with pytest.raises(ValidationError):
        ResolvedRunManifest.model_validate(payload, strict=True)
    payload = manifest_payload(tmp_path)
    payload["ordered_train_shard_ids"] = ("train-shard-1", "train-shard-0", "train-shard-2")
    with pytest.raises(ValidationError, match="three-shard"):
        ResolvedRunManifest.model_validate(payload, strict=True)


def test_component_hashes_and_schema_are_exact(tmp_path):
    manifest = ResolvedRunManifest.model_validate(manifest_payload(tmp_path), strict=True)
    validate_component_files(manifest)
    Path(manifest.dataset_manifest_path).write_bytes(b"changed")
    with pytest.raises(RunManifestError, match="hash mismatch"):
        validate_component_files(manifest)
    generated = json.loads(schema_bytes())
    assert generated["additionalProperties"] is False
    assert generated["properties"]["test_access_policy"]["const"] == "forbidden"
    schema_path = Path(__file__).parents[2] / "configs/run/run_manifest.schema.json"
    verify_schema_file(schema_path.resolve())
    assert schema_path.read_bytes().rstrip(b"\n") == schema_bytes()


def test_relative_and_symlink_paths_fail(tmp_path):
    payload = manifest_payload(tmp_path)
    payload["run_root"] = "runs/smoke/case"
    with pytest.raises(ValidationError, match="absolute"):
        ResolvedRunManifest.model_validate(payload, strict=True)
    target = tmp_path / "target"
    target.write_bytes(b"{}")
    link = tmp_path / "link"
    link.symlink_to(target)
    payload = manifest_payload(tmp_path)
    payload["dataset_manifest_path"] = str(link)
    with pytest.raises(ValidationError, match="non-symlinks"):
        ResolvedRunManifest.model_validate(payload, strict=True)
