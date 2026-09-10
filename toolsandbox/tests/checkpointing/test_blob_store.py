import json
import os
from pathlib import Path

import pytest

from toolsandbox_pipeline.checkpointing import (
    CheckpointStore,
    CheckpointStoreError,
    RunIdentity,
)
from toolsandbox_pipeline.checkpointing.blob_store import BlobStoreError


CONFIG = Path(__file__).parents[2] / "configs/reproducibility/checkpointing_v1.json"


def digest(character):
    return "sha256:" + character * 64


def identity():
    return RunIdentity(
        run_id="run-security",
        profile="offline",
        environment_identity=digest("1"),
        dataset_manifest_sha256=digest("2"),
        config_manifest_sha256=digest("3"),
        prompt_manifest_sha256=digest("4"),
        generation_manifest_sha256=digest("5"),
        fixture_manifest_sha256=digest("6"),
    )


def test_blob_reuse_hash_mode_and_tamper_detection(tmp_path):
    store = CheckpointStore.create(tmp_path / "run", identity(), CONFIG.resolve())
    try:
        data = b'{"a":1}'
        first = store.blobs.put(
            data,
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="fixture",
            schema_version=1,
        )
        second = store.blobs.put(
            data,
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="fixture",
            schema_version=1,
        )
        assert first == second and store.blobs.read(first) == data
        path = (
            tmp_path
            / "run/checkpointing/blobs/sha256"
            / first.sha256[7:9]
            / first.sha256[9:]
        )
        assert os.stat(path).st_mode & 0o777 == 0o600
        os.chmod(path, 0o644)
        with pytest.raises(BlobStoreError, match="mode"):
            store.blobs.read(first)
    finally:
        store.close()


def test_noncanonical_json_and_unknown_config_fields_fail(tmp_path):
    store = CheckpointStore.create(tmp_path / "run", identity(), CONFIG.resolve())
    try:
        with pytest.raises(BlobStoreError, match="noncanonical"):
            store.blobs.put(
                b'{"b": 2, "a": 1}',
                media_type="application/vnd.toolsandbox.canonical+json",
                schema_name="fixture",
                schema_version=1,
            )
    finally:
        store.close()
    payload = json.loads(CONFIG.read_text())
    payload["unknown"] = True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(Exception):
        CheckpointStore.create(tmp_path / "other", identity(), bad.resolve())


def test_relative_symlink_and_partial_schema_fail_closed(tmp_path):
    with pytest.raises(CheckpointStoreError, match="absolute"):
        CheckpointStore.create(Path("relative"), identity(), CONFIG.resolve())
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(CheckpointStoreError, match="symlink"):
        CheckpointStore.create(link, identity(), CONFIG.resolve())

    root = tmp_path / "run"
    store = CheckpointStore.create(root, identity(), CONFIG.resolve())
    store._connection.execute("DROP TABLE tool_execution_attempts")
    store.close()
    with pytest.raises(CheckpointStoreError, match="partial"):
        CheckpointStore.open(root, identity(), CONFIG.resolve())
