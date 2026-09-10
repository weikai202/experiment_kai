import json
import os
from pathlib import Path

from toolsandbox_pipeline.checkpointing import (
    CheckpointStore,
    LLMLedger,
    RunIdentity,
)


CONFIG = Path(__file__).parents[2] / "configs/reproducibility/checkpointing_v1.json"


def digest(character):
    return "sha256:" + character * 64


def test_derived_snapshot_and_latest_pointer_repair_from_database(tmp_path):
    identity = RunIdentity(
        run_id="run-export",
        profile="offline",
        environment_identity=digest("1"),
        dataset_manifest_sha256=digest("2"),
        config_manifest_sha256=digest("3"),
        prompt_manifest_sha256=digest("4"),
        generation_manifest_sha256=digest("5"),
        fixture_manifest_sha256=digest("6"),
    )
    store = CheckpointStore.create(tmp_path / "run", identity, CONFIG.resolve())
    try:
        ledger = LLMLedger(store)
        ledger.commit_checkpoint("checkpoint-one", "after_message", {"state": 1})
        snapshot = store.export_checkpoint("checkpoint-one")
        assert os.stat(snapshot).st_mode & 0o777 == 0o600
        assert json.loads(snapshot.read_bytes())["payload"] == {"state": 1}
        latest = snapshot.parent / "latest.json"
        latest.write_bytes(b"torn")
        assert store.export_checkpoint("checkpoint-one") == snapshot
        assert json.loads(latest.read_bytes())["checkpoint_id"] == "checkpoint-one"
        marks = store.high_water_marks()
        assert marks["checkpoint_event_ordinal"] == 1
        assert marks["logical_llm_requests"] == 0
    finally:
        store.close()
