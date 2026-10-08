from __future__ import annotations

import json

import pytest

from bfcl_pipeline.checkpoint import CheckpointStore


def test_checkpoint_detects_tampering(tmp_path):
    store = CheckpointStore(tmp_path)
    store.persist("run", 0, "episodes_complete", "g000", ("case",), {"value": 1})
    path = tmp_path / "run" / "round-0" / "episodes_complete.json"
    data = json.loads(path.read_text())
    data["payload"]["material"]["value"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        store.load("run", 0, "episodes_complete")
