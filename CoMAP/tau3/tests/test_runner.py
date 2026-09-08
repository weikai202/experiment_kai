import json
from pathlib import Path

import pytest
import tau2
import tau2.runner
from test_agent import Scripted
from test_mock_integration import MockUser
from test_splits import fixture_manifest

from comap_tau3 import backends, runner
from comap_tau3.training import load_transitions


def test_usage_missing_is_not_zero():
    result = runner.summarize_usage(
        [
            {"phase": "wm", "input_tokens": 5, "output_tokens": None, "cost": None},
            {"phase": "wm", "input_tokens": 2, "output_tokens": 3, "cost": 0.1},
        ]
    )["by_phase"]["wm"]
    assert result["input_tokens"] == 7
    assert result["output_tokens"] is None
    assert result["cost"] is None
    assert not result["usage_complete"]


@pytest.mark.parametrize("split", ["train", "dev"])
def test_runner_artifacts_using_mock_environment(tmp_path, monkeypatch, split):
    manifest, official = fixture_manifest()
    root = Path(tau2.__file__).resolve().parents[2]
    env = tau2.runner.build_environment("mock")
    task = tau2.runner.get_tasks("mock", task_ids=["create_task_1"])[0]
    task_id = (
        manifest["domains"]["airline"]["rounds"][0][0]
        if split == "train"
        else manifest["domains"]["airline"]["dev"][0]
    )
    task = task.model_copy(update={"id": task_id})
    monkeypatch.setattr(runner, "verify_repo", lambda _: root)
    monkeypatch.setattr(runner, "official_splits", lambda _: official)
    monkeypatch.setattr(tau2.runner, "build_environment", lambda _: env)
    monkeypatch.setattr(tau2.runner, "get_tasks", lambda *a, **k: [task])
    monkeypatch.setattr(tau2.runner, "build_user", lambda *a, **k: MockUser())
    create = {
        "type": "tool_calls",
        "calls": [
            {
                "name": "create_task",
                "arguments": {"user_id": "user_1", "title": "Important Meeting"},
            }
        ],
    }
    done = {"type": "message", "content": "The task was created successfully."}
    policy = Scripted([create, "invalid reflection", done, "invalid reflection"])
    world = Scripted(["predicted", "predicted"])
    monkeypatch.setattr(
        backends, "make_backend", lambda cfg: policy if cfg["model"] == "policy" else world
    )
    config = {
        "tau_root": str(root),
        "policy": {"model": "policy"},
        "world_model": {"model": "world"},
        "user_simulator": {"model": "scripted"},
    }
    output = tmp_path / "run"
    summary = runner.run(config, manifest, "airline", split, output, round_index=0, limit=1)
    assert summary["mean_reward"] == 1
    assert summary["status"] == "complete"
    assert (output / "summary.json").is_file()
    assert summary["usage"]["by_phase"]["draft"]["calls"] == 2
    if split == "train":
        rows = load_transitions(output / "transitions.jsonl", manifest, "airline", 0)
        assert len(rows) == 2
        assert rows[0]["next_observation"][0]["role"] == "tool"
        assert "reward" not in json.dumps(rows[0]["history"])
    else:
        assert not (output / "transitions.jsonl").exists()
