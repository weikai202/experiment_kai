import json
from pathlib import Path

import pytest

from toolsandbox_pipeline.orchestration.cli import COMMANDS, build_parser, run_cli
from toolsandbox_pipeline.orchestration.run_manifest import manifest_bytes
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest

from .test_run_manifest import manifest_payload


def write_manifest(tmp_path, *, purpose="train_smoke"):
    manifest = ResolvedRunManifest.model_validate(
        manifest_payload(tmp_path, purpose=purpose), strict=True
    )
    path = tmp_path / "run_manifest.json"
    path.write_bytes(manifest_bytes(manifest))
    return manifest, path


def test_parser_exposes_exact_command_allowlist_and_no_passthrough(capsys):
    parser = build_parser()
    actions = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    assert set(actions.choices) == set(COMMANDS)
    assert len(actions.choices) == len(COMMANDS)
    assert run_cli([]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "error",
        "error_class": "invalid_arguments",
    }
    assert run_cli(["all"]) == 2
    assert "Traceback" not in capsys.readouterr().out


def test_external_command_requires_fixed_launcher_map_and_sanitizes(tmp_path, capsys):
    _, manifest_path = write_manifest(tmp_path)
    smoke_path = Path(__file__).parents[2] / "configs/run/train_smoke_v1.json"
    args = [
        "train-smoke",
        "--manifest",
        str(manifest_path),
        "--smoke-config",
        str(smoke_path.resolve()),
    ]
    assert run_cli(args) == 2
    assert json.loads(capsys.readouterr().out)["error_class"] == (
        "command_requires_coordinator_runtime"
    )

    operation = lambda _: {
        "status": "complete",
        "total_running_time_seconds": 1.0,
        "total_tokens": 10,
        "usage_complete": True,
        "total_cost": 3,
        "cost_unit": "qwen_effective_output_tokens",
        "cost_complete": True,
    }
    assert run_cli(args, operations={"train-smoke": operation}) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["total_running_time_seconds"] == 1.0
    assert result["total_cost"] == 3

    assert run_cli(
        args,
        operations={"train-smoke": operation, "arbitrary-python": operation},
    ) == 2
    assert json.loads(capsys.readouterr().out)["error_class"] == (
        "invalid_coordinator_runtime"
    )

    unsafe = lambda _: {"status": "complete", "raw_response": "forbidden"}
    assert run_cli(args, operations={"train-smoke": unsafe}) == 2
    assert json.loads(capsys.readouterr().out)["error_class"] == "unsafe_result"


def test_unknown_option_is_rejected_without_echoing_input(capsys):
    secret_like = "sk-placeholder-never-print"
    assert run_cli(["query-checkpoints", "--registry", "/tmp/x", "--exec", secret_like]) == 2
    output = capsys.readouterr().out
    assert secret_like not in output and "Traceback" not in output


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "prefix SK-project-value suffix",
        "  bEaReR credential",
        "see HTTPS://example.invalid/path",
        "embedded http://example.invalid/path",
        "contains API_KEY material",
        "nested Authorization header",
        "copied PROMPT_TEXT fragment",
        "copied RAW_RESPONSE fragment",
    ],
)
def test_external_result_rejects_dangerous_fragments_anywhere_casefolded(
    tmp_path, capsys, unsafe_value
):
    _, manifest_path = write_manifest(tmp_path)
    args = ["train", "--manifest", str(manifest_path)]
    operation = lambda _: {"status": "complete", "detail": unsafe_value}
    assert run_cli(args, operations={"train": operation}) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "error",
        "error_class": "unsafe_result",
    }
