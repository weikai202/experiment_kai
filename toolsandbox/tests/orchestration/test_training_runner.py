from types import SimpleNamespace

import pytest

from toolsandbox_pipeline.orchestration.preflight_gate import PreflightGateResult
from toolsandbox_pipeline.orchestration.round_runner import (
    RoundExecutionError,
    RoundExecutionResult,
)
from toolsandbox_pipeline.orchestration.training_runner import (
    TrainingExecutionError,
    TrainingRunner,
)
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest

from .test_run_manifest import HASH, manifest_payload


class Round:
    def __init__(self, calls):
        self.calls = calls

    def run(self, **values):
        self.calls.append(values)
        index = values["round_index"]
        return SimpleNamespace(
            completion_status="complete",
            published_generation_id=f"g{index + 1:03d}",
        )


class Metrics:
    def __init__(self):
        self.calls = []

    def finalize_training(self, **values):
        self.calls.append(values)
        return values


def test_exact_three_round_mapping_and_updated_is_always_g003(tmp_path):
    payload = manifest_payload(tmp_path, purpose="formal_training")
    manifest = ResolvedRunManifest.model_validate(payload, strict=True)
    preflight = PreflightGateResult(
        manifest_sha256=manifest.manifest_sha256,
        evidence_sha256=HASH,
    )
    calls = []
    metrics = Metrics()
    result = TrainingRunner(
        manifest=manifest,
        preflight=preflight,
        boot_id="boot",
        round_factory=lambda _: Round(calls),
        metrics=metrics,
    ).run((("a",), ("b",), ("c",)))
    assert result.updated_generation_id == "g003"
    assert [(row["round_index"], row["input_generation_id"], row["train_shard_id"]) for row in calls] == [
        (0, "g000", "train-shard-0"),
        (1, "g001", "train-shard-1"),
        (2, "g002", "train-shard-2"),
    ]
    assert result.timing.timing_complete
    assert metrics.calls[0]["completion_status"] == "complete"


def test_duplicate_scenario_across_shards_is_rejected_before_round_dispatch(tmp_path):
    payload = manifest_payload(tmp_path, purpose="formal_training")
    manifest = ResolvedRunManifest.model_validate(payload, strict=True)
    preflight = PreflightGateResult(
        manifest_sha256=manifest.manifest_sha256,
        evidence_sha256=HASH,
    )
    calls = []
    metrics = Metrics()
    runner = TrainingRunner(
        manifest=manifest,
        preflight=preflight,
        boot_id="boot",
        round_factory=lambda _: Round(calls),
        metrics=metrics,
    )
    with pytest.raises(ValueError, match="three nonempty ordered train shards"):
        runner.run((("same",), ("same",), ("other",)))
    assert calls == [] and metrics.calls == []


def test_failed_round_result_remains_in_training_failure_record(tmp_path):
    payload = manifest_payload(tmp_path, purpose="formal_training")
    manifest = ResolvedRunManifest.model_validate(payload, strict=True)
    preflight = PreflightGateResult(
        manifest_sha256=manifest.manifest_sha256,
        evidence_sha256=HASH,
    )
    partial = RoundExecutionResult(
        round_index=0,
        input_generation_id="g000",
        train_shard_id="train-shard-0",
        published_generation_id=None,
        completion_status="failed",
        timing=SimpleNamespace(timing_complete=True),
        final_checkpoint=None,
        metric_record={"status": "failed"},
        archive_reference=None,
        memory_result=None,
        skill_result=None,
    )

    class FailedRound:
        def run(self, **_):
            raise RoundExecutionError(partial)

    metrics = Metrics()
    runner = TrainingRunner(
        manifest=manifest,
        preflight=preflight,
        boot_id="boot",
        round_factory=lambda _: FailedRound(),
        metrics=metrics,
    )
    with pytest.raises(TrainingExecutionError) as caught:
        runner.run((("a",), ("b",), ("c",)))
    assert caught.value.result.round_results == (partial,)
    assert metrics.calls[0]["round_results"] == (partial,)
