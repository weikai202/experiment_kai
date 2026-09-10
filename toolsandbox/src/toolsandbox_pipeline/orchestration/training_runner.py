"""The exact immutable G000 -> G003 three-round training lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from toolsandbox_pipeline.metrics.timing import ScopeTimer
from toolsandbox_pipeline.orchestration.preflight_gate import PreflightGateResult
from toolsandbox_pipeline.orchestration.round_runner import RoundExecutionError
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest


class RoundFactory(Protocol):
    def __call__(self, round_index: int): ...


class TrainingMetricSink(Protocol):
    def finalize_training(
        self, *, timing, completion_status: str, round_results: tuple[object, ...],
    ): ...


@dataclass(frozen=True)
class TrainingExecutionResult:
    run_id: str
    updated_generation_id: str | None
    completion_status: str
    timing: object
    round_results: tuple[object, ...]
    metric_record: object


class TrainingExecutionError(RuntimeError):
    def __init__(self, result: TrainingExecutionResult):
        super().__init__("formal training failed")
        self.result = result


class TrainingRunner:
    def __init__(
        self, *, manifest: ResolvedRunManifest, preflight: PreflightGateResult,
        boot_id: str, round_factory: RoundFactory,
        metrics: TrainingMetricSink, timer_factory=ScopeTimer,
    ) -> None:
        if manifest.purpose != "formal_training":
            raise ValueError("formal TrainingRunner requires formal manifest")
        if preflight.status != "pass" or preflight.manifest_sha256 != manifest.manifest_sha256:
            raise ValueError("manifest-bound preflight pass required")
        self.manifest = manifest
        self.preflight = preflight
        self.boot_id = boot_id
        self.round_factory = round_factory
        self.metrics = metrics
        self.timer_factory = timer_factory

    def run(self, shards: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]):
        scenario_ids = tuple(scenario_id for shard in shards for scenario_id in shard)
        if (
            len(shards) != 3
            or any(not shard for shard in shards)
            or len(set(scenario_ids)) != len(scenario_ids)
        ):
            raise ValueError("exactly three nonempty ordered train shards required")
        timer = self.timer_factory(
            "training_run", self.manifest.run_id, self.boot_id,
        )
        results: list[object] = []
        try:
            for index in range(3):
                runner = self.round_factory(index)
                result = runner.run(
                    round_index=index,
                    train_shard_id=self.manifest.ordered_train_shard_ids[index],
                    scenario_ids=shards[index],
                    input_generation_id=f"g{index:03d}",
                )
                if (
                    result.completion_status != "complete"
                    or result.published_generation_id != f"g{index + 1:03d}"
                ):
                    raise RuntimeError("round did not complete exact generation transition")
                results.append(result)
            timing = timer.close()
            metric = self.metrics.finalize_training(
                timing=timing,
                completion_status="complete",
                round_results=tuple(results),
            )
            return TrainingExecutionResult(
                self.manifest.run_id, "g003", "complete", timing, tuple(results), metric,
            )
        except BaseException as error:
            if isinstance(error, RoundExecutionError):
                results.append(error.result)
            timing = timer.close()
            metric = self.metrics.finalize_training(
                timing=timing,
                completion_status="failed",
                round_results=tuple(results),
            )
            partial = TrainingExecutionResult(
                self.manifest.run_id, None, "failed", timing, tuple(results), metric,
            )
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise TrainingExecutionError(partial) from error


__all__ = [
    "TrainingExecutionError", "TrainingExecutionResult", "TrainingRunner",
]
