"""Fail-closed resume decisions over an explicit immutable run root."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from toolsandbox_pipeline.schemas.run import FrozenRunRecord, ResolvedRunManifest


ResumeStage = Literal[
    "scenario", "offline_memory", "offline_skill", "dev_minibench",
    "generation_build", "generation_publication", "metric_materialization",
    "next_round", "run_complete", "reconciliation_required",
]


class RecoveryPlanner(Protocol):
    def plan_run_recovery(self, *, manifest_sha256: str): ...


class ResumeDecision(FrozenRunRecord):
    status: Literal["resume", "complete", "operator_action_required"]
    stage: ResumeStage
    round_index: Annotated[int, Field(ge=0, le=2)] | None = None
    sanitized_action: str | None = None
    opened_execution_dependencies: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self) -> "ResumeDecision":
        if self.status == "complete":
            if self.stage != "run_complete" or self.round_index != 2 or self.sanitized_action is not None:
                raise ValueError("completed resume decision mismatch")
        elif self.status == "operator_action_required":
            if self.stage != "reconciliation_required" or self.round_index is None or not self.sanitized_action:
                raise ValueError("operator-action resume decision mismatch")
        elif (
            self.stage in {"run_complete", "reconciliation_required"}
            or self.round_index is None
            or self.sanitized_action is not None
        ):
            raise ValueError("active resume decision mismatch")
        return self


def plan_resume(
    run_root: Path | str,
    manifest: ResolvedRunManifest,
    planner: RecoveryPlanner,
) -> ResumeDecision:
    root = Path(run_root)
    if (
        not root.is_absolute() or root.is_symlink() or not root.is_dir()
        or root.resolve(strict=True) != Path(manifest.run_root).resolve(strict=True)
    ):
        raise ValueError("explicit manifest-bound existing run root required")
    plan = planner.plan_run_recovery(manifest_sha256=manifest.manifest_sha256)
    stage = getattr(plan, "stage", None)
    round_index = getattr(plan, "round_index", None)
    if stage == "reconciliation_required":
        return ResumeDecision(
            status="operator_action_required",
            stage=stage,
            round_index=round_index,
            sanitized_action="Reconcile the recorded official-live external read.",
        )
    if stage == "run_complete":
        return ResumeDecision(status="complete", stage=stage, round_index=2)
    if stage not in ResumeStage.__args__:
        raise ValueError("unknown recovery stage")
    return ResumeDecision(status="resume", stage=stage, round_index=round_index)


__all__ = ["ResumeDecision", "ResumeStage", "plan_resume"]
