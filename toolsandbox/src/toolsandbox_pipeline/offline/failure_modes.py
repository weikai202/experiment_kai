"""Host-owned ADD/MERGE/SKIP semantics and capacity-five retention."""

from __future__ import annotations

from dataclasses import dataclass

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.offline_skill import (
    FailureModeAdd,
    FailureModeDecision,
    FailureModeMerge,
    FailureModeSkip,
)
from toolsandbox_pipeline.schemas.skill import SkillFailureMode


class FailureModeMutationError(ValueError):
    pass


@dataclass(frozen=True)
class FailureModeMutation:
    decision: FailureModeDecision
    before: tuple[SkillFailureMode, ...]
    after: tuple[SkillFailureMode, ...]
    before_sha256: str
    after_sha256: str
    substantive: bool
    observed_seq: int | None


def failure_mode_id(skill_id: str, task_condition: str, failure_mode: str) -> str:
    digest = canonical_sha256(
        {
            "skill_id": skill_id,
            "task_condition": task_condition,
            "failure_mode": failure_mode,
        }
    )
    return "fm_" + digest[7:]


def _buffer_hash(buffer: tuple[SkillFailureMode, ...]) -> str:
    return canonical_sha256([item.model_dump(mode="json") for item in buffer])


def _validate_buffer(buffer: tuple[SkillFailureMode, ...]) -> None:
    if len(buffer) > 5:
        raise FailureModeMutationError("failure buffer exceeds capacity")
    if tuple(item.mode_id for item in buffer) != tuple(
        sorted((item.mode_id for item in buffer), key=lambda value: value.encode("utf-8"))
    ):
        raise FailureModeMutationError("failure buffer must use stable mode_id order")
    if len({item.mode_id for item in buffer}) != len(buffer):
        raise FailureModeMutationError("duplicate failure mode ID")


def apply_failure_mode_decision(
    *,
    skill_id: str,
    buffer: tuple[SkillFailureMode, ...],
    decision: FailureModeDecision,
    next_observed_seq: int | None,
) -> FailureModeMutation:
    _validate_buffer(buffer)
    before_hash = _buffer_hash(buffer)
    staged = list(buffer)
    observed_seq: int | None = None
    if isinstance(decision, FailureModeSkip):
        if next_observed_seq is not None:
            raise FailureModeMutationError("SKIP cannot consume an observation sequence")
    else:
        if type(next_observed_seq) is not int or next_observed_seq < 1:
            raise FailureModeMutationError("positive host observation sequence required")
        if any(item.last_observed_seq >= next_observed_seq for item in buffer):
            raise FailureModeMutationError("observation sequence must increase globally")
        observed_seq = next_observed_seq
        if isinstance(decision, FailureModeAdd):
            mode_id = failure_mode_id(
                skill_id, decision.task_condition, decision.failure_mode
            )
            if any(
                item.mode_id == mode_id
                or (
                    item.task_condition == decision.task_condition
                    and item.failure_mode == decision.failure_mode
                )
                for item in buffer
            ):
                raise FailureModeMutationError("ADD collides with existing semantic mode")
            staged.append(
                SkillFailureMode(
                    mode_id=mode_id,
                    task_condition=decision.task_condition,
                    failure_mode=decision.failure_mode,
                    support_count=1,
                    last_observed_seq=next_observed_seq,
                )
            )
        elif isinstance(decision, FailureModeMerge):
            matches = [index for index, item in enumerate(buffer) if item.mode_id == decision.mode_id]
            if len(matches) != 1:
                raise FailureModeMutationError("MERGE target must be one supplied mode")
            index = matches[0]
            current = staged[index]
            staged[index] = current.model_copy(
                update={
                    "support_count": current.support_count + 1,
                    "last_observed_seq": next_observed_seq,
                }
            )
        else:
            raise TypeError("strict failure-mode decision required")
    retained = sorted(
        staged,
        key=lambda item: (
            -item.support_count,
            -item.last_observed_seq,
            item.mode_id.encode("utf-8"),
        ),
    )[:5]
    after = tuple(sorted(retained, key=lambda item: item.mode_id.encode("utf-8")))
    after_hash = _buffer_hash(after)
    return FailureModeMutation(
        decision=decision,
        before=buffer,
        after=after,
        before_sha256=before_hash,
        after_sha256=after_hash,
        substantive=after_hash != before_hash,
        observed_seq=observed_seq,
    )


__all__ = [
    "FailureModeMutation",
    "FailureModeMutationError",
    "apply_failure_mode_decision",
    "failure_mode_id",
]
