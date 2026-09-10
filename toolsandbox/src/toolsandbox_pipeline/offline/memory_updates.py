"""Host-owned deterministic ADD/MERGE/SKIP memory mutation rules."""

from __future__ import annotations

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryReviewAdd,
    MemoryReviewMerge,
    MemoryReviewSkip,
    PolicyMemoryCandidate,
    StagedMemoryMutation,
    WorldMemoryCandidate,
)


class MemoryUpdateError(ValueError):
    pass


def memory_id(candidate: PolicyMemoryCandidate | WorldMemoryCandidate) -> str:
    if type(candidate) is PolicyMemoryCandidate:
        prefix = "pm_"
    elif type(candidate) is WorldMemoryCandidate:
        prefix = "wm_"
    else:
        raise TypeError("concrete memory candidate required")
    return prefix + canonical_sha256(candidate.candidate.model_dump(mode="json"))[7:]


def _record_id(record: PolicyMemory | WorldMemory) -> str:
    return record.memory_id


def _add(
    candidate: PolicyMemoryCandidate | WorldMemoryCandidate,
    *,
    trajectory_id: str,
    next_generation_id: str,
    binary_label: bool,
    existing: tuple[PolicyMemory | WorldMemory, ...],
) -> StagedMemoryMutation:
    identifier = memory_id(candidate)
    collision = next((record for record in existing if _record_id(record) == identifier), None)
    if collision is not None:
        semantic = candidate.candidate.model_dump(mode="json")
        collision_data = collision.model_dump(mode="json")
        if all(collision_data.get(key) == value for key, value in semantic.items()):
            raise MemoryUpdateError("duplicate candidate must not be silently added")
        raise MemoryUpdateError("memory ID collision")
    fields = candidate.candidate.model_dump()
    if type(candidate) is PolicyMemoryCandidate:
        record: PolicyMemory | WorldMemory = PolicyMemory(
            memory_id=identifier,
            **fields,
            evidence_trajectory_ids=(trajectory_id,),
            support_count=1,
            success_rate=float(binary_label),
            confidence=1 / 3,
            created_version=next_generation_id,
            status="active",
        )
        role = "policy"
    else:
        record = WorldMemory(
            memory_id=identifier,
            **fields,
            evidence_trajectory_ids=(trajectory_id,),
            support_count=1,
            empirical_failure_rate=float(binary_label),
            confidence=1 / 3,
            created_version=next_generation_id,
            status="active",
        )
        role = "world"
    return StagedMemoryMutation.build(
        role=role,
        operation="ADD",
        trajectory_id=trajectory_id,
        source_memory_id=None,
        record=record,
    )


def _merge(
    candidate: PolicyMemoryCandidate | WorldMemoryCandidate,
    decision: MemoryReviewMerge,
    *,
    trajectory_id: str,
    binary_label: bool,
    supplied_matches: tuple[PolicyMemory | WorldMemory, ...],
) -> StagedMemoryMutation:
    target = next(
        (record for record in supplied_matches if record.memory_id == decision.target_memory_id),
        None,
    )
    expected_type = PolicyMemory if type(candidate) is PolicyMemoryCandidate else WorldMemory
    if target is None or type(target) is not expected_type or target.status != "active":
        raise MemoryUpdateError("MERGE target must be a supplied active matching-role memory")
    if trajectory_id in target.evidence_trajectory_ids:
        raise MemoryUpdateError("duplicate trajectory evidence would be a no-op")
    evidence = tuple(
        sorted((*target.evidence_trajectory_ids, trajectory_id), key=lambda value: value.encode("utf-8"))
    )
    support = target.support_count + 1
    if type(target) is PolicyMemory:
        rate = (target.success_rate * target.support_count + int(binary_label)) / support
        record: PolicyMemory | WorldMemory = target.model_copy(
            update={
                "evidence_trajectory_ids": evidence,
                "support_count": support,
                "success_rate": rate,
                "confidence": support / (support + 2),
            }
        )
        record = PolicyMemory.model_validate(record)
        role = "policy"
    else:
        rate = (
            target.empirical_failure_rate * target.support_count + int(binary_label)
        ) / support
        record = target.model_copy(
            update={
                "evidence_trajectory_ids": evidence,
                "support_count": support,
                "empirical_failure_rate": rate,
                "confidence": support / (support + 2),
            }
        )
        record = WorldMemory.model_validate(record)
        role = "world"
    return StagedMemoryMutation.build(
        role=role,
        operation="MERGE",
        trajectory_id=trajectory_id,
        source_memory_id=target.memory_id,
        record=record,
    )


def apply_memory_review(
    candidate: PolicyMemoryCandidate | WorldMemoryCandidate,
    review: MemoryReviewAdd | MemoryReviewMerge | MemoryReviewSkip,
    *,
    trajectory_id: str,
    next_generation_id: str,
    binary_label: bool,
    all_visible_records: tuple[PolicyMemory | WorldMemory, ...],
    supplied_matches: tuple[PolicyMemory | WorldMemory, ...],
) -> StagedMemoryMutation | None:
    if type(review) is MemoryReviewSkip:
        return None
    if type(review) is MemoryReviewAdd:
        return _add(
            candidate,
            trajectory_id=trajectory_id,
            next_generation_id=next_generation_id,
            binary_label=binary_label,
            existing=all_visible_records,
        )
    if type(review) is MemoryReviewMerge:
        return _merge(
            candidate,
            review,
            trajectory_id=trajectory_id,
            binary_label=binary_label,
            supplied_matches=supplied_matches,
        )
    raise TypeError("strict review decision required")


__all__ = ["MemoryUpdateError", "apply_memory_review", "memory_id"]
