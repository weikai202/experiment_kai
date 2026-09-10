import pytest

from toolsandbox_pipeline.offline.memory_updates import MemoryUpdateError, apply_memory_review, memory_id
from toolsandbox_pipeline.schemas.critic import CriticErrorCode
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryReviewAdd,
    MemoryReviewMerge,
    MemoryReviewSkip,
    PolicyMemoryCandidate,
    WorldMemoryCandidate,
)


def policy_candidate():
    return PolicyMemoryCandidate(
        result="CANDIDATE", role="policy",
        candidate={
            "scope": "Actions with prerequisites", "applicability": ("Required values are visible",),
            "action_guidance": "Validate prerequisites before execution", "avoid": ("Do not guess missing values",),
        },
    )


def world_candidate():
    return WorldMemoryCandidate(
        result="CANDIDATE", role="world",
        candidate={
            "action_pattern": "Calls with required arguments", "state_conditions": (),
            "schema_conditions": ("A required argument is absent",),
            "likely_error_codes": (CriticErrorCode.MISSING_REQUIRED_ARGUMENT,),
            "outcome_calibration": "Failure is likely before execution",
            "correction_principle": "Ask for the missing value",
        },
    )


def test_add_assigns_host_identity_evidence_rate_confidence_and_generation():
    candidate = policy_candidate()
    mutation = apply_memory_review(
        candidate, MemoryReviewAdd(decision="ADD", reason="Reusable and nonduplicate"),
        trajectory_id="sha256:" + "2" * 64, next_generation_id="g001", binary_label=True,
        all_visible_records=(), supplied_matches=(),
    )
    assert mutation.operation == "ADD"
    assert mutation.record.memory_id == memory_id(candidate)
    assert mutation.record.evidence_trajectory_ids == ("sha256:" + "2" * 64,)
    assert mutation.record.support_count == 1 and mutation.record.success_rate == 1.0
    assert mutation.record.confidence == 1 / 3 and mutation.record.created_version == "g001"


def test_merge_preserves_semantics_and_updates_sorted_evidence_and_statistics():
    target = PolicyMemory(
        memory_id="pm_" + "a" * 64,
        scope="Existing scope", applicability=("Existing condition",),
        action_guidance="Existing guidance", avoid=(),
        evidence_trajectory_ids=("sha256:" + "2" * 64,), support_count=1,
        success_rate=1.0, confidence=1 / 3, created_version="g000", status="active",
    )
    mutation = apply_memory_review(
        policy_candidate(),
        MemoryReviewMerge(decision="MERGE", target_memory_id=target.memory_id, reason="Same rule"),
        trajectory_id="sha256:" + "1" * 64, next_generation_id="g001", binary_label=False,
        all_visible_records=(target,), supplied_matches=(target,),
    )
    merged = mutation.record
    assert merged.memory_id == target.memory_id and merged.scope == target.scope
    assert merged.action_guidance == target.action_guidance and merged.created_version == "g000"
    assert merged.evidence_trajectory_ids == (
        "sha256:" + "1" * 64, "sha256:" + "2" * 64
    )
    assert merged.support_count == 2 and merged.success_rate == 0.5 and merged.confidence == 0.5


def test_skip_has_no_mutation_and_duplicate_add_or_evidence_fail_closed():
    candidate = policy_candidate()
    assert apply_memory_review(
        candidate, MemoryReviewSkip(decision="SKIP", reason="Not reusable"),
        trajectory_id="sha256:" + "1" * 64, next_generation_id="g001", binary_label=True,
        all_visible_records=(), supplied_matches=(),
    ) is None
    prior_add = apply_memory_review(
        candidate, MemoryReviewAdd(decision="ADD", reason="First"),
        trajectory_id="sha256:" + "1" * 64, next_generation_id="g001", binary_label=True,
        all_visible_records=(), supplied_matches=(),
    ).record
    with pytest.raises(MemoryUpdateError, match="duplicate candidate"):
        apply_memory_review(
            candidate, MemoryReviewAdd(decision="ADD", reason="Duplicate"),
            trajectory_id="sha256:" + "2" * 64, next_generation_id="g001", binary_label=True,
            all_visible_records=(prior_add,), supplied_matches=(),
        )
    with pytest.raises(MemoryUpdateError, match="duplicate trajectory"):
        apply_memory_review(
            candidate,
            MemoryReviewMerge(decision="MERGE", target_memory_id=prior_add.memory_id, reason="Same"),
            trajectory_id="sha256:" + "1" * 64, next_generation_id="g001", binary_label=True,
            all_visible_records=(prior_add,), supplied_matches=(prior_add,),
        )


def test_world_add_uses_attributable_failure_binary_label():
    mutation = apply_memory_review(
        world_candidate(), MemoryReviewAdd(decision="ADD", reason="New calibration"),
        trajectory_id="sha256:" + "3" * 64, next_generation_id="g001", binary_label=True,
        all_visible_records=(), supplied_matches=(),
    )
    assert type(mutation.record) is WorldMemory
    assert mutation.record.empirical_failure_rate == 1.0
