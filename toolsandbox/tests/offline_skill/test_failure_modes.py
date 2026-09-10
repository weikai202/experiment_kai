import pytest

from toolsandbox_pipeline.offline.failure_modes import (
    FailureModeMutationError,
    apply_failure_mode_decision,
    failure_mode_id,
)
from toolsandbox_pipeline.schemas.offline_skill import (
    FailureModeAdd,
    FailureModeMerge,
    FailureModeSkip,
)


def test_add_merge_skip_and_substantive_cost_boundary():
    add = FailureModeAdd(
        decision="ADD", task_condition="Missing input", failure_mode="Ask first"
    )
    first = apply_failure_mode_decision(
        skill_id="skill", buffer=(), decision=add, next_observed_seq=1
    )
    assert first.substantive
    assert first.after[0].mode_id == failure_mode_id(
        "skill", "Missing input", "Ask first"
    )
    merged = apply_failure_mode_decision(
        skill_id="skill",
        buffer=first.after,
        decision=FailureModeMerge(decision="MERGE", mode_id=first.after[0].mode_id),
        next_observed_seq=2,
    )
    assert merged.after[0].support_count == 2
    skipped = apply_failure_mode_decision(
        skill_id="skill",
        buffer=merged.after,
        decision=FailureModeSkip(decision="SKIP", reason="Case specific"),
        next_observed_seq=None,
    )
    assert not skipped.substantive
    with pytest.raises(FailureModeMutationError):
        apply_failure_mode_decision(
            skill_id="skill", buffer=first.after, decision=add, next_observed_seq=3
        )
