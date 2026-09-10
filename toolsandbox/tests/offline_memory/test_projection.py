import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.offline.memory_projection import (
    MemoryProjectionError,
    projection_json,
    reject_sensitive_candidate,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.offline_memory import (
    PolicyMemoryCandidate,
    PolicyTrajectoryProjection,
)


def digest(value="a"):
    return "sha256:" + value * 64


def projection(**changes):
    values = dict(
        trajectory_id=digest(),
        manifest_position=3,
        visible_states=({"content": "Transfer to account ALPHA-4839"},),
        retrieved_policy_memory=(),
        retrieved_skills=(),
        proposed_actions=(ActionEnvelope(action={"type": "assistant_message", "content": "checking"}),),
        final_actions=(ActionEnvelope(action={"type": "assistant_message", "content": "done"}),),
        controller_codes=(),
        visible_tool_outcomes=({"result": "receipt TEMP-9921"},),
        native_similarity=1.0,
        fully_successful=True,
        host_attribution="successful",
    )
    values.update(changes)
    return PolicyTrajectoryProjection(**values)


def test_projection_is_strict_canonical_and_host_labelled():
    item = projection()
    assert '"projection_version":"policy-trajectory-v1"' in projection_json(item)
    with pytest.raises(ValidationError):
        projection(fully_successful=False)
    with pytest.raises(ValidationError):
        PolicyTrajectoryProjection(**item.model_dump(), extra_field=True)


def test_sensitive_concrete_values_are_rejected_but_public_vocabulary_is_allowed():
    item = projection()
    leaked = PolicyMemoryCandidate(
        result="CANDIDATE",
        role="policy",
        candidate={
            "scope": "Transfer workflow",
            "applicability": [],
            "action_guidance": "Never repeat receipt TEMP-9921 in reusable guidance",
            "avoid": [],
        },
    )
    with pytest.raises(MemoryProjectionError, match="protected literal"):
        reject_sensitive_candidate(leaked, item)
    safe = leaked.model_copy(
        update={"candidate": leaked.candidate.model_copy(update={"action_guidance": "Validate tool arguments before execution"})}
    )
    reject_sensitive_candidate(safe, item)


def test_nested_forbidden_fields_fail_visibility_audit():
    item = projection(visible_states=({"hidden_database": "forbidden"},))
    with pytest.raises(MemoryProjectionError, match="forbidden"):
        projection_json(item)
