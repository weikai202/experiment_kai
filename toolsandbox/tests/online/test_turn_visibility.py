import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.online.controller_inputs import ReproducibilityProfile
from toolsandbox_pipeline.online.turn_context import SAFE_CLARIFICATION_TEXT
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnIdentity, OnlineTurnInput
from toolsandbox_pipeline.schemas.state import VisibleMessageInput, VisibleRole
from toolsandbox_pipeline.toolsandbox_adapter.contracts import (
    AdapterTurn,
    AgentTurnView,
    ControllerToolContext,
)


DIGEST = "sha256:" + "9" * 64


def identity():
    return OnlineTurnIdentity(
        run_id="run",
        profile=ReproducibilityProfile.STRICT_REPLAY,
        phase="train",
        round_index=0,
        shard_id="shard-0",
        family_id="family",
        scenario_id="scenario",
        episode_id="episode",
        agent_turn_index=0,
        expected_generation_id="g000",
        dataset_manifest_sha256=DIGEST,
        runtime_config_sha256=DIGEST,
        prompt_manifest_sha256=DIGEST,
        token_limit_config_sha256=DIGEST,
        fixture_manifest_sha256=DIGEST,
        environment_identity="environment",
    )


def adapter_turn():
    return AdapterTurn(
        agent_view=AgentTurnView(
            visible_messages=(
                VisibleMessageInput(
                    source_message_index=0,
                    sender=VisibleRole.USER,
                    recipient=VisibleRole.AGENT,
                    content="visible content",
                ),
            ),
            available_tools=(),
        ),
        controller_context=ControllerToolContext(
            agent_to_execution_name={},
            mapping_manifest_hash=canonical_sha256({}),
            tool_objects={},
        ),
    )


def test_turn_input_has_no_generic_hidden_or_evaluator_escape_hatch():
    payload = {
        "identity": identity(),
        "adapter_turn": adapter_turn(),
        "committed_tool_outcomes": (),
        "pending_dependencies": (),
        "prior_visible_failed_action_history": (),
        "structured_constraint_tension": False,
        "evaluator": {"minefield": "hidden"},
    }
    with pytest.raises(ValidationError, match="Extra inputs"):
        OnlineTurnInput.model_validate(payload, strict=True)
    fields = set(OnlineTurnInput.model_fields)
    assert not fields & {
        "raw_context",
        "tool_trace",
        "database",
        "evaluator",
        "milestones",
        "minefields",
        "secret",
        "raw_response",
    }


def test_safe_clarification_contains_no_controller_or_schema_evidence():
    lowered = SAFE_CLARIFICATION_TEXT.lower()
    for forbidden in (
        "canonical",
        "schema",
        "controller",
        "ungrounded_argument",
        "missing_dependency",
        "minefield",
    ):
        assert forbidden not in lowered
    assert "clarify" in lowered
