from __future__ import annotations

import pytest

from tool_sandbox.common.execution_context import RoleType
from toolsandbox_pipeline.schemas import ActionEnvelope
from toolsandbox_pipeline.toolsandbox_adapter import ControllerToolContext, action_to_messages

def alpha(value: int) -> dict[str, int]:
    return {"value": value}


def beta(text: str) -> str:
    return text


def context(mapping=None, tools=None):
    mapping = mapping or {"shown_alpha": "alpha", "shown_beta": "beta"}
    tools = tools or {"shown_alpha": alpha, "shown_beta": beta}
    return ControllerToolContext(mapping, "sha256:" + "0" * 64, tools)


def envelope(action):
    return ActionEnvelope.model_validate({"action": action})


def test_exact_assistant_message_conversion():
    message = action_to_messages(
        envelope({"type": "assistant_message", "content": "  verbatim  "}), context()
    )[0]
    assert message.sender == RoleType.AGENT
    assert message.recipient == RoleType.USER
    assert message.content == "  verbatim  "
    assert message.conversation_active is None


def test_function_call_uses_execution_name_in_code_and_agent_name_in_metadata():
    message = action_to_messages(
        envelope(
            {
                "type": "function_call",
                "call_id": "call_1",
                "selected_skill_id": None,
                "name": "shown_alpha",
                "arguments": {"value": 7},
            }
        ),
        context(),
    )[0]
    assert message.sender == RoleType.AGENT
    assert message.recipient == RoleType.EXECUTION_ENVIRONMENT
    assert message.openai_tool_call_id == "call_1"
    assert message.openai_function_name == "shown_alpha"
    assert "alpha(**call_1_parameters)" in message.content
    assert "shown_alpha(" not in message.content
    assert "selected_skill" not in message.content


def test_parallel_order_and_all_or_nothing_failure():
    action = envelope(
        {
            "type": "parallel_batch",
            "calls": [
                {"call_id": "call_1", "selected_skill_id": None, "name": "shown_alpha", "arguments": {"value": 1}},
                {"call_id": "call_2", "selected_skill_id": "skill:2", "name": "shown_beta", "arguments": {"text": "x"}},
            ],
        }
    )
    assert [x.openai_tool_call_id for x in action_to_messages(action, context())] == [
        "call_1",
        "call_2",
    ]

    stale = action.model_copy(
        update={"action": action.action.model_copy(update={"calls": [action.action.calls[0], action.action.calls[1].model_copy(update={"name": "stale"})]})}
    )
    with pytest.raises(KeyError, match="unavailable"):
        action_to_messages(stale, context())


def test_rejects_missing_and_duplicate_mapping_values():
    call = envelope(
        {"type": "function_call", "call_id": "call_1", "selected_skill_id": None, "name": "shown_alpha", "arguments": {}}
    )
    with pytest.raises(KeyError, match="missing"):
        action_to_messages(call, context(mapping={"shown_beta": "beta"}))
    with pytest.raises(ValueError, match="unique"):
        action_to_messages(
            call,
            context(mapping={"shown_alpha": "same", "shown_beta": "same"}),
        )
