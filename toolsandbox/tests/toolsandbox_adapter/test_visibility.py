from __future__ import annotations

import pytest

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message
from toolsandbox_pipeline.toolsandbox_adapter import PipelineAgent, extract_visible_messages


class NeverResponder:
    def respond(self, turn):
        raise AssertionError("responder must not be called")


def test_visible_messages_retain_original_indices_and_drop_hidden_fields(sandbox_context):
    messages = [
        Message(RoleType.SYSTEM, RoleType.AGENT, "system"),
        Message(RoleType.USER, RoleType.AGENT, "user", conversation_active=False),
        Message(
            RoleType.AGENT,
            RoleType.EXECUTION_ENVIRONMENT,
            "call",
            openai_tool_call_id="call_1",
            openai_function_name="public_alpha",
            tool_trace=["secret trace"],
        ),
        Message(
            RoleType.EXECUTION_ENVIRONMENT,
            RoleType.AGENT,
            "{'value': 1}",
            openai_tool_call_id="call_1",
            openai_function_name="public_alpha",
        ),
        Message(
            RoleType.USER,
            RoleType.SYSTEM,
            "hidden",
            visible_to=[RoleType.USER, RoleType.SYSTEM],
        ),
    ]
    PipelineAgent.add_messages(messages)

    visible = extract_visible_messages(PipelineAgent)
    assert [item.source_message_index for item in visible] == [0, 1, 2, 3]
    assert [item.content for item in visible] == [
        "system",
        "user",
        "call",
        "{'value': 1}",
    ]
    dumped = [item.model_dump() for item in visible]
    assert all("visible_to" not in item for item in dumped)
    assert all("conversation_active" not in item for item in dumped)
    assert all("tool_trace" not in item for item in dumped)
    assert "hidden" not in repr(visible)


def test_ending_index_matches_upstream_truncation(sandbox_context):
    PipelineAgent.add_messages(
        [
            Message(RoleType.USER, RoleType.AGENT, "first"),
            Message(RoleType.AGENT, RoleType.USER, "middle"),
            Message(RoleType.USER, RoleType.AGENT, "last"),
        ]
    )
    assert [x.content for x in extract_visible_messages(PipelineAgent, 1)] == [
        "first",
        "middle",
    ]
    assert [x.content for x in PipelineAgent.get_messages(1)] == ["first", "middle"]


def test_system_message_skips_responder_and_append(sandbox_context):
    PipelineAgent.add_messages([Message(RoleType.SYSTEM, RoleType.AGENT, "system")])
    before = PipelineAgent.get_messages()
    PipelineAgent(NeverResponder()).respond()
    assert PipelineAgent.get_messages() == before


def test_last_recipient_is_validated(sandbox_context):
    PipelineAgent.add_messages([Message(RoleType.AGENT, RoleType.USER, "not for agent")])
    with pytest.raises(KeyError):
        PipelineAgent(NeverResponder()).respond()
