from __future__ import annotations

import pytest

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole
from toolsandbox_pipeline.toolsandbox_adapter import PipelineAgent

def alpha(value: int) -> dict[str, int]:
    """Return a value.

    Args:
        value: Input value.
    """
    return {"value": value}


class Responder:
    def __init__(self, value):
        self.value = value
        self.calls = 0
        self.reset_calls = 0
        self.teardown_calls = 0

    def respond(self, turn):
        self.calls += 1
        return self.value

    def reset(self):
        self.reset_calls += 1

    def teardown(self):
        self.teardown_calls += 1


def configure_agent(monkeypatch, sandbox_context):
    monkeypatch.setattr(PipelineAgent, "get_available_tools", classmethod(lambda cls: {"shown_alpha": alpha}))
    monkeypatch.setattr(
        sandbox_context,
        "get_agent_to_execution_facing_tool_name",
        lambda: {"shown_alpha": "alpha"},
    )


def test_agent_identity_single_response_and_lifecycle(monkeypatch, sandbox_context):
    assert issubclass(PipelineAgent, BaseRole)
    assert PipelineAgent.role_type == RoleType.AGENT
    configure_agent(monkeypatch, sandbox_context)
    PipelineAgent.add_messages([Message(RoleType.USER, RoleType.AGENT, "hello")])
    responder = Responder({"action": {"type": "assistant_message", "content": "answer"}})
    agent = PipelineAgent(responder)
    agent.respond()
    assert responder.calls == 1
    assert PipelineAgent.get_messages()[-1].sender == RoleType.AGENT
    assert PipelineAgent.get_messages()[-1].recipient == RoleType.USER
    assert PipelineAgent.get_messages()[-1].content == "answer"
    assert PipelineAgent.get_messages()[-1].conversation_active is True
    agent.reset()
    agent.teardown()
    assert (responder.reset_calls, responder.teardown_calls) == (1, 1)


@pytest.mark.parametrize(
    "bad_value",
    [None, {}, {"action": {"type": "assistant_message", "content": ""}}],
)
def test_invalid_responder_output_appends_nothing(monkeypatch, sandbox_context, bad_value):
    configure_agent(monkeypatch, sandbox_context)
    PipelineAgent.add_messages([Message(RoleType.USER, RoleType.AGENT, "hello")])
    before = PipelineAgent.get_messages()
    with pytest.raises(Exception):
        PipelineAgent(Responder(bad_value)).respond()
    assert PipelineAgent.get_messages() == before

def test_conversion_failure_appends_nothing(monkeypatch, sandbox_context):
    configure_agent(monkeypatch, sandbox_context)
    PipelineAgent.add_messages([Message(RoleType.USER, RoleType.AGENT, "hello")])
    before = PipelineAgent.get_messages()
    responder = Responder(
        {
            "action": {
                "type": "function_call",
                "call_id": "call_1",
                "selected_skill_id": None,
                "name": "stale",
                "arguments": {},
            }
        }
    )
    with pytest.raises(KeyError):
        PipelineAgent(responder).respond()
    assert responder.calls == 1
    assert PipelineAgent.get_messages() == before
