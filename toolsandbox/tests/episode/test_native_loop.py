from dataclasses import dataclass

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.reproducibility.scenario_hashes import context_sha256
from toolsandbox_pipeline.toolsandbox_adapter.native_loop import NativeLoop


class Agent(BaseRole):
    role_type = RoleType.AGENT
    calls = 0

    def respond(self, ending_index=None):
        type(self).calls += 1
        self.add_messages([Message(RoleType.AGENT, RoleType.USER, "answer")])


class User(BaseRole):
    role_type = RoleType.USER
    calls = 0

    def respond(self, ending_index=None):
        type(self).calls += 1
        self.add_messages(
            [Message(RoleType.USER, RoleType.AGENT, "done", conversation_active=False)]
        )


@dataclass
class Scenario:
    starting_context: object
    max_messages: int = 4


def test_fresh_loop_deep_copies_and_terminates_exactly(start_context):
    Agent.calls = User.calls = 0
    original = context_sha256(start_context)
    result = NativeLoop().run_fresh(
        Scenario(start_context), {RoleType.AGENT: Agent(), RoleType.USER: User()}
    )
    assert result.termination_reason == "conversation_ended"
    assert result.final_message_index == 2
    assert Agent.calls == User.calls == 1
    assert context_sha256(start_context) == original
    assert result.context is get_current_context() and result.context is not start_context


def test_resume_does_not_copy_or_repeat_setup(start_context):
    User.calls = 0
    Agent().respond()
    result = NativeLoop().run_resumed(
        start_context,
        {RoleType.USER: User()},
        max_messages=4,
        initial_max_message_index=0,
    )
    assert result.context is start_context
    assert User.calls == 1
