import json

import pytest
from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
from tau2.environment.tool import as_tool

from comap_tau3.agent import CoMAPAgent
from comap_tau3.backends import Completion


def lookup(name: str) -> str:
    """Look up a customer by name.

    Args:
        name: Customer name.
    """
    raise AssertionError("Agent/world model must never execute tools directly")


class Scripted:
    model_id = "scripted"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def generate(self, messages, **kwargs):
        self.prompts.append(messages)
        value = next(self.replies)
        return Completion(value if isinstance(value, str) else json.dumps(value), 2, 3)


DRAFT = {"type": "tool_calls", "calls": [{"name": "lookup", "arguments": {"name": "Alice"}}]}
REVISED = {"type": "message", "content": "Please confirm your name."}


@pytest.mark.parametrize(
    "decision,probability,action,expected",
    [
        ("REVISE", 0.9, REVISED, True),
        ("KEEP", 0.9, REVISED, False),
        ("REVISE", 0.5, REVISED, False),
        ("REVISE", 0.9, DRAFT, False),
        ("REVISE", float("nan"), REVISED, False),
        (
            "REVISE",
            0.9,
            {"type": "tool_calls", "calls": [{"name": "hidden", "arguments": {}}]},
            False,
        ),
    ],
)
def test_revision_gate(decision, probability, action, expected):
    policy = Scripted(
        [DRAFT, {"decision": decision, "revise_probability": probability, "action": action}]
    )
    agent = CoMAPAgent([as_tool(lookup)], "policy", policy, Scripted(["HYPOTHETICAL_RESULT"]))
    response, state = agent.generate_next_message(UserMessage(role="user", content="Hello"), [])
    assert agent.traces[0]["used_revised_action"] is expected
    assert bool(response.tool_calls) is (not expected)
    assert "HYPOTHETICAL_RESULT" not in json.dumps(state)
    assert "HYPOTHETICAL_RESULT" in json.dumps(policy.prompts[-1])


def test_multi_tool_results_are_real_targets_and_initial_history_retained():
    policy = Scripted([DRAFT, "bad reflection", REVISED, "bad reflection"])
    agent = CoMAPAgent(
        [as_tool(lookup)], "policy", policy, Scripted(["imagined", "imagined again"])
    )
    state = agent.get_init_state([UserMessage(role="user", content="prior context")])
    first, state = agent.generate_next_message(UserMessage(role="user", content="Hello"), state)
    observed = MultiToolMessage(
        role="tool",
        tool_messages=[ToolMessage(role="tool", id=first.tool_calls[0].id, content="REAL_RESULT")],
    )
    second, state = agent.generate_next_message(observed, state)
    assert agent.traces[0]["next_observation"][0]["content"] == "REAL_RESULT"
    assert state[0]["content"] == "prior context"
    assert len(state) == 5
    assert second.content == REVISED["content"]


def test_private_tool_result_rejected():
    agent = CoMAPAgent([], "", Scripted([]), Scripted([]))
    with pytest.raises(ValueError, match="private"):
        agent.generate_next_message(
            ToolMessage(role="tool", id="private", requestor="user", content="secret"), []
        )


def test_bad_draft_retries_once_then_fails():
    agent = CoMAPAgent([], "", Scripted(["bad", "still bad"]), Scripted([]))
    with pytest.raises(ValueError):
        agent.generate_next_message(UserMessage(role="user", content="hello"), [])
    assert len(agent.calls) == 2
