"""Public core domain interface checks, without loading or running core tasks."""

import pytest
from tau2.data_model.message import UserMessage
from tau2.runner import build_environment
from test_agent import Scripted

from comap_tau3.agent import CoMAPAgent


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_public_domain_schema_and_policy_adapter(domain):
    env = build_environment(domain)
    action = {"type": "message", "content": "How can I help you?"}
    agent = CoMAPAgent(
        env.get_tools(),
        env.get_policy(),
        Scripted([action, {"decision": "KEEP", "revise_probability": 0, "action": action}]),
        Scripted(["The user describes their request."]),
    )
    response, state = agent.generate_next_message(UserMessage(role="user", content="Hello"), [])
    assert response.content == action["content"]
    assert agent.schemas
    assert len(state) == 2
