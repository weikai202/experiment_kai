"""Native Mock orchestrator/evaluator integration; no model APIs or core test data."""

from tau2.data_model.message import UserMessage
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, get_tasks, run_simulation
from tau2.user.user_simulator_base import HalfDuplexUser
from test_agent import Scripted

from comap_tau3.agent import CoMAPAgent


class MockUser(HalfDuplexUser):
    def __init__(self):
        super().__init__()

    def get_init_state(self, message_history=None):
        return 0

    def generate_next_message(self, message, state):
        content = (
            "Create a task called Important Meeting for user_1." if state == 0 else "###STOP###"
        )
        return UserMessage(role="user", content=content), state + 1

    @classmethod
    def is_stop(cls, message):
        return message.content == "###STOP###"


def test_real_mock_tool_execution_and_native_reward():
    env = build_environment("mock")
    task = get_tasks("mock", task_ids=["create_task_1"])[0]
    draft = {
        "type": "tool_calls",
        "calls": [
            {
                "name": "create_task",
                "arguments": {"user_id": "user_1", "title": "Important Meeting"},
            }
        ],
    }
    done = {"type": "message", "content": "The Important Meeting task was created successfully."}

    def keep(action):
        return {"decision": "KEEP", "revise_probability": 0.0, "action": action}

    agent = CoMAPAgent(
        env.get_tools(),
        env.get_policy(),
        Scripted([draft, keep(draft), done, keep(done)]),
        Scripted(["HYPOTHETICAL_NEXT_STATE", "User acknowledges"]),
    )
    result = run_simulation(
        Orchestrator(
            domain="mock",
            agent=agent,
            user=MockUser(),
            environment=env,
            task=task,
            max_steps=12,
            seed=42,
        )
    )
    assert result.reward_info.reward == 1.0
    assert len(agent.traces) == 2
    assert agent.traces[0]["next_observation"][0]["role"] == "tool"
    assert "HYPOTHETICAL_NEXT_STATE" not in result.model_dump_json()
