from dataclasses import dataclass

import pytest
from test_manifests import official_fixture

from tau3_evolution.access import OrdinaryDatasetAccess
from tau3_evolution.manifests import build_manifests
from tau3_evolution.native_adapter import (
    AssistantAction,
    TauAgentAdapter,
    visible_message,
)


def access():
    manifest, _, _ = build_manifests(official_fixture())
    return OrdinaryDatasetAccess(manifest, lambda domain, ids: [(domain, task) for task in ids])


def test_train_and_dev_access_are_exact_and_audited():
    dataset = access()
    train, train_audit = dataset.load(
        split="train", purpose="evolution", phase="round0", round_index=0
    )
    dev, dev_audit = dataset.load(
        split="dev", purpose="skill_ab_validation", phase="skill", round_index=None
    )
    assert len(train) == 48 and train_audit.task_count == 48
    assert len(dev) == 34 and dev_audit.task_count == 34


def test_loader_count_cannot_hide_wrong_task_identities():
    manifest, _, _ = build_manifests(official_fixture())
    dataset = OrdinaryDatasetAccess(
        manifest, lambda domain, ids: [(domain, f"wrong-{index}") for index, _ in enumerate(ids)]
    )
    with pytest.raises(ValueError, match="exact manifest-ordered"):
        dataset.load(split="dev", purpose="skill_ab_validation", phase="skill")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": "test", "purpose": "development", "phase": "debug"},
        {"split": "dev", "purpose": "development", "phase": "debug"},
        {"split": "train", "purpose": "evolution", "phase": "bad", "round_index": None},
    ],
)
def test_invalid_dataset_access_fails_before_loader(kwargs):
    called = False

    def loader(domain, ids):
        nonlocal called
        called = True
        return []

    manifest, _, _ = build_manifests(official_fixture())
    dataset = OrdinaryDatasetAccess(manifest, loader)
    with pytest.raises(PermissionError):
        dataset.load(**kwargs)
    assert called is False


@dataclass
class Tool:
    openai_schema: dict


class Policy:
    def __init__(self, action):
        self.action = action
        self.histories = []

    def respond(self, **kwargs):
        self.histories.append(kwargs["visible_history"])
        return self.action


def tool():
    return Tool(
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    )


def test_native_adapter_preserves_visible_history_and_validates_tools():
    policy = Policy(AssistantAction(tool_calls=(("lookup_order", {"order_id": "A"}),)))
    agent = TauAgentAdapter([tool()], "policy", policy)
    action, state = agent.generate_action({"role": "user", "content": "Help"}, [])
    assert action.tool_calls[0][0] == "lookup_order"
    assert state[-1]["tool_calls"][0]["id"] == "tau3_evolution_0_0"
    assert policy.histories[0] == ({"role": "user", "content": "Help"},)


def test_native_adapter_rejects_unknown_or_bad_tool_arguments():
    for action in (
        AssistantAction(tool_calls=(("missing", {}),)),
        AssistantAction(tool_calls=(("lookup_order", {"order_id": 1}),)),
    ):
        agent = TauAgentAdapter([tool()], "policy", Policy(action))
        with pytest.raises(ValueError):
            agent.generate_action({"role": "user", "content": "Help"}, [])


def test_private_user_tool_state_is_rejected():
    with pytest.raises(ValueError, match="private"):
        visible_message({"role": "user", "content": None, "tool_calls": [{"name": "secret"}]})
    with pytest.raises(ValueError, match="private"):
        visible_message({"role": "tool", "content": "secret", "requestor": "user"})
