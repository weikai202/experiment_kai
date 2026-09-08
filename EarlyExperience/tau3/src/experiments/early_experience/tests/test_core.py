"""Offline tests of counterfactual isolation, visibility, formatting and curricula."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.early_experience.agent import strip_reflection
from experiments.early_experience.core import (
    action_key,
    make_expert,
    make_iwm,
    make_reflection,
    parse_action,
    probe,
    validate_split,
    visible_context,
)
from experiments.early_experience.pipeline import write_jsonl
from experiments.early_experience.train import encode_example, stages
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.mock.environment import get_environment


def call(name, **arguments):
    return AssistantMessage(
        role="assistant", tool_calls=[ToolCall(id="a", name=name, arguments=arguments)]
    )


def test_branch_mutation_isolation_and_batch_order():
    env = get_environment()
    before = env.tools.db.model_dump()
    user_id = next(iter(env.tools.db.users))
    action = call("create_task", user_id=user_id, title="branch only")
    result = probe(env, action)
    assert "branch only" in result["messages"][0]["content"]
    assert env.tools.db.model_dump() == before
    # Both writes in a batch must see the same cloned DB, hence different IDs.
    action.tool_calls.append(
        ToolCall(
            id="b",
            name="create_task",
            arguments={"user_id": user_id, "title": "second"},
        )
    )
    import json

    results = probe(env, action)["messages"]
    assert (
        json.loads(results[0]["content"])["task_id"]
        != json.loads(results[1]["content"])["task_id"]
    )
    assert env.tools.db.model_dump() == before


def test_invalid_actions_are_observations():
    result = probe(get_environment(), call("does_not_exist"))
    assert "Error" in result["messages"][0]["content"]


def test_dedup_ignores_id_but_not_order_or_arguments():
    first = call("get_users")
    second = deepcopy(first)
    second.tool_calls[0].id = "other"
    assert action_key(first) == action_key(second)
    second.tool_calls[0].arguments = {"bad": True}
    assert action_key(first) != action_key(second)
    with pytest.raises(ValueError):
        parse_action(
            {"content": "hi", "tool_calls": [{"name": "get_users", "arguments": {}}]}
        )


def test_private_user_tools_are_hidden():
    history = [
        UserMessage(role="user", content="Visible request"),
        UserMessage(
            role="user",
            tool_calls=[
                ToolCall(id="private", name="secret", arguments={}, requestor="user")
            ],
        ),
        ToolMessage(role="tool", id="private", content="SECRET", requestor="user"),
    ]
    context = visible_context("policy", history)
    assert len(context) == 2
    assert "SECRET" not in str(context)


def test_user_branch_tools_do_not_leak_or_mutate():
    env = get_environment()
    before = env.tools.db.model_dump()

    class User:
        def get_init_state(self, history):
            return SimpleNamespace(n=0)

        def generate_next_message(self, message, state):
            state.n += 1
            if state.n == 1:
                return UserMessage(
                    role="user",
                    tool_calls=[
                        ToolCall(
                            id="u",
                            name="missing_private_tool",
                            arguments={},
                            requestor="user",
                        )
                    ],
                ), state
            return UserMessage(role="user", content="observed user response"), state

        def is_stop(self, message):
            return False

    response = probe(
        env, AssistantMessage(role="assistant", content="Please check"), User()
    )
    assert response["messages"] == [
        {"role": "user", "content": "observed user response"}
    ]
    assert env.tools.db.model_dump() == before


def test_sft_targets_and_reflection_transport():
    context = visible_context("policy", [UserMessage(role="user", content="Help")])
    action = call("get_users")
    expert = make_expert(context, [], action)
    reflection = make_reflection(expert, "I should inspect the available users.")
    assert (
        reflection["messages"][-1]["tool_calls"] == expert["messages"][-1]["tool_calls"]
    )
    assert strip_reflection(reflection["messages"][-1]["content"]) is None
    assert strip_reflection("<think>private</think>Hello") == "Hello"
    with pytest.raises(ValueError):
        strip_reflection("<think>truncated")
    iwm = make_iwm(context, [], action, {"messages": []})
    assert iwm["messages"][0] != expert["messages"][0]
    assert iwm["messages"][-1]["content"].startswith("Observation:")


def test_split_rejection():
    validate_split(["a"], ["b"])
    for train, test in [(["a"], ["a"]), ([], ["b"]), (["a", "a"], ["b"])]:
        with pytest.raises(ValueError):
            validate_split(train, test)


class Tokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        value = "".join(f"[{m['role']}]{m.get('content') or ''}[end]" for m in messages)
        return value + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(x) for x in text]}


def test_final_completion_only_loss_and_no_truncation():
    row = {
        "messages": [
            {"role": "user", "content": "secret prompt"},
            {"role": "assistant", "content": "target"},
        ]
    }
    encoded = encode_example(Tokenizer(), row, 1000)
    learned = "".join(chr(x) for x in encoded["labels"] if x != -100)
    assert learned == "target[end]"
    with pytest.raises(ValueError, match="no silent truncation"):
        encode_example(Tokenizer(), row, 3)


def test_curricula(tmp_path):
    for name, label in [("expert", "il"), ("iwm", "iwm"), ("reflection", "sr")]:
        write_jsonl(tmp_path / f"{name}_sft.jsonl", [{"label": label}])
    curriculum = stages("iwm", tmp_path, 2, 1)
    assert [(name, rows[0]["label"], epochs) for name, rows, epochs in curriculum] == [
        ("iwm_warmup", "iwm", 1),
        ("final", "il", 2),
    ]
    assert [r["label"] for r in stages("sr", tmp_path, 1, 1)[0][1]] == ["il", "sr"]


def test_native_results_replay_export_and_resume(tmp_path):
    from experiments.early_experience.pipeline import (
        export_sft,
        generate_records,
        read_jsonl,
    )
    from tau2.data_model.simulation import (
        Results,
        SimulationRun,
        TerminationReason,
        TextRunConfig,
    )
    from tau2.runner import get_info, get_tasks

    env = get_environment()
    task = get_tasks("mock", task_ids=["create_task_1"])[0]
    action = call("get_users")
    messages = [
        AssistantMessage(role="assistant", content="Hello"),
        UserMessage(role="user", content="List users"),
        action,
        env.get_response(action.tool_calls[0]),
    ]
    source = Results(
        info=get_info(
            TextRunConfig(domain="mock", llm_agent="test-teacher", llm_user="test-user")
        ),
        tasks=[task],
        simulations=[
            SimulationRun(
                id="offline",
                task_id=task.id,
                start_time="2026-01-01",
                end_time="2026-01-01",
                duration=0,
                termination_reason=TerminationReason.USER_STOP,
                messages=messages,
            )
        ],
    )
    source_path = tmp_path / "source.json"
    source.save(source_path)
    split = tmp_path / "split.json"
    split.write_text(
        '{"domain":"mock","train_ids":["create_task_1"],"eval_ids":["held_out"]}'
    )

    class OfflineGenerator:
        model = "offline-test"
        args = {}
        calls = 0

        def alternatives(self, context, tools, expert, k):
            self.calls += 1
            return [call("unknown_tool")], {}

        def reflection(self, *args):
            return (
                "I should use the available user lookup instead of an unsupported tool.",
                {},
            )

    generator = OfflineGenerator()
    directory = tmp_path / "data"
    generate_records(source_path, split, directory, generator, k=1)
    export_sft(directory)
    assert len(read_jsonl(directory / "expert_sft.jsonl")) == 1
    assert (
        "Error" in read_jsonl(directory / "iwm_sft.jsonl")[0]["messages"][-1]["content"]
    )
    generate_records(source_path, split, directory, generator, k=1)
    assert generator.calls == 1
    with pytest.raises(ValueError, match="configuration differs"):
        generate_records(source_path, split, directory, generator, k=2)


def test_native_evaluator_with_reflection_agent(monkeypatch):
    from experiments.early_experience.agent import create_agent
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import build_text_orchestrator, get_tasks, run_simulation

    if registry.get_agent_factory("ee_agent") is None:
        registry.register_agent_factory(create_agent, "ee_agent")
    action = call("create_task", user_id="user_1", title="Important Meeting")
    action.content = "<think>I should create the requested meeting task.</think>"
    replies = iter(
        [
            action,
            AssistantMessage(
                role="assistant", content="<think>The task is created.</think>Done."
            ),
        ]
    )
    users = iter(
        [
            AssistantMessage(
                role="assistant", content="Create Important Meeting for user_1."
            ),
            AssistantMessage(role="assistant", content="###STOP###"),
        ]
    )
    from tau2.utils.utils import get_now

    def reply(sequence):
        result = next(sequence)
        result.timestamp = get_now()
        return result

    monkeypatch.setattr(
        "tau2.agent.llm_agent.generate", lambda **kwargs: reply(replies)
    )
    monkeypatch.setattr(
        "tau2.user.user_simulator.generate", lambda **kwargs: reply(users)
    )
    config = TextRunConfig(
        domain="mock", agent="ee_agent", llm_agent="offline", llm_user="offline"
    )
    task = get_tasks("mock", task_ids=["create_task_1"])[0]
    orchestrator = build_text_orchestrator(config, task)
    result = run_simulation(orchestrator)
    assert result.reward_info.reward == 1
    assert all("<think>" not in (m.content or "") for m in result.messages)


@pytest.mark.parametrize("domain", ["retail", "airline", "telecom"])
def test_real_domain_lookup_branch(domain):
    from tau2.runner import build_environment

    env = build_environment(domain)
    before = env.tools.db.model_dump()
    if domain == "telecom":
        customer = env.tools.db.customers[0]
        action = call("get_customer_by_id", customer_id=customer.customer_id)
    else:
        user_id = next(iter(env.tools.db.users))
        action = call("get_user_details", user_id=user_id)
    result = probe(env, action)
    assert "Error:" not in result["messages"][0]["content"]
    assert env.tools.db.model_dump() == before
