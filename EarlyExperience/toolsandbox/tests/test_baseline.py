import json
import random

import polars as pl
import pytest
from tool_sandbox.common.evaluation import (
    Evaluation,
    Milestone,
    MilestoneMatcher,
    SnapshotConstraint,
    snapshot_similarity,
)
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
    new_context,
)
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.execution_environment import ExecutionEnvironment

from ee_toolsandbox.adapter import PolicyAgent, probe, visible_state
from ee_toolsandbox.pipeline import Recorder, prepare, read_jsonl
from ee_toolsandbox.protocol import (
    canonical,
    parse_action,
    policy_messages,
    sft_records,
    validate_manifest,
)
from ee_toolsandbox.training import encode_example


def call(name, **arguments):
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def context():
    ctx = ExecutionContext(tool_allow_list=["set_wifi_status", "get_wifi_status"])
    with new_context(ctx):
        BaseRole.add_messages(
            [
                Message(
                    sender=RoleType.SYSTEM,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content="from tool_sandbox.tools.setting import set_wifi_status, get_wifi_status",
                ),
                Message(
                    sender=RoleType.SYSTEM,
                    recipient=RoleType.AGENT,
                    content="Assist with phone settings.",
                ),
                Message(
                    sender=RoleType.SYSTEM,
                    recipient=RoleType.USER,
                    content="PRIVATE_USER_INSTRUCTIONS",
                ),
                Message(
                    sender=RoleType.USER,
                    recipient=RoleType.AGENT,
                    content="Turn off Wi-Fi.",
                ),
            ]
        )
        ExecutionEnvironment().respond(ending_index=0)
    return ctx


class EndUser(BaseRole):
    role_type = RoleType.USER

    def respond(self, ending_index=None):
        self.add_messages(
            [
                Message(
                    sender=RoleType.USER,
                    recipient=RoleType.AGENT,
                    content="Thanks.",
                    conversation_active=False,
                )
            ]
        )


class DemoPolicy:
    def action(self, state):
        if state["conversation"][-1]["role"] == "tool":
            return {"response": "Wi-Fi is off."}
        return call("set_wifi_status", on=False)


class Proposer:
    def alternatives(self, state, expert, k):
        choices = [
            call("get_wifi_status"),
            call("set_wifi_status", on=True),
            {"response": "Please clarify."},
        ]
        return [a for a in choices if a != expert][:k]


def test_branch_isolation_and_real_errors():
    ctx = context()
    before = ctx.to_dict(serialize_console=False)
    rng = random.getstate()
    with new_context(ctx):
        off = probe(ctx, call("set_wifi_status", on=False), EndUser())
        get = probe(ctx, call("get_wifi_status"), EndUser())
        bad = probe(ctx, call("set_wifi_status", on="bad"), EndUser())
        assert get_current_context() is ctx
        assert ctx.to_dict(serialize_console=False) == before
        assert random.getstate() == rng
        assert off["messages"][0]["content"] == "None"
        assert get["messages"][0]["content"] == "True"
        assert "Error" in bad["messages"][0]["content"]
        with pytest.raises(ValueError):
            probe(ctx, call("not_allowed"), EndUser())
        assert get_current_context() is ctx


def test_visibility_and_format():
    with new_context(context()):
        state = visible_state()
        assert "PRIVATE_USER_INSTRUCTIONS" not in canonical(state)
        assert "latitude" not in canonical(state)
        assert {t["function"]["name"] for t in state["tools"]} == {
            "set_wifi_status",
            "get_wifi_status",
        }
        action = call("set_wifi_status", on=False)
        assert (
            parse_action(
                "<reflection>Check settings.</reflection>\n" + canonical(action)
            )
            == action
        )
        with pytest.raises(ValueError):
            parse_action('{"response":"x","tool_calls":[]}')


def test_native_episode_and_dataset(tmp_path):
    records = []
    user = EndUser()
    recorder = Recorder("synthetic_wifi", Proposer(), user, 2, records.append)
    scenario = Scenario(
        starting_context=context(),
        max_messages=10,
        evaluation=Evaluation(
            milestone_matcher=MilestoneMatcher(
                milestones=[
                    Milestone(
                        snapshot_constraints=[
                            SnapshotConstraint(
                                database_namespace=DatabaseNamespace.SETTING,
                                snapshot_constraint=snapshot_similarity,
                                target_dataframe=pl.DataFrame({"wifi": [False]}),
                            )
                        ]
                    )
                ]
            )
        ),
    )
    result = scenario.play_and_evaluate(
        {
            RoleType.AGENT: PolicyAgent(DemoPolicy(), recorder),
            RoleType.USER: user,
            RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
        },
        tmp_path / "native",
        "synthetic_wifi",
    )
    assert result.evaluation_result.similarity == 1
    assert len(records) == 2
    assert all(len(r["alternatives"]) == 2 for r in records)
    for r in records:
        for a in r["alternatives"]:
            a["reflection"] = "I should act on the current request."
        expert, iwm, sr = sft_records(r)
        assert expert["messages"][:-1] == policy_messages(r["state"])
        assert sr[0]["messages"][:-1] == expert["messages"][:-1]
        assert iwm[0]["messages"][0] != expert["messages"][0]
        assert len(iwm) == 3
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in records))
    counts = prepare(
        raw, tmp_path / "sft", {"scenario_ids": ["synthetic_wifi"], "split": "train"}
    )
    assert counts == {"expert": 2, "iwm": 6, "reflection": 4}
    assert len(list(read_jsonl(tmp_path / "sft" / "iwm_sft.jsonl"))) == 6


def test_family_leakage():
    with pytest.raises(ValueError, match="family leakage"):
        validate_manifest(
            {
                "train": [{"id": "a", "family": "wifi"}],
                "dev": [],
                "test": [{"id": "a_augmented", "family": "wifi"}],
            }
        )
    assert validate_manifest({"train": [], "dev": [], "test": []})


class Tokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        return "".join(m["role"] + ":" + m["content"] + ";" for m in messages) + (
            "assistant:" if add_generation_prompt else ""
        )

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode())}


def test_completion_mask_and_overlength():
    messages = [
        {"role": "user", "content": "input"},
        {"role": "assistant", "content": "target"},
    ]
    encoded = encode_example(Tokenizer(), messages, 1000)
    labels = [x for x in encoded["labels"] if x != -100]
    assert bytes(labels).decode() == "target;"
    with pytest.raises(ValueError, match="no silent truncation"):
        encode_example(Tokenizer(), messages, 3)


def test_augmented_ids_cannot_evade_family_check():
    with pytest.raises(ValueError, match="augmented scenario leakage"):
        validate_manifest(
            {
                "train": [{"id": "synthetic", "family": "x"}],
                "dev": [],
                "test": [
                    {
                        "id": "synthetic_3_distraction_tools_tool_name_scrambled",
                        "family": "y",
                    }
                ],
            }
        )


def test_parallel_branch_restores_context_and_hashes():
    from ee_toolsandbox.adapter import advance, apply_action

    original = context()
    with new_context(original):
        before = original.to_dict(serialize_console=False)
        action = {
            "tool_calls": [
                call("get_wifi_status")["tool_calls"][0],
                call("get_wifi_status")["tool_calls"][0],
            ]
        }
        observed = probe(original, action, EndUser())
        assert len(observed["messages"]) == 2
        assert original.to_dict(serialize_console=False) == before
    states = []
    for _ in range(2):
        with new_context(context()):
            apply_action(call("get_wifi_status"))
            advance(EndUser())
            states.append(visible_state())
    assert states[0] == states[1]


def test_standalone_install_has_no_pipeline_dependency():
    from importlib.metadata import requires

    assert not any(
        "toolsandbox-pipeline" in requirement.lower()
        for requirement in requires("earlyexperience-toolsandbox")
    )


def test_standalone_model_nonthinking_and_failed_usage():
    from types import SimpleNamespace

    from openai.types.chat import ChatCompletion

    from ee_toolsandbox.clients import ModelClient

    seen = []
    client = object.__new__(ModelClient)
    client.model = "Qwen/Qwen3-32B"
    client.config, client.usage = {"seed": 0, "enable_thinking": False}, []

    def create(**kwargs):
        seen.append(kwargs)
        return ChatCompletion(
            id="fixture",
            created=0,
            model=client.model,
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"response":"Hello"}'},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    assert client.complete([]) == '{"response":"Hello"}'
    assert seen[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert seen[0]["seed"] == 0

    def fail(**kwargs):
        raise TimeoutError("fixture")

    client.client.chat.completions.create = fail
    with pytest.raises(TimeoutError):
        client.complete([])
    assert client.usage[-1]["usage"] is None and client.usage[-1]["status"] == "failed"


def test_prepare_respects_completed_expert_selection(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"scenario_id": "failed"}) + "\n")
    with pytest.raises(ValueError, match="No expert records"):
        prepare(
            raw,
            tmp_path / "out",
            {"scenario_ids": ["failed"], "accepted_scenario_ids": []},
        )
    assert not (tmp_path / "out/provenance.json").exists()


def test_standalone_evaluation_entrypoint_uses_native_evaluator(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import tool_sandbox.scenarios as registry

    from ee_toolsandbox import clients
    from ee_toolsandbox.cli import run

    scenario = Scenario(
        starting_context=context(),
        max_messages=10,
        evaluation=Evaluation(
            milestone_matcher=MilestoneMatcher(
                milestones=[
                    Milestone(
                        snapshot_constraints=[
                            SnapshotConstraint(
                                database_namespace=DatabaseNamespace.SETTING,
                                snapshot_constraint=snapshot_similarity,
                                target_dataframe=pl.DataFrame({"wifi": [False]}),
                            )
                        ]
                    )
                ]
            )
        ),
    )
    monkeypatch.setattr(
        registry, "named_scenarios", lambda **kwargs: {"synthetic_wifi": scenario}
    )
    monkeypatch.setattr(clients, "ModelClient", lambda *args: DemoPolicy())
    monkeypatch.setattr(clients, "UserSimulator", lambda *args: EndUser())
    config, manifest = tmp_path / "config.json", tmp_path / "manifest.json"
    config.write_text(json.dumps({"seed": 0, "policy": {}, "user": {}}))
    manifest.write_text(
        json.dumps(
            {
                "train": [],
                "dev": [{"id": "synthetic_wifi", "family": "synthetic_wifi"}],
                "test": [],
            }
        )
    )
    run(
        SimpleNamespace(
            command="evaluate",
            config=str(config),
            manifest=str(manifest),
            split="dev",
            output=str(tmp_path / "eval"),
            experts=None,
        )
    )
    result = json.loads((tmp_path / "eval/run.json").read_text())
    assert result["complete"] and result["mean_similarity"] == 1.0
