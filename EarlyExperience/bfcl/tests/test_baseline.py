import copy
import json
from pathlib import Path

import pytest

from ee_bfcl.client import transport_messages
from ee_bfcl.collect import IWM_SYSTEM, build_iwm, collect, parse_proposals
from ee_bfcl.environment import Environment, action_text, load_cases, parse_calls, score_case
from ee_bfcl.evaluate import evaluate
from ee_bfcl.io import read_jsonl, write_json, write_jsonl
from ee_bfcl.prepare import prepare, successful_ids
from ee_bfcl.train import encode_example, plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bad", ["__import__('os').system('id')", "pwd(__import__('os'))", "[x for x in []]",
                                 "pwd(**{})", "pwd(a=1,a=2)", "_load_scenario({})"])
def test_reject_executable_python(bad):
    with pytest.raises((ValueError, SyntaxError)):
        parse_calls(bad)


def test_decode_actions_and_reflection():
    assert parse_calls("<reflection>\nCheck the directory.\n</reflection>\n[pwd(), ls(a=True)]") == ["pwd()", "ls(a=True)"]
    assert parse_calls("[]") == []
    assert parse_calls("I need a file name.") == []
    with pytest.raises(ValueError):
        parse_calls("<reflection>unfinished")


def test_probe_isolation_and_official_output_parity():
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    import uuid
    case = load_cases("multi_turn_base")[0][0]
    env = Environment(case)
    before = env.snapshot()
    calls = ["cd(folder='document')", "mkdir(dir_name='temp')"]
    observation = env.probe(calls)
    assert env.snapshot() == before
    assert observation["post_state"] != before
    official, _ = execute_multi_turn_func_call(calls, case["initial_config"], case["involved_classes"],
                                              "ee_" + uuid.uuid4().hex, case["id"])
    assert observation["outputs"] == official
    assert env.probe(["cd(folder='does-not-exist')"])["outputs"]
    assert env.snapshot() == before


@pytest.mark.parametrize("category", ["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"])
def test_official_scoring_reference_replay(category):
    cases, truth = load_cases(category)
    case = cases[0]
    prediction = [[[call] for call in turn] for turn in truth[case["id"]]]
    assert score_case(case, prediction, truth[case["id"]])["valid"]
    assert not score_case(case, [[] for _ in prediction], truth[case["id"]])["valid"]
    # A second run must not inherit simulator state from the first run.
    assert score_case(case, prediction, truth[case["id"]])["valid"]


def test_scores_failures_only_and_truncated_rejected():
    results = [{"id": "a"}, {"id": "b"}]
    score = [{"total_count": 2, "correct_count": 1}, {"id": "b", "valid": False}]
    assert successful_ids(results, score) == ["a"]
    with pytest.raises(ValueError):
        successful_ids(results, score[:1])


def test_proposal_shape_and_iwm_separation():
    assert parse_proposals('[{"name":"pwd","arguments":{}}]', ["pwd"]) == [["pwd()"]]
    with pytest.raises(ValueError):
        parse_proposals('[{"name":"ls","arguments":{}}]', ["pwd"])
    row = build_iwm([{"role": "user", "content": "List files"}], ["pwd()"], "Current directory is workspace.")
    assert row["messages"][0]["content"] == IWM_SYSTEM
    assert row["messages"][-1]["content"].startswith("Observation:\n")


class TinyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return [10, 11, 12]

    def encode(self, text, **kwargs):
        return [20] * len(text)


def test_loss_mask_and_no_silent_truncation():
    messages = [{"role": "user", "content": "list"}, {"role": "assistant", "content": "[]"}]
    row = encode_example(TinyTokenizer(), messages, 10)
    assert row["labels"] == [-100, -100, -100, 20, 20, 2]
    with pytest.raises(ValueError):
        encode_example(TinyTokenizer(), messages, 5)


def test_tool_transport():
    original = [{"role": "tool", "name": "pwd()", "content": "one"},
                {"role": "tool", "name": "ls()", "content": "two"}]
    result = transport_messages(original)
    assert len(result) == 1
    assert result[0]["content"] == "<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response>"
    assert original[0]["role"] == "tool"


class FixtureGenerator:
    """TEST ONLY: actual simulator probes, synthetic LLM text. Never experiment data."""
    def ask(self, prompt):
        if "fill plausible arguments" in prompt:
            docs = json.loads(prompt.split("\nFunctions:\n", 1)[1])
            return json.dumps([{"name": d["name"], "arguments": {}} for d in docs])
        if prompt.startswith("Describe"):
            return "Fixture-only description of the provided real execution output."
        return "I inspect the available information before performing the requested operation."


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    source = ROOT / "data/source"
    if not (source / "opus_base_result.jsonl").exists():
        pytest.skip("Download public expert result and score files to run full harvester integration")
    path = tmp_path_factory.mktemp("prepared")
    stats = prepare(source / "opus_base_result.jsonl", source / "opus_base_score.jsonl", path)
    assert stats["successful_cases"] == 162
    assert stats["replay_mismatches"] == 0
    return path


def test_full_harvest_split(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert len(manifest["train_ids"]) == 121
    assert len(manifest["heldout_ids"]) == 41
    assert not set(manifest["train_ids"]) & set(manifest["heldout_ids"])
    for r in read_jsonl(prepared / "expert_records.jsonl"):
        assert r["case_id"] in manifest["train_ids"]
        assert r["messages"][-1]["role"] == "assistant"


def test_five_state_collection_and_stage_lineage(prepared, tmp_path):
    config = json.loads((ROOT / "configs/qwen3_32b.json").read_text())
    result = collect(prepared, tmp_path / "collected", config, FixtureGenerator(), limit=5)
    assert result["states"] == 5
    assert result["iwm_samples"] == 50
    assert result["reflection_samples"] == 5
    assert result["partial"]
    # Repeat from per-state checkpoints, without touching a generator.
    assert collect(prepared, tmp_path / "collected", config, None, limit=5) == result
    with pytest.raises(ValueError, match="Partial"):
        plan(prepared, tmp_path / "collected", tmp_path / "training")
    stages = plan(prepared, tmp_path / "collected", tmp_path / "training", allow_partial=True)["stages"]
    assert stages[2]["model"] == stages[1]["output"]
    assert len(stages[3]["data"]) == 2
    assert stages[3]["model"] == "Qwen/Qwen3-32B"
    for row in read_jsonl(tmp_path / "collected/reflection_sft_text.jsonl"):
        assert row["messages"][-1]["content"].startswith("<reflection>")
        assert parse_calls(row["messages"][-1]["content"])


def test_evaluation_driver_oracle_and_resume(prepared, tmp_path):
    manifest = json.loads((prepared / "manifest.json").read_text())
    cases, truth = load_cases("multi_turn_base")
    case = next(c for c in cases if c["id"] in manifest["heldout_ids"])
    responses = iter([text for turn in truth[case["id"]] for text in
                      [*[action_text([call]) for call in turn], "[]"]])

    class Oracle:
        def complete(self, messages):
            return next(responses)

    config = {"model": "TEST-ONLY-ORACLE", "enable_thinking": False}
    report = evaluate(prepared, tmp_path, "multi_turn_base", Oracle(), config, limit=1)
    assert report["accuracy"] == 1
    assert evaluate(prepared, tmp_path, "multi_turn_base", None, config, limit=1) == report


def test_real_qwen3_non_thinking_template_and_supervision():
    tokenizer_dir = ROOT / 'data/tokenizer'
    if not (tokenizer_dir / 'tokenizer.json').exists():
        pytest.skip('Optional official Qwen3 tokenizer files not downloaded')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    history = [{'role': 'system', 'content': 'Call tools.'}, {'role': 'user', 'content': 'List files'},
               {'role': 'assistant', 'content': '[pwd(), ls()]'},
               {'role': 'tool', 'name': 'pwd()', 'content': 'workspace'},
               {'role': 'tool', 'name': 'ls()', 'content': '[]'}]
    native = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    transported = tokenizer.apply_chat_template(transport_messages(history), tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
    assert native == transported
    assert native.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    messages = history + [{'role': 'assistant', 'content': '[]'}]
    row = encode_example(tokenizer, messages, 4096)
    prompt_ids = [x for x, label in zip(row['input_ids'], row['labels']) if label == -100]
    assert tokenizer.decode(prompt_ids) == native
    target_ids = [x for x in row['labels'] if x != -100]
    assert tokenizer.decode(target_ids) == '[]<|im_end|>'


def test_api_non_thinking_cache_and_truncation(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import openai
    from ee_bfcl.client import Client
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(model='fixture', usage=None, choices=[SimpleNamespace(
            finish_reason='length' if len(requests) == 1 else 'stop',
            message=SimpleNamespace(content='[]', reasoning_content=None))])

    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    client = Client({'model': 'fixture', 'base_url': 'http://fixture.invalid/v1',
                     'enable_thinking': False}, tmp_path)
    messages = [{'role': 'user', 'content': 'List files'}]
    with pytest.raises(RuntimeError, match='Incomplete'):
        client.complete(messages)
    assert not list(tmp_path.glob('*.json'))
    assert client.complete(messages) == '[]'
    assert client.complete(messages) == '[]'
    assert len(requests) == 2
    assert requests[-1]['extra_body']['chat_template_kwargs']['enable_thinking'] is False


def test_proposal_repair_is_bounded_and_grounded(prepared, tmp_path):
    class RepairGenerator(FixtureGenerator):
        repairs = 0

        def ask(self, prompt):
            if prompt.startswith('Given the conversation'):
                if '\nPrevious response:' not in prompt:
                    return 'invalid json'
                self.repairs += 1
                prompt = prompt.split('\nPrevious response:')[0]
            return super().ask(prompt)

    generator = RepairGenerator()
    config = json.loads((ROOT / 'configs/qwen3_32b.json').read_text())
    result = collect(prepared, tmp_path, config, generator, limit=1)
    assert result['states'] == 1
    assert generator.repairs == 1
    state = json.loads(next((tmp_path / 'states').glob('*.json')).read_text())
    assert len(state['alternatives']) == 10
    assert all('outputs' in alt and 'post_state' in alt for alt in state['alternatives'])
