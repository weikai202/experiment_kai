from __future__ import annotations

import pytest

from bfcl_pipeline.bfcl_adapter import parse_literal_calls, run_episode, score_with_official_checker
from bfcl_pipeline.online import CriticDecision, VisibleContext, run_policy_controller_critic_turn


def test_literal_parser_accepts_parallel_calls():
    calls = parse_literal_calls("[read('x'), write(key='x', value={'a': [1, 2]})]")
    assert [x.name for x in calls] == ["read", "write"]
    assert calls[1].kwargs[1][1] == {"a": [1, 2]}


@pytest.mark.parametrize("text", [
    "obj.read('x')", "read(*xs)", "read(**kw)", "__import__('os')", "read(x + 1)", "[read('x'), broken]", "<tool_call>bad",
])
def test_literal_parser_rejects_executable_or_nonliteral_forms(text):
    with pytest.raises((ValueError, TypeError)):
        parse_literal_calls(text)


class Policy:
    def draft(self, context): return "read('x')"


class Critic:
    def __init__(self, decision): self.decision = decision
    def review(self, context, draft, calls): return CriticDecision(self.decision, "ok", "bounded")


class Revision:
    def revise(self, context, draft, critique): return "write(key='x', value=1)"


def test_online_policy_controller_critic_revision():
    context = VisibleContext((), ({"name": "read", "parameters": {"properties": {"key": {"type": "string"}}, "required": ["key"]}}, {"name": "write", "parameters": {"properties": {"key": {"type": "string"}, "value": {"type": "integer"}}, "required": ["key", "value"]}}), (), (), (), case_id="case-a", turn_index=1)
    result = run_policy_controller_critic_turn(context, Policy(), Critic("REVISE"), Revision())
    assert result.used_revision and result.final_calls[0].name == "write"
    assert result.committed_call_ids


def test_controller_rejects_unexposed_function():
    context = VisibleContext((), ({"name": "read", "parameters": {"properties": {"key": {"type": "string"}}}},), (), (), ())
    with pytest.raises(ValueError, match="unavailable"):
        run_policy_controller_critic_turn(context, Policy(), Critic("REVISE"), Revision())


class SequencePolicy:
    def __init__(self, outputs): self.outputs = iter(outputs)
    def respond(self, messages, functions): return next(self.outputs)


class FakeEnvironment:
    def __init__(self): self.executed = []
    def execute(self, calls):
        self.executed.append(tuple(x.name for x in calls))
        return tuple("ok" for _ in calls)
    def snapshot(self): return {"executed": self.executed}


def test_episode_emits_official_turn_step_call_shape_and_repeats_until_empty():
    case = {"id": "multi_turn_base_1", "question": [[{"role": "user", "content": "x"}]], "function": [{"name": "read"}]}
    result = run_episode(case, SequencePolicy(["read('x')", "read('y')", "[]"]), FakeEnvironment())
    assert result.predictions == ((("read('x')",), ("read('y')",)),)
    assert result.observation_batches == ((('ok',), ('ok',)),)


def test_episode_blocks_unexposed_function_before_execution():
    env = FakeEnvironment()
    case = {"id": "multi_turn_miss_func_1", "question": [[{"role": "user", "content": "x"}]], "function": [{"name": "read"}]}
    with pytest.raises(ValueError, match="not currently exposed"):
        run_episode(case, SequencePolicy(["write('x')"]), env)
    assert env.executed == []


def test_official_checker_seam_receives_turn_step_call_shape(monkeypatch):
    pytest.importorskip("bfcl_eval")
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_checker as module

    seen = {}
    def checker(predictions, ground_truth, case, category, model_name):
        seen["predictions"] = predictions
        assert isinstance(predictions[0][0][0], str)
        return {"valid": False, "error_type": "test"}
    monkeypatch.setattr(module, "multi_turn_checker", checker)
    case = {"id": "multi_turn_base_1", "involved_classes": [], "initial_config": {}}
    predictions = [[['read("x")']]]
    result = score_with_official_checker(case, predictions, [[['read("x")']]])
    assert result["valid"] is False and seen["predictions"] == predictions
