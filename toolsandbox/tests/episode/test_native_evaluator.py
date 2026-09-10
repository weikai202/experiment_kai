from collections import OrderedDict

from tool_sandbox.common.evaluation import EvaluationResult

from toolsandbox_pipeline.toolsandbox_adapter.native_evaluator import NativeEvaluator


class Evaluation:
    def __init__(self):
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return EvaluationResult(
            milestone_mapping=OrderedDict([(0, (2, 1.0))]),
            minefield_mapping=OrderedDict(),
            milestone_similarity=1.0,
            minefield_similarity=0.0,
            turn_count=2,
        )


class Scenario:
    max_messages = 4

    def __init__(self):
        self.evaluation = Evaluation()


def test_evaluator_is_exactly_once_and_reused(
    trajectory_store, start_context, episode_identity
):
    scenario = Scenario()
    evaluator = NativeEvaluator(trajectory_store)
    first = evaluator.evaluate(
        identity=episode_identity, scenario=scenario, ending_context=start_context
    )
    second = evaluator.evaluate(
        identity=episode_identity, scenario=scenario, ending_context=start_context
    )
    assert len(scenario.evaluation.calls) == 1
    assert scenario.evaluation.calls[0] == {
        "execution_context": start_context,
        "max_turn_count": 4,
    }
    assert first.record.fully_successful and second.reused
    assert first.reference == second.reference
