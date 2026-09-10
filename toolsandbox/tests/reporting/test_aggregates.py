from toolsandbox_pipeline.reporting.aggregates import aggregate_system, deterministic_median
from toolsandbox_pipeline.schemas.reporting import ScenarioEvaluationRecord

D = "sha256:" + "3" * 64


def row(position, score, categories, turns):
    return ScenarioEvaluationRecord(
        system_id="generation_0", manifest_position=position,
        scenario_id=f"s-{position}", scenario_family_id="family",
        variant=f"v-{position}", categories=categories,
        episode_id=f"e-{position}", trajectory_sha256=D,
        evaluator_record_sha256=D, similarity=score,
        milestone_similarity=score, minefield_similarity=0.0,
        fully_successful=score == 1.0, effective_turn_count=turns,
        critic_trigger_count=1, critic_accept_count=int(position == 0),
        critic_revise_count=int(position == 1), revision_count=int(position == 1),
    )


def test_native_and_category_aggregates_keep_denominators_distinct():
    result = aggregate_system(
        (row(0, 1.0, ("a", "b"), 1), row(1, 0.0, ("b",), 3)),
        system_id="generation_0", total_running_time_seconds=4.0,
        total_tokens=100, usage_complete=True, total_cost=10,
        cost_complete=True,
    )
    assert result.mean_similarity == 0.5
    assert result.micro_category_mean_similarity == 0.5
    assert result.macro_category_mean_similarity == 0.75
    assert tuple(item.scenario_count for item in result.category_metrics) == (1, 2)
    assert result.median_effective_turn_count == 2.0
    assert result.critic_accept_rate == 0.5
    assert result.critic_revise_rate == 0.5
    assert result.critic_trigger_rate == 0.5
    assert result.revision_rate == 0.25
    assert result.total_running_time_seconds == 4.0
    assert result.total_cost == 10


def test_incomplete_usage_and_cost_are_explicit_nulls():
    result = aggregate_system(
        (), system_id="updated", total_running_time_seconds=None,
        total_tokens=None, usage_complete=False, total_cost=None,
        cost_complete=False,
    )
    assert result.mean_similarity is None
    assert result.total_tokens is None and not result.usage_complete
    assert result.total_cost is None and not result.cost_complete
    assert deterministic_median((4, 1, 3, 2)) == 2.5
