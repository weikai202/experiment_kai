"""Deterministic native-score and accounting headline aggregation."""

from __future__ import annotations

from math import fsum

from toolsandbox_pipeline.schemas.reporting import (
    CategoryMetric,
    ScenarioEvaluationRecord,
    SystemAggregate,
    SystemId,
)


class AggregateError(ValueError):
    """Scenario or accounting inputs cannot form a complete aggregate."""


def deterministic_median(values: tuple[int, ...]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def aggregate_system(
    records: tuple[ScenarioEvaluationRecord, ...],
    *,
    system_id: SystemId,
    total_running_time_seconds: float | None,
    total_tokens: int | None,
    usage_complete: bool,
    total_cost: int | None,
    cost_complete: bool,
    failure_count: int = 0,
    timeout_count: int = 0,
    reconciliation_count: int = 0,
) -> SystemAggregate:
    """Aggregate completed trusted records in manifest order using ``fsum``."""

    if any(record.system_id != system_id for record in records):
        raise AggregateError("cross-system aggregate input")
    if tuple(record.manifest_position for record in records) != tuple(range(len(records))):
        raise AggregateError("records must use complete manifest order")
    if usage_complete != (total_tokens is not None):
        raise AggregateError("usage completeness mismatch")
    if cost_complete != (total_cost is not None):
        raise AggregateError("cost completeness mismatch")
    count = len(records)
    successful = sum(record.fully_successful for record in records)
    categories: dict[str, list[ScenarioEvaluationRecord]] = {}
    for record in records:
        for category in record.categories:
            categories.setdefault(category, []).append(record)
    category_metrics = tuple(
        CategoryMetric(
            category=category,
            scenario_count=len(members),
            mean_similarity=fsum(item.similarity for item in members) / len(members),
            fully_successful_rate=sum(item.fully_successful for item in members) / len(members),
        )
        for category, members in sorted(categories.items(), key=lambda item: item[0].encode("utf-8"))
    )
    critic_triggers = sum(item.critic_trigger_count for item in records)
    decision_turns = sum(item.effective_turn_count for item in records)
    external_events = sum(
        item.fixture_hit_count + item.fixture_miss_count + item.external_tool_exception_count
        for item in records
    )
    revision_records = [item for item in records if item.revision_count > 0]
    return SystemAggregate(
        system_id=system_id,
        scenario_count=count,
        family_count=len({item.scenario_family_id for item in records}),
        mean_similarity=None if count == 0 else fsum(item.similarity for item in records) / count,
        mean_milestone_similarity=None if count == 0 else fsum(item.milestone_similarity for item in records) / count,
        mean_minefield_similarity=None if count == 0 else fsum(item.minefield_similarity for item in records) / count,
        fully_successful_count=successful,
        fully_successful_rate=_rate(successful, count),
        mean_effective_turn_count=None if count == 0 else fsum(item.effective_turn_count for item in records) / count,
        median_effective_turn_count=deterministic_median(tuple(item.effective_turn_count for item in records)),
        critic_trigger_rate=_rate(critic_triggers, decision_turns),
        critic_accept_rate=_rate(sum(item.critic_accept_count for item in records), critic_triggers),
        critic_revise_rate=_rate(sum(item.critic_revise_count for item in records), critic_triggers),
        critic_uncertain_rate=_rate(sum(item.critic_uncertain_count for item in records), critic_triggers),
        revision_rate=_rate(sum(item.revision_count for item in records), decision_turns),
        mean_post_revision_similarity=None if not revision_records else fsum(item.similarity for item in revision_records) / len(revision_records),
        fixture_hit_rate=_rate(sum(item.fixture_hit_count for item in records), external_events),
        fixture_miss_rate=_rate(sum(item.fixture_miss_count for item in records), external_events),
        external_tool_exception_rate=_rate(sum(item.external_tool_exception_count for item in records), external_events),
        failure_count=failure_count,
        timeout_count=timeout_count,
        reconciliation_count=reconciliation_count,
        category_metrics=category_metrics,
        macro_category_mean_similarity=None if not category_metrics else fsum(item.mean_similarity for item in category_metrics) / len(category_metrics),
        micro_category_mean_similarity=None if count == 0 else fsum(item.similarity for item in records) / count,
        total_running_time_seconds=total_running_time_seconds,
        total_tokens=total_tokens,
        usage_complete=usage_complete,
        total_cost=total_cost,
        cost_complete=cost_complete,
    )


__all__ = ["AggregateError", "aggregate_system", "deterministic_median"]
