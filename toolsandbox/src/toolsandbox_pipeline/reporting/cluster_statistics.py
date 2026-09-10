"""Fixed paired family-cluster bootstrap for final evaluation."""

from __future__ import annotations

from math import ceil, fsum
from random import Random

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.reporting import (
    ClusterBootstrapReport,
    PairwiseBootstrapResult,
    ScenarioEvaluationRecord,
)


class ClusterStatisticsError(ValueError):
    """The frozen family design is incomplete or inconsistent."""


def bootstrap_index_stream() -> tuple[tuple[int, ...], ...]:
    """Return the frozen Python-3.10 Random(0) family-index stream."""

    rng = Random(0)
    return tuple(tuple(rng.randrange(25) for _ in range(25)) for _ in range(10000))


def nearest_rank(values: list[float], probability: float) -> float:
    """Return the documented one-based nearest-rank percentile."""

    if not values or not 0.0 < probability <= 1.0:
        raise ClusterStatisticsError("invalid nearest-rank input")
    ordered = sorted(values)
    return ordered[ceil(probability * len(ordered)) - 1]


def paired_cluster_bootstrap(
    records: tuple[ScenarioEvaluationRecord, ...],
) -> ClusterBootstrapReport:
    """Run the frozen 25-family, 10,000-replicate paired bootstrap."""

    by_system_family: dict[tuple[str, str], list[float]] = {}
    score_by_scenario: dict[tuple[str, str], dict[str, float]] = {}
    membership: dict[tuple[str, str], dict[str, str]] = {}
    seen_system_scenarios: set[tuple[str, str]] = set()
    for record in records:
        scenario_key = (record.system_id, record.scenario_id)
        if scenario_key in seen_system_scenarios:
            raise ClusterStatisticsError("duplicate system/scenario result")
        seen_system_scenarios.add(scenario_key)
        by_system_family.setdefault(
            (record.system_id, record.scenario_family_id), []
        ).append(record.similarity)
        score_by_scenario.setdefault(
            (record.system_id, record.scenario_family_id), {}
        )[record.scenario_id] = record.similarity
        membership.setdefault(
            (record.system_id, record.scenario_family_id), {}
        )[record.scenario_id] = record.variant
    family_ids = sorted(
        {
            family_id
            for system_id, family_id in by_system_family
            if system_id == "vanilla"
        },
        key=lambda item: item.encode("utf-8"),
    )
    if len(family_ids) != 25:
        raise ClusterStatisticsError(
            "formal cluster bootstrap requires exactly 25 families"
        )
    systems = ("vanilla", "generation_0", "updated")
    if len(records) != 25 * 8 * 3:
        raise ClusterStatisticsError("cluster matrix must contain exactly 600 records")
    vanilla_family_set = set(family_ids)
    if any(
        {family for candidate, family in by_system_family if candidate == system_id}
        != vanilla_family_set
        for system_id in systems
    ):
        raise ClusterStatisticsError("family membership differs across systems")
    family_means: dict[tuple[str, str], float] = {}
    for system_id in systems:
        for family_id in family_ids:
            values = by_system_family.get((system_id, family_id))
            if values is None or len(values) != 8:
                raise ClusterStatisticsError(
                    "each system/family requires exactly eight variants"
                )
            if membership[(system_id, family_id)] != membership[("vanilla", family_id)]:
                raise ClusterStatisticsError(
                    "scenario/variant membership differs across systems"
                )
            if len(set(membership[(system_id, family_id)].values())) != 8:
                raise ClusterStatisticsError("family variant identities must be unique")
            ordered_scenarios = sorted(
                membership[("vanilla", family_id)], key=lambda item: item.encode("utf-8")
            )
            family_means[(system_id, family_id)] = fsum(
                score_by_scenario[(system_id, family_id)][scenario_id]
                for scenario_id in ordered_scenarios
            ) / 8
    index_stream = bootstrap_index_stream()
    pairs = (
        ("updated", "vanilla"),
        ("updated", "generation_0"),
        ("generation_0", "vanilla"),
    )
    pair_results: list[PairwiseBootstrapResult] = []
    for left, right in pairs:
        observed = fsum(
            family_means[(left, family_id)]
            - family_means[(right, family_id)]
            for family_id in family_ids
        ) / 25
        replicates = [
            fsum(
                family_means[(left, family_ids[index])]
                - family_means[(right, family_ids[index])]
                for index in indices
            )
            / 25
            for indices in index_stream
        ]
        pair_results.append(
            PairwiseBootstrapResult(
                left_system=left,
                right_system=right,
                observed_difference=observed,
                percentile_95_lower=nearest_rank(replicates, 0.025),
                percentile_95_upper=nearest_rank(replicates, 0.975),
            )
        )
    return ClusterBootstrapReport(
        index_stream_sha256=canonical_sha256([list(item) for item in index_stream]),
        pairs=tuple(pair_results),
    )


__all__ = [
    "ClusterStatisticsError", "bootstrap_index_stream", "nearest_rank",
    "paired_cluster_bootstrap",
]
