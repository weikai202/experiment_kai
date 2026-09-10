import pytest

from toolsandbox_pipeline.reporting.cluster_statistics import (
    ClusterStatisticsError,
    bootstrap_index_stream,
    nearest_rank,
    paired_cluster_bootstrap,
)
from toolsandbox_pipeline.schemas.reporting import ScenarioEvaluationRecord

D = "sha256:" + "2" * 64


def record(system, family, variant, position, score):
    return ScenarioEvaluationRecord(
        system_id=system,
        manifest_position=position,
        scenario_id=f"{family}-{variant}",
        scenario_family_id=family,
        variant=f"v{variant}",
        categories=("category",),
        episode_id=f"episode-{system}-{family}-{variant}",
        trajectory_sha256=D,
        evaluator_record_sha256=D,
        similarity=score,
        milestone_similarity=score,
        minefield_similarity=0.0,
        fully_successful=score == 1.0,
        effective_turn_count=1,
    )


def test_nearest_rank_golden_vector():
    assert nearest_rank([4.0, 1.0, 3.0, 2.0], 0.25) == 1.0
    assert nearest_rank([4.0, 1.0, 3.0, 2.0], 0.75) == 3.0


def test_fixed_bootstrap_is_deterministic_and_family_clustered():
    records = []
    for family_index in range(25):
        family = f"family-{family_index:02d}"
        for variant in range(8):
            position = family_index * 8 + variant
            records.extend((
                record("vanilla", family, variant, position, 0.0),
                record("generation_0", family, variant, position, 0.5),
                record("updated", family, variant, position, 1.0),
            ))
    first = paired_cluster_bootstrap(tuple(records))
    second = paired_cluster_bootstrap(tuple(reversed(records)))
    assert first == second
    assert tuple(item.observed_difference for item in first.pairs) == (1.0, 0.5, 0.5)
    # Python 3.10 random.Random(0), first replicate, sampled family indices.
    assert bootstrap_index_stream()[0] == (12, 24, 13, 1, 8, 16, 15, 12, 9, 15, 11, 18, 6, 16, 4, 9, 4, 24, 3, 19, 8, 17, 22, 19, 4)
    assert first.index_stream_sha256 == "sha256:1a5262520ca432ba23f97522af19eaaabb437e1d50d83fdd709877405ad00050"


def test_bootstrap_rejects_adaptive_family_count():
    with pytest.raises(ClusterStatisticsError, match="exactly 25"):
        paired_cluster_bootstrap(())


def test_bootstrap_rejects_extra_nonshared_family():
    records = []
    for family_index in range(25):
        family = f"family-{family_index:02d}"
        for variant in range(8):
            position = family_index * 8 + variant
            records.extend(record(system, family, variant, position, 1.0) for system in ("vanilla", "generation_0", "updated"))
    records[-1] = record("updated", "extra-family", 0, 199, 1.0)
    with pytest.raises(ClusterStatisticsError, match="family membership"):
        paired_cluster_bootstrap(tuple(records))
