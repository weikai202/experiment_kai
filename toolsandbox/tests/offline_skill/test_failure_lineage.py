import pytest

from toolsandbox_pipeline.offline.failure_lineage import (
    FailureLineageIndex,
    build_lineage,
    link_accepted_skill,
)


DIGEST = "sha256:" + "a" * 64


def test_lineage_is_exact_idempotent_and_acceptance_linked():
    record = build_lineage(
        skill_id="skill",
        evidence_kind="fixture_miss",
        canonical_tool_dependencies=("search_weather_around_lat_lon",),
        sanitized_outcome_class="EXTERNAL_FIXTURE_MISS",
        mode_id="fm_mode",
        source_evidence_sha256=DIGEST,
        producing_round=0,
        failure_mode_effect_id="effect-failure",
    )
    index = FailureLineageIndex()
    assert index.add(record) == index.add(record)
    linked = link_accepted_skill(
        record,
        accepted_skill_version="v1.1",
        accepted_skill_effect_id="effect-skill",
    )
    assert linked.accepted_skill_version == "v1.1"
    with pytest.raises(ValueError):
        build_lineage(
            skill_id="skill",
            evidence_kind="fixture_miss",
            canonical_tool_dependencies=("z", "a"),
            sanitized_outcome_class="MISS",
            mode_id="fm_mode",
            source_evidence_sha256=DIGEST,
            producing_round=0,
            failure_mode_effect_id="effect-failure",
        )
