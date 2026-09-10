from toolsandbox_pipeline.offline.failure_lineage import FailureLineageIndex, build_lineage


def test_identical_lineage_replay_is_idempotent():
    digest = "sha256:" + "e" * 64
    record = build_lineage(
        skill_id="skill", evidence_kind="controller_rejection",
        canonical_tool_dependencies=("search_stock",),
        sanitized_outcome_class="MISSING_DEPENDENCY", mode_id="fm_mode",
        source_evidence_sha256=digest, producing_round=1,
        failure_mode_effect_id="effect",
    )
    index = FailureLineageIndex()
    assert index.add(record) is record
    assert index.add(record) == record
    assert index.records() == (record,)
