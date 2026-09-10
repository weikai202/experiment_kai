import json
from pathlib import Path

import pytest

from toolsandbox_pipeline.reporting.artifacts import (
    ArtifactError,
    ArtifactPublisher,
    verify_report,
)
from toolsandbox_pipeline.reporting.aggregates import aggregate_system
from toolsandbox_pipeline.reporting.cluster_statistics import paired_cluster_bootstrap
from toolsandbox_pipeline.reporting.failure_mode_analysis import (
    analyze_failure_modes,
    compute_family_stability,
)
from toolsandbox_pipeline.reporting.reproducibility import evaluate_reproducibility
from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.reporting import (
    CategoryResultRecord,
    MainTable,
    ScenarioEvaluationRecord,
    TrainingRoundHeadline,
)
from tests.reporting.test_evaluation_plan import build_plan
from tests.reporting.test_reproducibility import checks, smoke

D = "sha256:" + "5" * 64


def complete(root):
    plan = build_plan(root)
    publisher = ArtifactPublisher(root, plan.plan_sha256)
    records = tuple(
        ScenarioEvaluationRecord(
            system_id=system_id,
            manifest_position=item.manifest_position,
            scenario_id=item.scenario_id,
            scenario_family_id=item.scenario_family_id,
            variant=item.variant,
            categories=item.categories,
            episode_id=f"episode-{system_id}-{item.manifest_position}",
            trajectory_sha256=D,
            evaluator_record_sha256=D,
            similarity=1.0,
            milestone_similarity=1.0,
            minefield_similarity=0.0,
            fully_successful=True,
            effective_turn_count=1,
        )
        for system_id in plan.ordered_system_ids
        for item in plan.scenarios
    )
    attributions, failure_summary = analyze_failure_modes(
        lineage_records=(), scenario_records=records,
        generation_0_trajectories=(),
        generation_0_external_attempts=(),
        updated_trajectories=(),
    )
    family_rows, family_summaries = compute_family_stability(records, attributions)
    aggregates = tuple(
        aggregate_system(
            tuple(item for item in records if item.system_id == system_id),
            system_id=system_id,
            total_running_time_seconds=1.0,
            total_tokens=0,
            usage_complete=True,
            total_cost=0,
            cost_complete=True,
        )
        for system_id in plan.ordered_system_ids
    )
    category_rows = tuple(
        CategoryResultRecord(system_id=aggregate.system_id, metric=metric)
        for aggregate in aggregates
        for metric in aggregate.category_metrics
    )
    rounds = tuple(
        TrainingRoundHeadline(
            round_index=index,
            total_running_time_seconds=1.0,
            total_tokens=0,
            usage_complete=True,
            total_cost=0,
            cost_complete=True,
            immutable_round_record_sha256=D,
        )
        for index in range(3)
    )
    smoke_evidence = smoke()
    reproducibility = evaluate_reproducibility(
        profile="strict_replay",
        checks=checks(smoke_evidence=smoke_evidence.evidence_sha256),
        strict_replay_smoke_evidence=smoke_evidence,
    )
    main = MainTable(
        plan_sha256=plan.plan_sha256,
        system_aggregates=aggregates,
        training_round_headlines=rounds,
        failure_mode_repair_summary=failure_summary,
        family_stability_summaries=family_summaries,
    )
    publisher.publish_json("plan.json", plan.model_dump(mode="json"))
    publisher.publish_text("plan.sha256", plan.plan_sha256)
    publisher.publish_jsonl("scenario_results.jsonl", (item.model_dump(mode="json") for item in records))
    publisher.publish_jsonl("family_results.jsonl", (item.model_dump(mode="json") for item in family_rows))
    publisher.publish_jsonl("category_results.jsonl", (item.model_dump(mode="json") for item in category_rows))
    publisher.publish_jsonl("failure_mode_case_attribution.jsonl", (item.model_dump(mode="json") for item in attributions))
    publisher.publish_json("failure_mode_repair_summary.json", failure_summary.model_dump(mode="json"))
    publisher.publish_json("pairwise_cluster_bootstrap.json", paired_cluster_bootstrap(records).model_dump(mode="json"))
    publisher.publish_json("reproducibility_checklist.json", reproducibility.model_dump(mode="json"))
    publisher.publish_json("training_round_headlines.json", {"schema_version": 1, "rounds": [item.model_dump(mode="json") for item in rounds]})
    publisher.publish_json("main_table.json", main.model_dump(mode="json"))
    publisher.publish_text(
        "report.md",
        "# Final evaluation\n\nQwen weights were never trained or modified.\n\n"
        "Failure attribution is observational and not causal.\n\n"
        "Latency: total_running_time_seconds. Cost: qwen_effective_output_tokens.",
    )
    for aggregate in aggregates:
        publisher.publish_json(
            f"systems/{aggregate.system_id}/metrics/headline.json",
            aggregate.model_dump(mode="json"),
        )
    return publisher


def test_canonical_immutable_report_and_verification(tmp_path):
    root = tmp_path / "final_evaluation"
    publisher = complete(root)
    manifest = publisher.finalize()
    assert verify_report(root) == manifest
    assert all((root / item.path).stat().st_mode & 0o777 == 0o600 for item in manifest.entries)


def test_different_overwrite_and_unlisted_file_are_rejected(tmp_path):
    root = tmp_path / "final_evaluation"
    publisher = complete(root)
    with pytest.raises(ArtifactError, match="overwrite"):
        publisher.publish_json("main_table.json", {"value": 2})
    publisher.finalize()
    extra = root / "extra.json"
    extra.write_text("{}", encoding="utf-8")
    extra.chmod(0o600)
    with pytest.raises(ArtifactError, match="unlisted"):
        verify_report(root)


def test_sensitive_sentinel_is_rejected_before_write(tmp_path):
    publisher = ArtifactPublisher(tmp_path / "final_evaluation", D)
    with pytest.raises(ArtifactError, match="sensitive"):
        publisher.publish_text("report.md", "Authorization: secret")
    assert not (tmp_path / "final_evaluation" / "report.md").exists()


def test_finalize_rejects_missing_structure_and_symlink_ancestry(tmp_path):
    publisher = ArtifactPublisher(tmp_path / "incomplete", D)
    publisher.publish_json("main_table.json", {"schema_version": 1})
    with pytest.raises(ArtifactError, match="incomplete"):
        publisher.finalize()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        ArtifactPublisher(link / "final", D).publish_json("main_table.json", {})
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        ArtifactPublisher(broken / "final", D).publish_json("main_table.json", {})


@pytest.mark.parametrize(
    ("relative_path", "mutate", "message"),
    (
        (
            "failure_mode_repair_summary.json",
            lambda value: value.__setitem__("generation_0_failure_case_count", 1),
            "failure summary",
        ),
        (
            "family_results.jsonl",
            lambda value: value[0].__setitem__("minimum_native_similarity", 0.5),
            "family stability",
        ),
        (
            "pairwise_cluster_bootstrap.json",
            lambda value: value["pairs"][0].__setitem__("observed_difference", 0.5),
            "cluster bootstrap",
        ),
    ),
)
def test_typed_but_false_semantic_artifacts_are_rejected(
    tmp_path, relative_path, mutate, message,
):
    root = tmp_path / "false_semantics"
    publisher = complete(root)
    path = root / relative_path
    if relative_path.endswith(".jsonl"):
        value = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mutate(value)
        path.write_bytes(b"".join(canonical_json_bytes(item) + b"\n" for item in value))
    else:
        value = json.loads(path.read_bytes())
        mutate(value)
        path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ArtifactError, match=message):
        publisher.finalize()
