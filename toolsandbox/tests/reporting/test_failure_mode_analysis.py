from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.reporting.failure_mode_analysis import (
    analyze_failure_modes,
    compute_family_stability,
    derive_failure_signature_observations,
)
from toolsandbox_pipeline.schemas.offline_skill import FailureModeLineageRecord
from toolsandbox_pipeline.schemas.reporting import (
    FailureSignatureObservation,
    ScenarioEvaluationRecord,
)
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.fixtures import ExternalReadAttempt, ExternalReadContext
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeIdentity,
    OnlineTurnRecord,
    SkillUseAttribution,
    ToolActionRecord,
    TrustedTrajectory,
)
from pydantic import ValidationError
import pytest

D = "sha256:" + "1" * 64


def external_attempts(two=False):
    calls = (
        (("call-1", "search_stock"), ("call-2", "search_weather_around_lat_lon"))
        if two else (("call-1", "search_stock"),)
    )
    return tuple(
        ExternalReadAttempt(
            schema_version=1, attempt_id=f"external-{index}",
            context=ExternalReadContext(
                schema_version=1, run_id="run", profile="strict_replay",
                phase="final-test", scenario_family_id="family",
                scenario_id="case-0", state_id=D, logical_tool_call_id=call_id,
                backend_manifest_sha256=D, fixture_manifest_sha256=D,
            ),
            mode="replay", canonical_tool_name=tool_id, effect="external_read",
            fixture_key=D, backend_version="synthetic-v1", status="fixture_miss",
            dispatched=False, status_code=None, response_body_sha256=None,
            sanitized_exception_class="FixtureMiss",
            started_at_utc="2026-09-10T00:00:00Z",
            ended_at_utc="2026-09-10T00:00:01Z", latency_seconds=1.0,
        )
        for index, (call_id, tool_id) in enumerate(calls, 1)
    )


def scenario(
    system, position, scenario_id, score, *, family="family", trajectory_sha256=D,
    fixture_miss_count=0,
):
    return ScenarioEvaluationRecord(
        system_id=system,
        manifest_position=position,
        scenario_id=scenario_id,
        scenario_family_id=family,
        variant=f"v{position}",
        categories=("category",),
        episode_id=f"episode-{system}-{position}",
        trajectory_sha256=trajectory_sha256,
        evaluator_record_sha256=D,
        similarity=score,
        milestone_similarity=score,
        minefield_similarity=0.0,
        fully_successful=score == 1.0,
        effective_turn_count=1,
        fixture_miss_count=fixture_miss_count,
        external_tool_exception_count=fixture_miss_count,
    )


def blob(name):
    return BlobReference(
        sha256=D, byte_count=1,
        media_type="application/vnd.toolsandbox.canonical+json",
        schema_name=name, schema_version=1, content_visibility="restricted",
    )


def updated_trajectory(scenario_id="case-0", position=0, family="family"):
    identity = EpisodeIdentity(
        run_id="run", profile="strict_replay", phase="final-test",
        family_id=family, scenario_id=scenario_id,
        episode_id=f"episode-updated-{position}", manifest_position=position,
        system_variant="updated", generation_id="g003",
        starting_context_sha256=D, evaluation_definition_sha256=D,
        agent_tool_schema_sha256=D, dataset_manifest_sha256=D,
        runtime_config_sha256=D, prompt_manifest_sha256=D,
        token_limit_config_sha256=D, fixture_manifest_sha256=D,
        environment_sha256=D, max_messages=10,
    )
    turn = OnlineTurnRecord(
        agent_turn_index=0, state_id="state-1",
        decision_reference=blob("decision"), decision_sha256=D,
        final_action_sha256=D, logical_request_ids=("logical-1",),
        source_attempt_ids=("attempt-1",), application_ids=("application-1",),
        executed_call_ids=("call-1",),
    )
    attribution = SkillUseAttribution(
        skill_id="skill-1", skill_version="v1.1", generation_id="g003",
        executed_call_ids=("call-1",), canonical_tool_ids=("tool-1",),
        evaluator_record_sha256=D, fully_successful=True,
    )
    action = ToolActionRecord(
        transaction_id="transaction-updated", action_sha256=D,
        call_ids=("call-1",), selected_skill_ids=("skill-1",),
        canonical_tool_ids=("tool-1",), effect_classes=("read",),
        pre_context_reference=blob("context"), pre_context_sha256=D,
        post_context_reference=blob("context"), post_context_sha256=D,
        executed=True, committed=True, rolled_back=False, failed=False,
    )
    return TrustedTrajectory.build(
        identity=identity, messages=(), online_turns=(turn,), tool_actions=(action,),
        logical_request_ids=("logical-1",), physical_attempt_ids=("attempt-1",),
        ending_context_reference=blob("context"), ending_context_sha256=D,
        evaluator_record_reference=blob("evaluator"), evaluator_record_sha256=D,
        skill_attributions=(attribution,),
        eligible_for_train_offline_consumption=False,
    )


def generation_0_trajectory(
    scenario_id="case-0", position=0, family="family", *, two_skills=False,
):
    identity = EpisodeIdentity(
        run_id="run", profile="strict_replay", phase="final-test",
        family_id=family, scenario_id=scenario_id,
        episode_id=f"episode-generation_0-{position}", manifest_position=position,
        system_variant="generation_0", generation_id="g000",
        starting_context_sha256=D, evaluation_definition_sha256=D,
        agent_tool_schema_sha256=D, dataset_manifest_sha256=D,
        runtime_config_sha256=D, prompt_manifest_sha256=D,
        token_limit_config_sha256=D, fixture_manifest_sha256=D,
        environment_sha256=D, max_messages=10,
    )
    call_ids = ("call-1", "call-2") if two_skills else ("call-1",)
    skill_ids = ("skill-1", "skill-2") if two_skills else ("skill-1",)
    tool_ids = (
        ("search_stock", "search_weather_around_lat_lon")
        if two_skills else ("search_stock",)
    )
    external_ids = tuple(item.attempt_id for item in external_attempts(two_skills))
    turn = OnlineTurnRecord(
        agent_turn_index=0, state_id="state-1",
        decision_reference=blob("decision"), decision_sha256=D,
        final_action_sha256=D, logical_request_ids=("logical-g0",),
        source_attempt_ids=("attempt-g0",), application_ids=("application-g0",),
        executed_call_ids=call_ids,
    )
    action = ToolActionRecord(
        transaction_id="transaction-g0", action_sha256=D,
        call_ids=call_ids, selected_skill_ids=skill_ids,
        canonical_tool_ids=tool_ids, effect_classes=tuple("read" for _ in call_ids),
        pre_context_reference=blob("context"), pre_context_sha256=D,
        post_context_reference=blob("context"), post_context_sha256=D,
        external_attempt_ids=external_ids,
        executed=True, committed=True, rolled_back=False, failed=False,
    )
    attributions = tuple(
        SkillUseAttribution(
            skill_id=skill_id, skill_version="v1.0", generation_id="g000",
            executed_call_ids=(call_id,), canonical_tool_ids=(tool_id,),
            evaluator_record_sha256=D, fully_successful=False,
        )
        for call_id, skill_id, tool_id in zip(
            call_ids, skill_ids, tool_ids, strict=True,
        )
    )
    return TrustedTrajectory.build(
        identity=identity, messages=(), online_turns=(turn,), tool_actions=(action,),
        logical_request_ids=("logical-g0",), physical_attempt_ids=("attempt-g0",),
        ending_context_reference=blob("context"), ending_context_sha256=D,
        evaluator_record_reference=blob("evaluator"), evaluator_record_sha256=D,
        skill_attributions=attributions,
        eligible_for_train_offline_consumption=False,
    )
def lineage(
    signature, *, skill_id="skill-1", tool_id="search_stock",
    mode_id="mode-1", effect_id="effect-1",
):
    payload = {
        "failure_signature_sha256": signature,
        "mode_id": mode_id,
        "source_evidence_sha256": D,
        "producing_round": 0,
        "failure_mode_effect_id": effect_id,
    }
    return FailureModeLineageRecord(
        lineage_id="lineage_" + canonical_sha256(payload)[7:],
        failure_signature_sha256=signature,
        skill_id=skill_id,
        evidence_kind="fixture_miss",
        canonical_tool_dependencies=(tool_id,),
        sanitized_outcome_class="FixtureMiss",
        mode_id=mode_id,
        source_evidence_sha256=D,
        producing_round=0,
        failure_mode_effect_id=effect_id,
        accepted_skill_version="v1.1",
        accepted_skill_effect_id="effect-2",
    )


def test_exact_repair_requires_linked_skill_use_proof():
    generation_0 = generation_0_trajectory()
    observation = FailureSignatureObservation.build(
        scenario_id="case-0", skill_id="skill-1", evidence_kind="fixture_miss",
        canonical_tool_dependencies=("search_stock",), sanitized_outcome_class="FixtureMiss",
    )
    trajectory = updated_trajectory()
    records = (
        scenario("vanilla", 0, "case-0", 0.0),
        scenario(
            "generation_0", 0, "case-0", 0.0,
            trajectory_sha256=generation_0.trajectory_id, fixture_miss_count=1,
        ),
        scenario("updated", 0, "case-0", 1.0, trajectory_sha256=trajectory.trajectory_id),
    )
    rows, summary = analyze_failure_modes(
        lineage_records=(lineage(observation.signature_sha256),),
        scenario_records=records,
        generation_0_trajectories=(generation_0,),
        generation_0_external_attempts=external_attempts(),
        updated_trajectories=(trajectory,),
    )
    assert {row.classification for row in rows} == {"related_repaired"}
    assert summary.failure_mode_repaired_case_count == 1
    assert summary.failure_mode_repair_rate == 1.0
    assert summary.updated_minus_generation_0_similarity_points_related_subset == 1.0


def test_zero_match_and_missing_signature_are_retained():
    generation_0 = generation_0_trajectory()
    records = tuple(
        scenario(
            system, position, f"case-{position}", 0.0,
            trajectory_sha256=(
                generation_0.trajectory_id
                if system == "generation_0" and position == 0 else D
            ),
            fixture_miss_count=int(system == "generation_0" and position == 0),
        )
        for position in range(2)
        for system in ("vanilla", "generation_0", "updated")
    )
    _, summary = analyze_failure_modes(
        lineage_records=(), scenario_records=records,
        generation_0_trajectories=(generation_0,),
        generation_0_external_attempts=external_attempts(),
        updated_trajectories=(),
    )
    assert summary.failure_mode_unmatched_case_count == 1
    assert summary.incomplete_evidence_case_count == 1
    assert summary.failure_mode_repair_rate is None
    assert summary.repair_rate_complete is False


def test_family_stability_uses_exact_eight_variant_denominator():
    records = tuple(
        scenario(
            system, family_index * 8 + variant,
            f"family-{family_index:02d}-case-{variant}", 1.0,
            family=f"family-{family_index:02d}",
        )
        for family_index in range(25)
        for variant in range(8)
        for system in ("vanilla", "generation_0", "updated")
    )
    rows, _ = analyze_failure_modes(
        lineage_records=(), scenario_records=records,
        generation_0_trajectories=(),
        generation_0_external_attempts=(),
        updated_trajectories=(),
    )
    families, summaries = compute_family_stability(records, rows)
    assert len(families) == 75
    assert all(item.variant_count == 8 for item in families)
    assert all(item.all_variants_fully_successful for item in families)
    assert all(item.all_variants_success_family_rate == 1.0 for item in summaries)


def test_failure_observation_hash_rejects_tampering():
    observation = FailureSignatureObservation.build(
        scenario_id="case-0", skill_id="skill-1",
        evidence_kind="controller_rejection",
        canonical_tool_dependencies=("tool-1",),
        sanitized_outcome_class="denied",
    )
    payload = observation.model_dump(mode="python")
    payload["sanitized_outcome_class"] = "different"
    with pytest.raises(ValidationError, match="observation hash"):
        FailureSignatureObservation(**payload)


def test_forged_or_missing_updated_trajectory_cannot_prove_repair():
    generation_0 = generation_0_trajectory()
    observation = FailureSignatureObservation.build(
        scenario_id="case-0", skill_id="skill-1",
        evidence_kind="fixture_miss",
        canonical_tool_dependencies=("search_stock",),
        sanitized_outcome_class="FixtureMiss",
    )
    trajectory = updated_trajectory()
    records = (
        scenario("vanilla", 0, "case-0", 0.0),
        scenario(
            "generation_0", 0, "case-0", 0.0,
            trajectory_sha256=generation_0.trajectory_id, fixture_miss_count=1,
        ),
        scenario("updated", 0, "case-0", 1.0, trajectory_sha256=trajectory.trajectory_id),
    )
    rows, _ = analyze_failure_modes(
        lineage_records=(lineage(observation.signature_sha256),),
        scenario_records=records,
        generation_0_trajectories=(generation_0,),
        generation_0_external_attempts=external_attempts(),
        updated_trajectories=(),
    )
    assert {row.classification for row in rows} == {"related_unrepaired"}
    forged = trajectory.model_copy(update={
        "identity": trajectory.identity.model_copy(update={"generation_id": "g000"})
    })
    with pytest.raises(Exception, match="Updated G003"):
        analyze_failure_modes(
            lineage_records=(lineage(observation.signature_sha256),),
            scenario_records=records,
            generation_0_trajectories=(generation_0,),
            generation_0_external_attempts=external_attempts(),
            updated_trajectories=(forged,),
        )


def test_observations_are_host_derived_and_caller_cannot_inject_signature():
    trajectory = generation_0_trajectory()
    records = (
        scenario("vanilla", 0, "case-0", 0.0),
        scenario(
            "generation_0", 0, "case-0", 0.0,
            trajectory_sha256=trajectory.trajectory_id, fixture_miss_count=1,
        ),
        scenario("updated", 0, "case-0", 0.0),
    )
    observations = derive_failure_signature_observations(
        (trajectory,), external_attempts(), records
    )
    assert len(observations) == 1
    assert observations[0].evidence_kind == "fixture_miss"
    with pytest.raises(TypeError, match="unexpected keyword"):
        analyze_failure_modes(
            lineage_records=(), scenario_records=records,
            generation_0_trajectories=(trajectory,),
            generation_0_external_attempts=external_attempts(),
            updated_trajectories=(),
            observations=observations,
        )


def test_multiple_signatures_are_ambiguous_but_duplicate_lineage_is_not():
    trajectory = generation_0_trajectory(two_skills=True)
    records = (
        scenario("vanilla", 0, "case-0", 0.0),
        scenario(
            "generation_0", 0, "case-0", 0.0,
            trajectory_sha256=trajectory.trajectory_id, fixture_miss_count=1,
        ),
        scenario("updated", 0, "case-0", 0.0),
    )
    observations = derive_failure_signature_observations(
        (trajectory,), external_attempts(True), records
    )
    assert len(observations) == 2
    first = lineage(observations[0].signature_sha256)
    rows, _ = analyze_failure_modes(
        lineage_records=(first, first), scenario_records=records,
        generation_0_trajectories=(trajectory,),
        generation_0_external_attempts=external_attempts(True),
        updated_trajectories=(),
    )
    assert {row.classification for row in rows} == {"related_unrepaired"}
    second = lineage(
        observations[1].signature_sha256, skill_id="skill-2",
        tool_id="search_weather_around_lat_lon",
        mode_id="mode-2", effect_id="effect-2",
    )
    rows, _ = analyze_failure_modes(
        lineage_records=(first, second), scenario_records=records,
        generation_0_trajectories=(trajectory,),
        generation_0_external_attempts=external_attempts(True),
        updated_trajectories=(),
    )
    assert {row.classification for row in rows} == {"ambiguous"}
