from __future__ import annotations

from dataclasses import replace

import pytest

from bfcl_pipeline.accounting import AttemptUsage, CommittedEffect, LogicalRequest, ResponseApplication, UsageLedger
from bfcl_pipeline.canonical import sha256_json
from bfcl_pipeline.evaluation import FinalExecutionOutcome, FinalUnitArtifact, FinalUnitDispatch, build_final_plan
from bfcl_pipeline.evolution import Generation, seed_generation
from bfcl_pipeline.minibench import MiniBenchOutcome, SharedMiniBenchConfig, SkillCandidate, decide, select_relevant_families
from bfcl_pipeline.online import CriticDecision, VisibleContext, run_policy_controller_critic_turn
from bfcl_pipeline.reporting import (
    AttributionRow,
    FailureLineage,
    FailureSignature,
    SystemCaseResult,
    attribute_repairs,
    attribution_summary,
    family_stability,
    paired_accuracy_uplift,
    system_results_from_artifacts,
)
from bfcl_pipeline.providers import MaxTokenCalibrationReceipt, MaxTokenObservation, QwenRequestConfig
from bfcl_pipeline.resources import GenerationResource, GenerationResourceManifest, GenerationResourceStore, QueryEmbedding, ResourceEmbedding
from bfcl_pipeline.splits import build_split_manifest
from bfcl_pipeline.trajectory import build_trusted_trajectory


def test_minibench_selects_at_most_five_complete_families(families):
    candidate = SkillCandidate("skill", "skill@1", ("read",), "hash")
    selected = select_relevant_families(candidate, families[:40])
    assert len(selected) == 5
    outcomes = []
    config = SharedMiniBenchConfig.build("Qwen/Qwen3-32B", "g000", 20, "environment")
    for family_index, family in enumerate(selected):
        for case in family.variants:
            outcomes.append(MiniBenchOutcome(family.family_id, case.case_id, False, family_index == 0, f"a-{case.case_id}", f"b-{case.case_id}", config.config_sha256, "a" * 64, "b" * 64))
    decision = decide(candidate, selected, outcomes, config)
    assert decision.accepted and len(decision.selected_case_ids) == 20


class Provider:
    model = "text-embedding-3-small"
    client_config_sha256 = "embedding-config"

    def embed(self, text):
        vector = (1.0,)
        return QueryEmbedding(sha256_json(text), self.model, self.client_config_sha256, vector, sha256_json(vector))


class Policy:
    def draft(self, context):
        return "read(key='x')"


class Critic:
    def review(self, context, draft, calls):
        return CriticDecision("KEEP", "ok", "bounded")


class Revision:
    def revise(self, context, draft, critique):
        raise AssertionError("KEEP must not revise")


def _trusted_skill_trajectory(
    subject_override=None,
    response_override=None,
    manifest_effect_override=None,
    case_id="multi_turn_base_1",
    generation_memory="memory",
):
    resource = GenerationResource.build("skill@1", "g003", "skill", "Use read safely.", ["read"], "skill-create-effect")
    manifest_resource = resource if manifest_effect_override is None else GenerationResource.build(
        "skill@1", "g003", "skill", "Use read safely.", ["read"], manifest_effect_override
    )
    manifest = GenerationResourceManifest.build("g003", (manifest_resource,))
    generation_core = {
        "generation_id": "g003", "parent_generation_id": "g002",
        "memory_sha256": sha256_json(generation_memory),
        "skill_library_sha256": manifest.manifest_sha256,
        "accepted_skill_versions": ("skill@1",),
    }
    generation = Generation(**generation_core, generation_sha256=sha256_json(generation_core))
    store = GenerationResourceStore(
        "g003",
        (resource,),
        (ResourceEmbedding.build(resource, Provider.client_config_sha256, (1.0,)),),
        Provider(),
    )
    receipt = store.retrieve_with_receipts(("read",), "read x", 1)[0]
    binding = ((resource.resource_id, resource.resource_sha256, resource.creation_effect_id),)
    context = VisibleContext(
        (),
        ({"name": "read", "parameters": {"properties": {"key": {"type": "string"}}, "required": ["key"]}},),
        (),
        (),
        (resource.content,),
        generation_id="g003",
        selected_skill_versions=("skill@1",),
        case_id=case_id,
        turn_index=2,
        selected_skill_bindings=binding,
        selected_skill_receipts=(receipt,),
    )
    turn = run_policy_controller_critic_turn(context, Policy(), Critic(), Revision())
    ledger = UsageLedger()
    request = LogicalRequest.build(
        "final-run",
        3,
        context.case_id,
        context.turn_index,
        "qwen",
        "Qwen/Qwen3-32B",
        "policy",
        sha256_json("input"),
        sha256_json("config"),
    )
    ledger.add_logical_request(request)
    attempt = AttemptUsage.build(request.logical_request_id, 0, 10, 4, response_override or sha256_json(turn.final_text))
    ledger.add_attempt(attempt)
    application = ResponseApplication.build(
        request.logical_request_id,
        attempt.attempt_id,
        attempt.response_sha256,
        "online_action",
        True,
    )
    ledger.apply_response(application)
    effect = CommittedEffect.build(
        "TOOL_STATE",
        sha256_json("before"),
        sha256_json("after"),
        (application.application_id,),
        subject_id=subject_override or turn.committed_call_ids[0],
    )
    ledger.commit_effect(effect)
    return build_trusted_trajectory(turn, application.application_id, (effect.effect_id,), generation, manifest, ledger)


def test_failure_repair_requires_exact_host_derived_lineage_and_effect():
    signature = FailureSignature.build("skill", "official_checker_failure", ["State.read"], "state_mismatch")
    lineage = [FailureLineage("mode", 0, signature, "skill@1", "evidence", "skill-create-effect")]
    before = [SystemCaseResult("g000", "family", "multi_turn_base_1", "base", False, signature.signature_sha256)]
    trajectory = _trusted_skill_trajectory()
    after = [SystemCaseResult("g003", "family", "multi_turn_base_1", "base", True, None, trajectory)]
    rows = attribute_repairs(before, after, lineage)
    assert rows[0].classification == "related_repaired"
    assert attribution_summary(rows)["repair_rate"] == 1.0


def test_authoritative_final_artifact_proves_related_repair(families):
    manifest = build_split_manifest(families, "revision")
    family_id = manifest.sealed_test_family_ids[0]
    family_index = int(family_id.rsplit("_", 1)[1])
    case_id = f"multi_turn_base_{family_index}"
    trajectory = _trusted_skill_trajectory(case_id=case_id)

    seed_resource = GenerationResource.build("seed@1", "g000", "skill", "seed", ["read"])
    g000 = seed_generation("memory-g000", GenerationResourceManifest.build("g000", (seed_resource,)))
    train_case_id = manifest.rounds[0].case_ids[0]
    calibration = MaxTokenCalibrationReceipt.build(
        evidence_source="live_train_smoke",
        dataset_manifest_sha256=manifest.manifest_sha256,
        train_case_ids=(train_case_id,),
        observations=(MaxTokenObservation(train_case_id, 800, "stop"),),
    )
    plan = build_final_plan(
        manifest,
        g000,
        trajectory.generation,
        qwen_config=QwenRequestConfig(max_tokens=calibration.chosen_max_tokens),
        calibration_receipt=calibration,
        embedding_client_config_sha256=sha256_json("embedding-config"),
        bfcl_revision="6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        environment_sha256=sha256_json("environment"),
        evaluator_sha256=sha256_json("evaluator"),
        resource_manifest_sha256=trajectory.generation.skill_library_sha256,
        experiment_config_sha256=sha256_json("experiment-config"),
    )
    dispatch = FinalUnitDispatch.build(plan, "owner", 320, "g003", "g003", case_id)
    ledger = UsageLedger.from_payload(trajectory.ledger_payload)
    artifact = FinalUnitArtifact.build(
        plan,
        dispatch,
        FinalExecutionOutcome(True, None, {"valid": True}, ledger, trajectory),
        0.25,
    )
    after = system_results_from_artifacts(plan, "owner", (artifact,))
    signature = FailureSignature.build("skill", "official_checker_failure", ["State.read"], "state_mismatch")
    before = (SystemCaseResult("g000", family_id, case_id, "base", False, signature.signature_sha256),)
    lineage = (FailureLineage("mode", 0, signature, "skill@1", "evidence", "skill-create-effect"),)

    rows = attribute_repairs(before, after, lineage)
    assert rows == (AttributionRow(family_id, case_id, "related_repaired", "mode"),)

    forged = _trusted_skill_trajectory(case_id=case_id, generation_memory="different-self-consistent-generation")
    forged_ledger = UsageLedger.from_payload(forged.ledger_payload)
    with pytest.raises(ValueError, match="trusted execution provenance"):
        FinalUnitArtifact.build(
            plan,
            dispatch,
            FinalExecutionOutcome(True, None, {"valid": True}, forged_ledger),
            0.25,
        )
    with pytest.raises(ValueError, match="exact frozen generation"):
        FinalUnitArtifact.build(
            plan,
            dispatch,
            FinalExecutionOutcome(True, None, {"valid": True}, forged_ledger, forged),
            0.25,
        )


def test_tampered_trajectory_is_rejected_not_counted_as_repair():
    signature = FailureSignature.build("skill", "official_checker_failure", ["State.read"], "state_mismatch")
    lineage = [FailureLineage("mode", 0, signature, "skill@1", "evidence", "skill-create-effect")]
    before = [SystemCaseResult("g000", "family", "multi_turn_base_1", "base", False, signature.signature_sha256)]
    trajectory = replace(_trusted_skill_trajectory(), trajectory_sha256="caller-authored")
    after = [SystemCaseResult("g003", "family", "multi_turn_base_1", "base", True, None, trajectory)]
    with pytest.raises(ValueError, match="trajectory hash"):
        attribute_repairs(before, after, lineage)


def test_family_stability_requires_all_four_variants():
    variants = ("base", "miss_func", "miss_param", "long_context")
    results = [SystemCaseResult("g003", "family", f"multi_turn_{v}_1", v, True, None) for v in variants]
    stability = family_stability(results, ())
    assert stability[0].all_four_variants_successful
    assert stability[0].valid_variant_count == 4


def test_paired_uplift_reports_overall_related_and_vanilla():
    ids = ("multi_turn_base_1", "multi_turn_base_2")
    vanilla = [SystemCaseResult("vanilla", "f", ids[0], "base", False, None), SystemCaseResult("vanilla", "g", ids[1], "base", True, None)]
    before = [SystemCaseResult("g000", "f", ids[0], "base", False, "s"), SystemCaseResult("g000", "g", ids[1], "base", True, None)]
    updated = [SystemCaseResult("g003", "f", ids[0], "base", True, None), SystemCaseResult("g003", "g", ids[1], "base", True, None)]
    summary = paired_accuracy_uplift(vanilla, before, updated, [AttributionRow("f", ids[0], "related_repaired", "mode")])
    assert summary["updated_minus_g000"] == 0.5
    assert summary["updated_minus_g000_related"] == 1.0
    assert summary["updated_minus_vanilla"] == 0.5

def test_effect_for_another_call_cannot_prove_skill_use():
    with pytest.raises(ValueError, match="exact executed call"):
        _trusted_skill_trajectory("different-call")


def test_decision_text_must_match_the_accepted_application_response():
    with pytest.raises(ValueError, match="final text"):
        _trusted_skill_trajectory(response_override=sha256_json("another response"))


def test_retrieved_skill_must_exist_in_the_trusted_generation_manifest():
    with pytest.raises(ValueError, match="trusted generation manifest"):
        _trusted_skill_trajectory(manifest_effect_override="fabricated-effect")
