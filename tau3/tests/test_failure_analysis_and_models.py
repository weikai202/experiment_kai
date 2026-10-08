import pytest

from tau3_evolution.canonical import canonical_sha256
from tau3_evolution.failure_analysis import (
    CommittedAction,
    DecisionReceipt,
    EvaluationRecord,
    FailureLineage,
    RetrievalReceipt,
    TrustedEpisodeTrace,
    analyze_repairs,
    derive_failure_signatures,
    trusted_trajectory_sha256,
)
from tau3_evolution.model_boundary import ModelProfile, qwen_request, validate_qwen_request


def retrieval(generation, version):
    payload = {
        "generation_id": generation,
        "domain": "retail",
        "tool_names": ["refund"],
        "query_sha256": "sha256:query",
        "policy": [],
        "world": [],
        "skills": [["refund_skill", version]],
    }
    return RetrievalReceipt(
        generation,
        "retail",
        ("refund",),
        "sha256:query",
        (),
        (),
        (("refund_skill", version),),
        canonical_sha256(payload),
    )


def decision(decision_id="decision"):
    return DecisionReceipt(
        decision_id, "logical", "attempt", "application", "sha256:response", "executed-effect"
    )


def action(
    *,
    version=None,
    effect=None,
    outcome="tool_error",
    committed=True,
    decision_id="decision",
    generation="g003",
):
    selected_version = 0 if version is None else version
    return CommittedAction(
        "call",
        decision_id,
        "refund",
        "refund_skill",
        version,
        effect,
        outcome,
        True,
        committed,
        retrieval(generation, selected_version),
        decision(decision_id),
    )


def generation_document(generation, action_row):
    version = 0 if action_row.skill_version is None else action_row.skill_version
    base = {
        "generation_id": generation,
        "parent_generation_id": None if generation == "g000" else "g002",
        "memories": [],
        "skills": [
            {
                "skill_id": "refund_skill",
                "version": version,
                "domains": ["retail"],
                "tool_dependencies": ["refund"],
                "content": "refund skill",
                "accepted_effect_id": action_row.accepted_effect_id,
                "dev_evidence_sha256": None,
            }
        ],
        "vectors": [
            {
                "record_kind": "skill",
                "record_id": "refund_skill",
                "version": version,
                "content_sha256": canonical_sha256("refund skill"),
                "model": "text-embedding-3-small",
                "dimension": 2,
                "vector": [1.0, 0.0],
            }
        ],
    }
    return {**base, "state_sha256": canonical_sha256(base)}


def ledger_document(action_row):
    receipt = action_row.decision
    assert receipt is not None
    return {
        "schema_version": 2,
        "logical_requests": [
            {
                "logical_request_id": receipt.logical_request_id,
                "scope_id": "episode",
                "provider_role": "qwen",
                "purpose": "policy",
                "request_sha256": "sha256:request",
                "model_id": "Qwen/Qwen3-32B",
                "config_sha256": "sha256:config",
            }
        ],
        "physical_attempts": [
            {
                "attempt_id": receipt.attempt_id,
                "logical_request_id": receipt.logical_request_id,
                "input_tokens": 1,
                "output_tokens": 2,
                "completed": True,
                "response_sha256": receipt.response_sha256,
            }
        ],
        "applications": [
            {
                "application_id": receipt.application_id,
                "logical_request_id": receipt.logical_request_id,
                "attempt_id": receipt.attempt_id,
                "effect_id": receipt.executed_effect_id,
                "application_kind": "executed_action_decision",
                "committed": True,
                "causal_order": 0,
            }
        ],
        "effects": [
            {
                "effect_id": receipt.executed_effect_id,
                "scope_id": "episode",
                "effect_kind": "executed_action",
                "artifact_sha256": "sha256:call",
                "substantive": True,
                "committed": True,
            }
        ],
    }


def trace(
    system,
    generation,
    task,
    reward,
    *,
    action_row=None,
    version=None,
    effect=None,
    outcome="tool_error",
):
    action_row = action_row or action(
        version=version, effect=effect, outcome=outcome, generation=generation
    )
    episode_id = f"{system}-{task}"
    evaluator = f"sha256:evaluator-{system}-{task}"
    generation_doc = generation_document(generation, action_row)
    ledger = ledger_document(action_row)
    trajectory = trusted_trajectory_sha256(
        episode_id=episode_id,
        domain="retail",
        task_id=task,
        simulator_seed=7,
        system_id=system,
        generation_id=generation,
        evaluator_record_sha256=evaluator,
        reward=float(reward),
        actions=(action_row,),
        generation_state_document=generation_doc,
        accounting_ledger_document=ledger,
    )
    return TrustedEpisodeTrace(
        episode_id,
        "retail",
        task,
        7,
        system,
        generation,
        trajectory,
        evaluator,
        float(reward),
        (action_row,),
        generation_doc,
        ledger,
    )


def records(g0, updated, g0_reward=None, updated_reward=None):
    return (
        EvaluationRecord(
            "vanilla", "retail", "task", 7, 0.0, 0, "sha256:vanilla", "sha256:vanilla-eval"
        ),
        EvaluationRecord(
            "generation_0",
            "retail",
            "task",
            7,
            g0.reward if g0_reward is None else float(g0_reward),
            0,
            g0.trajectory_sha256,
            g0.evaluator_record_sha256,
        ),
        EvaluationRecord(
            "updated",
            "retail",
            "task",
            7,
            updated.reward if updated_reward is None else float(updated_reward),
            0,
            updated.trajectory_sha256,
            updated.evaluator_record_sha256,
        ),
    )


def test_exact_committed_call_decision_effect_and_skill_prove_repair():
    g0 = trace("generation_0", "g000", "task", 0.0)
    signature = derive_failure_signatures(g0)[0]
    lineage = (FailureLineage("lineage", signature.signature_sha256, "refund_skill", 3, "effect"),)
    updated = trace("updated", "g003", "task", 1.0, version=3, effect="effect")
    rows, summary = analyze_repairs(
        lineage=lineage,
        records=records(g0, updated),
        generation_0_traces=(g0,),
        updated_traces=(updated,),
    )
    assert rows[0].classification == "related_repaired"
    assert rows[0].committed_call_proof_sha256
    assert rows[0].simulator_seed == 7
    assert summary.repaired_case_count == 1 and summary.repair_rate == 1.0


@pytest.mark.parametrize(
    "updated_action",
    [
        action(version=2, effect="effect"),
        action(version=3, effect="wrong"),
        action(version=3, effect="effect", decision_id=""),
        action(version=3, effect="effect", committed=False),
    ],
)
def test_episode_membership_without_exact_committed_call_proof_is_not_repair(updated_action):
    g0 = trace("generation_0", "g000", "task", 0.0)
    signature = derive_failure_signatures(g0)[0]
    updated = trace("updated", "g003", "task", 1.0, action_row=updated_action)
    rows, summary = analyze_repairs(
        lineage=(
            FailureLineage("lineage", signature.signature_sha256, "refund_skill", 3, "effect"),
        ),
        records=records(g0, updated),
        generation_0_traces=(g0,),
        updated_traces=(updated,),
    )
    assert rows[0].classification == "related_unrepaired"
    assert summary.repaired_case_count == 0


def test_missing_committed_observable_signature_is_incomplete():
    g0 = trace("generation_0", "g000", "task", 0.0, outcome=None)
    updated = trace("updated", "g003", "task", 1.0)
    rows, summary = analyze_repairs(
        lineage=(),
        records=records(g0, updated),
        generation_0_traces=(g0,),
        updated_traces=(updated,),
    )
    assert rows[0].classification == "incomplete_evidence"
    assert summary.incomplete_evidence_count == 1


def test_lineage_without_positive_accepted_version_and_effect_cannot_prove_repair():
    g0 = trace("generation_0", "g000", "task", 0.0)
    signature = derive_failure_signatures(g0)[0]
    updated = trace("updated", "g003", "task", 1.0, version=3, effect="effect")
    for version, effect in ((None, None), (0, "effect"), (3, None)):
        rows, summary = analyze_repairs(
            lineage=(
                FailureLineage(
                    "lineage", signature.signature_sha256, "refund_skill", version, effect
                ),
            ),
            records=records(g0, updated),
            generation_0_traces=(g0,),
            updated_traces=(updated,),
        )
        assert rows[0].classification == "related_unrepaired"
        assert summary.repaired_case_count == 0


def test_trace_reward_must_match_evaluation_record():
    g0 = trace("generation_0", "g000", "task", 0.0)
    signature = derive_failure_signatures(g0)[0]
    updated = trace("updated", "g003", "task", 0.5, version=3, effect="effect")
    rows, summary = analyze_repairs(
        lineage=(
            FailureLineage("lineage", signature.signature_sha256, "refund_skill", 3, "effect"),
        ),
        records=records(g0, updated, updated_reward=1.0),
        generation_0_traces=(g0,),
        updated_traces=(updated,),
    )
    assert rows[0].classification == "related_unrepaired"
    assert summary.repaired_case_count == 0


def test_trajectory_hash_is_recomputed_from_persisted_material():
    valid = trace("updated", "g003", "task", 1.0, version=3, effect="effect")
    with pytest.raises(ValueError, match="trajectory hash"):
        TrustedEpisodeTrace(
            valid.episode_id,
            valid.domain,
            valid.task_id,
            valid.simulator_seed,
            valid.system_id,
            valid.generation_id,
            "sha256:fabricated",
            valid.evaluator_record_sha256,
            valid.reward,
            valid.actions,
            valid.generation_state_document,
            valid.accounting_ledger_document,
        )


def test_qwen_boundary_is_exact_and_non_thinking():
    ModelProfile().validate()
    request = qwen_request([{"role": "user", "content": "x"}], max_tokens=128)
    validate_qwen_request(request)
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    "mutation",
    [
        {"model": "other"},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"temperature": 0.1},
        {"tools": []},
    ],
)
def test_qwen_boundary_rejects_drift(mutation):
    request = qwen_request([{"role": "user", "content": "x"}], max_tokens=128)
    request.update(mutation)
    with pytest.raises(ValueError):
        validate_qwen_request(request)
