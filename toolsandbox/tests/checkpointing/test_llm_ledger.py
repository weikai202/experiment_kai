from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path

import pytest

from toolsandbox_pipeline.checkpointing import (
    CheckpointStore,
    LLMLedger,
    LedgerConflictError,
    LLMRecoveryAction,
    LogicalLLMRequestIdentity,
    NonSubstantiveOutcome,
    QwenEffectKind,
    RunIdentity,
)
from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    PhysicalAttemptResult,
    PhysicalAttemptStatus,
    ProviderRole,
)
from toolsandbox_pipeline.schemas.usage import PhysicalAttemptMetrics, TokenUsage
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope


CONFIG = Path(__file__).parents[2] / "configs/reproducibility/checkpointing_v1.json"


def fingerprint(character: str) -> str:
    return "sha256:" + character * 64


@pytest.fixture
def ledger(tmp_path):
    root = tmp_path / "run"
    identity = RunIdentity(
        run_id="run-1",
        profile="offline",
        environment_identity=fingerprint("1"),
        dataset_manifest_sha256=fingerprint("2"),
        config_manifest_sha256=fingerprint("3"),
        prompt_manifest_sha256=fingerprint("4"),
        generation_manifest_sha256=fingerprint("5"),
        fixture_manifest_sha256=fingerprint("6"),
    )
    store = CheckpointStore.create(root, identity, CONFIG.resolve())
    try:
        yield LLMLedger(store), root, identity
    finally:
        store.close()


def prepare_application(
    ledger,
    *,
    role=ProviderRole.POLICY,
    unit="state-1",
    output_tokens=7,
    complete=True,
):
    identity = LogicalLLMRequestIdentity(
        run_id="run-1",
        role=role,
        phase=(
            "online"
            if role
            in (ProviderRole.POLICY, ProviderRole.CRITIC, ProviderRole.REVISION)
            else "offline"
        ),
        unit_reference=unit,
        input_fingerprint=fingerprint("a"),
        model=(
            "text-embedding-3-small"
            if role is ProviderRole.EMBEDDING
            else "Qwen/Qwen3-32B"
        ),
        decoding_configuration_sha256=fingerprint("b"),
        output_schema_sha256=fingerprint("c"),
    )
    record = ledger.prepare_request(identity)
    context = ledger.allocate_attempt(
        record.logical_request_id,
        manifest_identity=fingerprint("d"),
        replayed_after_unknown_outcome=False,
    )
    ledger.mark_in_flight(context)
    raw = ("{\"unit\":\"" + unit + "\"}").encode()
    now = datetime.now(timezone.utc)
    usage = (
        TokenUsage(
            input_tokens=3,
            uncached_input_tokens=3,
            cache_read_input_tokens=0,
            cache_write_input_tokens=0,
            output_tokens=output_tokens,
            total_tokens=3 + output_tokens,
            usage_complete=True,
        )
        if complete
        else TokenUsage()
    )
    attempt = PhysicalAttemptResult(
        context=context,
        status=PhysicalAttemptStatus.COMPLETED,
        model=identity.model,
        returned_model=identity.model,
        finish_reason="stop",
        response_hash="sha256:" + sha256(raw).hexdigest(),
        metrics=PhysicalAttemptMetrics(
            started_at=now, completed_at=now, latency_seconds=0.1, usage=usage
        ),
    )
    ledger.complete_response(
        GatewayResponse(value={"action": unit}, attempt=attempt, raw_response_body=raw),
        validated_output={"action": unit},
        output_schema_name="synthetic_action",
        output_schema_version=1,
    )
    application = ledger.commit_checkpoint_and_apply(
        checkpoint_id=f"checkpoint-{unit}",
        event_kind="after_llm_application",
        checkpoint_payload={"state": unit},
        logical_request_id=record.logical_request_id,
        source_attempt_id=context.attempt_id,
        application_artifact_id=f"artifact-{unit}",
        application_artifact_sha256=fingerprint("e"),
    )
    return record, context, application


def prepare_request_only(ledger, unit):
    return ledger.prepare_request(
        LogicalLLMRequestIdentity(
            run_id="run-1",
            role=ProviderRole.POLICY,
            phase="online",
            unit_reference=unit,
            input_fingerprint=fingerprint("a"),
            model="Qwen/Qwen3-32B",
            decoding_configuration_sha256=fingerprint("b"),
            output_schema_sha256=fingerprint("c"),
        )
    )


def accounting_scope(**changes):
    values = dict(
        run_id="run-1",
        task_id="episode-1",
        scenario_family_id="family-1",
        scenario_id="scenario-1",
        system_variant="generation_0",
    )
    values.update(changes)
    return AccountingScope(**values)


def test_accounting_snapshot_is_exact_hash_verified_and_raw_free(ledger):
    record, context, application = prepare_application(
        ledger[0], unit="snapshot", output_tokens=7
    )
    ledger[0].bind_accounting_scope(record.logical_request_id, accounting_scope())
    effect = ledger[0].commit_checkpoint_and_effect(
        checkpoint_id="checkpoint-snapshot-effect",
        event_kind="online_action_committed",
        checkpoint_payload={"committed": True},
        effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
        effect_artifact_id="snapshot-action",
        effect_artifact_sha256=fingerprint("f"),
        ordered_application_ids=(application.application_id,),
    )
    snapshot = ledger[0].snapshot_accounting(accounting_scope())
    assert snapshot.population.logical_request_ids == (record.logical_request_id,)
    assert snapshot.population.physical_attempt_ids == (context.attempt_id,)
    assert snapshot.population.application_ids == (application.application_id,)
    assert snapshot.population.effect_ids == (effect.effect_id,)
    assert snapshot.physical_attempts[0].usage.output_tokens == 7
    assert snapshot.logical_requests[0].application_id == application.application_id
    assert "raw" not in snapshot.model_dump_json().lower()
    stored = ledger[0].store.blobs.read(snapshot.population_reference)
    assert b"snapshot-action" not in stored


def test_accounting_scope_is_immutable_and_tamper_is_rejected(ledger):
    record = prepare_request_only(ledger[0], "scoped")
    ledger[0].bind_accounting_scope(record.logical_request_id, accounting_scope())
    ledger[0].bind_accounting_scope(record.logical_request_id, accounting_scope())
    with pytest.raises(LedgerConflictError, match="scope conflict"):
        ledger[0].bind_accounting_scope(
            record.logical_request_id,
            accounting_scope(scenario_id="different"),
        )
    ledger[0].store._connection.execute(
        "UPDATE logical_llm_requests SET role='critic' WHERE logical_request_id=?",
        (record.logical_request_id,),
    )
    with pytest.raises(LedgerConflictError, match="tampered"):
        ledger[0].snapshot_accounting(accounting_scope())


def test_snapshot_rejects_any_unbound_request_and_includes_recovery_population(ledger):
    first = prepare_request_only(ledger[0], "recoverable")
    ledger[0].bind_accounting_scope(first.logical_request_id, accounting_scope())
    original = ledger[0].allocate_attempt(
        first.logical_request_id,
        manifest_identity=fingerprint("d"),
        replayed_after_unknown_outcome=False,
    )
    ledger[0].mark_in_flight(original)
    replacement = ledger[0].reconcile_and_allocate_attempt(
        first.logical_request_id, manifest_identity=fingerprint("d")
    )
    ledger[0].mark_in_flight(replacement)
    raw = b'{"recovered":true}'
    now = datetime.now(timezone.utc)
    recovered_result = PhysicalAttemptResult(
        context=replacement,
        status=PhysicalAttemptStatus.COMPLETED,
        model="Qwen/Qwen3-32B",
        returned_model="Qwen/Qwen3-32B",
        finish_reason="stop",
        response_hash="sha256:" + sha256(raw).hexdigest(),
        metrics=PhysicalAttemptMetrics(
            started_at=now,
            completed_at=now,
            latency_seconds=0.01,
            usage=TokenUsage(
                input_tokens=1,
                uncached_input_tokens=1,
                cache_read_input_tokens=0,
                cache_write_input_tokens=0,
                output_tokens=1,
                total_tokens=2,
                usage_complete=True,
            ),
        ),
    )
    ledger[0].complete_response(
        GatewayResponse({"recovered": True}, recovered_result, raw),
        validated_output={"recovered": True},
        output_schema_name="synthetic_recovery",
        output_schema_version=1,
    )
    unbound = prepare_request_only(ledger[0], "unbound")
    with pytest.raises(LedgerConflictError, match="unbound"):
        ledger[0].snapshot_accounting(accounting_scope())
    ledger[0].bind_accounting_scope(
        unbound.logical_request_id,
        accounting_scope(scenario_id="scenario-2"),
    )
    snapshot = ledger[0].snapshot_accounting(accounting_scope())
    assert snapshot.population.physical_attempt_ids == (
        original.attempt_id,
        replacement.attempt_id,
    )
    assert [item.status for item in snapshot.physical_attempts] == [
        "unknown_outcome",
        "completed",
    ]
    assert snapshot.physical_attempts[0].unknown_outcome_kind == "other"


def test_v1_store_migrates_additive_accounting_scope_table(ledger):
    _, root, identity = ledger
    ledger[0].store._connection.execute("DROP TABLE llm_accounting_scopes")
    ledger[0].store._connection.execute("PRAGMA user_version=1")
    ledger[0].store.close()
    reopened = CheckpointStore.open(root, identity, CONFIG.resolve())
    try:
        assert reopened._connection.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {
            row[0]
            for row in reopened._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "llm_accounting_scopes" in tables
    finally:
        reopened.close()


def test_recovery_reconciles_allocated_without_claiming_unknown_outcome(ledger):
    record = prepare_request_only(ledger[0], "allocated-crash")
    original = ledger[0].allocate_attempt(
        record.logical_request_id,
        manifest_identity=fingerprint("d"),
        replayed_after_unknown_outcome=False,
    )
    plan = ledger[0].plan_recovery(record.logical_request_id)
    assert plan.action is LLMRecoveryAction.DISPATCH_FIRST_ATTEMPT
    replacement = ledger[0].reconcile_and_allocate_attempt(
        record.logical_request_id, manifest_identity=fingerprint("d")
    )
    assert ledger[0].attempt_status(original.attempt_id).value == "abandoned_before_dispatch"
    assert replacement.attempt_id.endswith("00000002")
    assert replacement.replayed_after_unknown_outcome is False


def test_recovery_reconciles_in_flight_as_unknown_and_sets_replay_flag(ledger):
    record = prepare_request_only(ledger[0], "inflight-crash")
    original = ledger[0].allocate_attempt(
        record.logical_request_id,
        manifest_identity=fingerprint("d"),
        replayed_after_unknown_outcome=False,
    )
    ledger[0].mark_in_flight(original)
    plan = ledger[0].plan_recovery(record.logical_request_id)
    assert plan.action is LLMRecoveryAction.DISPATCH_RECOVERY_ATTEMPT
    assert plan.replayed_after_unknown_outcome is True
    replacement = ledger[0].reconcile_and_allocate_attempt(
        record.logical_request_id, manifest_identity=fingerprint("d")
    )
    assert ledger[0].attempt_status(original.attempt_id).value == "unknown_outcome"
    assert replacement.attempt_id.endswith("00000002")
    assert replacement.replayed_after_unknown_outcome is True


def test_applied_recovery_restores_checkpoint_without_dispatch(ledger):
    record, _, _ = prepare_application(ledger[0], unit="already-applied")
    plan = ledger[0].plan_recovery(record.logical_request_id)
    assert plan.action is LLMRecoveryAction.RESTORE_APPLIED_CHECKPOINT
    with pytest.raises(LedgerConflictError, match="not recoverable"):
        ledger[0].reconcile_and_allocate_attempt(
            record.logical_request_id, manifest_identity=fingerprint("d")
        )


def test_effect_is_immutable_idempotent_ordered_and_costed(ledger):
    target, _, policy = prepare_application(ledger[0], unit="policy", output_tokens=7)
    _, _, critic = prepare_application(
        ledger[0], role=ProviderRole.CRITIC, unit="critic", output_tokens=11
    )
    ledger[0].commit_checkpoint("checkpoint-action", "after_action_commit", {"ok": True})
    effect = ledger[0].record_effect(
        effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
        effect_artifact_id="final-action-1",
        effect_artifact_sha256=fingerprint("f"),
        ordered_application_ids=(policy.application_id, critic.application_id),
        committed_checkpoint_id="checkpoint-action",
    )
    assert effect == ledger[0].record_effect(
        effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
        effect_artifact_id="final-action-1",
        effect_artifact_sha256=fingerprint("f"),
        ordered_application_ids=(policy.application_id, critic.application_id),
        committed_checkpoint_id="checkpoint-action",
    )
    assert effect.ordered_application_ids == (
        policy.application_id,
        critic.application_id,
    )
    assert ledger[0].effective_output_cost() == (18, True)
    assert ledger[0].load_completed_output(target.logical_request_id).output == {
        "action": "policy"
    }
    material = ledger[0].load_completed_response_material(target.logical_request_id)
    assert material.validated_output == {"action": "policy"}
    assert material.raw_response_body == b'{"unit":"policy"}'
    assert '{"unit":"policy"}' not in repr(material)


def test_checkpoint_and_effect_commit_is_atomic_and_recoverable(ledger):
    _, _, application = prepare_application(ledger[0], unit="atomic")
    effect = ledger[0].commit_checkpoint_and_effect(
        checkpoint_id="atomic-effect-cp",
        event_kind="after_action_commit",
        checkpoint_payload={"final_action": "artifact-atomic"},
        effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
        effect_artifact_id="artifact-atomic",
        effect_artifact_sha256=fingerprint("f"),
        ordered_application_ids=(application.application_id,),
    )
    assert ledger[0].get_effect(effect.effect_id) == effect
    assert effect == ledger[0].commit_checkpoint_and_effect(
        checkpoint_id="atomic-effect-cp",
        event_kind="after_action_commit",
        checkpoint_payload={"final_action": "artifact-atomic"},
        effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
        effect_artifact_id="artifact-atomic",
        effect_artifact_sha256=fingerprint("f"),
        ordered_application_ids=(application.application_id,),
    )


def test_failed_effect_validation_rolls_back_new_checkpoint(ledger):
    with pytest.raises(LedgerConflictError, match="not durably applied"):
        ledger[0].commit_checkpoint_and_effect(
            checkpoint_id="must-rollback",
            event_kind="after_action_commit",
            checkpoint_payload={"bad": True},
            effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
            effect_artifact_id="bad-action",
            effect_artifact_sha256=fingerprint("f"),
            ordered_application_ids=("application-" + "0" * 64,),
        )
    assert ledger[0].store._connection.execute(
        "SELECT 1 FROM checkpoint_events WHERE checkpoint_id='must-rollback'"
    ).fetchone() is None


@pytest.mark.parametrize("outcome", list(NonSubstantiveOutcome))
def test_none_skip_noop_and_rejected_outcomes_create_no_cost_effect(ledger, outcome):
    _, _, application = prepare_application(ledger[0], unit=outcome.value)
    ledger[0].record_non_substantive_outcome(
        outcome=outcome, ordered_application_ids=(application.application_id,)
    )
    assert ledger[0].effective_output_cost() == (0, True)
    count = ledger[0].store._connection.execute(
        "SELECT COUNT(*) FROM qwen_effective_effects"
    ).fetchone()[0]
    assert count == 0


def test_one_application_cannot_be_reclassified_into_a_second_effect(ledger):
    _, _, application = prepare_application(ledger[0], unit="one")
    ledger[0].commit_checkpoint("effect-cp-1", "effect", {"n": 1})
    ledger[0].commit_checkpoint("effect-cp-2", "effect", {"n": 2})
    ledger[0].record_effect(
        effect_kind=QwenEffectKind.POLICY_MEMORY_MUTATION,
        effect_artifact_id="memory-1",
        effect_artifact_sha256=fingerprint("7"),
        ordered_application_ids=(application.application_id,),
        committed_checkpoint_id="effect-cp-1",
    )
    with pytest.raises(LedgerConflictError, match="already belongs"):
        ledger[0].record_effect(
            effect_kind=QwenEffectKind.WORLD_MEMORY_MUTATION,
            effect_artifact_id="memory-2",
            effect_artifact_sha256=fingerprint("8"),
            ordered_application_ids=(application.application_id,),
            committed_checkpoint_id="effect-cp-2",
        )


def test_non_qwen_application_is_never_cost_eligible(ledger):
    _, _, application = prepare_application(
        ledger[0], role=ProviderRole.EMBEDDING, unit="embedding", output_tokens=0
    )
    ledger[0].commit_checkpoint("effect-cp", "effect", {"n": 1})
    with pytest.raises(LedgerConflictError, match="non-Qwen"):
        ledger[0].record_effect(
            effect_kind=QwenEffectKind.POLICY_MEMORY_MUTATION,
            effect_artifact_id="memory",
            effect_artifact_sha256=fingerprint("9"),
            ordered_application_ids=(application.application_id,),
            committed_checkpoint_id="effect-cp",
        )


def test_missing_effect_output_usage_makes_only_cost_incomplete(ledger):
    _, _, application = prepare_application(
        ledger[0], unit="unknown-usage", complete=False
    )
    ledger[0].commit_checkpoint("effect-cp", "effect", {"n": 1})
    ledger[0].record_effect(
        effect_kind=QwenEffectKind.FAILURE_MODE_MUTATION,
        effect_artifact_id="failure-mode",
        effect_artifact_sha256=fingerprint("0"),
        ordered_application_ids=(application.application_id,),
        committed_checkpoint_id="effect-cp",
    )
    assert ledger[0].effective_output_cost() == (None, False)


def test_reopen_preserves_effect_and_rejects_changed_identity(ledger):
    _, _, application = prepare_application(ledger[0], unit="restart")
    ledger[0].commit_checkpoint("effect-cp", "effect", {"n": 1})
    effect = ledger[0].record_effect(
        effect_kind=QwenEffectKind.ACCEPTED_SKILL_MUTATION,
        effect_artifact_id="skill-v2",
        effect_artifact_sha256=fingerprint("f"),
        ordered_application_ids=(application.application_id,),
        committed_checkpoint_id="effect-cp",
    )
    ledger[0].store.close()
    reopened_store = CheckpointStore.open(ledger[1], ledger[2], CONFIG.resolve())
    reopened = LLMLedger(reopened_store)
    try:
        assert reopened.get_effect(effect.effect_id) == effect
        assert reopened.effective_output_cost() == (7, True)
    finally:
        reopened_store.close()


def test_store_permissions_and_sqlite_pragmas(ledger):
    database = ledger[1] / "checkpointing" / "ledger.sqlite3"
    assert os.stat(ledger[1]).st_mode & 0o777 == 0o700
    assert os.stat(database).st_mode & 0o777 == 0o600
    connection = ledger[0].store._connection
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
