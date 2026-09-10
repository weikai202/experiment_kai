from pathlib import Path

import pytest

from toolsandbox_pipeline.checkpointing import (
    CheckpointStore,
    RunIdentity,
    ToolActionIdentity,
    ToolLedger,
    ToolLedgerConflictError,
    ToolRecoveryAction,
)


CONFIG = Path(__file__).parents[2] / "configs/reproducibility/checkpointing_v1.json"


def digest(character):
    return "sha256:" + character * 64


@pytest.fixture
def tool_ledger(tmp_path):
    identity = RunIdentity(
        run_id="run-1",
        profile="offline",
        environment_identity=digest("1"),
        dataset_manifest_sha256=digest("2"),
        config_manifest_sha256=digest("3"),
        prompt_manifest_sha256=digest("4"),
        generation_manifest_sha256=digest("5"),
        fixture_manifest_sha256=digest("6"),
    )
    store = CheckpointStore.create(tmp_path / "run", identity, CONFIG.resolve())
    try:
        yield ToolLedger(store), store
    finally:
        store.close()


def prepare(tool_ledger, *, profile="offline", effects=("sandbox_write",), ordinal=1):
    ledger, store = tool_ledger
    pre = store.blobs.put(
        b'{"context":"pre"}',
        media_type="application/vnd.toolsandbox.execution-context+json",
        schema_name="ExecutionContextEnvelope",
        schema_version=1,
    )
    identity = ToolActionIdentity(
        run_id="run-1",
        scenario_id=f"scenario-{ordinal}",
        pre_action_context_sha256=digest("7"),
        action_fingerprint=digest("8"),
        ordered_call_ids=(f"call-{ordinal}",),
        action_ordinal=ordinal,
        profile=profile,
        effect_classes=effects,
        fixture_manifest_sha256=digest("9"),
        backend_manifest_sha256=digest("a"),
    )
    return ledger.prepare_transaction(identity, pre_context_reference=pre)


def test_tool_commit_is_at_most_once_and_restores_post_context(tool_ledger):
    record = prepare(tool_ledger)
    attempt = tool_ledger[0].allocate_attempt(record.transaction_id)
    tool_ledger[0].mark_in_flight(attempt)
    post = tool_ledger[1].blobs.put(
        b'{"context":"post"}',
        media_type="application/vnd.toolsandbox.execution-context+json",
        schema_name="ExecutionContextEnvelope",
        schema_version=1,
    )
    committed = tool_ledger[0].commit_attempt(
        attempt,
        post_context_reference=post,
        post_context_sha256=digest("b"),
        visible_result_identity=digest("c"),
    )
    assert committed.status.value == "committed"
    assert tool_ledger[0].plan_recovery(record.transaction_id).action is ToolRecoveryAction.RESTORE_COMMITTED_CONTEXT
    assert tool_ledger[0].restore_committed_context_reference(record.transaction_id) == post
    with pytest.raises(ToolLedgerConflictError, match="not in flight"):
        tool_ledger[0].commit_attempt(
            attempt,
            post_context_reference=post,
            post_context_sha256=digest("b"),
            visible_result_identity=digest("c"),
        )


def test_local_and_fixture_unknown_attempts_are_safely_replayed(tool_ledger):
    local = prepare(tool_ledger, ordinal=1)
    local_attempt = tool_ledger[0].allocate_attempt(local.transaction_id)
    tool_ledger[0].mark_in_flight(local_attempt)
    assert tool_ledger[0].plan_recovery(local.transaction_id).action is ToolRecoveryAction.REPLAY_LOCAL_FROM_PRE_CONTEXT
    assert tool_ledger[0].reconcile_and_allocate_attempt(local.transaction_id).endswith("00000002")

    fixture = prepare(
        tool_ledger,
        profile="strict_replay",
        effects=("external_read",),
        ordinal=2,
    )
    fixture_attempt = tool_ledger[0].allocate_attempt(fixture.transaction_id)
    tool_ledger[0].mark_in_flight(fixture_attempt)
    assert tool_ledger[0].plan_recovery(fixture.transaction_id).action is ToolRecoveryAction.REPLAY_FIXTURE_FROM_PRE_CONTEXT
    assert tool_ledger[0].reconcile_and_allocate_attempt(fixture.transaction_id).endswith("00000002")


def test_conversation_control_unknown_attempt_replays_locally(tool_ledger):
    record = prepare(
        tool_ledger,
        effects=("conversation_control",),
        ordinal=5,
    )
    attempt = tool_ledger[0].allocate_attempt(record.transaction_id)
    tool_ledger[0].mark_in_flight(attempt)

    plan = tool_ledger[0].plan_recovery(record.transaction_id)

    assert plan.action is ToolRecoveryAction.REPLAY_LOCAL_FROM_PRE_CONTEXT
    assert tool_ledger[0].reconcile_and_allocate_attempt(
        record.transaction_id
    ).endswith("00000002")


def test_official_live_external_unknown_requires_reconciliation(tool_ledger):
    record = prepare(
        tool_ledger,
        profile="official_live",
        effects=("external_read",),
        ordinal=3,
    )
    attempt = tool_ledger[0].allocate_attempt(record.transaction_id)
    tool_ledger[0].mark_in_flight(attempt)
    assert tool_ledger[0].plan_recovery(record.transaction_id).action is ToolRecoveryAction.RECONCILIATION_REQUIRED
    with pytest.raises(ToolLedgerConflictError, match="reconciliation_required"):
        tool_ledger[0].reconcile_and_allocate_attempt(record.transaction_id)
    assert tool_ledger[0].plan_recovery(record.transaction_id).action is ToolRecoveryAction.RECONCILIATION_REQUIRED


def test_allocated_crash_is_abandoned_before_execution(tool_ledger):
    record = prepare(tool_ledger, ordinal=4)
    first = tool_ledger[0].allocate_attempt(record.transaction_id)
    assert tool_ledger[0].plan_recovery(record.transaction_id).action is ToolRecoveryAction.EXECUTE_FROM_PRE_CONTEXT
    second = tool_ledger[0].reconcile_and_allocate_attempt(record.transaction_id)
    assert first != second and second.endswith("00000002")
