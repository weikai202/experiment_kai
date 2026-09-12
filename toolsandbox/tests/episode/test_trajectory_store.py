import os

from toolsandbox_pipeline.schemas.trajectory import (
    TrustedEvaluatorRecord,
    TrustedTrajectory,
)


def test_context_round_trip_and_restricted_mode(trajectory_store, start_context):
    stored = trajectory_store.persist_context(start_context)
    restored = trajectory_store.load_context(stored)
    assert trajectory_store.persist_context(restored) == stored
    digest = stored.reference.sha256[7:]
    root = trajectory_store.store._database_path.parents[1]
    path = root / "checkpointing/blobs/sha256" / digest[:2] / digest[2:]
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_context_checkpoint_is_idempotent(trajectory_store, start_context, episode_identity):
    stored = trajectory_store.persist_context(start_context)
    one = trajectory_store.commit_context_checkpoint(
        episode_identity,
        event_kind="unit_boundary",
        stored=stored,
        recipient="AGENT",
        boundary_ordinal=1,
    )
    two = trajectory_store.commit_context_checkpoint(
        episode_identity,
        event_kind="unit_boundary",
        stored=stored,
        recipient="AGENT",
        boundary_ordinal=1,
    )
    assert one == two


def test_trusted_trajectory_build_normalizes_tuple_id_columns(
    trajectory_store, start_context, episode_identity
):
    ending = trajectory_store.persist_context(start_context)
    evaluator = TrustedEvaluatorRecord(
        milestone_similarity=0.0,
        minefield_similarity=0.0,
        similarity=0.0,
        turn_count=0,
        milestone_mapping=(),
        minefield_mapping=(),
        fully_successful=False,
        evaluation_definition_sha256=episode_identity.evaluation_definition_sha256,
        ending_context_sha256=ending.context_sha256,
    )
    evaluator_ref, _ = trajectory_store.persist_evaluator(episode_identity, evaluator)
    trajectory = TrustedTrajectory.build(
        identity=episode_identity,
        messages=(),
        online_turns=(),
        tool_actions=(),
        logical_request_ids=("llm-a",),
        physical_attempt_ids=("attempt-a",),
        ending_context_reference=ending.reference,
        ending_context_sha256=ending.context_sha256,
        evaluator_record_reference=evaluator_ref,
        evaluator_record_sha256=evaluator_ref.sha256,
        skill_attributions=(),
        eligible_for_train_offline_consumption=True,
    )
    assert trajectory.trajectory_id.startswith("sha256:")


def test_user_termination_is_retained_without_agent_attribution(
    trajectory_store, start_context, episode_identity
):
    import pytest
    from toolsandbox_pipeline.schemas.trajectory import ToolActionRecord
    ending = trajectory_store.persist_context(start_context)
    control = ToolActionRecord(
        transaction_id="user-end", action_sha256="sha256:" + "a" * 64,
        call_ids=("end-1",), selected_skill_ids=(None,), canonical_tool_ids=("end_conversation",),
        effect_classes=("conversation_control",), pre_context_reference=ending.reference,
        pre_context_sha256=ending.context_sha256, post_context_reference=ending.reference,
        post_context_sha256=ending.context_sha256, executed=True, committed=True,
        rolled_back=False, failed=False,
    )
    args = dict(identity=episode_identity, messages=(), online_turns=(), logical_request_ids=(),
                physical_attempt_ids=(), ending_context_reference=ending.reference,
                ending_context_sha256=ending.context_sha256, evaluator_record_reference=ending.reference,
                evaluator_record_sha256=ending.reference.sha256, skill_attributions=(),
                eligible_for_train_offline_consumption=False)
    trajectory = TrustedTrajectory.build(tool_actions=(control,), **args)
    assert trajectory.tool_actions == (control,)
    assert not trajectory.online_turns
    for change in ({"effect_classes": ("sandbox_read",)}, {"canonical_tool_ids": ("search_messages",)},
                   {"selected_skill_ids": ("skill-1",)}):
        with pytest.raises(ValueError, match="originating turns"):
            TrustedTrajectory.build(tool_actions=(control.model_copy(update=change),), **args)
    with pytest.raises(ValueError, match="identity reused"):
        TrustedTrajectory.build(tool_actions=(control, control.model_copy(update={"transaction_id": "other"})), **args)
