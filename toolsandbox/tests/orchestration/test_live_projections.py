from dataclasses import replace

import pytest

from toolsandbox_pipeline.orchestration.live_projections import (
    CapturedEpisodeInputs, CapturedOnlineTurn, CurrentRoundScope,
    LiveMemoryProjectionAdapter, LiveFailureEvidenceAdapter, LiveTrajectoryLoader,
)
from toolsandbox_pipeline.reproducibility import canonical_sha256, canonical_json_bytes
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnDecision, OnlineTurnIdentity
from toolsandbox_pipeline.online.controller_inputs import ReproducibilityProfile
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeIdentity, TrustedTrajectory, TrustedEvaluatorRecord, NativeMessageRecord,
    OnlineTurnRecord, ToolActionRecord, SkillUseAttribution,
)

D = 'sha256:' + 'a' * 64


def blob(schema='context', raw=b'x'):
    from hashlib import sha256
    return BlobReference(sha256='sha256:' + sha256(raw).hexdigest(), byte_count=len(raw),
        media_type='application/vnd.toolsandbox.canonical+json', schema_name=schema,
        schema_version=1, content_visibility='restricted')


def sample(round_index=0, *, critic=True, exception=True, success=False, revised=False):
    identity = EpisodeIdentity(run_id='run', profile='offline', phase='train_round', round_index=round_index,
        shard_id=f'train-shard-{round_index}', family_id='family', scenario_id='scenario', episode_id='episode',
        manifest_position=0, system_variant='generation_0' if round_index == 0 else 'updated', generation_id=f'g{round_index:03d}',
        starting_context_sha256=D, evaluation_definition_sha256=D, agent_tool_schema_sha256=D,
        dataset_manifest_sha256=D, runtime_config_sha256=D, prompt_manifest_sha256=D,
        token_limit_config_sha256=D, fixture_manifest_sha256=D, environment_sha256=D, max_messages=8)
    evaluation = TrustedEvaluatorRecord(milestone_similarity=float(success), minefield_similarity=0., similarity=float(success),
        turn_count=1, milestone_mapping=(), minefield_mapping=(), fully_successful=success,
        evaluation_definition_sha256=D, ending_context_sha256=D)
    evaluator_hash = canonical_sha256(evaluation.model_dump(mode='json'))
    action = ActionEnvelope(action={'type': 'function_call', 'call_id': 'model-1', 'name': 'search_messages', 'arguments': {}, 'selected_skill_id': 'skill-1'})
    draft = action if not revised else ActionEnvelope(action={'type': 'assistant_message', 'content': 'Different draft'})
    online_identity = OnlineTurnIdentity(run_id='run', profile=ReproducibilityProfile.STRICT_REPLAY, phase='train_round',
        round_index=round_index, shard_id=identity.shard_id, family_id='family', scenario_id='scenario', episode_id='episode',
        agent_turn_index=0, expected_generation_id=identity.generation_id, dataset_manifest_sha256=D,
        runtime_config_sha256=D, prompt_manifest_sha256=D, token_limit_config_sha256=D, fixture_manifest_sha256=D, environment_identity=D)
    controller = {'blocking_codes': [], 'critic_trigger_codes': [], 'evidence': []}
    decision = OnlineTurnDecision(identity=online_identity, state_id='state-1', generation_id=identity.generation_id,
        final_action=action, initial_policy_logical_request_id='policy-1', critic_logical_request_id='critic-1' if critic else None,
        revision_logical_request_id='revision-1' if revised else None,
        initial_controller_decision=controller, post_revision_controller_decision=controller if revised else None,
        critic_verdict={'verdict': 'accept', 'error_codes': [], 'predicted_outcome': 'success', 'predicted_effect': 'Will succeed', 'correction': ''} if critic else None,
        revision_count=int(revised), retrieval_references=(), audit_record_sha256=D)
    logical_ids = ('policy-1',) + (('critic-1',) if critic else ()) + (('revision-1',) if revised else ())
    action_hash = canonical_sha256(action.model_dump(mode='json'))
    turn = OnlineTurnRecord(agent_turn_index=0, state_id='state-1', decision_reference=blob('OnlineTurnDecision'),
        decision_sha256=canonical_sha256(decision.model_dump(mode='json')), final_action_sha256=action_hash,
        logical_request_ids=logical_ids, source_attempt_ids=(), application_ids=(), executed_call_ids=('exec-1',))
    transaction = ToolActionRecord(transaction_id='txn-1', action_sha256=action_hash, call_ids=('exec-1',),
        selected_skill_ids=('skill-1',), canonical_tool_ids=('search_messages',), effect_classes=('sandbox_read',),
        pre_context_reference=blob(), pre_context_sha256=D, post_context_reference=blob(), post_context_sha256=D,
        result_message_indices=(1,), executed=True, committed=True, rolled_back=False, failed=exception)
    messages = (NativeMessageRecord(sandbox_message_index=1, sender='EXECUTION_ENVIRONMENT', recipient='AGENT',
        content='private entity value', openai_tool_call_id='exec-1', tool_call_exception='ValueError: private entity' if exception else None,
        visible_to=('AGENT',)), NativeMessageRecord(sandbox_message_index=2, sender='AGENT', recipient='USER', content='Done', visible_to=('AGENT', 'USER')))
    attribution = SkillUseAttribution(skill_id='skill-1', skill_version='v1.0', generation_id=identity.generation_id,
        executed_call_ids=('exec-1',), canonical_tool_ids=('search_messages',), evaluator_record_sha256=evaluator_hash, fully_successful=success)
    trajectory = TrustedTrajectory.build(identity=identity, messages=messages, online_turns=(turn,), tool_actions=(transaction,),
        logical_request_ids=logical_ids, physical_attempt_ids=(), ending_context_reference=blob(), ending_context_sha256=D,
        evaluator_record_reference=blob('TrustedEvaluatorRecord'), evaluator_record_sha256=evaluator_hash,
        skill_attributions=(attribution,), eligible_for_train_offline_consumption=True)
    capture = CapturedEpisodeInputs(evaluation, (CapturedOnlineTurn({'state': {'state_id': 'state-1'}, 'policy_memory': [], 'skills': []}, draft, decision),))
    scope = CurrentRoundScope('run', round_index, D, D)
    return trajectory, capture, scope


@pytest.mark.parametrize('round_index', [0, 1, 2])
def test_three_rounds_project_only_permitted_labels_and_executed_evidence(round_index):
    trajectory, captured, scope = sample(round_index)
    policy, world = LiveMemoryProjectionAdapter(scope=scope, capture_loader=lambda _: captured).project(trajectory)
    assert policy.native_similarity == 0. and world.attributable_failure
    assert world.attribution_kind == 'real_tool_exception'
    evidence = LiveFailureEvidenceAdapter(scope=scope, capture_loader=lambda _: captured).project(trajectory)
    assert len(evidence) == 1 and evidence[0].evidence_kind == 'visible_tool_exception'
    assert 'private entity' not in evidence[0].model_dump_json()
    for projection in (policy, world):
        assert 'milestone_mapping' not in projection.model_dump_json()
        assert 'evaluator_record' not in projection.model_dump_json()


@pytest.mark.parametrize('options', [{'critic': False}, {'exception': False}, {'revised': True}])
def test_world_does_not_infer_draft_failure_from_episode_or_prediction(options):
    trajectory, captured, scope = sample(**options)
    _, world = LiveMemoryProjectionAdapter(scope=scope, capture_loader=lambda _: captured).project(trajectory)
    assert world is None


def test_successful_episode_creates_no_skill_failure():
    trajectory, captured, scope = sample(success=True, exception=False)
    assert LiveFailureEvidenceAdapter(scope=scope, capture_loader=lambda _: captured).project(trajectory) == ()


def test_loader_validates_blob_and_current_round_before_returning():
    trajectory, _, scope = sample()
    raw = canonical_json_bytes(trajectory.model_dump(mode='json'))
    class Blobs:
        def read(self, reference):
            return raw
    loader = LiveTrajectoryLoader(blob_store=Blobs(), scope=scope)
    assert loader.load_trajectory(blob('TrustedTrajectory', raw)) == trajectory
    with pytest.raises(ValueError, match='blob identity'):
        loader.load_trajectory(blob('TrustedTrajectory', b'wrong'))
    with pytest.raises(ValueError, match='current train round'):
        LiveTrajectoryLoader(blob_store=Blobs(), scope=replace(scope, round_index=1)).load_trajectory(blob('TrustedTrajectory', raw))


def test_drift_and_incomplete_captures_fail_closed():
    trajectory, captured, scope = sample()
    bad_capture = replace(captured, turns=())
    with pytest.raises(ValueError, match='complete ordered'):
        LiveMemoryProjectionAdapter(scope=scope, capture_loader=lambda _: bad_capture).project(trajectory)
    bad_capture = replace(captured, evaluator=captured.evaluator.model_copy(update={'ending_context_sha256': 'sha256:' + 'b' * 64}))
    with pytest.raises(ValueError, match='evaluator binding'):
        LiveMemoryProjectionAdapter(scope=scope, capture_loader=lambda _: bad_capture).project(trajectory)
    bad_turn = replace(captured.turns[0], decision=captured.turns[0].decision.model_copy(update={'state_id': 'wrong'}))
    with pytest.raises(ValueError, match='decision binding'):
        LiveMemoryProjectionAdapter(scope=scope, capture_loader=lambda _: replace(captured, turns=(bad_turn,))).project(trajectory)


def test_skill_calls_must_match_actual_selected_skill_and_visible_result():
    trajectory, captured, scope = sample()
    values = {name: getattr(trajectory, name) for name in trajectory.model_fields if name != 'trajectory_id'}
    values['skill_attributions'] = (trajectory.skill_attributions[0].model_copy(update={'canonical_tool_ids': ('other_tool',)}),)
    altered = TrustedTrajectory.build(**values)
    with pytest.raises(ValueError, match='executed-call binding'):
        LiveFailureEvidenceAdapter(scope=scope, capture_loader=lambda _: captured).project(altered)


def test_terminal_skill_failure_is_generalized_and_requires_post_execution_message():
    trajectory, captured, scope = sample(exception=False)
    adapter = LiveFailureEvidenceAdapter(scope=scope, capture_loader=lambda _: captured)
    assert adapter.project(trajectory)[0].evidence_kind == 'visible_terminal_state_failure'
    values = {name: getattr(trajectory, name) for name in trajectory.model_fields if name != 'trajectory_id'}
    values['messages'] = (trajectory.messages[1].model_copy(update={'sandbox_message_index': 0}), trajectory.messages[0])
    assert adapter.project(TrustedTrajectory.build(**values)) == ()


def test_hidden_exception_cannot_be_world_evidence_and_hidden_state_keys_are_rejected():
    trajectory, captured, scope = sample()
    values = {name: getattr(trajectory, name) for name in trajectory.model_fields if name != 'trajectory_id'}
    values['messages'] = (trajectory.messages[0].model_copy(update={'visible_to': ('SYSTEM',)}), trajectory.messages[1])
    assert LiveMemoryProjectionAdapter(scope=scope, capture_loader=lambda _: captured).project(TrustedTrajectory.build(**values))[1] is None
    envelope = {'state': {'state_id': 'state-1', 'hidden_database': 'never project'}, 'policy_memory': [], 'skills': []}
    altered = replace(captured, turns=(replace(captured.turns[0], policy_envelope=envelope),))
    with pytest.raises(ValueError, match='forbidden'):
        LiveMemoryProjectionAdapter(scope=scope, capture_loader=lambda _: altered).project(trajectory)
