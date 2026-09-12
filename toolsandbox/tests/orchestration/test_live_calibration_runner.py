"""Durable real role replay uses offline transports, never actual providers."""
import pytest
from toolsandbox_pipeline.orchestration.live_calibration_runner import DurableCalibrationReplay
from toolsandbox_pipeline.orchestration.live_calibration import PilotReceipt, PilotCapturedRequest
from toolsandbox_pipeline.online.token_limit_calibration import CapturedCalibrationRequest, with_ceiling
from tests.orchestration.test_live_turn_context import setup, real_ledger
from tests.retrieval.test_queries import state
from tests.online.test_durable_roles import digest


def captured(setup):
    factory, request, kwargs, chat, _, _ = setup
    context = factory(request)
    current = state()
    prepared = context.role_request_builders.initial_policy(current,
        context.retrieval.retrieve_policy_skills(current, canonical_to_agent={'search_contacts': 'scrambled'})).request
    original = context.durable_roles.execute(prepared)
    replay = DurableCalibrationReplay(ledger=factory.ledger, gateways=kwargs['gateways'],
        qwen_config=kwargs['qwen_config'], manifest_identity=kwargs['manifest_identity'], run_id=request.identity.run_id)
    receipt = PilotReceipt(request.identity.scenario_id, request.identity.family_id, request.identity.episode_id,
        'completed_evaluated', (PilotCapturedRequest(CapturedCalibrationRequest(scenario_id=request.identity.scenario_id,
            scenario_family_id=request.identity.family_id, prepared=prepared), original.logical_request_id,
            original.source_attempt_id),), 'synthetic-pilot-checkpoint')
    replay.persist_receipt(receipt)
    return replay, with_ceiling(prepared, 256, kwargs['qwen_config'], digest('a')), chat


def test_same_corpus_request_replayed_with_distinct_durable_ids(setup):
    replay, prepared, chat = captured(setup)
    first, second = replay(prepared), replay(prepared)
    assert first.output == second.output
    assert first.attempt.context.logical_request_id != second.attempt.context.logical_request_id
    assert first.attempt.context.attempt_id != second.attempt.context.attempt_id
    assert replay.ledger.load_completed_output(first.attempt.context.logical_request_id).output == first.output.model_dump(mode='json')
    assert len(chat.calls) == 3


def test_length_is_recorded_and_next_larger_ceiling_can_replay(setup):
    replay, prepared, chat = captured(setup)
    chat.finish = 'length'
    result = replay(prepared)
    assert result.truncated and result.output is None
    assert replay.ledger.request_status(result.attempt.context.logical_request_id).value != 'in_flight'
    chat.finish = 'stop'
    result = replay(with_ceiling(prepared, 512, replay.qwen, digest('a')))
    assert not result.truncated and result.output is not None


def test_unbound_request_cannot_dispatch(setup):
    replay, prepared, chat = captured(setup)
    before = len(chat.calls)
    with pytest.raises(ValueError, match='natural G000'):
        replay(prepared.model_copy(update={'state_id': 'unseen'}))
    assert len(chat.calls) == before
