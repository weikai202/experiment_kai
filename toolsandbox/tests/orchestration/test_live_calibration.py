"""Synthetic callbacks only; these tests are not real model calibration."""
from types import SimpleNamespace
import json

import pytest

from toolsandbox_pipeline.orchestration import live_calibration as live
from toolsandbox_pipeline.online.token_limit_calibration import (
    CalibrationRuntimeInputs, CalibrationScenario, CapturedCalibrationRequest,
    VARIANTS, select_calibration_sample,
)
from toolsandbox_pipeline.online.prompt_contracts import InitialPolicyContext, CriticContext, RevisionContext
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.runtime import QwenConfig
from tests.online.test_prompt_builder import contexts, prepared
from tests.retrieval.test_queries import state

D = 'sha256:' + 'a' * 64


def setup(tmp_path, monkeypatch, available=lambda index: ('policy', 'critic', 'revision')):
    templates = contexts(tmp_path)
    records = tuple(CalibrationScenario(scenario_id=f'synthetic-{i}-{j}', variant=variant, split='train')
        for i, variant in enumerate(VARIANTS) for j in range(8))
    sample = select_calibration_sample(records, D)
    events, persisted, replayed = [], [], []

    class Gate:
        def load(self, **kwargs):
            assert kwargs['split'] == 'train' and kwargs['purpose'] == live.PURPOSE
            assert kwargs['phase'] == live.PHASE and kwargs['manifest_sha256'] == D
            sid = kwargs['requested_ids'][0]
            events.append(sid)
            return (SimpleNamespace(record=SimpleNamespace(scenario_id=sid, scenario_family_id='synthetic-family')),)

    def pilot(*, lease, run_id, purpose):
        assert purpose == live.PURPOSE
        sid = lease.record.scenario_id
        current = state(sid)
        first, second, third = templates
        envelope = json.loads(first.user_envelope)
        envelope['state'] = current.model_dump(mode='json', by_alias=True)
        identity = first.identity_payload()
        identity.update(state_id=current.state_id, user_envelope=envelope)
        first = InitialPolicyContext(**{**first.model_dump(), 'state_id': current.state_id,
            'user_envelope': canonical_json_bytes(envelope).decode(), 'policy_context_hash': canonical_sha256(identity)})
        second = CriticContext(**{**second.model_dump(), 'initial': first})
        third = RevisionContext(initial=first, critic=second, critic_feedback_json=third.critic_feedback_json)
        requests = tuple(live.PilotCapturedRequest(
            CapturedCalibrationRequest(scenario_id=sid, scenario_family_id='synthetic-family', prepared=prepared(context, index)),
            f'logical-{sid}-{index}', f'attempt-{sid}-{index}')
            for index, context in enumerate((first, second, third))
            if ('policy', 'critic', 'revision')[index] in available(len(events)-1))
        return live.PilotReceipt(sid, 'synthetic-family', f'episode-{sid}', 'completed_evaluated', requests, f'checkpoint-{sid}')

    def calibration(captured, **kwargs):
        # Stub records orchestration dispatch, not fabricated model results.
        replayed.append((captured, kwargs))
        return None, {}

    monkeypatch.setattr(live, 'calibrate_and_write', calibration)
    runtime = CalibrationRuntimeInputs(**{name: D for name in CalibrationRuntimeInputs.model_fields if name.endswith('sha256')},
        user_simulator_model='gpt-4o-mini-2024-07-18', online_orchestrator_version='synthetic-test')
    args = dict(records=records, train_manifest_sha256=D, run_id='synthetic-run', gate=Gate(), pilot_executor=pilot,
        persist_receipt=persisted.append, runtime_inputs=runtime, qwen=QwenConfig(structured_output_wire_mode='guided_json'),
        hard_ceiling=1024, invoke=lambda _: pytest.fail('unexpected direct replay'), output_directory=tmp_path / 'recommendation')
    return args, sample, events, persisted, replayed


def test_initial32_are_complete_before_replay(tmp_path, monkeypatch):
    args, sample, events, receipts, replay = setup(tmp_path, monkeypatch)
    result = live.collect_and_calibrate(**args)
    assert events == list(sample.initial) and len(receipts) == 32
    assert result.role_counts == {'policy': 32, 'critic': 32, 'revision': 32}
    assert len(replay) == 1 and len(replay[0][0]) == 96


def test_reserve_is_bounded_prefix_and_never_forces_missing_roles(tmp_path, monkeypatch):
    args, sample, events, receipts, replay = setup(tmp_path, monkeypatch,
        available=lambda index: ('policy',) if index < 8 else ('policy', 'critic', 'revision'))
    result = live.collect_and_calibrate(**args)
    assert events == list(sample.initial + sample.reserve[:8])
    assert result.role_counts == {'policy': 40, 'critic': 32, 'revision': 32}
    assert len(receipts) == 40 and len(replay) == 1


def test_insufficient_critic_or_revision_stops_before_any_replay(tmp_path, monkeypatch):
    args, sample, events, receipts, replay = setup(tmp_path, monkeypatch, available=lambda _: ('policy',))
    with pytest.raises(live.CalibrationCollectionError, match='insufficient_natural') as caught:
        live.collect_and_calibrate(**args)
    assert events == list(sample.initial + sample.reserve) and len(receipts) == 64
    assert caught.value.receipts == tuple(receipts) and not replay
    assert caught.value.role_counts == {'policy': 64, 'critic': 0, 'revision': 0}


def test_invalid_output_root_rejected_before_pilots(tmp_path, monkeypatch):
    args, _, events, _, replay = setup(tmp_path, monkeypatch)
    args['output_directory'] = tmp_path
    with pytest.raises(ValueError, match='output directory'):
        live.collect_and_calibrate(**args)
    assert not events and not replay


def test_duplicate_request_source_fails_and_retains_receipt(tmp_path, monkeypatch):
    from dataclasses import replace
    args, _, _, persisted, replay = setup(tmp_path, monkeypatch)
    original = args['pilot_executor']
    def duplicate(**kwargs):
        receipt = original(**kwargs)
        return replace(receipt, requests=receipt.requests + (receipt.requests[0],))
    args['pilot_executor'] = duplicate
    with pytest.raises(live.CalibrationCollectionError, match='request_binding') as caught:
        live.collect_and_calibrate(**args)
    assert caught.value.receipts == tuple(persisted) and len(persisted) == 1 and not replay


def test_late_role_identity_conflict_is_found_before_policy_replay(tmp_path, monkeypatch):
    from dataclasses import replace
    args, _, _, receipts, replay = setup(tmp_path, monkeypatch)
    original = args['pilot_executor']
    def conflict(**kwargs):
        receipt = original(**kwargs)
        if len(receipts) == 31:
            request = receipt.requests[-1]
            altered = request.captured.model_copy(update={'prepared': request.captured.prepared.model_copy(update={'generation_id': 'g001'})})
            receipt = replace(receipt, requests=receipt.requests[:-1] + (replace(request, captured=altered),))
        return receipt
    args['pilot_executor'] = conflict
    with pytest.raises(live.CalibrationCollectionError, match='mixed_role_corpus'):
        live.collect_and_calibrate(**args)
    assert len(receipts) == 32 and not replay


def test_duplicate_input_keeps_all_receipts_but_only_one_corpus_entry(tmp_path, monkeypatch):
    from dataclasses import replace
    args, _, _, receipts, replay = setup(tmp_path, monkeypatch)
    original = args['pilot_executor']
    def repeat(**kwargs):
        receipt = original(**kwargs)
        duplicated = tuple(replace(request, logical_request_id=request.logical_request_id + '-repeat',
            source_attempt_id=request.source_attempt_id + '-repeat') for request in receipt.requests)
        return replace(receipt, requests=receipt.requests + duplicated)
    args['pilot_executor'] = repeat
    result = live.collect_and_calibrate(**args)
    assert sum(len(receipt.requests) for receipt in receipts) == 192
    assert len(replay[0][0]) == 96
    assert result.role_counts == {'policy': 32, 'critic': 32, 'revision': 32}


def test_failed_pilot_receipt_is_preserved_without_replay(tmp_path, monkeypatch):
    from dataclasses import replace
    args, _, _, receipts, replay = setup(tmp_path, monkeypatch)
    original = args['pilot_executor']
    args['pilot_executor'] = lambda **kwargs: replace(original(**kwargs), status='failed')
    with pytest.raises(live.CalibrationCollectionError, match='pilot_did_not_complete') as caught:
        live.collect_and_calibrate(**args)
    assert caught.value.receipts == tuple(receipts) and len(receipts) == 1 and not replay
