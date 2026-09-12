"""Injected fake native roles/ledger only; never counted as real calibration."""
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from toolsandbox_pipeline.orchestration import live_episode as live
from toolsandbox_pipeline.orchestration.live_pilot import LiveCalibrationPilot
from toolsandbox_pipeline.orchestration.live_calibration import PURPOSE, PHASE
from toolsandbox_pipeline.online.durable_roles import DurableRoleExecution
from toolsandbox_pipeline.online.turn_context import TurnContext, TurnRoleRequestBuilders, PreparedInitialPolicy, PreparedCritic
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.critic import CriticOutput
from toolsandbox_pipeline.schemas.accounting import TaskAccountingInput, ScopeTimingInput
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.trajectory import EpisodeResult, EpisodeExecutionStatus
from toolsandbox_pipeline.reproducibility.dataset_access import ScenarioLease
from tests.orchestration.test_live_episode import executor, record, D
from tests.online.test_prompt_builder import contexts, prepared


def setup_pilot(tmp_path, monkeypatch, roles=('policy',), completed=True):
    engine, calls, captures, _ = executor(monkeypatch)
    templates = contexts(tmp_path)
    requests = tuple(prepared(template, index) for index, template in enumerate(templates))
    config_hash = requests[0].token_limit_config_sha256
    assert all(request.token_limit_config_sha256 == config_hash for request in requests)
    engine.scope = replace(engine.scope, round_index=0, shard_id='train-shard-0', generation_id='g000',
        token_limit_config_sha256=config_hash)
    engine.generation.manifest.generation_id = 'g000'
    source_records = {}
    engine.ledger.load_completed_output = lambda logical: source_records[logical]
    proposal = ActionEnvelope(action={'type': 'assistant_message', 'content': 'Observed answer'})
    critic = CriticOutput(verdict='accept', error_codes=[], predicted_outcome='success', predicted_effect='Answer is grounded', correction='')

    class Roles:
        def execute(self, request):
            calls.append(('natural_execute', request.role))
            logical = 'logical-' + request.role
            source = 'source-attempt-' + request.role
            output = critic if request.role == 'critic' else proposal
            source_records[logical] = SimpleNamespace(logical_request_id=logical, source_attempt_id=source, output=output.model_dump(mode='json'))
            return DurableRoleExecution(request, output, logical, source)
        def apply(self, *args, **kwargs):
            calls.append(('apply_forwarded', None))
        def commit_online_action(self, *args, **kwargs):
            calls.append(('commit_forwarded', None))

    def context(request):
        return TurnContext(identity=request.identity, generation=engine.generation,
            retrieval=SimpleNamespace(snapshot=engine.generation), state_builder=object(), controller=object(),
            controller_tool_metadata=(), role_request_builders=TurnRoleRequestBuilders(
                lambda *args: PreparedInitialPolicy(templates[0], requests[0]),
                lambda *args: PreparedCritic(templates[1], requests[1]), lambda *args: requests[2]),
            durable_roles=Roles(), checkpoint_sink=object(), committed_tool_outcomes=request.committed_tool_outcomes)
    engine.context_factory = context

    class NativeRunner:
        def __init__(self, **kwargs):
            pass
        def run(self, native):
            calls.append(('native_calibration', native))
            assert native.eligible_for_train_offline_consumption is False
            assert native.identity.phase == PHASE and native.identity.round_index is None
            assert native.identity.shard_id is None
            agent = native.roles[live.RoleType.AGENT]
            responder = agent.factory(0)
            builders = responder.context.role_request_builders
            for role in roles:
                calls.append(('native_route', role))
                request = builders.initial_policy(None, None).request if role == 'policy' else builders.critic(None).request if role == 'critic' else builders.revision(None)
                result = responder.context.durable_roles.execute(request)
                responder.context.durable_roles.apply(result)
            if 'policy' in roles:
                agent.decisions.append(SimpleNamespace(identity=SimpleNamespace(agent_turn_index=0), initial_policy_logical_request_id='logical-policy'))
            reference = BlobReference(sha256=D, byte_count=1, media_type='application/vnd.toolsandbox.canonical+json',
                schema_name='synthetic', schema_version=1, content_visibility='restricted')
            identity = native.identity
            return EpisodeResult(identity=identity,
                status=EpisodeExecutionStatus.COMPLETED_EVALUATED if completed else EpisodeExecutionStatus.TERMINAL_FAILURE_BEFORE_EVALUATION,
                ending_context_reference=reference, ending_context_sha256=D,
                trusted_trajectory_reference=reference if completed else None,
                evaluator_record_reference=reference if completed else None,
                sanitized_failure_class=None if completed else 'synthetic_failure',
                task_accounting_input=TaskAccountingInput(run_id=identity.run_id, task_id=identity.episode_id,
                    scenario_family_id=identity.family_id, scenario_id=identity.scenario_id, system_variant='generation_0',
                    timing=ScopeTimingInput(scope_kind='scenario_task', scope_id=identity.episode_id, boot_id='test',
                        started_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc), start_monotonic_ns=0),
                    completion_status='complete' if completed else 'failed', final_context_sha256=D),
                last_checkpoint_ordinal=17)
    monkeypatch.setattr(live, 'EpisodeRunner', NativeRunner)
    def authorize(**kwargs):
        calls.append(('pilot_authorize', kwargs))
    pilot = LiveCalibrationPilot(executor=engine, authorization=authorize)
    lease = ScenarioLease(record(), SimpleNamespace(max_messages=4))
    return pilot, engine, lease, calls, captures, source_records


@pytest.mark.parametrize('roles', [('policy',), ('policy', 'critic'), ('policy', 'critic', 'revision')])
def test_only_naturally_routed_roles_enter_receipt(tmp_path, monkeypatch, roles):
    pilot, engine, lease, calls, captures, _ = setup_pilot(tmp_path, monkeypatch, roles)
    receipt = pilot(lease=lease, run_id='run', purpose=PURPOSE)
    assert receipt.status == 'completed_evaluated'
    assert tuple(request.captured.prepared.role for request in receipt.requests) == roles
    assert tuple(request.source_attempt_id for request in receipt.requests) == tuple('source-attempt-' + role for role in roles)
    assert not any(kind == 'load' or kind == 'authorize' for kind, _ in calls)
    assert any(kind == 'pilot_authorize' for kind, _ in calls)
    assert captures[0].identity.phase == PHASE
    assert engine.ledger.get_checkpoint(receipt.checkpoint_id).payload['prepared_request_count'] == len(roles)
    assert [value for kind, value in calls if kind == 'natural_execute'] == list(roles)
    with pytest.raises(live.LiveEpisodeResumeRequired):
        pilot(lease=lease, run_id='run', purpose=PURPOSE)


def test_wrong_purpose_and_authorization_fail_before_native_loop(tmp_path, monkeypatch):
    pilot, engine, lease, calls, _, _ = setup_pilot(tmp_path, monkeypatch)
    with pytest.raises(PermissionError):
        pilot(lease=lease, run_id='run', purpose='train_round')
    assert calls == []
    pilot.authorization = lambda **kwargs: (_ for _ in ()).throw(PermissionError('denied'))
    with pytest.raises(PermissionError, match='denied'):
        pilot(lease=lease, run_id='run', purpose=PURPOSE)
    assert not any(kind == 'native_calibration' for kind, _ in calls)


def test_failed_episode_is_not_relabelled_completed(tmp_path, monkeypatch):
    pilot, _, lease, _, _, _ = setup_pilot(tmp_path, monkeypatch, completed=False)
    receipt = pilot(lease=lease, run_id='run', purpose=PURPOSE)
    assert receipt.status == 'failed' and len(receipt.requests) == 1


def test_prepared_is_persisted_before_observing_durable_execution(tmp_path, monkeypatch):
    pilot, engine, lease, calls, _, _ = setup_pilot(tmp_path, monkeypatch)
    pilot(lease=lease, run_id='run', purpose=PURPOSE)
    capture_index = next(i for i, (kind, key) in enumerate(calls) if kind == 'capture_checkpoint' and key.startswith('pilot-prepared-'))
    execute_index = next(i for i, (kind, _) in enumerate(calls) if kind == 'natural_execute')
    assert capture_index < execute_index


def test_conflicting_durable_source_is_rejected(tmp_path, monkeypatch):
    pilot, engine, lease, calls, _, source_records = setup_pilot(tmp_path, monkeypatch)
    def wrong_source(logical):
        item = source_records[logical]
        return SimpleNamespace(logical_request_id=logical, source_attempt_id='unrelated-attempt', output=item.output)
    engine.ledger.load_completed_output = wrong_source
    with pytest.raises(ValueError, match='source attempt'):
        pilot(lease=lease, run_id='run', purpose=PURPOSE)
    assert not any(kind == 'capture_checkpoint' and key.startswith('pilot-receipt-') for kind, key in calls)


@pytest.mark.parametrize('changes', [
    {'phase': 'train_round'}, {'round_index': 0, 'shard_id': 'train-shard-0'},
    {'generation_id': 'g001'}, {'dataset_manifest_sha256': 'sha256:' + 'f' * 64},
])
def test_calibration_entry_rejects_formal_or_drifted_identity(tmp_path, monkeypatch, changes):
    pilot, engine, lease, calls, _, _ = setup_pilot(tmp_path, monkeypatch)
    original = engine._identity(lease.record, 0)
    identity = type(original).model_validate({**original.model_dump(), 'phase': PHASE,
        'round_index': None, 'shard_id': None, **changes})
    with pytest.raises(PermissionError):
        engine.run_calibration_lease(lease=lease, identity=identity, purpose=PURPOSE,
            authorization=lambda **kwargs: pytest.fail('unexpected authorization'),
            prepared_request_sink=lambda *args: None, role_execution_sink=lambda *args: None)
    assert calls == []
