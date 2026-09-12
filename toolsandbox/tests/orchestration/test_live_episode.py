"""Synthetic coordinator/gate/role seams; no scenario registry or provider access."""
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest

from toolsandbox_pipeline.orchestration import live_episode as live
from toolsandbox_pipeline.online.turn_context import TurnContext, TurnRoleRequestBuilders
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.dataset import ScenarioRecord
from toolsandbox_pipeline.reproducibility.dataset_access import ScenarioLease

D = 'sha256:' + '1' * 64


def scope():
    return live.LiveEpisodeScope(run_id='run', round_index=1, shard_id='train-shard-1', generation_id='g001',
        profile='strict_replay', dataset_manifest_sha256=D, runtime_config_sha256=D,
        prompt_manifest_sha256=D, token_limit_config_sha256=D, fixture_manifest_sha256=D,
        environment_sha256=D, scenario_positions={'family': 0})


def record():
    return ScenarioRecord(scenario_id='family', scenario_family_id='family', variant='no_distraction',
        categories=('NO_DISTRACTION_TOOLS',), max_messages=4, starting_context_sha256=D,
        evaluation_definition_sha256=D, agent_facing_tool_names_sha256=D, agent_facing_tool_schema_sha256=D)


def executor(monkeypatch, *, authorized=True, calibrated=True):
    calls, checkpoints, captures = [], {}, []
    generation = SimpleNamespace(manifest=SimpleNamespace(generation_id='g001'), skills=())
    lease = ScenarioLease(record(), SimpleNamespace(max_messages=4))
    def authorize(value):
        calls.append(('authorize', value))
        if not authorized:
            raise PermissionError('formal gate failed')
    def load(**kwargs):
        calls.append(('load', kwargs)); return (lease,)
    ledger = SimpleNamespace(store=object(), get_checkpoint=lambda key: checkpoints.get(key))
    def commit(key, kind, payload):
        calls.append(('capture_checkpoint', key))
        checkpoints[key] = SimpleNamespace(payload=payload)
    ledger.commit_checkpoint = commit
    proposal = ActionEnvelope.model_validate({'action': {'type': 'assistant_message', 'content': 'Done.'}})
    ledger.load_completed_output = lambda _: SimpleNamespace(output=proposal.model_dump(mode='json'))
    def context(request):
        calls.append(('context', request))
        prepared_request = SimpleNamespace(token_limit_config_status='calibrated' if calibrated else 'calibration',
                                           token_limit_config_sha256=D)
        prepared = SimpleNamespace(request=prepared_request, context=SimpleNamespace(user_envelope='{"state":{"visible":"exact"}}'))
        return TurnContext(identity=request.identity, generation=generation,
            retrieval=SimpleNamespace(snapshot=generation), state_builder=object(), controller=object(),
            controller_tool_metadata=(), role_request_builders=TurnRoleRequestBuilders(
                lambda *args: prepared, lambda *args: prepared, lambda *args: prepared_request),
            durable_roles=object(), checkpoint_sink=object(), committed_tool_outcomes=request.committed_tool_outcomes)
    service = live.LiveEpisodeExecutor(scope=scope(), gate=SimpleNamespace(load=load), generation=generation,
        ledger=ledger, trajectory_store=object(), metadata=(), context_factory=context,
        user_factory=lambda *args: calls.append(('user', args)) or object(), authorization=authorize,
        physical_attempt_provider=lambda identity, ids: ('actual-user-attempt', 'actual-qwen-attempt'),
        boot_id='boot', capture_sink=captures.append)
    monkeypatch.setattr(live, 'extract_visible_messages', lambda _: ())
    adapter = SimpleNamespace(controller_context=SimpleNamespace(agent_to_execution_name={}))
    monkeypatch.setattr(live, 'build_adapter_turn', lambda *args: adapter)
    monkeypatch.setattr(live, 'ToolLedger', lambda _: object())
    monkeypatch.setattr(live, 'NativeEvaluator', lambda _: object())
    monkeypatch.setattr(live, 'ExecutionEnvironment', lambda: object())
    monkeypatch.setattr(live, 'TransactionalExecutionEnvironment', lambda *args, **kwargs: SimpleNamespace(records=[]))
    monkeypatch.setattr(live, 'TransactionalUserRole', lambda *args, **kwargs: object())
    def agent(**kwargs):
        return SimpleNamespace(factory=kwargs['responder_factory'], decisions=[])
    monkeypatch.setattr(live, 'TransactionalAgentRole', agent)
    monkeypatch.setattr(live, 'DurableTurnResponder', lambda context: SimpleNamespace(context=context))
    result = SimpleNamespace(status='synthetic_native_result')
    class Runner:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def run(self, request):
            calls.append(('native_run', request))
            assert request.eligible_for_train_offline_consumption is True
            assert self.kwargs['physical_attempt_provider'](()) == ('actual-user-attempt', 'actual-qwen-attempt')
            agent = request.roles[live.RoleType.AGENT]
            responder = agent.factory(0)
            responder.context.role_request_builders.initial_policy(None, None)
            # Capture checkpoint must precede the simulated physical dispatch.
            assert any(kind == 'capture_checkpoint' and key.startswith('live-initial-') for kind, key in calls)
            calls.append(('model_dispatch', None))
            agent.decisions.append(SimpleNamespace(identity=SimpleNamespace(agent_turn_index=0),
                                                   initial_policy_logical_request_id='initial'))
            return result
    monkeypatch.setattr(live, 'EpisodeRunner', Runner)
    return service, calls, captures, result


@pytest.mark.parametrize('scenario,position,generation', [('other', 0, 'g001'), ('family', 1, 'g001'), ('family', 0, 'g002')])
def test_wrong_scope_fails_before_authorization_or_dataset(monkeypatch, scenario, position, generation):
    service, calls, _, _ = executor(monkeypatch)
    assert calls == []
    with pytest.raises(PermissionError):
        service.run_scenario(scenario_id=scenario, manifest_position=position, generation_id=generation)
    assert calls == []


def test_formal_gate_precedes_materialization(monkeypatch):
    service, calls, _, _ = executor(monkeypatch, authorized=False)
    with pytest.raises(PermissionError, match='formal gate'):
        service.run_scenario(scenario_id='family', manifest_position=0, generation_id='g001')
    assert [item[0] for item in calls] == ['authorize']


def test_native_seams_capture_exact_inputs_and_same_instance_reuse(monkeypatch):
    service, calls, captures, result = executor(monkeypatch)
    assert service.run_scenario(scenario_id='family', manifest_position=0, generation_id='g001') is result
    assert calls[1][1] == dict(requested_ids=['family'], run_id='run', phase='train_round',
        manifest_sha256=D, split='train', purpose='train_round', train_shard=1)
    capture = captures[0]
    assert capture.identity.round_index == 1 and capture.identity.generation_id == 'g001'
    assert capture.initial_envelopes == ((0, '{"state":{"visible":"exact"}}'),)
    assert capture.proposed_actions[0].action.content == 'Done.'
    assert capture.result is result
    before = len(calls)
    assert service.run_scenario(scenario_id='family', manifest_position=0, generation_id='g001') is result
    assert len(calls) == before and len(captures) == 1


def test_provisional_context_stops_before_model_dispatch(monkeypatch):
    service, calls, captures, _ = executor(monkeypatch, calibrated=False)
    with pytest.raises(ValueError, match='formal calibrated'):
        service.run_scenario(scenario_id='family', manifest_position=0, generation_id='g001')
    assert not any(item[0] == 'model_dispatch' or (item[0] == 'capture_checkpoint' and item[1].startswith('live-initial-')) for item in calls)
    assert captures == [] and not service._active


def test_scope_mapping_immutable_and_round_relationship_enforced():
    original = scope()
    with pytest.raises(TypeError): original.scenario_positions['other'] = 1
    with pytest.raises(ValueError): replace(original, generation_id='g002')
    with pytest.raises(ValueError): replace(original, shard_id='train-shard-0')
    with pytest.raises(ValueError): replace(original, phase='dev')


def test_capture_sink_failure_and_restart_fail_closed_without_duplicate_dispatch(monkeypatch):
    service, calls, captures, _ = executor(monkeypatch)
    def fail(_): raise RuntimeError('capture sink failed')
    service.capture_sink = fail
    with pytest.raises(RuntimeError, match='capture sink'):
        service.run_scenario(scenario_id='family', manifest_position=0, generation_id='g001')
    before = len(calls)
    with pytest.raises(live.LiveEpisodeResumeRequired):
        service.run_scenario(scenario_id='family', manifest_position=0, generation_id='g001')
    assert len(calls) == before
    # A new coordinator instance with the same authoritative ledger has no memory
    # cache, but must still reject a fresh dispatch.
    restarted, restart_calls, _, _ = executor(monkeypatch)
    restarted.ledger = service.ledger
    with pytest.raises(live.LiveEpisodeResumeRequired):
        restarted.run_scenario(scenario_id='family', manifest_position=0, generation_id='g001')
    assert restart_calls == []


def test_invalid_hex_and_noncanonical_training_phase_rejected():
    with pytest.raises(ValueError, match='digest'):
        replace(scope(), runtime_config_sha256='sha256:' + 'z' * 64)
    with pytest.raises(ValueError, match='scope'):
        replace(scope(), phase='train_dev_bypass')
