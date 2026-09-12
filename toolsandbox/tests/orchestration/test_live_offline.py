from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from toolsandbox_pipeline.checkpointing import CheckpointStore, LLMLedger, RunIdentity
from toolsandbox_pipeline.orchestration.live_offline import LiveOfflineRequests, LiveOfflineDurability, LiveMemoryUpdater, LiveSkillUpdater
from toolsandbox_pipeline.providers.qwen import QwenGateway
from toolsandbox_pipeline.schemas.runtime import QwenConfig
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.offline.memory_roles import prepare_candidate_request, load_token_limits
from toolsandbox_pipeline.offline.memory_prompts import load_memory_prompts
from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
from toolsandbox_pipeline.offline.memory_orchestrator import SealedMemoryTrajectoryBuffer
from toolsandbox_pipeline.reproducibility import canonical_sha256
from tests.providers.test_request_identity import FakeTransport, chat_response
from tests.offline_memory.test_orchestrator import reference, buffer, Resolver

ROOT = Path(__file__).parents[2]
D = 'sha256:' + 'a' * 64


@pytest.fixture
def live(tmp_path):
    identity = RunIdentity(run_id='run', profile='offline', environment_identity=D,
        dataset_manifest_sha256='sha256:' + '1' * 64, config_manifest_sha256='sha256:' + '2' * 64,
        prompt_manifest_sha256=D, generation_manifest_sha256=D, fixture_manifest_sha256=D)
    with CheckpointStore.create(tmp_path / 'run', identity, ROOT / 'configs/reproducibility/checkpointing_v1.json') as store:
        ledger = LLMLedger(store)
        transport = FakeTransport(chat_response('{"result":"NONE"}'))
        gateway = QwenGateway(QwenConfig(structured_output_wire_mode='guided_json', context_limit=32768), transport=transport)
        scope = AccountingScope(run_id='run', round_index=0, task_id='offline-round-0',
            scenario_family_id='offline', scenario_id='offline', system_variant='generation_0')
        requests = LiveOfflineRequests(ledger=ledger, gateway=gateway, scope=scope,
            manifest_identity=identity.config_manifest_sha256, count_tokens=lambda messages: 10)
        prompts = load_memory_prompts(ROOT, ROOT / 'prompts/offline/memory_manifest.json')
        limits, limits_hash = load_token_limits(ROOT / 'configs/offline_memory_token_limits.provisional.json')
        yield SimpleNamespace(requests=requests, ledger=ledger, transport=transport, prompts=prompts,
                              limits=limits, limits_hash=limits_hash)


def prepared(live):
    return prepare_candidate_request(projection=reference().policy_projection,
        unit_reference='unit', prompts=live.prompts, limits=live.limits,
        limits_sha256=live.limits_hash, structured_output_wire_mode='guided_json')


def test_memory_real_ledger_response_application_recovery_and_none_cost(live):
    executor = live.requests.memory_executor()
    first = executor.execute_and_apply(prepared(live))
    second = executor.execute_and_apply(prepared(live))
    assert first == second
    assert len(live.transport.calls) == 1
    assert len(live.ledger.applications()) == 1
    assert live.ledger.effective_output_cost() == (0, True)
    live.requests.memory_executor().execute_and_apply(prepared(live))
    assert len(live.transport.calls) == 1


def test_context_limit_blocks_before_request_or_dispatch(live):
    live.requests.count_tokens = lambda messages: 32768
    before = live.ledger.store.high_water_marks()
    with pytest.raises(ValueError, match='context limit'):
        live.requests.memory_executor().execute_and_apply(prepared(live))
    assert not live.transport.calls
    assert before == live.ledger.store.high_water_marks()


def test_failed_output_is_recorded_and_never_implicitly_retried(live):
    from toolsandbox_pipeline.providers.contracts import ProviderRequestError
    live.transport.payload = chat_response('bad JSON')
    with pytest.raises(ProviderRequestError):
        live.requests.memory_executor().execute_and_apply(prepared(live))
    with pytest.raises(RuntimeError, match='terminal failure'):
        live.requests.memory_executor().execute_and_apply(prepared(live))
    assert len(live.transport.calls) == 1
    assert not live.ledger.applications()


def memory_service(live, **kwargs):
    snapshot = SimpleNamespace(manifest=SimpleNamespace(generation_id='g000', model_dump=lambda **kw: {'generation_id':'g000'}),
                               policy_memory=(), world_memory=())
    retriever = MemoryCandidateRetriever(generation_id='g000', policy_records=(), world_records=(),
                                         indexes=(), embedding_resolver=Resolver())
    return LiveMemoryUpdater(requests=live.requests, snapshot=snapshot, prompts=live.prompts,
        limits=live.limits, limits_sha256=live.limits_hash, retriever=retriever, **kwargs)


def test_memory_orchestrator_sealed_run_and_resume(live):
    original = buffer((reference(),))
    payload = original.model_dump(mode='json')
    payload['identity']['prompt_manifest_sha256'] = live.prompts.manifest_sha256
    sealed = SealedMemoryTrajectoryBuffer.model_validate_json(json.dumps(payload))
    result = memory_service(live).run(sealed)
    restored = memory_service(live).run(sealed)
    assert result.policy_counts['NONE'] == restored.policy_counts['NONE'] == 1
    assert result.units == restored.units
    assert len(live.transport.calls) == 1
    assert live.ledger.effective_output_cost() == (0, True)


def test_mixed_world_format_is_explicitly_selected(live):
    service = memory_service(live, policy_input_representation='packed-v2', world_input_representation='v1')
    assert service.input_representation_manifest == {'policy':'packed-v2','world':'v1'}
    assert service.orchestrator.input_representation == 'packed-v2'
    assert service.orchestrator.world_input_representation == 'v1'
    assert not live.transport.calls


def test_real_effect_sink_cost_is_idempotent(live):
    from toolsandbox_pipeline.offline.skill_roles import OfflineSkillRoleRunner
    from toolsandbox_pipeline.providers.contracts import ProviderRole
    live.transport.payload = chat_response('{"decision":"ADD","task_condition":"Missing input","failure_mode":"Ask first"}')
    runner = OfflineSkillRoleRunner(live.requests.gateway, role=ProviderRole.FAILURE_MODE_UPDATE, max_tokens=512)
    request = runner.prepare(unit_id='unit', messages=[{'role':'user','content':'synthetic failure'}])
    applied = live.requests.execute(role=request.role, unit_id=request.unit_id,
        fingerprint=request.input_fingerprint, messages=request.messages, output_model=request.output_model,
        max_tokens=request.max_tokens, invoke=lambda context: runner.run(request, context))
    sink = LiveOfflineDurability(live.ledger)
    effect = sink.commit_failure_mutation(unit_id='unit', application_id=applied.application_id,
                                         mutation_artifact_sha256=D)
    assert sink.commit_failure_mutation(unit_id='unit', application_id=applied.application_id,
                                       mutation_artifact_sha256=D) == effect
    assert live.ledger.effective_output_cost() == (3, True)



def skill_service(live, **changes):
    from hashlib import sha256
    from tests.offline_skill.test_orchestrator import current_skill, Never
    prompts = {role: (ROOT / 'prompts/offline' / filename).read_text() for role, filename in
               [('failure_mode_update', 'failure_mode_update_v1.txt'), ('skill_candidate', 'skill_candidate_v1.txt')]}
    kwargs = dict(requests=live.requests, current_skills=(current_skill(),), public_tool_inventory=('search_stock',),
        public_tool_schemas=(), prompts=prompts,
        prompt_hashes={role: 'sha256:' + sha256(text.encode()).hexdigest() for role, text in prompts.items()},
        prompt_manifest_sha256=D, token_limits_sha256=D,
        max_tokens={'failure_mode_update':512, 'skill_candidate':2048}, mini_bench_executor=Never())
    kwargs.update(changes)
    return LiveSkillUpdater(**kwargs)


def test_skill_failure_output_roundtrips_and_reuses_real_application(live):
    from toolsandbox_pipeline.schemas.offline_skill import SkillFailureEvidence
    live.transport.payload = chat_response('{"decision":"SKIP","reason":"Case specific"}')
    service = skill_service(live)
    payload = dict(skill_id='skill', skill_version='v1.0', trajectory_id=D, episode_id='episode', manifest_position=0,
        evidence_kind='visible_tool_exception', canonical_tool_dependencies=['search_stock'],
        sanitized_outcome_class='ValueError', generalized_failure='A required input was missing.')
    payload['source_evidence_sha256'] = canonical_sha256({'protocol':'skill-failure-evidence-v1', **payload})
    evidence = SkillFailureEvidence.model_validate_json(json.dumps(payload))
    first = service.orchestrator.failure_executor.decide(evidence=evidence, current_buffer=(), unit_id='failure-unit')
    second = service.orchestrator.failure_executor.decide(evidence=evidence, current_buffer=(), unit_id='failure-unit')
    assert first == second
    assert first.decision.decision == 'SKIP'
    assert len(live.transport.calls) == 1
    assert live.ledger.effective_output_cost() == (0, True)


def test_skill_calibrated_cap_is_not_silently_bootstrapped(live):
    with pytest.raises(ValueError, match='verified selection'):
        skill_service(live, max_tokens={'failure_mode_update':576, 'skill_candidate':2048})
    assert not live.transport.calls


def test_changed_prepared_bytes_cannot_reuse_completed_output(live):
    from toolsandbox_pipeline.offline.memory_roles import OfflinePromptMessage
    p = prepared(live)
    live.requests.memory_executor().execute_and_apply(p)
    corrupted = p.model_copy(update={'messages': (p.messages[0], OfflinePromptMessage(role='user', content='{}'))})
    with pytest.raises(Exception, match='conflict'):
        live.requests.memory_executor().execute_and_apply(corrupted)
    assert len(live.transport.calls) == 1



def test_skill_round_adapter_uses_supplied_empty_buffer_without_models_or_dev(live):
    from toolsandbox_pipeline.schemas.offline_skill import SkillRoundIdentity
    from toolsandbox_pipeline.offline.skill_orchestrator import SealedSkillTrajectoryBuffer
    sealed = canonical_sha256(dict(protocol='sealed-skill-buffer-v1', run_id='run', round_index=0,
                                   shard_id='train-shard-0', generation_id='g000', entries=[]))
    identity = SkillRoundIdentity(run_id='run', round_index=0, shard_id='train-shard-0',
        current_generation_id='g000', next_generation_id='g001',
        dataset_manifest_sha256='sha256:' + '1' * 64, config_manifest_sha256='sha256:' + '2' * 64,
        prompt_manifest_sha256=D, token_limit_config_sha256=D, sealed_input_buffer_sha256=sealed)
    result = skill_service(live).run(SealedSkillTrajectoryBuffer(identity, ()))
    assert result.completion_status == 'completed'
    assert not result.accepted_skill_ids
    assert not live.transport.calls
    assert live.ledger.effective_output_cost() == (0, True)


def test_buffer_manifest_drift_rejected_before_model(live):
    with pytest.raises(ValueError, match='identity mismatch'):
        live.requests.validate_buffer(buffer((reference(),)).identity.model_copy(update={'round_index':1}))
    assert not live.transport.calls


def test_memory_add_real_ledger_links_candidate_and_review_once(live):
    from toolsandbox_pipeline.providers.contracts import TransportResponse
    outputs = [dict(result='CANDIDATE', role='policy', candidate=dict(scope='General prerequisites', applicability=[],
        action_guidance='Validate required arguments before execution', avoid=[])),
        dict(decision='ADD', reason='Reusable new rule')]
    class SequentialTransport:
        calls = 0
        def create(self, **request):
            value = chat_response(json.dumps(outputs[self.calls]))
            self.calls += 1
            return TransportResponse(json.dumps(value).encode(), value)
    transport = SequentialTransport()
    live.requests.gateway = QwenGateway(live.requests.gateway.config, transport=transport)
    payload = buffer((reference(),)).model_dump(mode='json')
    payload['identity']['prompt_manifest_sha256'] = live.prompts.manifest_sha256
    sealed = SealedMemoryTrajectoryBuffer.model_validate_json(json.dumps(payload))
    result = memory_service(live).run(sealed)
    assert result.policy_counts['ADD'] == 1
    assert len(result.staged_policy_memory) == 1
    assert len(live.ledger.applications()) == 2
    assert len(live.ledger.effects()) == 1
    assert live.ledger.effective_output_cost() == (6, True)
    resumed = memory_service(live).run(sealed)
    assert resumed.staged_policy_memory == result.staged_policy_memory
    assert transport.calls == 2
    assert live.ledger.effective_output_cost() == (6, True)



@pytest.mark.parametrize('verifier', [None, lambda selection, digest: False])
def test_nonbootstrap_skill_selection_requires_positive_evidence_verification(live, verifier):
    from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig
    selection = RoleTokenLimitConfig(version='synthetic-calibration', role='failure_mode_update',
        stage='calibrated', max_tokens=576, evidence_manifest_identity=D)
    with pytest.raises(ValueError, match='train token-limit evidence'):
        skill_service(live, max_tokens={'failure_mode_update':576,'skill_candidate':2048},
            token_limit_configs={'failure_mode_update':selection}, verify_token_limit_evidence=verifier)
    assert not live.transport.calls


def test_explicit_synthetic_skill_calibration_evidence_is_forwarded(live):
    from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig
    selection = RoleTokenLimitConfig(version='synthetic-calibration', role='failure_mode_update',
        stage='calibration', max_tokens=576, evidence_manifest_identity=D)
    verified = []
    def verify(config, config_hash):
        verified.append((config,config_hash))
        return True  # Synthetic fixture only; production caller verifies actual artifacts.
    service = skill_service(live, max_tokens={'failure_mode_update':576,'skill_candidate':2048},
        token_limit_configs={'failure_mode_update':selection}, verify_token_limit_evidence=verify)
    assert verified == [(selection,D)]
    assert not live.transport.calls
