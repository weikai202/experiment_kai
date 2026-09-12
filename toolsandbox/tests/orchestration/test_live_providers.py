from types import SimpleNamespace
import pytest
from openai import NOT_GIVEN

from toolsandbox_pipeline.orchestration.live_providers import LiveProviderServices
from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig, UserSimulatorConfig
from toolsandbox_pipeline.providers.contracts import ProviderRequestError
from tests.providers.test_request_identity import FakeTransport, chat_response
from tests.retrieval.test_embedding_cache import Transport
from tests.orchestration.test_live_offline import live


def services(live, *, phase='train'):
    embedding = Transport()
    user = FakeTransport(chat_response('synthetic user response', model='gpt-4o-mini-2024-07-18'))
    svc = LiveProviderServices(ledger=live.ledger,qwen_config=QwenConfig(structured_output_wire_mode='guided_json'),
        embedding_config=EmbeddingConfig(expected_dimension=2),user_config=UserSimulatorConfig(),
        manifest_identity=live.requests.manifest_identity,phase=phase,embedding_transport=embedding,user_transport=user)
    return svc, embedding, user


def test_embedding_durable_recovery_has_one_physical_attempt(live):
    svc, transport, _ = services(live)
    scope = live.requests.scope
    context = svc.embedding_context_factory(scope,0,['synthetic text'])
    response = svc.embedding_gateway.embed(context,['synthetic text'])
    assert svc.embedding_record_durable(response) is True
    restored_context = svc.embedding_context_factory(scope,0,['synthetic text'])
    restored = svc.embedding_gateway.embed(restored_context,['synthetic text'])
    assert restored_context == context and restored == response
    assert svc.embedding_record_durable(restored) is True
    assert len(transport.calls) == 1
    assert live.ledger.snapshot_accounting(scope=scope).physical_attempts[0].phase == 'train'


def test_user_factory_recovers_exact_native_response_without_dispatch(live):
    svc, _, transport = services(live)
    identity = SimpleNamespace(run_id='run',round_index=0,episode_id='episode',family_id='family',
        scenario_id='scenario',system_variant='generation_0',phase='dev_minibench',runtime_config_sha256=svc.manifest_identity)
    user = svc.user_factory(identity,None,live.ledger)
    assert not transport.calls
    messages = [{'role':'user','content':'synthetic conversation'}]
    first = user.model_inference(messages,NOT_GIVEN)
    second = svc.user_factory(identity,None).model_inference(messages,NOT_GIVEN)
    assert first == second
    assert len(transport.calls) == 1
    sent = transport.calls[0]
    assert set(sent) == {'model','messages','tools'}
    assert sent['tools'] is NOT_GIVEN
    assert sent['model'] == 'gpt-4o-mini-2024-07-18'


def test_user_failure_recorded_and_terminal_not_retried(live):
    svc, _, transport = services(live)
    transport.payload = chat_response('x',model='wrong')
    identity = SimpleNamespace(run_id='run',round_index=0,episode_id='episode',family_id='family',
        scenario_id='scenario',system_variant='generation_0',phase='train',runtime_config_sha256=svc.manifest_identity)
    user = svc.user_factory(identity,None)
    with pytest.raises(ProviderRequestError):
        user.model_inference([{'role':'user','content':'synthetic'}],NOT_GIVEN)
    with pytest.raises(RuntimeError,match='terminal'):
        user.model_inference([{'role':'user','content':'synthetic'}],NOT_GIVEN)
    assert len(transport.calls) == 1


def test_construction_uses_no_client_or_secret_lookup(live,monkeypatch):
    import toolsandbox_pipeline.providers.openai_clients as clients
    monkeypatch.setattr(clients,'create_transport',lambda *a,**kw: pytest.fail('eager client or secret access'))
    service = LiveProviderServices(ledger=live.ledger,qwen_config=QwenConfig(structured_output_wire_mode='guided_json'),
        embedding_config=EmbeddingConfig(),user_config=UserSimulatorConfig(),
        manifest_identity=live.requests.manifest_identity,phase='generation_build')
    assert set(service.gateways) == {'policy','critic','revision'}
    assert service.embedding_gateway.dimension is None


def test_embedding_phase_separation_and_input_drift(live):
    svc, transport, _ = services(live,phase='calibration')
    scope = live.requests.scope
    ctx = svc.embedding_context_factory(scope,0,['one'])
    with pytest.raises(ValueError,match='differ'):
        svc.embedding_gateway.embed(ctx,['different'])
    assert not transport.calls
    other, _, _ = services(live,phase='dev_minibench')
    other_ctx = other.embedding_context_factory(scope,0,['one'])
    assert ctx.logical_request_id != other_ctx.logical_request_id
    assert (ctx.phase,other_ctx.phase) == ('calibration','dev_minibench')


def test_factory_exact_native_user_passes_real_official_transaction_guard(live):
    from toolsandbox_pipeline.providers.user_simulator import InstrumentedGPT4oMiniUser
    from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import TransactionalUserRole, TransactionalRoleError
    svc, _, transport = services(live)
    identity = SimpleNamespace(run_id='run',round_index=0,episode_id='episode',family_id='family',
        scenario_id='scenario',system_variant='generation_0',phase='online_token_calibration_pilot',
        runtime_config_sha256=svc.manifest_identity,profile='official_live')
    delegate=svc.user_factory(identity,None,live.ledger)
    assert type(delegate) is InstrumentedGPT4oMiniUser
    role=TransactionalUserRole(delegate,identity=identity,trajectory_store=None)
    assert role.delegate is delegate and delegate._durability_seam is not None
    assert not transport.calls
    messages=[{'role':'user','content':'synthetic native user turn'}]
    first=role.delegate.model_inference(messages,NOT_GIVEN)
    restored=TransactionalUserRole(svc.user_factory(identity,None),identity=identity,trajectory_store=None)
    assert restored.delegate.model_inference(messages,NOT_GIVEN)==first
    assert len(transport.calls)==1
    identity.profile='strict_replay'
    with pytest.raises(TransactionalRoleError,match='forbidden'):
        TransactionalUserRole(delegate,identity=identity,trajectory_store=None)
