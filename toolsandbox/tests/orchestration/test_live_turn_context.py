"""Real factory, retrieval, role request and ledger wiring with fake transports."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from toolsandbox_pipeline.orchestration.live_turn_context import LiveTurnContextFactory
from toolsandbox_pipeline.orchestration.live_episode import LiveTurnRequest
from toolsandbox_pipeline.online.prompt_loader import load_prompts
from toolsandbox_pipeline.online.token_limits import ProvisionalTokenLimits
from toolsandbox_pipeline.online.controller_inputs import ReproducibilityProfile
from toolsandbox_pipeline.providers.qwen import QwenGateway
from toolsandbox_pipeline.schemas.runtime import QwenConfig
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnIdentity
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.critic import CriticOutput
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
from toolsandbox_pipeline.retrieval.index import file_hash
from tests.online.test_durable_roles import real_ledger, digest
from tests.online.test_qwen_roles import Chat
from tests.memory.test_generation_store import generation, load
from tests.retrieval.test_embedding_cache import dependencies
from tests.retrieval.test_queries import state, action

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def setup(tmp_path, real_ledger):
    snapshot = load(generation(tmp_path))
    raw = (ROOT / 'configs/online_token_limits.provisional.json').read_bytes()
    limits = ProvisionalTokenLimits.model_validate_json(raw)
    qc = QwenConfig(structured_output_wire_mode='guided_json')
    chat = Chat('{"action":{"type":"assistant_message","content":"What would identify the record?"}}')
    gateway = QwenGateway(qc, transport=chat)
    transport, durable, deps = dependencies()
    scopes = []
    def embedding_context(scope, ordinal, texts):
        scopes.append(scope)
        return deps['context_factory'](ordinal, texts)
    with EmbeddingCache(tmp_path / 'cache', EmbeddingIdentity(), 2) as cache:
        kwargs = dict(generation=snapshot, ledger=real_ledger, qwen_config=qc,
            gateways={role: gateway for role in ('policy', 'critic', 'revision')},
            prompts={p.entry.role: p for p in load_prompts(ROOT)}, token_limits=limits,
            token_limit_config_sha256=file_hash(raw), manifest_identity=digest('3'), metadata=(),
            embedding_cache=cache, embedding_gateway=deps['gateway'],
            embedding_context_factory=embedding_context, embedding_record_durable=deps['record_durable'],
            count_prompt_tokens=lambda messages: 100, mode='calibration')
        factory = LiveTurnContextFactory(**kwargs)
        identity = OnlineTurnIdentity(run_id='run-1', profile=ReproducibilityProfile.OFFICIAL_LIVE,
            phase='online_token_calibration_pilot', family_id='family-secret', scenario_id='scenario-secret',
            episode_id='episode-secret', agent_turn_index=0, expected_generation_id='g000',
            dataset_manifest_sha256=digest('2'), runtime_config_sha256=digest('3'),
            prompt_manifest_sha256=digest('4'), token_limit_config_sha256=file_hash(raw),
            fixture_manifest_sha256=digest('6'), environment_identity=digest('1'))
        scope = AccountingScope(run_id='run-1', round_index=None, task_id=identity.episode_id,
            scenario_family_id=identity.family_id, scenario_id=identity.scenario_id, system_variant='generation_0')
        def public_tool(name: str) -> list[dict]:
            return []
        adapter = SimpleNamespace(controller_context=SimpleNamespace(
            agent_to_execution_name={'scrambled': 'search_contacts'}, tool_objects={'scrambled': public_tool}))
        request = LiveTurnRequest(identity, scope, adapter, ())
        yield factory, request, kwargs, chat, transport, scopes


def test_real_retrieval_preparation_and_durable_qwen_execution(setup):
    factory, request, _, chat, transport, scopes = setup
    context = factory(request)
    current = state()
    bundle = context.retrieval.retrieve_policy_skills(current, canonical_to_agent={'search_contacts': 'scrambled'})
    initial = context.role_request_builders.initial_policy(current, bundle)
    assert json.loads(initial.context.user_envelope)['state']['available_tools'][0]['name'] == 'scrambled'
    execution = context.durable_roles.execute(initial.request)
    assert execution.output.action.content == 'What would identify the record?'
    assert factory.ledger.load_completed_output(execution.logical_request_id).output == execution.output.model_dump(mode='json')
    assert len(chat.calls) == len(transport.calls) == 1
    assert scopes == [request.accounting_scope]
    assert context.generation is factory.generation
    assert context.committed_tool_outcomes == request.committed_tool_outcomes
    assert initial.request.token_limit_config_status == 'provisional'


def test_critic_revision_keep_original_visible_context(setup):
    factory, request, _, _, _, _ = setup
    context = factory(request)
    current = state()
    policy = context.role_request_builders.initial_policy(current,
        context.retrieval.retrieve_policy_skills(current, canonical_to_agent={'search_contacts': 'scrambled'}))
    decision = ControllerDecision.model_validate_json('{"blocking_codes":[],"critic_trigger_codes":["MEDIUM_OR_HIGH_RISK"],"evidence":[{"code":"MEDIUM_OR_HIGH_RISK","source_kind":"tool_metadata","source_ref":"search_contacts:private"}]}')
    world = context.retrieval.retrieve_world(current, action(), controller_decision=decision)
    critic = context.role_request_builders.critic(policy.context, action(), decision, world)
    output = CriticOutput.model_validate_json(json.dumps(dict(verdict='accept', predicted_outcome='success',
        predicted_effect='A visible result', error_codes=[], correction='')))
    revision = context.role_request_builders.revision(policy.context, critic.context, output)
    assert json.loads(revision.messages[1].content)['state'] == json.loads(policy.context.user_envelope)['state']
    assert 'search_contacts' not in critic.request.messages[1].content
    assert revision.role == 'revision'


@pytest.mark.parametrize('field,value', [('runtime_config_sha256', digest('9')),
    ('token_limit_config_sha256', digest('9')), ('phase', 'train_round'), ('round_index', 0)])
def test_mismatched_scope_fails_before_provider(setup, field, value):
    factory, request, _, chat, transport, _ = setup
    with pytest.raises(ValueError):
        factory(replace(request, identity=request.identity.model_copy(update={field: value})))
    assert not chat.calls and not transport.calls


def test_formal_rejects_bootstrap_and_configuration_mismatch(setup):
    _, _, kwargs, chat, transport, _ = setup
    with pytest.raises(ValueError):
        LiveTurnContextFactory(**{**kwargs, 'mode': 'formal', 'runtime_inputs_sha256': digest('8')})
    other = QwenGateway(QwenConfig(structured_output_wire_mode='structured_outputs_json'))
    with pytest.raises(ValueError, match='configuration mismatch'):
        LiveTurnContextFactory(**{**kwargs, 'gateways': {**kwargs['gateways'], 'critic': other}})
    assert not chat.calls and not transport.calls
