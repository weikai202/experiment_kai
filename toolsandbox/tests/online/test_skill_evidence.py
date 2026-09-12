import json
import pytest
from toolsandbox_pipeline.online.prompt_builder import safe_controller, prepare_request
from toolsandbox_pipeline.online.prompt_contracts import (
    InitialPolicyContext, CriticContext, RevisionContext, PreparedRoleRequest,
    feedback_skill_bindings,
)
from toolsandbox_pipeline.online.prompt_loader import load_prompts
from toolsandbox_pipeline.online.token_limits import ProvisionalTokenLimits
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.runtime import QwenConfig
from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.retrieval.index import file_hash
from tests.online.test_prompt_builder import contexts, ROOT

MAPPING = {'search_contacts': 'scrambled', 'get_current_timestamp': 'scrambled_clock'}


def setup_case(tmp_path):
    original = contexts(tmp_path)[0]
    envelope = json.loads(original.user_envelope)
    envelope['state']['available_tools'].append({'name': 'scrambled_clock', 'schema': {
        'type': 'function', 'function': {'name': 'scrambled_clock', 'parameters': {'type': 'object'}}}})
    envelope['state'].pop('state_id')
    envelope['state']['state_id'] = canonical_sha256(envelope['state'])
    envelope['skills'][0]['tool_dependencies'] = ['scrambled_clock']
    data = original.model_dump(mode='json')
    data['state_id'] = envelope['state']['state_id']
    data['user_envelope'] = canonical_json_bytes(envelope).decode()
    identity = {k: v for k, v in data.items() if k != 'policy_context_hash'}
    identity['user_envelope'] = envelope
    data['policy_context_hash'] = canonical_sha256(identity)
    initial = InitialPolicyContext.model_validate_json(json.dumps(data))
    skill = envelope['skills'][0]
    action = ActionEnvelope.model_validate(dict(action=dict(type='function_call', call_id='c',
        name='scrambled', arguments={'creation_timestamp_upperbound': 123.0}, selected_skill_id=skill['skill_id'])))
    decision = ControllerDecision.model_validate(dict(blocking_codes=['CONSTRAINT_VIOLATION'], critic_trigger_codes=[],
        evidence=[dict(code='CONSTRAINT_VIOLATION', source_kind='skill', source_ref=f"skill:{skill['skill_id']}@{skill['version']}")]))
    return initial, action, decision, envelope['skills']


def make_critic(initial, action, decision, skills):
    feedback = safe_controller(decision, action, retrieved_skills=skills)
    return CriticContext(initial=initial, proposed_action_json=action.model_dump_json(),
        proposed_action_sha256=canonical_sha256(action.model_dump(mode='json')), controller_feedback=feedback,
        controller_feedback_sha256=canonical_sha256(feedback.model_dump(mode='json')), world_memory=())


def request(context, role):
    raw = (ROOT / 'configs/online_token_limits.provisional.json').read_bytes()
    return prepare_request(context, prompt=load_prompts(ROOT)[role],
        token_limits=ProvisionalTokenLimits.model_validate_json(raw), token_limit_config_sha256=file_hash(raw),
        qwen_config=QwenConfig(structured_output_wire_mode='guided_json'), canonical_to_agent=MAPPING)


def test_mismatch_minimal_evidence_is_shared_with_revision(tmp_path):
    initial, action, decision, skills = setup_case(tmp_path)
    critic = make_critic(initial, action, decision, skills)
    evidence = critic.controller_feedback.evidence[0]
    assert evidence.reason == 'selected_skill_tool_mismatch'
    assert evidence.action_pointer == '/action/selected_skill_id'
    assert evidence.evidence_ref == safe_controller(decision).evidence[0].evidence_ref
    revision = RevisionContext(initial=initial, critic=critic, critic_feedback_json=json.dumps(dict(
        verdict='revise', predicted_outcome='failure', predicted_effect='Skill does not cover this tool.',
        error_codes=['CONSTRAINT_VIOLATION'], correction='Choose a matching skill or null.')))
    requests = [request(critic, 1), request(revision, 2)]
    envelopes = [json.loads(item.messages[1].content) for item in requests]
    assert envelopes[0]['controller_feedback'] == envelopes[1]['controller_feedback']
    assert envelopes[0]['retrieved_skill_bindings'] == envelopes[1]['retrieved_skill_bindings'] == [
        {'skill_id': skills[0]['skill_id'], 'tool_dependencies': ['scrambled_clock']}]
    assert 'skills' not in envelopes[0]
    for item in requests:
        assert 'get_current_timestamp' not in item.messages[1].content
        assert 'search_contacts' not in item.messages[1].content
        assert 'source_ref' not in item.messages[1].content
        assert PreparedRoleRequest.model_validate_json(item.model_dump_json()) == item


def test_matching_unknown_and_ambiguous_bindings_do_not_guess(tmp_path):
    initial, action, decision, skills = setup_case(tmp_path)
    for replacement in ('scrambled_clock',):
        data = action.model_dump(mode='json'); data['action']['name'] = replacement
        feedback = safe_controller(decision, ActionEnvelope.model_validate(data), retrieved_skills=skills)
        assert feedback.evidence[0].reason is None
    assert safe_controller(decision, action, retrieved_skills=[]).evidence[0].reason is None
    call = action.model_dump(mode='json')['action']; call.pop('type')
    batch = ActionEnvelope.model_validate({'action': {'type': 'parallel_batch', 'calls': [call, dict(call, call_id='c2')]}})
    assert safe_controller(decision, batch, retrieved_skills=skills).evidence[0].reason is None
    other = dict(call, call_id='other', selected_skill_id=None)
    batch = ActionEnvelope.model_validate({'action': {'type': 'parallel_batch', 'calls': [other, call]}})
    feedback = safe_controller(decision, batch, retrieved_skills=skills)
    assert feedback.evidence[0].action_pointer == '/action/calls/1/selected_skill_id'
    assert feedback_skill_bindings(batch, feedback, skills)[0]['skill_id'] == call['selected_skill_id']


@pytest.mark.parametrize('damage', ['missing', 'matching', 'canonical', 'unknown_skill', 'wrong_pointer'])
def test_direct_restoration_rejects_unproven_mismatch(tmp_path, damage):
    initial, action, decision, skills = setup_case(tmp_path)
    critic = make_critic(initial, action, decision, skills)
    prepared = request(critic, 1)
    data = prepared.model_dump(mode='json')
    envelope = json.loads(data['messages'][1]['content'])
    if damage == 'missing':
        envelope.pop('retrieved_skill_bindings')
    elif damage == 'matching':
        envelope['retrieved_skill_bindings'][0]['tool_dependencies'] = ['scrambled']
    elif damage == 'canonical':
        envelope['retrieved_skill_bindings'][0]['tool_dependencies'] = ['get_current_timestamp']
    elif damage == 'unknown_skill':
        envelope['retrieved_skill_bindings'][0]['skill_id'] = 'not_retrieved'
    else:
        envelope['controller_feedback']['evidence'][0]['action_pointer'] = '/action/arguments/creation_timestamp_upperbound'
    data['messages'][1]['content'] = canonical_json_bytes(envelope).decode()
    data['user_envelope_sha256'] = canonical_sha256(envelope)
    with pytest.raises(ValueError, match='Skill'):
        PreparedRoleRequest.model_validate_json(json.dumps(data))


def test_context_rejects_wrong_binding_even_with_new_hash(tmp_path):
    initial, action, decision, skills = setup_case(tmp_path)
    critic = make_critic(initial, action, decision, skills)
    data = critic.model_dump(mode='json')
    changed = json.loads(data['proposed_action_json']); changed['action']['name'] = 'scrambled_clock'
    data['proposed_action_json'] = json.dumps(changed)
    data['proposed_action_sha256'] = canonical_sha256(changed)
    with pytest.raises(ValueError, match='Skill'):
        CriticContext.model_validate_json(json.dumps(data))


def test_actual_controller_wrong_tool_binding_diagnosed_without_changing_judgment(tmp_path):
    from toolsandbox_pipeline.online.controller import Controller
    from toolsandbox_pipeline.online.controller_inputs import RetrievedSkillControllerView, SkillStatus
    from tests.online.test_controller import data, function
    _, _, _, skills = setup_case(tmp_path)
    view = dict(skills[0], tool_dependencies=['other_public'])
    selected = RetrievedSkillControllerView(skill_id=view['skill_id'], version=view['version'],
        status=SkillStatus.ACTIVE, generation_id='g0', tool_dependencies=('other_canonical',))
    controller_input = data(function({'value': 1}, skill=view['skill_id']), skills=(selected,))
    decision = Controller()(controller_input)
    before = decision.model_dump_json()
    assert [code.value for code in decision.blocking_codes] == ['CONSTRAINT_VIOLATION']
    feedback = safe_controller(decision, controller_input.action, retrieved_skills=[view])
    assert feedback.evidence[0].reason == 'selected_skill_tool_mismatch'
    assert decision.model_dump_json() == before
    assert Controller()(data(function({'value': 1}), skills=(selected,))).blocking_codes == []


def test_revision_minimal_binding_must_match_initial_skills(tmp_path):
    initial, action, decision, skills = setup_case(tmp_path)
    critic = make_critic(initial, action, decision, skills)
    revision = RevisionContext(initial=initial, critic=critic, critic_feedback_json=json.dumps(dict(
        verdict='revise', predicted_outcome='failure', predicted_effect='Mismatch.',
        error_codes=['CONSTRAINT_VIOLATION'], correction='Use a matching Skill.')))
    data = request(revision, 2).model_dump(mode='json')
    envelope = json.loads(data['messages'][1]['content'])
    envelope['skills'][0]['tool_dependencies'] = ['scrambled']
    data['messages'][1]['content'] = canonical_json_bytes(envelope).decode()
    data['user_envelope_sha256'] = canonical_sha256(envelope)
    with pytest.raises(ValueError, match='Skill'):
        PreparedRoleRequest.model_validate_json(json.dumps(data))
