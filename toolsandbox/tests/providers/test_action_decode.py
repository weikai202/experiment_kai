"""Host labels preserve raw actions while keeping the execution contract strict."""
import json
from dataclasses import replace
import pytest
from toolsandbox_pipeline.toolsandbox_adapter.action_decode import decode_action, action_decoding_audit
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from tests.providers.test_request_identity import context


def raw_batch():
    return {'action': {'type': 'parallel_batch', 'calls': [
        {'call_id': 'copied-exec-label', 'selected_skill_id': None, 'name': 'lookup', 'arguments': {'item': 11}},
        {'call_id': 'copied-exec-label', 'selected_skill_id': None, 'name': 'lookup', 'arguments': {'item': 29}},
    ]}}


def test_duplicate_labels_become_stable_position_ids_without_removing_calls():
    source=raw_batch();content=json.dumps(source)
    with pytest.raises(ValueError):ActionEnvelope.model_validate_json(content)
    result=decode_action(content,context())
    assert len(result.action.action.calls)==2
    assert len({call.call_id for call in result.action.action.calls})==2
    assert [call.arguments for call in result.action.action.calls]==[{'item':11},{'item':29}]
    assert [item.model_call_id for item in result.call_identities]==['copied-exec-label']*2
    assert [item.position for item in result.call_identities]==[0,1]
    assert decode_action(content,context(attempt_id='recovery-attempt')).action==result.action
    another=decode_action(content,context(logical_request_id='next-turn',unit_reference='next-state'))
    assert {call.call_id for call in another.action.action.calls}.isdisjoint(call.call_id for call in result.action.action.calls)
    assert json.loads(content)==source


@pytest.mark.parametrize('mutate',[
    lambda p:p['action']['calls'][0].update(call_id=''),
    lambda p:p['action']['calls'][0].pop('call_id'),
    lambda p:p['action']['calls'][0].update(call_id=3),
    lambda p:p['action']['calls'][0].update(arguments=[]),
    lambda p:p['action']['calls'][0].update(selected_skill_id=''),
    lambda p:p['action']['calls'][0].update(extra_field=True),
    lambda p:p['action'].update(calls=[]),
])
def test_non_identity_contract_is_still_strict(mutate):
    payload=raw_batch();mutate(payload)
    with pytest.raises(ValueError):decode_action(json.dumps(payload),context())


@pytest.mark.parametrize('raw',['{"action":{},"action":{}}','{"action":{"type":"function_call","arguments":{"x":NaN}}}'])
def test_invalid_json_not_sanitized_into_valid_action(raw):
    with pytest.raises(ValueError):decode_action(raw,context())


def test_legacy_committed_action_is_recognized_without_reinterpretation():
    payload=raw_batch();payload['action']['calls'][1]['call_id']='legacy-second'
    content=json.dumps(payload);legacy=ActionEnvelope.model_validate_json(content)
    assert action_decoding_audit(content=content,raw_response_body=b'original',context=context(),action=legacy,allow_legacy=True) is None
    with pytest.raises(ValueError):
        action_decoding_audit(content=content,raw_response_body=b'original',context=context(),action=legacy)
