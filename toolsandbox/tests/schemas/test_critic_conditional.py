import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from toolsandbox_pipeline.schemas.critic import CriticOutput, CRITIC_MAX_WORDS

BASE = dict(verdict='accept', predicted_outcome='success', predicted_effect='Respond to user.', error_codes=[], correction='')

@pytest.mark.parametrize('changes,valid', [({},True),({'correction':'none'},False),({'error_codes':['INSUFFICIENT_CONTEXT']},False),({'predicted_effect':''},False),({'verdict':'revise'},False),({'verdict':'revise','error_codes':['INSUFFICIENT_CONTEXT'],'correction':'Ask user.'},True),({'verdict':'uncertain','error_codes':['INSUFFICIENT_CONTEXT'],'correction':'Ask user.'},False),({'verdict':'uncertain','predicted_outcome':'uncertain','error_codes':['INSUFFICIENT_CONTEXT'],'correction':'Ask user.'},True)])
def test_conditional_wire_and_existing_semantics_agree(changes, valid):
    payload={**BASE,**changes}
    assert Draft202012Validator(CriticOutput.model_json_schema()).is_valid(payload) is valid
    if valid: CriticOutput.model_validate(payload)
    else:
        with pytest.raises(ValidationError): CriticOutput.model_validate(payload)

def test_word_limit_and_duplicate_error_code_checks_remain_strict():
    assert CRITIC_MAX_WORDS == 128
    for changes in ({'predicted_effect':'word '*129},{'verdict':'revise','correction':'Ask user.','error_codes':['INSUFFICIENT_CONTEXT']*2}):
        with pytest.raises(ValidationError): CriticOutput.model_validate({**BASE,**changes})

def test_schema_preserves_all_properties_and_forbids_extra_fields():
    schema=CriticOutput.model_json_schema()
    Draft202012Validator.check_schema(schema)
    assert len(schema['anyOf'])==3
    for branch in schema['anyOf']:
        assert branch['additionalProperties'] is False
        assert set(branch['required'])==set(BASE)


def test_error_code_wire_bound_matches_finite_unique_code_contract():
    from toolsandbox_pipeline.schemas.critic import CriticErrorCode
    codes = [code.value for code in CriticErrorCode]
    schema = CriticOutput.model_json_schema()
    for branch in schema['anyOf']:
        expected = 0 if branch['properties']['verdict']['const'] == 'accept' else len(codes)
        assert branch['properties']['error_codes']['maxItems'] == expected
    payload = {**BASE, 'verdict': 'revise', 'correction': 'Correct the constraints.', 'error_codes': codes}
    assert Draft202012Validator(schema).is_valid(payload)
    CriticOutput.model_validate(payload)
    too_many = {**payload, 'error_codes': codes + [codes[0]]}
    assert not Draft202012Validator(schema).is_valid(too_many)
    with pytest.raises(ValidationError):
        CriticOutput.model_validate(too_many)
    # The finite bound is not a replacement for strict uniqueness validation.
    with pytest.raises(ValidationError, match='duplicates'):
        CriticOutput.model_validate({**payload, 'error_codes': codes[:1] * 2})
