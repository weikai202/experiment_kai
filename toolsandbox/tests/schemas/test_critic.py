import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.schemas import BlockingCode, CriticErrorCode, CriticOutput


VALID = [
    {
        "verdict": "accept", "predicted_outcome": "success",
        "predicted_effect": "The lookup succeeds.", "error_codes": [], "correction": "",
    },
    {
        "verdict": "revise", "predicted_outcome": "failure",
        "predicted_effect": "The call fails.", "error_codes": ["INVALID_PARAMETER"],
        "correction": "Use the declared parameter.",
    },
    {
        "verdict": "uncertain", "predicted_outcome": "uncertain",
        "predicted_effect": "The outcome is unknown.", "error_codes": ["INSUFFICIENT_CONTEXT"],
        "correction": "Obtain the missing context.",
    },
]


@pytest.mark.parametrize("payload", VALID)
def test_valid_critic_combinations(payload):
    assert CriticOutput.model_validate(payload).model_dump(mode="json") == payload


def test_error_enum_is_blocking_codes_plus_five():
    values = {code.value for code in CriticErrorCode}
    assert {code.value for code in BlockingCode} <= values
    assert values - {code.value for code in BlockingCode} == {
        "PREMATURE_ACTION", "UNNECESSARY_RISK", "NO_RELEVANT_TOOL",
        "INSUFFICIENT_CONTEXT", "LIKELY_MINEFIELD_BEHAVIOR",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"error_codes": ["INVALID_FUNCTION"]},
        {"correction": "change it"},
        {"verdict": "revise", "predicted_outcome": "failure", "error_codes": [], "correction": "fix"},
        {"verdict": "revise", "predicted_outcome": "failure", "error_codes": ["INVALID_FUNCTION"], "correction": ""},
        {"verdict": "uncertain", "predicted_outcome": "success", "error_codes": ["INSUFFICIENT_CONTEXT"], "correction": "check"},
        {"verdict": "uncertain", "predicted_outcome": "uncertain", "error_codes": [], "correction": "check"},
        {"verdict": "uncertain", "predicted_outcome": "uncertain", "error_codes": ["INSUFFICIENT_CONTEXT"], "correction": ""},
        {"verdict": "revise", "predicted_outcome": "failure", "error_codes": ["INVALID_FUNCTION", "INVALID_FUNCTION"], "correction": "fix"},
        {"predicted_effect": ""},
    ],
)
def test_invalid_cross_field_combinations(changes):
    payload = dict(VALID[0])
    payload.update(changes)
    with pytest.raises(ValidationError):
        CriticOutput.model_validate(payload)


@pytest.mark.parametrize("field", ["predicted_effect", "correction"])
def test_40_word_boundary(field):
    payload = dict(VALID[1])
    payload[field] = "word " * 40
    CriticOutput.model_validate(payload)
    payload[field] += "overflow"
    with pytest.raises(ValidationError):
        CriticOutput.model_validate(payload)


def test_critic_strictness_and_schema():
    payload = dict(VALID[0], predicted_effect=1)
    with pytest.raises(ValidationError):
        CriticOutput.model_validate(payload)
    schema = CriticOutput.model_json_schema()
    assert schema["additionalProperties"] is False
