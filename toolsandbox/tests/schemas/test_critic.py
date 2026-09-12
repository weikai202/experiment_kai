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
def test_128_word_boundary(field):
    payload = dict(VALID[1])
    payload[field] = "word " * 128
    CriticOutput.model_validate(payload)
    payload[field] += "overflow"
    with pytest.raises(ValidationError):
        CriticOutput.model_validate(payload)


def test_critic_strictness_and_schema():
    payload = dict(VALID[0], predicted_effect=1)
    with pytest.raises(ValidationError):
        CriticOutput.model_validate(payload)
    schema = CriticOutput.model_json_schema()
    assert all(branch["additionalProperties"] is False for branch in schema["anyOf"])


@pytest.mark.parametrize("field", ["predicted_effect", "correction"])
def test_complete_explanation_above_old_limit_is_preserved(field):
    text = ("The proposed action attempts to shift a timestamp by -7 days, but there is no prior "
            "evidence that the timestamp provided (1700000000) is relevant or valid in the current "
            "context. The call is ungrounded as it lacks a clear connection to the user's request "
            "to find the most recent message.")
    assert len(text.split()) == 50
    payload = dict(VALID[1], **{field: text})
    assert CriticOutput.model_validate(payload).model_dump()[field] == text


def test_word_limit_is_visible_in_decoding_schema():
    schema = CriticOutput.model_json_schema()
    for branch in schema["anyOf"]:
        for field in ("predicted_effect", "correction"):
            assert "128 whitespace-delimited words" in branch["properties"][field]["description"]
