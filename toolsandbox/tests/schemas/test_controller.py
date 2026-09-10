import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.schemas import (
    BlockingCode,
    ControllerDecision,
    ControllerSourceKind,
    CriticTriggerCode,
)


def evidence(code, source_kind="state"):
    return {"code": code, "source_kind": source_kind, "source_ref": "opaque:1"}


def test_all_enum_values_and_input_order_round_trip():
    blocking = [code.value for code in BlockingCode]
    triggers = [code.value for code in CriticTriggerCode]
    items = [
        evidence(code, list(ControllerSourceKind)[index % len(ControllerSourceKind)].value)
        for index, code in enumerate(blocking + triggers)
    ]
    model = ControllerDecision.model_validate(
        {"blocking_codes": blocking, "critic_trigger_codes": triggers, "evidence": items}
    )
    assert [code.value for code in model.blocking_codes] == blocking
    assert [code.value for code in model.critic_trigger_codes] == triggers
    assert {kind.value for kind in ControllerSourceKind} == {
        "state", "schema", "skill", "tool_metadata", "action_history"
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"blocking_codes": ["INVALID_FUNCTION"], "critic_trigger_codes": [], "evidence": []},
        {"blocking_codes": [], "critic_trigger_codes": [], "evidence": [evidence("INVALID_FUNCTION")]},
        {
            "blocking_codes": ["INVALID_FUNCTION", "INVALID_FUNCTION"],
            "critic_trigger_codes": [],
            "evidence": [evidence("INVALID_FUNCTION")],
        },
        {
            "blocking_codes": [],
            "critic_trigger_codes": ["EXTERNAL_READ_REVIEW", "EXTERNAL_READ_REVIEW"],
            "evidence": [evidence("EXTERNAL_READ_REVIEW")],
        },
        {
            "blocking_codes": ["INVALID_FUNCTION"],
            "critic_trigger_codes": [],
            "evidence": [{**evidence("INVALID_FUNCTION"), "source_ref": ""}],
        },
    ],
)
def test_invalid_controller_decisions(payload):
    with pytest.raises(ValidationError):
        ControllerDecision.model_validate(payload)


def test_cross_list_collision_guard_is_explicit():
    trigger = CriticTriggerCode.CRITIC_REQUIRED_TOOL
    original_value = trigger._value_
    try:
        trigger._value_ = BlockingCode.INVALID_FUNCTION.value
        with pytest.raises(ValidationError, match="both blocking"):
            ControllerDecision.model_validate(
                {
                    "blocking_codes": [BlockingCode.INVALID_FUNCTION],
                    "critic_trigger_codes": [trigger],
                    "evidence": [{
                        "code": BlockingCode.INVALID_FUNCTION,
                        "source_kind": ControllerSourceKind.STATE,
                        "source_ref": "opaque:1",
                    }],
                }
            )
    finally:
        trigger._value_ = original_value


def test_controller_is_strict_and_schema_forbids_extra_properties():
    with pytest.raises(ValidationError):
        ControllerDecision.model_validate(
            {"blocking_codes": [], "critic_trigger_codes": [], "evidence": [], "extra": False}
        )
    schema = ControllerDecision.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ControllerEvidence"]["additionalProperties"] is False
