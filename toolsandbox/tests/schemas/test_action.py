import math

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.schemas import ActionEnvelope


VALID_ACTIONS = [
    {
        "action": {
            "type": "function_call",
            "call_id": " c1 ",
            "selected_skill_id": None,
            "name": "lookup",
            "arguments": {"nested": [None, True, 2, 2.5, "\u503c"]},
        }
    },
    {
        "action": {
            "type": "parallel_batch",
            "calls": [
                {
                    "call_id": "c1",
                    "selected_skill_id": "skill-1",
                    "name": "first",
                    "arguments": {},
                },
                {
                    "call_id": "c2",
                    "selected_skill_id": None,
                    "name": "second",
                    "arguments": {"x": 1},
                },
            ],
        }
    },
    {"action": {"type": "assistant_message", "content": " answer "}},
]


@pytest.mark.parametrize("payload", VALID_ACTIONS)
def test_action_variants_round_trip_exactly(payload):
    assert ActionEnvelope.model_validate(payload).model_dump(mode="json") == payload


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action": {"type": "assistant_message", "content": "ok"}, "extra": 1},
        {"action": {"type": "assistant_message", "content": "", "extra": 1}},
        {"action": {"type": "assistant_message", "content": 1}},
        {
            "action": {
                "type": "function_call",
                "call_id": "",
                "selected_skill_id": None,
                "name": "x",
                "arguments": {},
            }
        },
        {
            "action": {
                "type": "function_call",
                "call_id": "c",
                "name": "x",
                "arguments": {},
            }
        },
        {
            "action": {
                "type": "function_call",
                "call_id": "c",
                "selected_skill_id": "",
                "name": "x",
                "arguments": {},
            }
        },
        {
            "action": {
                "type": "function_call",
                "call_id": "c",
                "selected_skill_id": None,
                "name": "x",
                "arguments": [],
            }
        },
        {"action": {"type": "parallel_batch", "calls": []}},
        {
            "action": {
                "type": "parallel_batch",
                "calls": [
                    {"call_id": "c", "selected_skill_id": None, "name": "x", "arguments": {}},
                    {"call_id": "c", "selected_skill_id": None, "name": "y", "arguments": {}},
                ],
            }
        },
    ],
)
def test_invalid_actions_fail_closed(payload):
    with pytest.raises(ValidationError):
        ActionEnvelope.model_validate(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_action_rejects_non_finite_arguments(value):
    payload = {
        "action": {
            "type": "function_call",
            "call_id": "c",
            "selected_skill_id": None,
            "name": "x",
            "arguments": {"value": value},
        }
    }
    with pytest.raises(ValidationError):
        ActionEnvelope.model_validate(payload)


def test_action_schema_has_discriminator_and_forbids_extra_properties():
    schema = ActionEnvelope.model_json_schema()
    action_schema = schema["properties"]["action"]
    assert action_schema["discriminator"]["propertyName"] == "type"
    assert len(action_schema["oneOf"]) == 3
    assert schema["additionalProperties"] is False
    for definition in schema["$defs"].values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False
