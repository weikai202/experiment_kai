import json

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.schemas.state import (
    AgentFacingToolInput,
    CompactVerifiedState,
    StateBuildInput,
    VisibleMessageInput,
)


def test_input_contracts_are_strict_and_forbid_visibility_escape_hatches() -> None:
    valid = {
        "source_message_index": 0,
        "sender": "USER",
        "recipient": "AGENT",
        "content": "  unchanged \n",
    }
    assert VisibleMessageInput.model_validate_json(json.dumps(valid)).content == "  unchanged \n"
    for forbidden in (
        "tool_trace",
        "conversation_active",
        "visible_to",
        "hidden_database",
        "evaluator",
        "milestone",
        "minefield",
        "target_data",
        "canonical_tool_name",
    ):
        with pytest.raises(ValidationError):
            VisibleMessageInput.model_validate_json(json.dumps({**valid, forbidden: "secret"}))
    with pytest.raises(ValidationError):
        VisibleMessageInput.model_validate({**valid, "source_message_index": "0"})


def test_augmented_schema_is_preserved_and_unknown_fields_fail() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "scrambled_name",
            "description": "",
            "parameters": {"type": "object", "x-extra": [1, True, None]},
        },
    }
    tool = AgentFacingToolInput(name="scrambled_name", schema=schema)
    assert tool.model_dump(mode="json", by_alias=True)["schema"] == schema
    with pytest.raises(ValidationError):
        AgentFacingToolInput.model_validate({"name": "x", "schema": {}, "canonical_name": "hidden"})


def test_public_state_schema_has_no_private_or_generic_metadata_fields() -> None:
    schema_text = str(CompactVerifiedState.model_json_schema())
    for forbidden in (
        "canonical_tool_name",
        "tool_mapping_manifest_hash",
        "sidecar",
        "hidden_database",
        "evaluator",
        "visible_to",
    ):
        assert forbidden not in schema_text
    assert CompactVerifiedState.model_json_schema()["additionalProperties"] is False


def test_state_build_input_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        StateBuildInput.model_validate_json(
            json.dumps({
                "episode_id": "e",
                "scenario_id": "s",
                "scenario_family_id": "f",
                "visible_messages": [],
                "database": {},
            })
        )
