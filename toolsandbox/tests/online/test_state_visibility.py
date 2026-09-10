from copy import deepcopy
import json

import pytest
from pydantic import TypeAdapter

from toolsandbox_pipeline.online.state_builder import StateBuilder, serialize_agent_visible_state
from toolsandbox_pipeline.schemas.state import StateBuildInput


def make_input(canonical_name: str, mapping_hash: str) -> StateBuildInput:
    return StateBuildInput.model_validate_json(
        json.dumps({
            "episode_id": "e",
            "scenario_id": "s",
            "scenario_family_id": "f",
            "visible_messages": [
                {"source_message_index": 0, "sender": "USER", "recipient": "AGENT", "content": "go"},
                {
                    "source_message_index": 1,
                    "sender": "EXECUTION_ENVIRONMENT",
                    "recipient": "AGENT",
                    "content": "7",
                    "openai_tool_call_id": "c",
                    "openai_function_name": "public",
                },
            ],
            "committed_tool_outcomes": [
                {
                    "call_id": "c",
                    "agent_facing_tool_name": "public",
                    "arguments": {},
                    "result_source_message_index": 1,
                    "public_return_contract_id": "integer",
                    "canonical_tool_name": canonical_name,
                    "tool_mapping_manifest_hash": mapping_hash,
                }
            ],
            "pending_dependencies": [
                {
                    "call_id": "next",
                    "agent_facing_tool_name": "public-next",
                    "prerequisite_code": "needs_value",
                    "description": "Need the prior value",
                    "metadata_record_hash": "sha256:opaque",
                    "canonical_tool_name": canonical_name,
                    "tool_mapping_manifest_hash": mapping_hash,
                }
            ],
        })
    )


def test_private_identity_is_sidecar_only_and_does_not_change_state_id() -> None:
    builder = StateBuilder({"integer": TypeAdapter(int)})
    first = builder.build(make_input("private.one", "sha256:one"))
    second = builder.build(make_input("private.two", "sha256:two"))
    assert first.state == second.state
    assert first.state.state_id == second.state.state_id
    state_bytes = serialize_agent_visible_state(first.state)
    assert b"private.one" not in state_bytes
    assert b"sha256:one" not in state_bytes
    sidecar = first.provenance_sidecar.model_dump_json()
    assert "private.one" in sidecar and "sha256:one" in sidecar
    assert first.provenance_sidecar.state_id == first.state.state_id


def test_online_serializer_rejects_result_and_sidecar() -> None:
    result = StateBuilder({"integer": TypeAdapter(int)}).build(make_input("private", "mapping"))
    with pytest.raises(TypeError):
        serialize_agent_visible_state(result)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        serialize_agent_visible_state(result.provenance_sidecar)  # type: ignore[arg-type]


def test_equivalent_fresh_inputs_and_caller_data_are_unchanged() -> None:
    raw = make_input("private", "mapping").model_dump(mode="python", by_alias=True)
    before = deepcopy(raw)
    builder = StateBuilder({"integer": TypeAdapter(int)})
    one = builder.build(StateBuildInput.model_validate(raw))
    two = builder.build(StateBuildInput.model_validate(deepcopy(raw)))
    assert one == two
    assert raw == before
