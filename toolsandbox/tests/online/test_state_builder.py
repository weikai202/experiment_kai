from copy import deepcopy
import json

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from toolsandbox_pipeline.online.state_builder import StateBuilder
from toolsandbox_pipeline.reproducibility import verify_state_id
from toolsandbox_pipeline.schemas.state import StateBuildInput


class ResultContract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    value: int
    nested: dict[str, list[str]]


def payload() -> dict:
    return {
        "episode_id": "episode-α",
        "scenario_id": "scenario",
        "scenario_family_id": "family",
        "visible_messages": [
            {"source_message_index": 0, "sender": "SYSTEM", "recipient": "AGENT", "content": "\u89c4\u5219"},
            {"source_message_index": 1, "sender": "USER", "recipient": "AGENT", "content": "\u67e5\u4e00\u4e0b"},
            {
                "source_message_index": 2,
                "sender": "AGENT",
                "recipient": "EXECUTION_ENVIRONMENT",
                "content": "call one",
                "openai_tool_call_id": "c1",
                "openai_function_name": "visible/tool",
            },
            {
                "source_message_index": 3,
                "sender": "AGENT",
                "recipient": "EXECUTION_ENVIRONMENT",
                "content": "call parallel",
                "openai_tool_call_id": "c-parallel",
                "openai_function_name": "other",
            },
            {
                "source_message_index": 4,
                "sender": "EXECUTION_ENVIRONMENT",
                "recipient": "AGENT",
                "content": "{'value': 3, 'nested': {'a/b~c': ['\u96ea', 'x']}}",
                "openai_tool_call_id": "c1",
                "openai_function_name": "visible/tool",
            },
        ],
        "committed_tool_outcomes": [
            {
                "call_id": "c1",
                "agent_facing_tool_name": "visible/tool",
                "arguments": {"q": "\u67e5\u4e00\u4e0b"},
                "result_source_message_index": 4,
                "public_return_contract_id": "result-v1",
                "canonical_tool_name": "private.canonical",
                "tool_mapping_manifest_hash": "sha256:private-map",
            }
        ],
        "available_tools": [
            {"name": "visible/tool", "schema": {"description": "exact", "parameters": {"z": 1}}},
            {"name": "other", "schema": {"description": "second"}},
        ],
    }


def builder() -> StateBuilder:
    return StateBuilder({"result-v1": TypeAdapter(ResultContract)})


def parsed(data: dict) -> StateBuildInput:
    return StateBuildInput.model_validate_json(json.dumps(data, ensure_ascii=False))


def test_deterministic_reduction_turn_count_facts_and_ids() -> None:
    first = builder().build(parsed(payload()))
    second = builder().build(parsed(deepcopy(payload())))
    assert first == second
    assert first.state.agent_turn_index == 1
    assert first.state.current_observation.content.startswith("{'value'")
    assert verify_state_id(first.state.model_dump(mode="json", by_alias=True))
    assert len(first.state.verified_facts) == 3
    assert [event.fact.result_pointer for event in first.fact_events] == [
        "/nested/a~1b~0c/0",
        "/nested/a~1b~0c/1",
        "/value",
    ]
    assert first.state.available_tools[0].schema_["description"] == "exact"


def test_visible_change_or_source_index_changes_identity_without_mutating_input() -> None:
    source = payload()
    before = deepcopy(source)
    baseline = builder().build(parsed(source)).state
    assert source == before
    changed = payload()
    changed["visible_messages"][1]["content"] += "!"
    assert builder().build(parsed(changed)).state.state_id != baseline.state_id
    shifted = payload()
    for message in shifted["visible_messages"]:
        message["source_message_index"] += 10
    shifted["committed_tool_outcomes"][0]["result_source_message_index"] += 10
    result = builder().build(parsed(shifted)).state
    assert result.state_id != baseline.state_id
    assert result.visible_messages[0].message_id != baseline.visible_messages[0].message_id


def test_later_result_replaces_active_slot_but_both_events_remain() -> None:
    data = payload()
    data["visible_messages"].extend(
        [
            {"source_message_index": 5, "sender": "AGENT", "recipient": "EXECUTION_ENVIRONMENT", "content": "again"},
            {
                "source_message_index": 6,
                "sender": "EXECUTION_ENVIRONMENT",
                "recipient": "AGENT",
                "content": "{'value': 9, 'nested': {'a/b~c': ['new', 'x']}}",
                "openai_tool_call_id": "c2",
                "openai_function_name": "visible/tool",
            },
        ]
    )
    second = deepcopy(data["committed_tool_outcomes"][0])
    second.update(call_id="c2", result_source_message_index=6)
    data["committed_tool_outcomes"].append(second)
    result = builder().build(parsed(data))
    assert len(result.state.verified_facts) == 3
    assert len(result.fact_events) == 6
    assert sorted(fact.value for fact in result.state.verified_facts.values() if isinstance(fact.value, int)) == [9]


@pytest.mark.parametrize(
    ("content", "exception", "structured", "failed"),
    [
        ("not a literal", None, False, False),
        ("{'value': '3', 'nested': {'a/b~c': []}}", None, False, False),
        ("visible exception", "ValueError", None, True),
        ("EXTERNAL_FIXTURE_MISS", None, None, True),
    ],
)
def test_parse_contract_and_visible_failure_paths(content, exception, structured, failed) -> None:
    data = payload()
    message = data["visible_messages"][4]
    message["content"] = content
    if exception is not None:
        message["tool_call_exception"] = exception
    result = builder().build(parsed(data))
    assert bool(result.state.failed_actions) is failed
    if failed:
        assert not result.state.completed_tool_calls
        assert not result.state.verified_facts
    else:
        assert result.state.completed_tool_calls[0].result_structured is structured
        if not structured:
            assert result.state.completed_tool_calls[0].result == content


@pytest.mark.parametrize("field", ["result_source_message_index", "call_id", "agent_facing_tool_name"])
def test_outcome_binding_mismatches_fail(field: str) -> None:
    data = payload()
    data["committed_tool_outcomes"][0][field] = 99 if field == "result_source_message_index" else "wrong"
    with pytest.raises(ValueError):
        builder().build(parsed(data))


def test_missing_contract_and_missing_observation_fail() -> None:
    data = payload()
    data["committed_tool_outcomes"][0]["public_return_contract_id"] = "missing"
    with pytest.raises(ValueError):
        builder().build(parsed(data))
    data = payload()
    data["visible_messages"] = [
        {"source_message_index": 0, "sender": "AGENT", "recipient": "USER", "content": "x"}
    ]
    data["committed_tool_outcomes"] = []
    with pytest.raises(ValueError, match="addressed to AGENT"):
        builder().build(parsed(data))


def test_duplicate_or_unsorted_indices_fail() -> None:
    data = payload()
    data["visible_messages"][1]["source_message_index"] = 0
    with pytest.raises(ValueError, match="strictly increasing"):
        builder().build(parsed(data))


def test_scalar_and_container_contracts_and_nonfinite_rejection() -> None:
    for literal, expected_count in (("None", 1), ("True", 1), ("4", 1), ("[1, {'x': 2}]", 2), ("{}", 0)):
        data = payload()
        data["visible_messages"][4]["content"] = literal
        result = StateBuilder({"result-v1": TypeAdapter(object)}).build(parsed(data))
        # object validation succeeds, but canonical JSON validation decides persistability.
        assert len(result.fact_events) == expected_count
    data = payload()
    data["visible_messages"][4]["content"] = "float('nan')"
    result = StateBuilder({"result-v1": TypeAdapter(object)}).build(parsed(data))
    assert result.state.completed_tool_calls[0].result_structured is False
