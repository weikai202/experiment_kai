"""Versioned, allowlisted semantic projections, never heuristic summaries."""
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.base import JsonValue, JsonObject
from toolsandbox_pipeline.schemas.memory import FrozenRecord, PolicyMemory, WorldMemory
from toolsandbox_pipeline.schemas.skill import SkillRecord, freeze_json
from toolsandbox_pipeline.schemas.state import CompactVerifiedState
from .contracts import RetrievalError

DOCUMENT_VERSION = "retrieval-document-v1"
STATE_VERSION = "retrieval-state-v1"
ACTION_VERSION = "retrieval-action-v1"


class RetrievalMessage(FrozenRecord):
    sender: Literal["SYSTEM", "USER", "AGENT", "EXECUTION_ENVIRONMENT"]
    recipient: Literal["SYSTEM", "USER", "AGENT", "EXECUTION_ENVIRONMENT"]
    content: str


class RetrievalFact(FrozenRecord):
    fact_pointer: str
    value: JsonValue
    agent_facing_tool_name: str
    result_pointer: str

    @field_validator("value")
    @classmethod
    def immutable(cls, value):
        canonical_json_bytes(value)
        return freeze_json(value)


class RetrievalCompletedCall(FrozenRecord):
    agent_facing_tool_name: str
    arguments: JsonObject
    result: JsonValue
    result_structured: bool

    @field_validator("arguments", "result")
    @classmethod
    def immutable(cls, value):
        canonical_json_bytes(value)
        return freeze_json(value)


class RetrievalFailure(FrozenRecord):
    agent_facing_tool_name: str
    error: str


class RetrievalDependency(FrozenRecord):
    agent_facing_tool_name: str
    prerequisite_code: str
    description: str


class RetrievalStateProjectionV1(FrozenRecord):
    query_version: Literal["retrieval-state-v1"] = STATE_VERSION
    visible_messages: tuple[RetrievalMessage, ...]
    current_observation: str
    verified_facts: tuple[RetrievalFact, ...]
    completed_tool_calls: tuple[RetrievalCompletedCall, ...]
    failed_actions: tuple[RetrievalFailure, ...]
    pending_dependencies: tuple[RetrievalDependency, ...]
    available_tool_names: tuple[str, ...]
    conversation_status: Literal["agent_turn"]


class RetrievalCall(FrozenRecord):
    name: Annotated[str, Field(min_length=1)]
    arguments: JsonObject

    @field_validator("arguments")
    @classmethod
    def immutable(cls, value):
        canonical_json_bytes(value)
        return freeze_json(value)


class RetrievalActionProjectionV1(FrozenRecord):
    type: Literal["function_call", "parallel_batch", "assistant_message"]
    content: str | None = None
    name: str | None = None
    arguments: JsonObject | None = None
    calls: tuple[RetrievalCall, ...] | None = None

    @field_validator("arguments")
    @classmethod
    def immutable(cls, value):
        return freeze_json(value)

    @model_validator(mode="after")
    def action_shape(self):
        fields = {"function_call": {"type", "name", "arguments"}, "parallel_batch": {"type", "calls"}, "assistant_message": {"type", "content"}}[self.type]
        if self.model_fields_set != fields:
            raise ValueError("action projection field mismatch")
        if self.type == "function_call" and (not self.name or self.arguments is None):
            raise ValueError("name and arguments required")
        if self.type == "parallel_batch" and not self.calls:
            raise ValueError("non-empty batch required")
        if self.type == "assistant_message" and not self.content:
            raise ValueError("non-empty assistant content required")
        return self


def state_projection(state: CompactVerifiedState):
    if type(state) is not CompactVerifiedState:
        raise TypeError("only CompactVerifiedState may be projected")
    data = state.model_dump(mode="json", by_alias=True)
    def fields(items, names):
        return [{name: item[name] for name in names} for item in items]
    return RetrievalStateProjectionV1.model_validate_json(canonical_json_bytes(dict(
        visible_messages=fields(data["visible_messages"], ("sender", "recipient", "content")),
        current_observation=state.current_observation.content,
        verified_facts=[{"fact_pointer": "/verified_facts/" + key.replace("~", "~0").replace("/", "~1"),
                         **{name: value[name] for name in ("value", "agent_facing_tool_name", "result_pointer")}}
                        for key, value in sorted(data["verified_facts"].items())],
        completed_tool_calls=fields(data["completed_tool_calls"], ("agent_facing_tool_name", "arguments", "result", "result_structured")),
        failed_actions=fields(data["failed_actions"], ("agent_facing_tool_name", "error")),
        pending_dependencies=fields(data["pending_dependencies"], ("agent_facing_tool_name", "prerequisite_code", "description")),
        available_tool_names=[tool.name for tool in state.available_tools],
        conversation_status=state.conversation_status,
    )))


def action_projection(action: ActionEnvelope):
    if type(action) is not ActionEnvelope:
        raise TypeError("only ActionEnvelope may be projected")
    action = ActionEnvelope.model_validate_json(action.model_dump_json())
    import json
    data = json.loads(action.model_dump_json())["action"]
    data.pop("call_id", None)
    data.pop("selected_skill_id", None)
    for call in data.get("calls", []):
        call.pop("call_id")
        call.pop("selected_skill_id")
    return RetrievalActionProjectionV1.model_validate_json(canonical_json_bytes(data))


def validate_input(text):
    from .long_inputs import validate_retrieval_input
    try:
        return validate_retrieval_input(text)
    except ValueError as error:
        raise RetrievalError(str(error)) from error


def policy_query(state):
    return validate_input(canonical_json_bytes(state_projection(state).model_dump(mode="json")).decode("utf-8"))


def world_query(state, action):
    return validate_input(canonical_json_bytes({
        "state": state_projection(state).model_dump(mode="json"),
        "action": action_projection(action).model_dump(mode="json", exclude_unset=True),
    }).decode("utf-8"))


def document(record):
    if type(record) is PolicyMemory:
        keys = ("scope", "applicability", "action_guidance", "avoid")
    elif type(record) is WorldMemory:
        keys = ("action_pattern", "state_conditions", "schema_conditions", "likely_error_codes", "outcome_calibration", "correction_principle")
    elif type(record) is SkillRecord:
        keys = ("name", "description", "applicability", "required_inputs", "expected_outputs", "success_criteria", "cost_profile", "risk_profile", "instruction")
    else:
        raise TypeError("validated generation record required")
    data = record.model_dump(mode="json")
    return validate_input(canonical_json_bytes({key: data[key] for key in keys}).decode("utf-8"))


def input_hash(text):
    from hashlib import sha256
    return "sha256:" + sha256(validate_input(text).encode("utf-8")).hexdigest()
