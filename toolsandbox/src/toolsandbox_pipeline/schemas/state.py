"""Strict contracts for compact Agent-visible state and private provenance."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field

from .base import JsonObject, JsonValue, StrictModel


class _FrozenStrictModel(StrictModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)


class VisibleRole(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    AGENT = "AGENT"
    EXECUTION_ENVIRONMENT = "EXECUTION_ENVIRONMENT"


class VisibleMessageInput(_FrozenStrictModel):
    source_message_index: int = Field(ge=0)
    sender: VisibleRole
    recipient: VisibleRole
    content: str
    openai_tool_call_id: str | None = None
    openai_function_name: str | None = None
    tool_call_exception: str | None = None


class CommittedToolOutcomeInput(_FrozenStrictModel):
    call_id: str = Field(min_length=1)
    agent_facing_tool_name: str = Field(min_length=1)
    arguments: JsonObject
    result_source_message_index: int = Field(ge=0)
    public_return_contract_id: str = Field(min_length=1)
    canonical_tool_name: str = Field(min_length=1)
    tool_mapping_manifest_hash: str = Field(min_length=1)


class AgentFacingToolInput(_FrozenStrictModel):
    name: str = Field(min_length=1)
    schema_: JsonObject = Field(alias="schema", serialization_alias="schema")


class PendingDependencyInput(_FrozenStrictModel):
    call_id: str = Field(min_length=1)
    agent_facing_tool_name: str = Field(min_length=1)
    prerequisite_code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    metadata_record_hash: str = Field(min_length=1)
    canonical_tool_name: str = Field(min_length=1)
    tool_mapping_manifest_hash: str = Field(min_length=1)


class StateBuildInput(_FrozenStrictModel):
    episode_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_family_id: str = Field(min_length=1)
    visible_messages: tuple[VisibleMessageInput, ...]
    committed_tool_outcomes: tuple[CommittedToolOutcomeInput, ...] = ()
    available_tools: tuple[AgentFacingToolInput, ...] = ()
    pending_dependencies: tuple[PendingDependencyInput, ...] = ()


class VisibleMessage(_FrozenStrictModel):
    message_id: str = Field(min_length=1)
    sender: VisibleRole
    recipient: VisibleRole
    content: str


class CurrentObservation(_FrozenStrictModel):
    message_id: str = Field(min_length=1)
    content: str


class VerifiedFact(_FrozenStrictModel):
    value: JsonValue
    source_message_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    result_pointer: str
    agent_facing_tool_name: str = Field(min_length=1)


class CompletedToolCall(_FrozenStrictModel):
    call_id: str = Field(min_length=1)
    agent_facing_tool_name: str = Field(min_length=1)
    arguments: JsonObject
    source_message_id: str = Field(min_length=1)
    result: JsonValue | str
    result_structured: bool


class FailedAction(_FrozenStrictModel):
    call_id: str = Field(min_length=1)
    agent_facing_tool_name: str = Field(min_length=1)
    source_message_id: str = Field(min_length=1)
    error: str


class PendingDependency(_FrozenStrictModel):
    call_id: str = Field(min_length=1)
    agent_facing_tool_name: str = Field(min_length=1)
    prerequisite_code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    metadata_record_hash: str = Field(min_length=1)


class AgentFacingTool(_FrozenStrictModel):
    name: str = Field(min_length=1)
    schema_: JsonObject = Field(alias="schema", serialization_alias="schema")


class CompactVerifiedState(_FrozenStrictModel):
    episode_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_family_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    agent_turn_index: int = Field(ge=0)
    visible_messages: tuple[VisibleMessage, ...]
    current_observation: CurrentObservation
    verified_facts: dict[str, VerifiedFact]
    completed_tool_calls: tuple[CompletedToolCall, ...]
    failed_actions: tuple[FailedAction, ...]
    pending_dependencies: tuple[PendingDependency, ...]
    available_tools: tuple[AgentFacingTool, ...]
    conversation_status: Literal["agent_turn"]


class ControllerFactProvenance(_FrozenStrictModel):
    fact_pointer: str
    call_id: str = Field(min_length=1)
    canonical_tool_name: str = Field(min_length=1)
    tool_mapping_manifest_hash: str = Field(min_length=1)


class ControllerDependencyProvenance(_FrozenStrictModel):
    metadata_record_hash: str = Field(min_length=1)
    canonical_tool_name: str = Field(min_length=1)
    tool_mapping_manifest_hash: str = Field(min_length=1)


class ControllerProvenanceSidecar(_FrozenStrictModel):
    state_id: str = Field(min_length=1)
    fact_provenance: tuple[ControllerFactProvenance, ...]
    dependency_provenance: tuple[ControllerDependencyProvenance, ...]


class FactEvent(_FrozenStrictModel):
    fact_pointer: str
    fact: VerifiedFact


class StateBuildResult(_FrozenStrictModel):
    state: CompactVerifiedState
    provenance_sidecar: ControllerProvenanceSidecar
    fact_events: tuple[FactEvent, ...]


__all__ = [
    "AgentFacingTool",
    "AgentFacingToolInput",
    "CommittedToolOutcomeInput",
    "CompactVerifiedState",
    "CompletedToolCall",
    "ControllerFactProvenance",
    "ControllerProvenanceSidecar",
    "CurrentObservation",
    "FailedAction",
    "PendingDependency",
    "PendingDependencyInput",
    "StateBuildInput",
    "StateBuildResult",
    "VerifiedFact",
    "VisibleMessage",
    "VisibleMessageInput",
    "VisibleRole",
]
