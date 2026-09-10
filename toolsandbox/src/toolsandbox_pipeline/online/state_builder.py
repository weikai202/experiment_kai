"""Pure deterministic reducer for compact Agent-visible state."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol

from toolsandbox_pipeline.reproducibility import canonical_sha256, compute_state_id
from toolsandbox_pipeline.schemas.base import JsonValue
from toolsandbox_pipeline.schemas.state import (
    AgentFacingTool,
    CommittedToolOutcomeInput,
    CompactVerifiedState,
    CompletedToolCall,
    ControllerDependencyProvenance,
    ControllerFactProvenance,
    ControllerProvenanceSidecar,
    CurrentObservation,
    FactEvent,
    FailedAction,
    PendingDependency,
    StateBuildInput,
    StateBuildResult,
    VerifiedFact,
    VisibleMessage,
    VisibleMessageInput,
    VisibleRole,
)


class PublicReturnValidator(Protocol):
    def validate_python(self, value: Any, *, strict: bool = ...) -> Any: ...
    def dump_python(self, value: Any, *, mode: str = ...) -> Any: ...


def _message_id(episode_id: str, message: VisibleMessageInput) -> str:
    payload = {
        "episode_id": episode_id,
        **message.model_dump(mode="json"),
    }
    return "message:" + canonical_sha256(payload).removeprefix("sha256:")


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _scalar_leaves(value: JsonValue, pointer: str = "") -> list[tuple[str, JsonValue]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, JsonValue]] = []
        for key in sorted(value):
            leaves.extend(_scalar_leaves(value[key], f"{pointer}/{_escape_pointer(key)}"))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_scalar_leaves(item, f"{pointer}/{index}"))
        return leaves
    return [(pointer, value)]


def _fact_slot(outcome: CommittedToolOutcomeInput, result_pointer: str) -> str:
    argument_hash = canonical_sha256(outcome.arguments)
    digest = canonical_sha256(
        {
            "agent_facing_tool_name": outcome.agent_facing_tool_name,
            "arguments_hash": argument_hash,
            "result_pointer": result_pointer,
        }
    )
    return "fact:" + digest.removeprefix("sha256:")


def serialize_agent_visible_state(state: CompactVerifiedState) -> bytes:
    """Serialize only a compact state for prompt/retrieval-facing consumers."""

    from toolsandbox_pipeline.reproducibility import canonical_json_bytes

    if not isinstance(state, CompactVerifiedState):
        raise TypeError("only CompactVerifiedState may be serialized for online consumers")
    return canonical_json_bytes(state.model_dump(mode="json", by_alias=True))


class StateBuilder:
    """Reduce sanitized visible events without inspecting upstream runtime state."""

    def __init__(self, public_return_contracts: Mapping[str, PublicReturnValidator]):
        self._contracts = dict(public_return_contracts)

    def build(self, build_input: StateBuildInput) -> StateBuildResult:
        if not isinstance(build_input, StateBuildInput):
            build_input = StateBuildInput.model_validate(build_input, strict=True)

        messages = list(build_input.visible_messages)
        indices = [message.source_message_index for message in messages]
        if any(current <= previous for previous, current in zip(indices, indices[1:])):
            raise ValueError("visible message indices must be unique and strictly increasing")

        message_ids = {message.source_message_index: _message_id(build_input.episode_id, message) for message in messages}
        visible_messages = tuple(
            VisibleMessage(
                message_id=message_ids[message.source_message_index],
                sender=message.sender,
                recipient=message.recipient,
                content=message.content,
            )
            for message in messages
        )

        observations = [message for message in messages if message.recipient == VisibleRole.AGENT]
        if not observations:
            raise ValueError("no visible message is addressed to AGENT")
        observation = observations[-1]
        observation_position = messages.index(observation)
        current_observation = CurrentObservation(
            message_id=message_ids[observation.source_message_index],
            content=observation.content,
        )

        completed_agent_runs = 0
        in_agent_run = False
        for message in messages[:observation_position]:
            if message.sender == VisibleRole.AGENT:
                if not in_agent_run:
                    completed_agent_runs += 1
                    in_agent_run = True
            else:
                in_agent_run = False

        by_index = {message.source_message_index: message for message in messages}
        completed_calls: list[CompletedToolCall] = []
        failed_actions: list[FailedAction] = []
        active_facts: dict[str, VerifiedFact] = {}
        fact_events: list[FactEvent] = []
        fact_provenance: list[ControllerFactProvenance] = []

        for outcome in build_input.committed_tool_outcomes:
            result_message = by_index.get(outcome.result_source_message_index)
            if result_message is None:
                raise ValueError("committed outcome result index is not visible")
            if result_message.sender != VisibleRole.EXECUTION_ENVIRONMENT or result_message.recipient != VisibleRole.AGENT:
                raise ValueError("committed outcome must bind EXECUTION_ENVIRONMENT -> AGENT")
            if result_message.openai_tool_call_id != outcome.call_id:
                raise ValueError("committed outcome call ID mismatch")
            if result_message.openai_function_name != outcome.agent_facing_tool_name:
                raise ValueError("committed outcome agent-facing tool name mismatch")
            contract = self._contracts.get(outcome.public_return_contract_id)
            if contract is None:
                raise ValueError(f"unknown public return contract: {outcome.public_return_contract_id}")

            source_message_id = message_ids[result_message.source_message_index]
            is_fixture_miss = (
                result_message.tool_call_exception == "EXTERNAL_FIXTURE_MISS"
                or result_message.content == "EXTERNAL_FIXTURE_MISS"
            )
            if result_message.tool_call_exception is not None or is_fixture_miss:
                failed_actions.append(
                    FailedAction(
                        call_id=outcome.call_id,
                        agent_facing_tool_name=outcome.agent_facing_tool_name,
                        source_message_id=source_message_id,
                        error=result_message.content,
                    )
                )
                continue

            structured = False
            parsed_result: JsonValue | str = result_message.content
            try:
                literal = ast.literal_eval(result_message.content)
                validated = contract.validate_python(literal, strict=True)
                dumped = contract.dump_python(validated, mode="json")
                # Canonical hashing is also the final recursive JSON/finite-value guard.
                canonical_sha256(dumped)
                parsed_result = deepcopy(dumped)
                structured = True
            except (ValueError, TypeError, SyntaxError, OverflowError):
                pass

            completed_calls.append(
                CompletedToolCall(
                    call_id=outcome.call_id,
                    agent_facing_tool_name=outcome.agent_facing_tool_name,
                    arguments=deepcopy(outcome.arguments),
                    source_message_id=source_message_id,
                    result=parsed_result,
                    result_structured=structured,
                )
            )
            if not structured:
                continue
            for result_pointer, value in _scalar_leaves(parsed_result):
                slot = _fact_slot(outcome, result_pointer)
                fact_pointer = f"/verified_facts/{_escape_pointer(slot)}"
                fact = VerifiedFact(
                    value=value,
                    source_message_id=source_message_id,
                    call_id=outcome.call_id,
                    result_pointer=result_pointer,
                    agent_facing_tool_name=outcome.agent_facing_tool_name,
                )
                active_facts[slot] = fact
                fact_events.append(FactEvent(fact_pointer=fact_pointer, fact=fact))
                fact_provenance.append(
                    ControllerFactProvenance(
                        fact_pointer=fact_pointer,
                        call_id=outcome.call_id,
                        canonical_tool_name=outcome.canonical_tool_name,
                        tool_mapping_manifest_hash=outcome.tool_mapping_manifest_hash,
                    )
                )

        pending_dependencies = tuple(
            PendingDependency(
                call_id=item.call_id,
                agent_facing_tool_name=item.agent_facing_tool_name,
                prerequisite_code=item.prerequisite_code,
                description=item.description,
                metadata_record_hash=item.metadata_record_hash,
            )
            for item in build_input.pending_dependencies
        )
        dependency_provenance = tuple(
            ControllerDependencyProvenance(
                metadata_record_hash=item.metadata_record_hash,
                canonical_tool_name=item.canonical_tool_name,
                tool_mapping_manifest_hash=item.tool_mapping_manifest_hash,
            )
            for item in build_input.pending_dependencies
        )
        available_tools = tuple(
            AgentFacingTool(name=item.name, schema=deepcopy(item.schema_))
            for item in build_input.available_tools
        )

        state_without_id = {
            "episode_id": build_input.episode_id,
            "scenario_id": build_input.scenario_id,
            "scenario_family_id": build_input.scenario_family_id,
            "agent_turn_index": completed_agent_runs,
            "visible_messages": [item.model_dump(mode="json", by_alias=True) for item in visible_messages],
            "current_observation": current_observation.model_dump(mode="json", by_alias=True),
            "verified_facts": {key: value.model_dump(mode="json", by_alias=True) for key, value in active_facts.items()},
            "completed_tool_calls": [item.model_dump(mode="json", by_alias=True) for item in completed_calls],
            "failed_actions": [item.model_dump(mode="json", by_alias=True) for item in failed_actions],
            "pending_dependencies": [item.model_dump(mode="json", by_alias=True) for item in pending_dependencies],
            "available_tools": [item.model_dump(mode="json", by_alias=True) for item in available_tools],
            "conversation_status": "agent_turn",
        }
        state_id = compute_state_id(state_without_id)
        state = CompactVerifiedState(
            episode_id=build_input.episode_id,
            scenario_id=build_input.scenario_id,
            scenario_family_id=build_input.scenario_family_id,
            state_id=state_id,
            agent_turn_index=completed_agent_runs,
            visible_messages=visible_messages,
            current_observation=current_observation,
            verified_facts=active_facts,
            completed_tool_calls=tuple(completed_calls),
            failed_actions=tuple(failed_actions),
            pending_dependencies=pending_dependencies,
            available_tools=available_tools,
            conversation_status="agent_turn",
        )
        sidecar = ControllerProvenanceSidecar(
            state_id=state_id,
            fact_provenance=tuple(fact_provenance),
            dependency_provenance=dependency_provenance,
        )
        return StateBuildResult(
            state=state,
            provenance_sidecar=sidecar,
            fact_events=tuple(fact_events),
        )


__all__ = ["StateBuilder", "serialize_agent_visible_state"]
