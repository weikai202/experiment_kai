"""Pure deterministic routing checks for validated Policy actions."""

from __future__ import annotations

from collections.abc import Iterable

from jsonschema import Draft202012Validator

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.action import AssistantMessageAction, FunctionCall, FunctionCallAction, ParallelBatchAction
from toolsandbox_pipeline.schemas.controller import BlockingCode, ControllerDecision, ControllerEvidence, ControllerSourceKind, CriticTriggerCode
from toolsandbox_pipeline.schemas.tool_metadata import ControllerToolMetadata, MetadataPredicate, PredicateOp, ToolEffect, ToolRisk

from .controller_inputs import ControllerInput, RetrievedSkillControllerView, SkillStatus
from .grounding import ground_arguments, resolve_json_pointer, strict_json_equal
from .resources import calls_are_independent


def action_fingerprint(canonical_tool_name: str, arguments: dict) -> str:
    return canonical_sha256({"arguments": arguments, "canonical_tool_name": canonical_tool_name})


def assistant_fingerprint(content: str) -> str:
    return canonical_sha256({"content": content})


def evaluate_predicate(document: object, predicate: MetadataPredicate) -> bool:
    found, selected = resolve_json_pointer(document, predicate.path)
    if predicate.op is PredicateOp.EXISTS:
        return found
    if predicate.op is PredicateOp.NOT_EXISTS:
        return not found
    if not found:
        return False
    if predicate.op is PredicateOp.EQ:
        return strict_json_equal(selected, predicate.value)
    if predicate.op is PredicateOp.NEQ:
        return not strict_json_equal(selected, predicate.value)
    if predicate.op is PredicateOp.IN:
        return any(strict_json_equal(selected, item) for item in predicate.value)  # type: ignore[union-attr]
    if predicate.op is PredicateOp.CONTAINS:
        return isinstance(selected, list) and any(strict_json_equal(item, predicate.value) for item in selected)
    return False


def _schema_codes(arguments: dict, parameters: dict) -> tuple[BlockingCode, ...]:
    codes: set[BlockingCode] = set()
    validator = Draft202012Validator(parameters)
    for error in validator.iter_errors(arguments):
        if error.validator == "additionalProperties":
            codes.add(BlockingCode.INVALID_PARAMETER)
        elif error.validator == "required":
            codes.add(BlockingCode.MISSING_REQUIRED_ARGUMENT)
        elif error.validator == "type":
            codes.add(BlockingCode.INVALID_ARGUMENT_TYPE)
        else:
            codes.add(BlockingCode.INVALID_ARGUMENT_VALUE)
    return tuple(code for code in BlockingCode if code in codes)


class _DecisionBuilder:
    def __init__(self) -> None:
        self.blocking: dict[BlockingCode, ControllerEvidence] = {}
        self.triggers: dict[CriticTriggerCode, ControllerEvidence] = {}

    def block(self, code: BlockingCode, kind: ControllerSourceKind, ref: str) -> None:
        self.blocking.setdefault(code, ControllerEvidence(code=code, source_kind=kind, source_ref=ref))

    def trigger(self, code: CriticTriggerCode, kind: ControllerSourceKind, ref: str) -> None:
        self.triggers.setdefault(code, ControllerEvidence(code=code, source_kind=kind, source_ref=ref))

    def build(self) -> ControllerDecision:
        blocking = [code for code in BlockingCode if code in self.blocking]
        triggers = [code for code in CriticTriggerCode if code in self.triggers]
        evidence = [self.blocking[code] for code in blocking] + [self.triggers[code] for code in triggers]
        return ControllerDecision(blocking_codes=blocking, critic_trigger_codes=triggers, evidence=evidence)


class Controller:
    """Stateless Controller; it never repairs, replaces, or executes an action."""

    def __call__(self, controller_input: ControllerInput) -> ControllerDecision:
        return self.decide(controller_input)

    def decide(self, controller_input: ControllerInput) -> ControllerDecision:
        out = _DecisionBuilder()
        sidecar_hashes = {item.tool_mapping_manifest_hash for item in controller_input.provenance_sidecar.fact_provenance + controller_input.provenance_sidecar.dependency_provenance}
        if controller_input.provenance_sidecar.state_id != controller_input.state.state_id or any(value != controller_input.mapping_manifest_hash for value in sidecar_hashes):
            out.block(BlockingCode.CONSTRAINT_VIOLATION, ControllerSourceKind.STATE, "controller_sidecar:identity")
        action = controller_input.action.action
        if isinstance(action, AssistantMessageAction):
            out.trigger(CriticTriggerCode.ASSISTANT_MESSAGE_REVIEW, ControllerSourceKind.STATE, "action:/content")
            fp = assistant_fingerprint(action.content)
            if any(item.visible and item.failed and item.action_fingerprint == fp for item in controller_input.action_history):
                out.block(BlockingCode.REPEATED_FAILED_ACTION, ControllerSourceKind.ACTION_HISTORY, f"fingerprint:{fp}")
            if controller_input.structured_constraint_tension:
                out.trigger(CriticTriggerCode.STRUCTURED_CONSTRAINT_TENSION, ControllerSourceKind.STATE, "structured_constraint_tension")
            return out.build()

        calls: tuple[FunctionCall, ...]
        if isinstance(action, FunctionCallAction):
            calls = (FunctionCall.model_validate(action.model_dump(exclude={"type"}), strict=True),)
        else:
            assert isinstance(action, ParallelBatchAction)
            calls = tuple(action.calls)
            out.trigger(CriticTriggerCode.PARALLEL_BATCH_REVIEW, ControllerSourceKind.TOOL_METADATA, "action:/calls")

        metadata_by_name = {item.canonical_tool_name: item for item in controller_input.tool_metadata}
        schemas = {tool.name: tool.schema_ for tool in controller_input.state.available_tools}
        resolved_metadata: list[ControllerToolMetadata | None] = []
        state_payload = controller_input.state.model_dump(mode="json", by_alias=True)

        for index, call in enumerate(calls):
            ref = f"action:/calls/{index}" if len(calls) > 1 else "action:/"
            canonical = controller_input.agent_to_canonical_name.get(call.name)
            metadata = metadata_by_name.get(canonical) if canonical is not None else None
            schema_wrapper = schemas.get(call.name)
            if call.name == "end_conversation" or canonical == "end_conversation" or (metadata is not None and metadata.agent_forbidden):
                out.block(BlockingCode.AGENT_FORBIDDEN_TOOL, ControllerSourceKind.TOOL_METADATA, ref + "/name")
            if canonical is None or metadata is None or schema_wrapper is None:
                if not (call.name == "end_conversation" or canonical == "end_conversation"):
                    out.block(BlockingCode.INVALID_FUNCTION, ControllerSourceKind.SCHEMA, ref + "/name")
                resolved_metadata.append(metadata)
                self._history_check(out, controller_input, canonical, call, ref)
                continue
            resolved_metadata.append(metadata)
            parameters = self._parameters(schema_wrapper)
            if parameters is None:
                out.block(BlockingCode.INVALID_FUNCTION, ControllerSourceKind.SCHEMA, ref + "/name")
                self._history_check(out, controller_input, canonical, call, ref)
                continue
            for code in _schema_codes(call.arguments, parameters):
                out.block(code, ControllerSourceKind.SCHEMA, ref + "/arguments")
            grounding = ground_arguments(call.arguments, parameters, controller_input.state)
            if grounding.ungrounded_pointers:
                out.block(BlockingCode.UNGROUNDED_ARGUMENT, ControllerSourceKind.STATE, ref + "/arguments" + grounding.ungrounded_pointers[0])
            for predicate in metadata.prerequisites:
                if not evaluate_predicate(state_payload, predicate):
                    out.block(BlockingCode.MISSING_DEPENDENCY, ControllerSourceKind.TOOL_METADATA, f"metadata:{canonical}:{predicate.code}")
            self._skill_checks(out, controller_input, call, canonical, state_payload, ref)
            if metadata.effect is ToolEffect.EXTERNAL_WRITE:
                out.block(BlockingCode.UNAUTHORIZED_EXTERNAL_SIDE_EFFECT, ControllerSourceKind.TOOL_METADATA, f"metadata:{canonical}:effect")
            if metadata.critic_required:
                out.trigger(CriticTriggerCode.CRITIC_REQUIRED_TOOL, ControllerSourceKind.TOOL_METADATA, f"metadata:{canonical}:critic_required")
            if metadata.risk in (ToolRisk.MEDIUM, ToolRisk.HIGH):
                out.trigger(CriticTriggerCode.MEDIUM_OR_HIGH_RISK, ControllerSourceKind.TOOL_METADATA, f"metadata:{canonical}:risk")
            if metadata.effect is ToolEffect.EXTERNAL_READ:
                out.trigger(CriticTriggerCode.EXTERNAL_READ_REVIEW, ControllerSourceKind.TOOL_METADATA, f"metadata:{canonical}:effect")
            self._history_check(out, controller_input, canonical, call, ref)

        if len(calls) > 1:
            if any(item is None for item in resolved_metadata) or not calls_are_independent(calls, tuple(item for item in resolved_metadata if item is not None)):
                out.block(BlockingCode.DEPENDENT_PARALLEL_CALLS, ControllerSourceKind.TOOL_METADATA, "action:/calls")
        if controller_input.structured_constraint_tension:
            out.trigger(CriticTriggerCode.STRUCTURED_CONSTRAINT_TENSION, ControllerSourceKind.STATE, "structured_constraint_tension")
        return out.build()

    @staticmethod
    def _parameters(schema_wrapper: dict) -> dict | None:
        function = schema_wrapper.get("function")
        parameters = function.get("parameters") if isinstance(function, dict) else None
        return parameters if isinstance(parameters, dict) else None

    @staticmethod
    def _history_check(out: _DecisionBuilder, data: ControllerInput, canonical: str | None, call: FunctionCall, ref: str) -> None:
        if canonical is None:
            return
        fp = action_fingerprint(canonical, call.arguments)
        if any(item.visible and item.failed and item.action_fingerprint == fp for item in data.action_history):
            out.block(BlockingCode.REPEATED_FAILED_ACTION, ControllerSourceKind.ACTION_HISTORY, f"fingerprint:{fp}")

    @staticmethod
    def _skill_checks(out: _DecisionBuilder, data: ControllerInput, call: FunctionCall, canonical: str, state_payload: dict, ref: str) -> None:
        if call.selected_skill_id is None:
            return
        skill = next((item for item in data.retrieved_skills if item.skill_id == call.selected_skill_id), None)
        if skill is None:
            out.block(BlockingCode.CONSTRAINT_VIOLATION, ControllerSourceKind.SKILL, ref + "/selected_skill_id")
            return
        valid = skill.status is SkillStatus.ACTIVE and skill.generation_id == data.generation_id and canonical in skill.tool_dependencies
        valid = valid and all(evaluate_predicate(state_payload, p) for p in skill.applicability_required + skill.required_inputs)
        valid = valid and not any(evaluate_predicate(state_payload, p) for p in skill.applicability_forbidden)
        if not valid:
            out.block(BlockingCode.CONSTRAINT_VIOLATION, ControllerSourceKind.SKILL, f"skill:{skill.skill_id}@{skill.version}")


__all__ = ["Controller", "action_fingerprint", "assistant_fingerprint", "evaluate_predicate"]
