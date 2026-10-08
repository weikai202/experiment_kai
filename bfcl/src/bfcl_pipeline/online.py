from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .bfcl_adapter import LiteralCall, parse_literal_calls
from .canonical import sha256_json
from .resources import RetrievalReceipt


@dataclass(frozen=True)
class VisibleContext:
    messages: tuple[dict[str, Any], ...]
    functions: tuple[dict[str, Any], ...]
    policy_memory: tuple[str, ...]
    world_memory: tuple[str, ...]
    skills: tuple[str, ...]
    generation_id: str = "g000"
    selected_memory_ids: tuple[str, ...] = ()
    selected_skill_versions: tuple[str, ...] = ()
    case_id: str = "case"
    turn_index: int = 0
    selected_skill_bindings: tuple[tuple[str, str, str], ...] = ()
    selected_skill_receipts: tuple[RetrievalReceipt, ...] = ()


@dataclass(frozen=True)
class CriticDecision:
    decision: str
    outcome_class: str
    rationale: str

    def __post_init__(self) -> None:
        if self.decision not in {"KEEP", "REVISE"}:
            raise ValueError("Critic decision must be KEEP or REVISE")
        if not self.outcome_class or not self.rationale:
            raise ValueError("Critic must return a non-empty bounded schema")


class PolicyRole(Protocol):
    def draft(self, context: VisibleContext) -> str: ...


class CriticRole(Protocol):
    def review(self, context: VisibleContext, draft: str, calls: tuple[LiteralCall, ...]) -> CriticDecision: ...


class RevisionRole(Protocol):
    def revise(self, context: VisibleContext, draft: str, critique: CriticDecision) -> str: ...


@dataclass(frozen=True)
class OnlineTurnResult:
    draft: str
    final_text: str
    final_calls: tuple[LiteralCall, ...]
    decision: str
    used_revision: bool
    committed_call_ids: tuple[str, ...]
    selected_memory_ids: tuple[str, ...]
    selected_skill_versions: tuple[str, ...]
    decision_lineage_sha256: str
    case_id: str
    turn_index: int
    generation_id: str
    selected_skill_bindings: tuple[tuple[str, str, str], ...]
    selected_skill_receipts: tuple[RetrievalReceipt, ...]


def _function_names(functions: tuple[dict[str, Any], ...]) -> set[str]:
    names: set[str] = set()
    for item in functions:
        name = item.get("name") or item.get("function", {}).get("name")
        if not isinstance(name, str):
            raise ValueError("BFCL function schema has no name")
        names.add(name)
    return names


def _matches_type(value: Any, expected: str) -> bool:
    return {"string": isinstance(value, str), "integer": type(value) is int, "number": type(value) in (int, float), "boolean": type(value) is bool, "array": isinstance(value, list), "object": isinstance(value, dict), "null": value is None}.get(expected, True)


def _validate_offered(calls: tuple[LiteralCall, ...], functions: tuple[dict[str, Any], ...], *, case_id: str, turn_index: int, generation_id: str, action_sha256: str) -> tuple[str, ...]:
    offered = _function_names(functions)
    unknown = [call.name for call in calls if call.name not in offered]
    if unknown:
        raise ValueError(f"Calls unavailable BFCL functions: {unknown}")
    schemas = {(item.get("name") or item.get("function", {}).get("name")): item.get("parameters", item.get("function", {}).get("parameters", {})) for item in functions}
    call_ids = []
    for index, call in enumerate(calls):
        schema = schemas[call.name] or {}
        properties = schema.get("properties", {})
        property_names = tuple(properties)
        if len(call.args) > len(property_names):
            raise ValueError("Too many positional arguments for BFCL schema")
        assigned = set(property_names[:len(call.args)]) | {key for key, _ in call.kwargs}
        if len(assigned) != len(call.args) + len(call.kwargs):
            raise ValueError("BFCL argument is assigned more than once")
        required = set(schema.get("required", ()))
        if not required <= assigned:
            raise ValueError("BFCL call is missing required arguments")
        values = dict(zip(property_names, call.args)) | dict(call.kwargs)
        if any(name not in properties for name in values):
            raise ValueError("BFCL call contains an unknown argument")
        for name, value in values.items():
            expected = properties.get(name, {}).get("type")
            if expected and not _matches_type(value, expected):
                raise ValueError(f"BFCL argument {name} does not match type {expected}")
        call_ids.append(sha256_json({"case_id": case_id, "turn_index": turn_index, "generation_id": generation_id, "action_sha256": action_sha256, "call_index": index, "name": call.name, "args": call.args, "kwargs": call.kwargs}))
    return tuple(call_ids)


def run_policy_controller_critic_turn(context: VisibleContext, policy: PolicyRole, critic: CriticRole, revision: RevisionRole) -> OnlineTurnResult:
    """BFCL adaptation of Policy -> Controller -> Critic -> optional Revision.

    Only conversation, currently exposed functions, generation-scoped memories,
    and Skills are visible. Ground truth, evaluator details, and future delayed
    functions never enter this context.
    """
    for receipt in context.selected_skill_receipts:
        receipt.validate()
        if receipt.kind != "skill" or receipt.generation_id != context.generation_id:
            raise ValueError("Selected Skill receipt has the wrong kind or generation")
    skill_bindings = tuple(
        (receipt.resource_id, receipt.resource_sha256, receipt.creation_effect_id)
        for receipt in context.selected_skill_receipts
    )
    if tuple(x[0] for x in skill_bindings) != context.selected_skill_versions:
        raise ValueError("Selected Skill versions do not match host retrieval receipts")
    if context.selected_skill_bindings != skill_bindings:
        raise ValueError("Selected Skill bindings do not match host retrieval receipts")
    draft = policy.draft(context)
    calls = parse_literal_calls(draft)
    call_ids = _validate_offered(calls, context.functions, case_id=context.case_id, turn_index=context.turn_index, generation_id=context.generation_id, action_sha256=sha256_json(draft))
    critique = critic.review(context, draft, calls)
    if critique.decision == "KEEP":
        core = {"case_id": context.case_id, "turn_index": context.turn_index, "generation_id": context.generation_id, "decision": "KEEP", "final_text_sha256": sha256_json(draft), "call_ids": call_ids, "memory_ids": context.selected_memory_ids, "skill_bindings": skill_bindings}
        return OnlineTurnResult(draft, draft, calls, "KEEP", False, call_ids, context.selected_memory_ids, context.selected_skill_versions, sha256_json(core), context.case_id, context.turn_index, context.generation_id, skill_bindings, context.selected_skill_receipts)
    final = revision.revise(context, draft, critique)
    final_calls = parse_literal_calls(final)
    final_call_ids = _validate_offered(final_calls, context.functions, case_id=context.case_id, turn_index=context.turn_index, generation_id=context.generation_id, action_sha256=sha256_json(final))
    if final == draft:
        core = {"case_id": context.case_id, "turn_index": context.turn_index, "generation_id": context.generation_id, "decision": "KEEP", "final_text_sha256": sha256_json(draft), "call_ids": call_ids, "memory_ids": context.selected_memory_ids, "skill_bindings": skill_bindings}
        return OnlineTurnResult(draft, draft, calls, "KEEP", False, call_ids, context.selected_memory_ids, context.selected_skill_versions, sha256_json(core), context.case_id, context.turn_index, context.generation_id, skill_bindings, context.selected_skill_receipts)
    core = {"case_id": context.case_id, "turn_index": context.turn_index, "generation_id": context.generation_id, "decision": "REVISE", "final_text_sha256": sha256_json(final), "call_ids": final_call_ids, "memory_ids": context.selected_memory_ids, "skill_bindings": skill_bindings}
    return OnlineTurnResult(draft, final, final_calls, "REVISE", True, final_call_ids, context.selected_memory_ids, context.selected_skill_versions, sha256_json(core), context.case_id, context.turn_index, context.generation_id, skill_bindings, context.selected_skill_receipts)
