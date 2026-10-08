"""Single-revision Policy-Controller-Critic turn pipeline with complete attempt evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .accounting import UsageAttempt
from .canonical import canonical_sha256
from .ledger import LogicalRequest, OutputApplication, PhysicalAttempt, SubstantiveEffect
from .model_boundary import (
    QWEN_MODEL,
    OutputLimitCalibration,
    qwen_request,
    validate_qwen_request,
)
from .native_adapter import AssistantAction, TauToolView, validate_action


class QwenTransport(Protocol):
    def complete(self, request: dict[str, Any], *, logical_request_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RetrievedContext:
    policy_memory_records: tuple[str, ...]
    world_memory_records: tuple[str, ...]
    skill_records: tuple[str, ...]
    selected_policy_memory_ids: tuple[str, ...]
    selected_world_memory_ids: tuple[str, ...]
    selected_policy_memory_versions: tuple[tuple[str, int], ...]
    selected_world_memory_versions: tuple[tuple[str, int], ...]
    selected_skill_versions: tuple[tuple[str, int], ...]
    retrieval_sha256: str


@dataclass(frozen=True)
class OnlineAttemptEvidence:
    logical_request_id: str
    attempt_id: str
    role: str
    request_sha256: str
    response_sha256: str
    model_id: str
    config_sha256: str
    completed: bool


@dataclass(frozen=True)
class OnlineTurnResult:
    action: AssistantAction
    used_revision: bool
    controller_code: str
    request_ids: tuple[str, ...]
    usage_attempts: tuple[UsageAttempt, ...]
    attempt_evidence: tuple[OnlineAttemptEvidence, ...]
    applied_request_ids: tuple[str, ...]


class _AttemptFailure(RuntimeError):
    def __init__(self, usage: UsageAttempt, evidence: OnlineAttemptEvidence):
        super().__init__("Qwen physical attempt failed before a complete response")
        self.usage = usage
        self.evidence = evidence


def _parse_action(value: Any) -> AssistantAction:
    if type(value) is not dict:
        raise ValueError("action must be an object")
    if value.get("type") == "message" and set(value) == {"type", "content"}:
        return AssistantAction(content=value["content"])
    if value.get("type") == "tool_calls" and set(value) == {"type", "calls"}:
        calls = value["calls"]
        if type(calls) is not list or not calls:
            raise ValueError("non-empty calls required")
        if any(type(call) is not dict or set(call) != {"name", "arguments"} for call in calls):
            raise ValueError("invalid call fields")
        return AssistantAction(
            tool_calls=tuple((call["name"], call["arguments"]) for call in calls)
        )
    raise ValueError("unknown action shape")


def _decode_content(content: Any) -> dict[str, Any]:
    if type(content) is str:
        content = json.loads(content)
    if type(content) is not dict:
        raise ValueError("Qwen role output must be one JSON object")
    return content


class TauOnlineTurnPipeline:
    def __init__(self, transport: QwenTransport, *, calibration: OutputLimitCalibration):
        calibration.validate()
        self.transport = transport
        self.calibration = calibration
        self.max_tokens = calibration.chosen_max_tokens

    def _call(self, role: str, payload: dict[str, Any], turn_id: str):
        logical_id = canonical_sha256({"turn_id": turn_id, "role": role})
        request = qwen_request(
            [
                {"role": "system", "content": f"tau3 evolution role: {role}"},
                {"role": "user", "content": json.dumps(payload, sort_keys=True)},
            ],
            max_tokens=self.max_tokens,
        )
        validate_qwen_request(request)
        request_sha = canonical_sha256(request)
        config_sha = canonical_sha256(
            {
                "model": request["model"],
                "temperature": request["temperature"],
                "seed": request["seed"],
                "max_tokens": request["max_tokens"],
                "chat_template_kwargs": request["chat_template_kwargs"],
            }
        )
        if config_sha != self.calibration.qwen_config_sha256:
            raise ValueError("online request does not match calibrated Qwen configuration")
        try:
            response = self.transport.complete(request, logical_request_id=logical_id)
        except Exception as error:
            attempt_id = canonical_sha256(
                {"logical_request_id": logical_id, "status": "unreported"}
            )
            usage = UsageAttempt(attempt_id, logical_id, "qwen", None, None)
            evidence = OnlineAttemptEvidence(
                logical_id,
                attempt_id,
                role,
                request_sha,
                canonical_sha256({"failure_type": type(error).__name__}),
                QWEN_MODEL,
                config_sha,
                False,
            )
            raise _AttemptFailure(usage, evidence) from error
        required = {"attempt_id", "logical_request_id", "input_tokens", "output_tokens", "content"}
        if set(response) != required or response["logical_request_id"] != logical_id:
            attempt_id = response.get("attempt_id", canonical_sha256(response))
            usage = UsageAttempt(attempt_id, logical_id, "qwen", None, None)
            evidence = OnlineAttemptEvidence(
                logical_id,
                attempt_id,
                role,
                request_sha,
                canonical_sha256(response),
                QWEN_MODEL,
                config_sha,
                False,
            )
            raise _AttemptFailure(usage, evidence)
        usage = UsageAttempt(
            response["attempt_id"],
            logical_id,
            "qwen",
            response["input_tokens"],
            response["output_tokens"],
        )
        evidence = OnlineAttemptEvidence(
            logical_id,
            response["attempt_id"],
            role,
            request_sha,
            canonical_sha256(response),
            QWEN_MODEL,
            config_sha,
            True,
        )
        return response["content"], usage, evidence

    def respond(
        self,
        *,
        turn_id: str,
        domain_policy: str,
        tools: tuple[TauToolView, ...],
        visible_history: tuple[dict[str, Any], ...],
        retrieved: RetrievedContext,
    ) -> OnlineTurnResult:
        if not turn_id:
            raise ValueError("turn ID required")
        context = {
            "domain_policy": domain_policy,
            "tools": [
                {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
                for tool in tools
            ],
            "visible_history": list(visible_history),
            "policy_memory_records": list(retrieved.policy_memory_records),
            "world_memory_records": list(retrieved.world_memory_records),
            "skill_records": list(retrieved.skill_records),
            "selected_policy_memory_ids": list(retrieved.selected_policy_memory_ids),
            "selected_world_memory_ids": list(retrieved.selected_world_memory_ids),
            "selected_policy_memory_versions": [
                list(value) for value in retrieved.selected_policy_memory_versions
            ],
            "selected_world_memory_versions": [
                list(value) for value in retrieved.selected_world_memory_versions
            ],
            "selected_skill_versions": [list(value) for value in retrieved.selected_skill_versions],
            "retrieval_sha256": retrieved.retrieval_sha256,
        }
        attempts: list[UsageAttempt] = []
        evidence: list[OnlineAttemptEvidence] = []

        def call(role: str, payload: dict[str, Any]):
            try:
                raw, usage, proof = self._call(role, payload, turn_id)
            except _AttemptFailure as error:
                attempts.append(error.usage)
                evidence.append(error.evidence)
                raise
            attempts.append(usage)
            evidence.append(proof)
            return _decode_content(raw)

        try:
            initial = _parse_action(call("policy", context))
            validate_action(initial, tools)
        except (_AttemptFailure, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return OnlineTurnResult(
                AssistantAction(
                    content="I could not produce a safe valid action. Please clarify your request."
                ),
                False,
                "invalid_initial_action_clarify",
                tuple(row.logical_request_id for row in evidence),
                tuple(attempts),
                tuple(evidence),
                (),
            )
        critic_payload = {**context, "initial_action": _action_json(initial)}
        try:
            critic = call("critic", critic_payload)
        except (_AttemptFailure, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return OnlineTurnResult(
                initial,
                False,
                "critic_incomplete_keep_initial",
                tuple(row.logical_request_id for row in evidence),
                tuple(attempts),
                tuple(evidence),
                tuple(row.logical_request_id for row in evidence[:1]),
            )
        if set(critic) != {"decision", "reason"} or critic["decision"] not in {"KEEP", "REVISE"}:
            return OnlineTurnResult(
                initial,
                False,
                "critic_invalid_keep_initial",
                tuple(row.logical_request_id for row in evidence),
                tuple(attempts),
                tuple(evidence),
                tuple(row.logical_request_id for row in evidence[:1]),
            )
        if critic["decision"] == "KEEP":
            return OnlineTurnResult(
                initial,
                False,
                "critic_keep",
                tuple(row.logical_request_id for row in evidence),
                tuple(attempts),
                tuple(evidence),
                tuple(row.logical_request_id for row in evidence[:2]),
            )
        try:
            revised = _parse_action(
                call("revision", {**critic_payload, "critic_reason": critic["reason"]})
            )
            validate_action(revised, tools)
        except (_AttemptFailure, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return OnlineTurnResult(
                initial,
                False,
                "revision_invalid_keep_initial",
                tuple(row.logical_request_id for row in evidence),
                tuple(attempts),
                tuple(evidence),
                tuple(row.logical_request_id for row in evidence[:2]),
            )
        return OnlineTurnResult(
            revised,
            True,
            "revision_accepted",
            tuple(row.logical_request_id for row in evidence),
            tuple(attempts),
            tuple(evidence),
            tuple(row.logical_request_id for row in evidence),
        )


def committed_action_ledger_events(
    result: OnlineTurnResult,
    *,
    scope_id: str,
    effect_id: str,
    artifact_sha256: str,
    committed: bool = True,
) -> tuple[object, ...]:
    """Bind every attempt and the exact applied Qwen chain to one native action effect."""
    if len(result.usage_attempts) != len(result.attempt_evidence):
        raise ValueError("online usage and evidence are not one-to-one")
    events: list[object] = []
    if committed:
        events.append(
            SubstantiveEffect(effect_id, scope_id, "executed_action", artifact_sha256, True, True)
        )
    for order, (usage, proof) in enumerate(
        zip(result.usage_attempts, result.attempt_evidence, strict=True)
    ):
        if (
            usage.attempt_id != proof.attempt_id
            or usage.logical_request_id != proof.logical_request_id
        ):
            raise ValueError("online usage/evidence identity mismatch")
        events.extend(
            (
                LogicalRequest(
                    proof.logical_request_id,
                    scope_id,
                    "qwen",
                    proof.role,
                    proof.request_sha256,
                    proof.model_id,
                    proof.config_sha256,
                ),
                PhysicalAttempt(
                    proof.attempt_id,
                    proof.logical_request_id,
                    usage.input_tokens,
                    usage.output_tokens,
                    proof.completed,
                    proof.response_sha256,
                ),
            )
        )
        applied = committed and proof.logical_request_id in result.applied_request_ids
        events.append(
            OutputApplication(
                canonical_sha256(
                    {"effect_id": effect_id, "attempt_id": proof.attempt_id, "order": order}
                ),
                proof.logical_request_id,
                proof.attempt_id,
                effect_id if applied else None,
                "executed_action_decision",
                applied,
                sum(
                    prior.logical_request_id in result.applied_request_ids
                    for prior in result.attempt_evidence[:order]
                ),
            )
        )
    return tuple(events)


def _action_json(action: AssistantAction) -> dict[str, Any]:
    if action.content is not None:
        return {"type": "message", "content": action.content}
    return {
        "type": "tool_calls",
        "calls": [{"name": name, "arguments": args} for name, args in action.tool_calls],
    }
