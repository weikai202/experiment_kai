"""One-dispatch frozen-Qwen roles for Task 016."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import model_validator

from toolsandbox_pipeline.providers.contracts import ProviderRole, RequestContext
from toolsandbox_pipeline.providers.qwen import QwenGateway
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.offline_skill import (
    FailureModeAdd,
    FailureModeDecision,
    FailureModeMerge,
    FailureModeSkip,
    SkillContentCandidate,
)


class FailureModeDecisionOutput(StrictModel):
    decision: str
    task_condition: str | None = None
    failure_mode: str | None = None
    mode_id: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def exact_variant(self) -> "FailureModeDecisionOutput":
        provided = self.model_fields_set
        expected = {
            "ADD": {"decision", "task_condition", "failure_mode"},
            "MERGE": {"decision", "mode_id"},
            "SKIP": {"decision", "reason"},
        }.get(self.decision)
        if expected is None or provided != expected:
            raise ValueError("failure-mode conditional fields mismatch")
        self.as_decision()
        return self

    def as_decision(self) -> FailureModeDecision:
        if self.decision == "ADD":
            return FailureModeAdd(
                decision="ADD",
                task_condition=self.task_condition,
                failure_mode=self.failure_mode,
            )
        if self.decision == "MERGE":
            return FailureModeMerge(decision="MERGE", mode_id=self.mode_id)
        return FailureModeSkip(decision="SKIP", reason=self.reason)


@dataclass(frozen=True)
class PreparedOfflineSkillRequest:
    role: ProviderRole
    unit_id: str
    messages: list[dict[str, str]]
    output_model: type[StrictModel]
    max_tokens: int
    input_fingerprint: str


class OfflineSkillRoleRunner:
    def __init__(self, gateway: QwenGateway, *, role: ProviderRole, max_tokens: int):
        if role not in (ProviderRole.FAILURE_MODE_UPDATE, ProviderRole.SKILL_CANDIDATE):
            raise ValueError("Task 016 Qwen role required")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("positive role token limit required")
        self.gateway = gateway
        self.role = role
        self.max_tokens = max_tokens

    def prepare(self, *, unit_id: str, messages: list[dict[str, str]]) -> PreparedOfflineSkillRequest:
        output_model = (
            FailureModeDecisionOutput
            if self.role is ProviderRole.FAILURE_MODE_UPDATE
            else SkillContentCandidate
        )
        fingerprint = canonical_sha256(
            {
                "role": self.role.value,
                "model": "Qwen/Qwen3-32B",
                "temperature": 0.0,
                "seed": 0,
                "top_p": "omitted",
                "enable_thinking": False,
                "messages": messages,
                "output_schema": output_model.model_json_schema(),
                "max_tokens": self.max_tokens,
            }
        )
        return PreparedOfflineSkillRequest(
            role=self.role,
            unit_id=unit_id,
            messages=messages,
            output_model=output_model,
            max_tokens=self.max_tokens,
            input_fingerprint=fingerprint,
        )

    def run(self, prepared: PreparedOfflineSkillRequest, context: RequestContext):
        if prepared.role is not self.role or context.role is not self.role:
            raise ValueError("offline Skill role mismatch")
        if context.unit_reference != prepared.unit_id:
            raise ValueError("offline Skill unit mismatch")
        if context.input_fingerprint != prepared.input_fingerprint:
            raise ValueError("offline Skill input fingerprint mismatch")
        return self.gateway.generate(
            context,
            prepared.messages,
            prepared.output_model,
            max_tokens=prepared.max_tokens,
        )


__all__ = [
    "FailureModeDecisionOutput",
    "OfflineSkillRoleRunner",
    "PreparedOfflineSkillRequest",
]
