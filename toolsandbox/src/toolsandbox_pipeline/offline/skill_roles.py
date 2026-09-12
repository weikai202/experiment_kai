"""One-dispatch frozen-Qwen roles for Task 016."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import model_validator

from toolsandbox_pipeline.providers.contracts import ProviderRole, RequestContext
from toolsandbox_pipeline.providers.qwen import QwenGateway
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig
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
    token_limit_config: RoleTokenLimitConfig | None = None


class OfflineSkillRoleRunner:
    def __init__(self, gateway: QwenGateway, *, role: ProviderRole, max_tokens: int,
                 token_limit_config: RoleTokenLimitConfig | None = None):
        if role not in (ProviderRole.FAILURE_MODE_UPDATE, ProviderRole.SKILL_CANDIDATE):
            raise ValueError("Task 016 Qwen role required")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("positive role token limit required")
        if token_limit_config is not None:
            if type(token_limit_config) is not RoleTokenLimitConfig:
                raise TypeError("validated role token-limit selection required")
            token_limit_config = RoleTokenLimitConfig.model_validate_json(token_limit_config.model_dump_json())
            if token_limit_config.role != role.value or token_limit_config.max_tokens != max_tokens:
                raise ValueError("Skill role token-limit selection mismatch")
        self.token_limit_config = token_limit_config
        self.gateway = gateway
        self.role = role
        self.max_tokens = max_tokens

    def prepare(self, *, unit_id: str, messages: list[dict[str, str]]) -> PreparedOfflineSkillRequest:
        output_model = (
            FailureModeDecisionOutput
            if self.role is ProviderRole.FAILURE_MODE_UPDATE
            else SkillContentCandidate
        )
        fingerprint_payload = {
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
        if self.token_limit_config is not None:
            fingerprint_payload["token_limit_config"] = self.token_limit_config.model_dump(mode="json")
            fingerprint_payload["qwen_config"] = self.gateway.config.model_dump(mode="json")
        fingerprint = canonical_sha256(fingerprint_payload)
        return PreparedOfflineSkillRequest(
            role=self.role,
            unit_id=unit_id,
            messages=messages,
            output_model=output_model,
            max_tokens=self.max_tokens,
            input_fingerprint=fingerprint,
            token_limit_config=self.token_limit_config,
        )

    def run(self, prepared: PreparedOfflineSkillRequest, context: RequestContext):
        expected = self.prepare(unit_id=prepared.unit_id, messages=prepared.messages)
        if prepared != expected:
            raise ValueError("offline Skill prepared request or token-limit selection mismatch")
        if prepared.role is not self.role or context.role is not self.role:
            raise ValueError("offline Skill role mismatch")
        if context.unit_reference != prepared.unit_id:
            raise ValueError("offline Skill unit mismatch")
        if context.input_fingerprint != prepared.input_fingerprint:
            raise ValueError("offline Skill input fingerprint mismatch")
        selection = {} if self.token_limit_config is None else {"token_limit_config": self.token_limit_config}
        return self.gateway.generate(
            context,
            prepared.messages,
            prepared.output_model,
            max_tokens=prepared.max_tokens, **selection,
        )


__all__ = [
    "FailureModeDecisionOutput",
    "OfflineSkillRoleRunner",
    "PreparedOfflineSkillRequest",
]
