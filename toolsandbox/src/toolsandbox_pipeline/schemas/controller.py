"""Deterministic Controller decision contracts."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator, model_validator

from .base import StrictModel


class BlockingCode(str, Enum):
    INVALID_FUNCTION = "INVALID_FUNCTION"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    INVALID_ARGUMENT_TYPE = "INVALID_ARGUMENT_TYPE"
    INVALID_ARGUMENT_VALUE = "INVALID_ARGUMENT_VALUE"
    UNGROUNDED_ARGUMENT = "UNGROUNDED_ARGUMENT"
    MISSING_REQUIRED_ARGUMENT = "MISSING_REQUIRED_ARGUMENT"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    DEPENDENT_PARALLEL_CALLS = "DEPENDENT_PARALLEL_CALLS"
    UNAUTHORIZED_EXTERNAL_SIDE_EFFECT = "UNAUTHORIZED_EXTERNAL_SIDE_EFFECT"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    REPEATED_FAILED_ACTION = "REPEATED_FAILED_ACTION"
    AGENT_FORBIDDEN_TOOL = "AGENT_FORBIDDEN_TOOL"


class CriticTriggerCode(str, Enum):
    CRITIC_REQUIRED_TOOL = "CRITIC_REQUIRED_TOOL"
    MEDIUM_OR_HIGH_RISK = "MEDIUM_OR_HIGH_RISK"
    PARALLEL_BATCH_REVIEW = "PARALLEL_BATCH_REVIEW"
    ASSISTANT_MESSAGE_REVIEW = "ASSISTANT_MESSAGE_REVIEW"
    STRUCTURED_CONSTRAINT_TENSION = "STRUCTURED_CONSTRAINT_TENSION"
    EXTERNAL_READ_REVIEW = "EXTERNAL_READ_REVIEW"


class ControllerSourceKind(str, Enum):
    STATE = "state"
    SCHEMA = "schema"
    SKILL = "skill"
    TOOL_METADATA = "tool_metadata"
    ACTION_HISTORY = "action_history"


class ControllerEvidence(StrictModel):
    code: BlockingCode | CriticTriggerCode
    source_kind: ControllerSourceKind
    source_ref: str = Field(min_length=1)

    @field_validator("code", mode="before")
    @classmethod
    def parse_code(cls, value: object) -> BlockingCode | CriticTriggerCode:
        if isinstance(value, (BlockingCode, CriticTriggerCode)):
            return value
        if not isinstance(value, str):
            raise ValueError("code must be a string")
        try:
            return BlockingCode(value)
        except ValueError:
            return CriticTriggerCode(value)

    @field_validator("source_kind", mode="before")
    @classmethod
    def parse_source_kind(cls, value: object) -> ControllerSourceKind:
        if isinstance(value, ControllerSourceKind):
            return value
        if not isinstance(value, str):
            raise ValueError("source_kind must be a string")
        return ControllerSourceKind(value)


class ControllerDecision(StrictModel):
    blocking_codes: list[BlockingCode]
    critic_trigger_codes: list[CriticTriggerCode]
    evidence: list[ControllerEvidence]

    @field_validator("blocking_codes", mode="before")
    @classmethod
    def parse_blocking_codes(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [item if isinstance(item, BlockingCode) else BlockingCode(item) if isinstance(item, str) else item for item in value]

    @field_validator("critic_trigger_codes", mode="before")
    @classmethod
    def parse_trigger_codes(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [item if isinstance(item, CriticTriggerCode) else CriticTriggerCode(item) if isinstance(item, str) else item for item in value]

    @model_validator(mode="after")
    def validate_code_evidence(self) -> "ControllerDecision":
        blocking_values = [code.value for code in self.blocking_codes]
        trigger_values = [code.value for code in self.critic_trigger_codes]
        if len(blocking_values) != len(set(blocking_values)):
            raise ValueError("blocking_codes must not contain duplicates")
        if len(trigger_values) != len(set(trigger_values)):
            raise ValueError("critic_trigger_codes must not contain duplicates")

        selected = set(blocking_values) | set(trigger_values)
        if set(blocking_values) & set(trigger_values):
            raise ValueError("a code cannot be both blocking and a critic trigger")

        evidence_values = [item.code.value for item in self.evidence]
        missing = selected - set(evidence_values)
        if missing:
            raise ValueError("every selected code must have matching evidence")
        if set(evidence_values) - selected:
            raise ValueError("evidence cannot cite an unselected code")
        return self
