"""Critic output contract."""

from __future__ import annotations

from enum import Enum

from pydantic import field_validator, model_validator

from .base import StrictModel


class CriticVerdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    UNCERTAIN = "uncertain"


class PredictedOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"


class CriticErrorCode(str, Enum):
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
    PREMATURE_ACTION = "PREMATURE_ACTION"
    UNNECESSARY_RISK = "UNNECESSARY_RISK"
    NO_RELEVANT_TOOL = "NO_RELEVANT_TOOL"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    LIKELY_MINEFIELD_BEHAVIOR = "LIKELY_MINEFIELD_BEHAVIOR"


class CriticOutput(StrictModel):
    verdict: CriticVerdict
    predicted_outcome: PredictedOutcome
    predicted_effect: str
    error_codes: list[CriticErrorCode]
    correction: str

    @field_validator("verdict", mode="before")
    @classmethod
    def parse_verdict(cls, value: object) -> CriticVerdict | object:
        if isinstance(value, CriticVerdict):
            return value
        return CriticVerdict(value) if isinstance(value, str) else value

    @field_validator("predicted_outcome", mode="before")
    @classmethod
    def parse_predicted_outcome(cls, value: object) -> PredictedOutcome | object:
        if isinstance(value, PredictedOutcome):
            return value
        return PredictedOutcome(value) if isinstance(value, str) else value

    @field_validator("error_codes", mode="before")
    @classmethod
    def parse_error_codes(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [item if isinstance(item, CriticErrorCode) else CriticErrorCode(item) if isinstance(item, str) else item for item in value]

    @field_validator("predicted_effect", "correction")
    @classmethod
    def validate_word_limit(cls, value: str) -> str:
        if len(value.split()) > 40:
            raise ValueError("text must contain at most 40 whitespace-delimited words")
        return value

    @model_validator(mode="after")
    def validate_verdict_fields(self) -> "CriticOutput":
        if not self.predicted_effect:
            raise ValueError("predicted_effect must be non-empty")
        if len(self.error_codes) != len(set(self.error_codes)):
            raise ValueError("error_codes must not contain duplicates")

        if self.verdict is CriticVerdict.ACCEPT:
            if self.error_codes or self.correction != "":
                raise ValueError("accept requires no errors and an empty correction")
        elif self.verdict is CriticVerdict.REVISE:
            if not self.error_codes or not self.correction:
                raise ValueError("revise requires errors and a non-empty correction")
        else:
            if self.predicted_outcome is not PredictedOutcome.UNCERTAIN:
                raise ValueError("uncertain verdict requires uncertain predicted_outcome")
            if not self.error_codes or not self.correction:
                raise ValueError("uncertain requires errors and a non-empty correction")
        return self
