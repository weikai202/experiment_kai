"""Immutable memory records; rate comparisons use absolute tolerance 1e-9."""
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from .base import StrictModel
from .critic import CriticErrorCode

Text = Annotated[str, Field(min_length=1, max_length=512)]
Identifier = Annotated[str, Field(min_length=1)]
GenerationId = Annotated[str, Field(pattern=r"^g[0-9]{3}$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Count = Annotated[int, Field(ge=0)]
Rate = Annotated[float, Field(ge=0, le=1)]
Status = Literal["active", "deprecated"]
TOLERANCE = 1e-9


class FrozenRecord(StrictModel):
    model_config = ConfigDict(frozen=True, revalidate_instances="always")


def unique(values, *, ordered=False):
    if len(values) != len(set(values)):
        raise ValueError("duplicate items")
    if ordered and tuple(values) != tuple(sorted(values, key=lambda v: v.encode("utf-8"))):
        raise ValueError("items must be in UTF-8 order")


class PolicyMemory(FrozenRecord):
    memory_id: Annotated[str, Field(pattern=r"^pm_[0-9a-f]{64}$")]
    scope: Text
    applicability: Annotated[tuple[Text, ...], Field(max_length=5)]
    action_guidance: Text
    avoid: Annotated[tuple[Text, ...], Field(max_length=5)]
    evidence_trajectory_ids: tuple[Identifier, ...]
    support_count: Count
    success_rate: Rate
    confidence: Rate
    created_version: GenerationId
    status: Status

    @model_validator(mode="after")
    def invariants(self):
        unique(self.applicability)
        unique(self.avoid)
        validate_statistics(self, self.success_rate)
        return self


def validate_statistics(record, rate):
    unique(record.evidence_trajectory_ids, ordered=True)
    product = rate * record.support_count
    if abs(product - round(product)) > TOLERANCE:
        raise ValueError("rate must represent an integral count")
    if abs(record.confidence - record.support_count / (record.support_count + 2)) > TOLERANCE:
        raise ValueError("confidence disagrees with support")


class WorldMemory(FrozenRecord):
    memory_id: Annotated[str, Field(pattern=r"^wm_[0-9a-f]{64}$")]
    action_pattern: Text
    state_conditions: Annotated[tuple[Text, ...], Field(max_length=5)]
    schema_conditions: Annotated[tuple[Text, ...], Field(max_length=5)]
    likely_error_codes: Annotated[tuple[CriticErrorCode, ...], Field(max_length=5)]
    outcome_calibration: Text
    correction_principle: Text
    evidence_trajectory_ids: tuple[Identifier, ...]
    support_count: Count
    empirical_failure_rate: Rate
    confidence: Rate
    created_version: GenerationId
    status: Status

    @model_validator(mode="after")
    def invariants(self):
        for values in (self.state_conditions, self.schema_conditions, self.likely_error_codes):
            unique(values)
        validate_statistics(self, self.empirical_failure_rate)
        return self
