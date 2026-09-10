"""Skill wire records, frozen JSON predicates, and host-owned statistics."""
import re
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from .base import JsonValue
from .memory import Count, FrozenRecord, Identifier, Rate, Status, TOLERANCE, unique
from .tool_metadata import PredicateOp


class FrozenDict(dict):
    def _deny(self, *args, **kwargs):
        raise TypeError("immutable JSON object")
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _deny


class FrozenList(list):
    def _deny(self, *args, **kwargs):
        raise TypeError("immutable JSON array")
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _deny


def freeze_json(value):
    if isinstance(value, dict):
        return FrozenDict({k: freeze_json(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_json(v) for v in value)
    return value


class SkillStatePredicate(FrozenRecord):
    path: str
    op: Literal["exists", "not_exists", "eq", "neq", "in", "contains"]
    value: JsonValue | None = None

    @model_validator(mode="before")
    @classmethod
    def presence(cls, data):
        if isinstance(data, dict):
            exists = data.get("op") in ("exists", "not_exists")
            if exists == ("value" in data):
                raise ValueError("predicate value presence mismatch")
        return data

    @model_validator(mode="after")
    def predicate(self):
        from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes
        canonical_json_bytes(self.value)
        if (self.path and not self.path.startswith("/")) or re.search(r"~(?![01])", self.path):
            raise ValueError("invalid RFC 6901 pointer")
        if self.op == "in" and not isinstance(self.value, list):
            raise ValueError("in requires an array")
        object.__setattr__(self, "value", freeze_json(self.value))
        return self

    @model_serializer(mode="wrap")
    def wire(self, handler):
        data = handler(self)
        if self.op in (PredicateOp.EXISTS, PredicateOp.NOT_EXISTS):
            data.pop("value", None)
        return data


def unique_records(values):
    from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
    unique(tuple(canonical_sha256(v.model_dump(mode="json")) for v in values))


class SkillApplicability(FrozenRecord):
    required_state: tuple[SkillStatePredicate, ...]
    forbidden_state: tuple[SkillStatePredicate, ...]
    best_used_when: tuple[Identifier, ...]

    @model_validator(mode="after")
    def invariants(self):
        unique_records(self.required_state)
        unique_records(self.forbidden_state)
        unique(self.best_used_when)
        return self


Level = Literal["low", "medium", "high"]


class SkillCostProfile(FrozenRecord):
    expected_tool_calls: Count
    latency: Level
    token_cost: Level


class SkillRiskProfile(FrozenRecord):
    risk_if_skipped: Level
    risk_if_wrong: Level


class SkillOnlineStatistics(FrozenRecord):
    evaluated_uses: Count
    successes: Count
    failures: Count
    success_rate: Rate
    last_update_attempt_at_use_count: Count

    @model_validator(mode="after")
    def invariants(self):
        if self.evaluated_uses != self.successes + self.failures:
            raise ValueError("inconsistent use counts")
        expected = self.successes / self.evaluated_uses if self.evaluated_uses else 0.0
        if abs(self.success_rate - expected) > TOLERANCE:
            raise ValueError("inconsistent success rate")
        if self.last_update_attempt_at_use_count > self.evaluated_uses:
            raise ValueError("update count exceeds uses")
        return self


class SkillFailureMode(FrozenRecord):
    mode_id: Identifier
    task_condition: Annotated[str, Field(min_length=1, max_length=240)]
    failure_mode: Annotated[str, Field(min_length=1, max_length=240)]
    support_count: Count
    last_observed_seq: Count


class SkillValidation(FrozenRecord):
    scenario_count: Annotated[int, Field(ge=1, le=20)]
    previous_full_success_count: Count
    candidate_full_success_count: Count
    previous_similarity_sum: Annotated[float, Field(ge=0)]
    candidate_similarity_sum: Annotated[float, Field(ge=0)]
    previous_minefield_hit_count: Count
    candidate_minefield_hit_count: Count

    @model_validator(mode="after")
    def invariants(self):
        for branch in ("previous", "candidate"):
            successes = getattr(self, branch + "_full_success_count")
            similarity = getattr(self, branch + "_similarity_sum")
            minefields = getattr(self, branch + "_minefield_hit_count")
            if not successes <= similarity <= self.scenario_count - minefields:
                raise ValueError("impossible validation totals")
        return self


class SkillRecord(FrozenRecord):
    skill_id: Identifier
    name: Identifier
    description: Identifier
    applicability: SkillApplicability
    required_inputs: tuple[SkillStatePredicate, ...]
    expected_outputs: tuple[Identifier, ...]
    tool_dependencies: tuple[Identifier, ...]
    success_criteria: tuple[Identifier, ...]
    failure_mode_buffer: Annotated[tuple[SkillFailureMode, ...], Field(max_length=5)]
    cost_profile: SkillCostProfile
    risk_profile: SkillRiskProfile
    instruction: Identifier
    online_statistics: SkillOnlineStatistics
    validation: SkillValidation | None
    version: Annotated[str, Field(pattern=r"^v1\.(0|[1-9][0-9]*)$")]
    status: Status

    @model_validator(mode="after")
    def invariants(self):
        unique_records(self.required_inputs)
        unique(self.expected_outputs)
        unique(self.success_criteria)
        unique(self.tool_dependencies, ordered=True)
        unique(tuple(v.mode_id for v in self.failure_mode_buffer))
        return self


def validate_skill_inventory(skill: SkillRecord, inventory: tuple[str, ...]):
    if not set(skill.tool_dependencies) <= set(inventory):
        raise ValueError("unknown Skill dependency")
    prose = (skill.name, skill.description, skill.instruction,
             *skill.applicability.best_used_when, *skill.expected_outputs,
             *skill.success_criteria,
             *(v.task_condition for v in skill.failure_mode_buffer),
             *(v.failure_mode for v in skill.failure_mode_buffer))
    for text in prose:
        if any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text) for name in inventory):
            raise ValueError("canonical tool identifier in Skill prose")
