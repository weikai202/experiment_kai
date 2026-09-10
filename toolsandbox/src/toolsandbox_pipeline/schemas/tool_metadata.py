"""Strict contracts for deterministic Controller tool metadata."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field, RootModel, model_validator

from .base import JsonValue, StrictModel


class _FrozenModel(StrictModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)


class ToolEffect(str, Enum):
    SANDBOX_READ = "sandbox_read"
    SANDBOX_WRITE = "sandbox_write"
    EXTERNAL_READ = "external_read"
    EXTERNAL_WRITE = "external_write"
    CONVERSATION_CONTROL = "conversation_control"


class ToolRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredicateOp(str, Enum):
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    CONTAINS = "contains"


class MetadataPredicate(_FrozenModel):
    code: str = Field(min_length=1)
    path: str
    op: PredicateOp
    value: JsonValue | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_value_presence(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        op = data.get("op")
        has_value = "value" in data
        if op in ("exists", "not_exists", PredicateOp.EXISTS, PredicateOp.NOT_EXISTS):
            if has_value:
                raise ValueError("exists/not_exists predicates must omit value")
        elif op in ("eq", "neq", "in", "contains", PredicateOp.EQ, PredicateOp.NEQ, PredicateOp.IN, PredicateOp.CONTAINS):
            if not has_value:
                raise ValueError("comparison predicates require value")
        return data

    @model_validator(mode="after")
    def validate_pointer_and_operator(self) -> "MetadataPredicate":
        if self.path != "" and not self.path.startswith("/"):
            raise ValueError("predicate path must be an RFC 6901 pointer")
        if self.op is PredicateOp.IN and not isinstance(self.value, list):
            raise ValueError("in predicate value must be an array")
        return self


class ResourceTemplate(RootModel[str]):
    model_config = ConfigDict(strict=True, frozen=True)
    root: str = Field(min_length=1)

    @property
    def template(self) -> str:
        return self.root


class ControllerToolMetadata(_FrozenModel):
    canonical_tool_name: str = Field(min_length=1)
    effect: ToolEffect
    risk: ToolRisk
    prerequisites: tuple[MetadataPredicate, ...] = ()
    parallel_safe: bool
    read_resources: tuple[ResourceTemplate, ...] = ()
    write_resources: tuple[ResourceTemplate, ...] = ()
    critic_required: bool
    agent_forbidden: bool = False

    @model_validator(mode="after")
    def validate_effect_defaults(self) -> "ControllerToolMetadata":
        if self.effect is ToolEffect.EXTERNAL_WRITE:
            if self.risk is not ToolRisk.HIGH or self.parallel_safe or not self.critic_required:
                raise ValueError("external_write must be high risk, non-parallel, and critic-required")
        if self.effect is ToolEffect.CONVERSATION_CONTROL and self.risk is not ToolRisk.HIGH:
            raise ValueError("conversation_control must be high risk")
        return self


class ToolMetadataManifest(_FrozenModel):
    schema_version: Literal["1.0"]
    upstream_repository: str = Field(min_length=1)
    upstream_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    record_count: int = Field(ge=0)
    effect_counts: dict[ToolEffect, int]
    metadata_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    public_inventory_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    risk_policy: str = Field(min_length=1)
    generated_at_utc: str = Field(min_length=1)


__all__ = [
    "ControllerToolMetadata", "MetadataPredicate", "PredicateOp", "ResourceTemplate",
    "ToolEffect", "ToolMetadataManifest", "ToolRisk",
]
