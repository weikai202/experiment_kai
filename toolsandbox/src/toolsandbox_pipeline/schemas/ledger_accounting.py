"""Cycle-safe immutable schemas for Task011 accounting snapshots."""

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.accounting import (
    LogicalRequestAccountingInput,
    PhysicalAttemptAccountingInput,
)
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.checkpoint import (
    BlobReference,
    LLMResponseApplication,
    QwenEffectiveEffect,
)


class _Frozen(StrictModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class AccountingScope(_Frozen):
    run_id: str = Field(min_length=1, pattern=r"^\S+$")
    round_index: int | None = Field(default=None, ge=0)
    task_id: str = Field(min_length=1, pattern=r"^\S+$")
    scenario_family_id: str = Field(min_length=1, pattern=r"^\S+$")
    scenario_id: str = Field(min_length=1, pattern=r"^\S+$")
    system_variant: Literal["vanilla", "generation_0", "updated"]


class Task011LedgerPopulation(_Frozen):
    schema_version: Literal[1] = 1
    scope: AccountingScope
    high_water_identity: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    logical_request_ids: tuple[str, ...]
    physical_attempt_ids: tuple[str, ...]
    application_ids: tuple[str, ...]
    effect_ids: tuple[str, ...]
    population_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_and_uniqueness(self):
        for values in (
            self.logical_request_ids,
            self.physical_attempt_ids,
            self.application_ids,
            self.effect_ids,
        ):
            if any(not value or any(c.isspace() for c in value) for value in values):
                raise ValueError("invalid Task011 population identity")
            if len(values) != len(set(values)):
                raise ValueError("duplicate Task011 population identity")
        payload = self.model_dump(mode="json", exclude={"population_sha256"})
        if self.population_sha256 != canonical_sha256(
            ["task011-ledger-population-v1", payload]
        ):
            raise ValueError("Task011 population identity mismatch")
        return self

    @classmethod
    def build(cls, **values: Any) -> "Task011LedgerPopulation":
        provisional = cls.model_construct(
            schema_version=1,
            population_sha256="sha256:" + "0" * 64,
            **values,
        )
        payload = provisional.model_dump(mode="json", exclude={"population_sha256"})
        values["population_sha256"] = canonical_sha256(
            ["task011-ledger-population-v1", payload]
        )
        return cls(schema_version=1, **values)


class Task011AccountingSnapshot(_Frozen):
    population: Task011LedgerPopulation
    population_reference: BlobReference
    physical_attempts: tuple[PhysicalAttemptAccountingInput, ...]
    logical_requests: tuple[LogicalRequestAccountingInput, ...]
    applications: tuple[LLMResponseApplication, ...]
    substantive_effects: tuple[QwenEffectiveEffect, ...]

    @model_validator(mode="after")
    def exact_population(self):
        population = self.population
        if (
            tuple(item.attempt_id for item in self.physical_attempts)
            != population.physical_attempt_ids
            or tuple(item.logical_request_id for item in self.logical_requests)
            != population.logical_request_ids
            or tuple(item.application_id for item in self.applications)
            != population.application_ids
            or tuple(item.effect_id for item in self.substantive_effects)
            != population.effect_ids
            or self.population_reference.sha256
            != canonical_sha256(population.model_dump(mode="json"))
            or self.population_reference.schema_name != "Task011LedgerPopulation"
            or self.population_reference.schema_version != 1
        ):
            raise ValueError("Task011 accounting snapshot population mismatch")
        return self


__all__ = [
    "AccountingScope",
    "Task011AccountingSnapshot",
    "Task011LedgerPopulation",
]
