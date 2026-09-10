"""Strict, shared schema primitives."""

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue


JsonObject: TypeAlias = dict[str, JsonValue]


class StrictModel(BaseModel):
    """Base for persisted and model-facing contracts."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
    )
