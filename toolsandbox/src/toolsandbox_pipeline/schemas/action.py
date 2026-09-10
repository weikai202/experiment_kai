"""Policy and Revision action contracts."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import JsonObject, StrictModel


class ActionType(str, Enum):
    FUNCTION_CALL = "function_call"
    PARALLEL_BATCH = "parallel_batch"
    ASSISTANT_MESSAGE = "assistant_message"


class FunctionCall(StrictModel):
    call_id: str = Field(min_length=1)
    selected_skill_id: str | None = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: JsonObject


class FunctionCallAction(FunctionCall):
    type: Literal[ActionType.FUNCTION_CALL]


class ParallelBatchAction(StrictModel):
    type: Literal[ActionType.PARALLEL_BATCH]
    calls: list[FunctionCall] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_call_ids(self) -> "ParallelBatchAction":
        call_ids = [call.call_id for call in self.calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("parallel batch call_id values must be unique")
        return self


class AssistantMessageAction(StrictModel):
    type: Literal[ActionType.ASSISTANT_MESSAGE]
    content: str = Field(min_length=1)


Action = Annotated[
    FunctionCallAction | ParallelBatchAction | AssistantMessageAction,
    Field(discriminator="type"),
]


class ActionEnvelope(StrictModel):
    action: Action
