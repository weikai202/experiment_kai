"""Immutable actual-usage records. Unknown counts are never estimated."""
from datetime import datetime
from typing import Annotated
from pydantic import ConfigDict, Field, model_validator
from toolsandbox_pipeline.schemas.base import StrictModel

Count = Annotated[int, Field(ge=0)] | None


class TokenUsage(StrictModel):
    model_config = ConfigDict(frozen=True)
    input_tokens: Count = None
    uncached_input_tokens: Count = None
    cache_read_input_tokens: Count = None
    cache_write_input_tokens: Count = None
    output_tokens: Count = None
    total_tokens: Count = None
    usage_complete: bool = False

    @model_validator(mode="after")
    def consistent(self):
        fields = (self.input_tokens, self.uncached_input_tokens,
                  self.cache_read_input_tokens, self.cache_write_input_tokens,
                  self.output_tokens, self.total_tokens)
        if self.usage_complete != all(v is not None for v in fields):
            raise ValueError("incorrect usage completeness")
        parts = fields[1:4]
        if self.input_tokens is not None and all(v is not None for v in parts):
            if self.input_tokens != sum(parts):
                raise ValueError("inconsistent input tokens")
        if all(v is not None for v in (self.input_tokens, self.output_tokens, self.total_tokens)):
            if self.total_tokens != self.input_tokens + self.output_tokens:
                raise ValueError("inconsistent total tokens")
        return self


class PhysicalAttemptMetrics(StrictModel):
    model_config = ConfigDict(frozen=True)
    started_at: datetime
    completed_at: datetime
    latency_seconds: Annotated[float, Field(ge=0)]
    usage: TokenUsage

    @model_validator(mode="after")
    def utc(self):
        for value in (self.started_at, self.completed_at):
            if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
                raise ValueError("UTC timestamp required")
        return self
