"""Strict, content-free projections and immutable metrics records."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.usage import TokenUsage


NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
SystemVariant = Literal["vanilla", "generation_0", "updated"] | None
ProviderRole = Literal[
    "vanilla", "policy", "critic", "revision", "memory_candidate", "memory_review",
    "failure_mode_update", "skill_candidate", "embedding", "user_simulator",
]
CostUnit = Literal["qwen_effective_output_tokens"]
CompletionStatus = Literal["complete", "failed", "partial"]


def _require_utc(value: datetime | None, field: str) -> None:
    if value is not None and (
        value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value)
    ):
        raise ValueError(f"{field} must be UTC")


class FrozenAccountingModel(StrictModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False,
        protected_namespaces=(),
    )


class PhysicalAttemptAccountingInput(FrozenAccountingModel):
    run_id: Identifier
    round_index: NonNegativeInt | None = None
    task_id: Identifier | None = None
    scenario_family_id: Identifier | None = None
    scenario_id: Identifier | None = None
    system_variant: SystemVariant = None
    logical_request_id: Identifier
    attempt_id: Identifier
    attempt_ordinal: NonNegativeInt
    role: ProviderRole
    phase: Identifier
    provider: Identifier
    model: Identifier
    endpoint_kind: Literal["chat", "embedding"]
    dispatched: bool
    replayed_after_unknown_outcome: bool
    status: Literal[
        "allocated", "abandoned_before_dispatch", "rejected_before_dispatch",
        "completed", "failed", "unknown_outcome",
    ]
    unknown_outcome_kind: Literal["timeout", "connection", "other"] | None = None
    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None
    latency_seconds: NonNegativeDecimal | None = None
    usage: TokenUsage | None = None
    response_sha256: Digest | None = None

    @model_validator(mode="after")
    def invariants(self):
        _require_utc(self.started_at_utc, "started_at_utc")
        _require_utc(self.completed_at_utc, "completed_at_utc")
        before = self.status in {"allocated", "abandoned_before_dispatch", "rejected_before_dispatch"}
        if self.dispatched == before:
            raise ValueError("dispatch flag conflicts with attempt status")
        if not self.dispatched and any(
            value is not None for value in (
                self.started_at_utc, self.completed_at_utc, self.latency_seconds,
                self.usage, self.response_sha256,
            )
        ):
            raise ValueError("undispatched attempt cannot have provider evidence")
        if self.status == "completed" and (
            self.completed_at_utc is None or self.latency_seconds is None
        ):
            raise ValueError("completed attempt requires completion timing")
        if self.response_sha256 is not None and self.status != "completed":
            raise ValueError("only completed attempts have a response hash")
        if (self.unknown_outcome_kind is not None) != (self.status == "unknown_outcome"):
            raise ValueError("unknown outcome classification mismatch")
        if self.replayed_after_unknown_outcome and not self.dispatched:
            raise ValueError("only dispatched attempts may be recovery dispatches")
        return self


class LogicalRequestAccountingInput(FrozenAccountingModel):
    run_id: Identifier
    round_index: NonNegativeInt | None = None
    task_id: Identifier | None = None
    scenario_family_id: Identifier | None = None
    scenario_id: Identifier | None = None
    system_variant: SystemVariant = None
    logical_request_id: Identifier
    role: ProviderRole
    phase: Identifier
    provider: Identifier
    model: Identifier
    status: Literal["prepared", "response_completed", "applied", "terminal_failure"]
    source_attempt_id: Identifier | None = None
    application_id: Identifier | None = None

    @model_validator(mode="after")
    def source_only_for_applied(self):
        if (self.source_attempt_id is not None) != (self.status == "applied"):
            raise ValueError("source_attempt_id is required exactly for applied status")
        if (self.application_id is not None) != (self.status == "applied"):
            raise ValueError("application_id is required exactly for applied status")
        return self


class ToolAttemptAccountingInput(FrozenAccountingModel):
    run_id: Identifier
    round_index: NonNegativeInt | None = None
    task_id: Identifier | None = None
    scenario_family_id: Identifier | None = None
    scenario_id: Identifier | None = None
    system_variant: SystemVariant = None
    logical_call_id: Identifier
    tool_attempt_id: Identifier
    profile: Literal["offline", "live"]
    execution_mode: Literal["fixture", "live", "local"]
    effect_class: Identifier
    fixture_status: Literal["hit", "miss"] | None = None
    status: Literal["completed", "failed", "timeout", "unknown_outcome"]
    started_at_utc: datetime
    completed_at_utc: datetime | None = None
    latency_seconds: NonNegativeDecimal

    @model_validator(mode="after")
    def timing_is_utc(self):
        _require_utc(self.started_at_utc, "started_at_utc")
        _require_utc(self.completed_at_utc, "completed_at_utc")
        return self


class ScopeTimingInput(FrozenAccountingModel):
    scope_kind: Literal[
        "scenario_task", "round_online", "round_offline", "round_total",
        "evaluation_run", "training_run",
    ]
    scope_id: Identifier
    boot_id: Identifier
    ended_boot_id: Identifier | None = None
    started_at_utc: datetime
    ended_at_utc: datetime | None = None
    start_monotonic_ns: NonNegativeInt
    end_monotonic_ns: NonNegativeInt | None = None
    timing_complete: bool = False
    total_running_time_seconds: NonNegativeDecimal | None = None

    @model_validator(mode="after")
    def timing_invariants(self):
        _require_utc(self.started_at_utc, "started_at_utc")
        _require_utc(self.ended_at_utc, "ended_at_utc")
        closed = self.end_monotonic_ns is not None
        if closed != (self.ended_boot_id is not None):
            raise ValueError("end boot identity mismatch")
        same_boot = closed and self.ended_boot_id == self.boot_id
        if same_boot and self.end_monotonic_ns < self.start_monotonic_ns:
            raise ValueError("negative monotonic elapsed time")
        if self.timing_complete != (
            same_boot and self.total_running_time_seconds is not None
        ):
            raise ValueError("timing completeness mismatch")
        if self.timing_complete:
            expected = Decimal(self.end_monotonic_ns - self.start_monotonic_ns) / Decimal(1_000_000_000)
            if self.total_running_time_seconds != expected or self.ended_at_utc is None:
                raise ValueError("direct monotonic timing mismatch")
        return self


class AccountingTotals(FrozenAccountingModel):
    input_tokens: NonNegativeInt | None
    uncached_input_tokens: NonNegativeInt | None
    cache_read_input_tokens: NonNegativeInt | None
    cache_write_input_tokens: NonNegativeInt | None
    output_tokens: NonNegativeInt | None
    total_tokens: NonNegativeInt | None
    usage_complete: bool
    total_cost: NonNegativeInt | None
    cost_unit: CostUnit = "qwen_effective_output_tokens"
    cost_complete: bool
    prepared_logical_request_count: NonNegativeInt = 0
    dispatched_logical_request_count: NonNegativeInt = 0
    applied_logical_response_count: NonNegativeInt = 0
    physical_dispatch_count: NonNegativeInt = 0
    recovery_dispatch_count: NonNegativeInt = 0
    abandoned_before_dispatch_count: NonNegativeInt = 0
    completed_attempt_count: NonNegativeInt = 0
    failed_attempt_count: NonNegativeInt = 0
    timeout_attempt_count: NonNegativeInt = 0
    unknown_outcome_attempt_count: NonNegativeInt = 0
    incomplete_usage_attempt_count: NonNegativeInt = 0

    @model_validator(mode="after")
    def completeness(self):
        token_values = (
            self.input_tokens, self.uncached_input_tokens,
            self.cache_read_input_tokens, self.cache_write_input_tokens,
            self.output_tokens, self.total_tokens,
        )
        if self.usage_complete != all(value is not None for value in token_values):
            raise ValueError("usage completeness mismatch")
        if self.cost_complete != (self.total_cost is not None):
            raise ValueError("cost completeness mismatch")
        return self


class AccountingBreakdown(FrozenAccountingModel):
    dimension: Literal["provider", "model", "role", "phase", "system_variant", "provider_model", "role_phase"]
    key: tuple[str, ...]
    totals: AccountingTotals


class TaskAccountingInput(FrozenAccountingModel):
    run_id: Identifier
    task_id: Identifier
    scenario_family_id: Identifier
    scenario_id: Identifier
    system_variant: Literal["vanilla", "generation_0", "updated"]
    timing: ScopeTimingInput
    completion_status: CompletionStatus
    evaluator_result_sha256: Digest | None = None
    final_context_sha256: Digest | None = None


class RoundAccountingInput(FrozenAccountingModel):
    run_id: Identifier
    round_index: Annotated[int, Field(ge=0, le=2)]
    input_generation_id: Identifier
    train_shard_id: Identifier
    published_generation_id: Identifier | None = None
    online_timing: ScopeTimingInput | None = None
    offline_timing: ScopeTimingInput | None = None
    publication_running_time_seconds: NonNegativeDecimal | None = None
    total_timing: ScopeTimingInput
    completion_status: CompletionStatus


class RunAccountingInput(FrozenAccountingModel):
    run_id: Identifier
    run_kind: Literal["training", "evaluation"]
    timing: ScopeTimingInput
    profile: Identifier
    system_variant: SystemVariant = None
    manifest_sha256: Digest
    config_sha256: Digest
    completion_status: CompletionStatus


def _record_identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    selected = {
        key: value for key, value in payload.items()
        if key not in {"record_id", "content_sha256"}
    }
    return to_jsonable_python(selected)


class MetricRecord(FrozenAccountingModel):
    schema_version: Literal[1] = 1
    record_id: Digest
    recorded_at_utc: datetime
    run_id: Identifier
    record_kind: Literal["request", "task", "round", "run"]
    content_sha256: Digest

    @model_validator(mode="after")
    def identity_and_time(self):
        _require_utc(self.recorded_at_utc, "recorded_at_utc")
        payload = _record_identity_payload(self.model_dump(mode="json"))
        expected_content = canonical_sha256(["metrics-record-content-v1", payload])
        expected_id = canonical_sha256(["metrics-record-id-v1", self.record_kind, payload])
        if self.content_sha256 != expected_content or self.record_id != expected_id:
            raise ValueError("metric record identity mismatch")
        return self

    @classmethod
    def build(cls, **values: Any):
        values.setdefault("record_kind", cls.model_fields["record_kind"].default)
        placeholder = "sha256:" + "0" * 64
        provisional = cls.model_construct(
            schema_version=1, record_id=placeholder,
            content_sha256=placeholder, **values,
        )
        payload = _record_identity_payload(provisional.model_dump(mode="json"))
        values["content_sha256"] = canonical_sha256(["metrics-record-content-v1", payload])
        values["record_id"] = canonical_sha256([
            "metrics-record-id-v1", payload["record_kind"], payload,
        ])
        return cls(schema_version=1, **values)


class RequestMetricRecord(MetricRecord):
    record_kind: Literal["request"] = "request"
    attempt: PhysicalAttemptAccountingInput
    source_of_applied_response: bool
    effective_output_tokens: NonNegativeInt | None


class TaskMetricRecord(MetricRecord):
    record_kind: Literal["task"] = "task"
    task: TaskAccountingInput
    totals: AccountingTotals
    breakdowns: tuple[AccountingBreakdown, ...] = ()
    model_call_counts: dict[str, NonNegativeInt] = {}
    tool_attempt_count: NonNegativeInt = 0
    tool_total_running_time_seconds: NonNegativeDecimal = Decimal(0)


class RoundMetricRecord(MetricRecord):
    record_kind: Literal["round"] = "round"
    round: RoundAccountingInput
    totals: AccountingTotals
    started_at_utc: datetime
    ended_at_utc: datetime | None
    total_running_time_seconds: NonNegativeDecimal | None
    total_tokens: NonNegativeInt | None
    usage_complete: bool
    total_cost: NonNegativeInt | None
    cost_unit: CostUnit = "qwen_effective_output_tokens"
    cost_complete: bool
    scenario_count: NonNegativeInt

    @model_validator(mode="after")
    def headline_matches_direct_timing(self):
        _require_utc(self.started_at_utc, "started_at_utc")
        _require_utc(self.ended_at_utc, "ended_at_utc")
        timing = self.round.total_timing
        if (
            self.started_at_utc != timing.started_at_utc
            or self.ended_at_utc != timing.ended_at_utc
            or self.total_running_time_seconds != timing.total_running_time_seconds
        ):
            raise ValueError("round headline must use direct total timing")
        if (
            self.total_tokens != self.totals.total_tokens
            or self.usage_complete != self.totals.usage_complete
            or self.total_cost != self.totals.total_cost
            or self.cost_unit != self.totals.cost_unit
            or self.cost_complete != self.totals.cost_complete
        ):
            raise ValueError("round headline accounting mismatch")
        return self


class RunMetricRecord(MetricRecord):
    record_kind: Literal["run"] = "run"
    run: RunAccountingInput
    totals: AccountingTotals
    total_running_time_seconds: NonNegativeDecimal | None
    total_tokens: NonNegativeInt | None
    usage_complete: bool
    total_cost: NonNegativeInt | None
    cost_unit: CostUnit = "qwen_effective_output_tokens"
    cost_complete: bool
    round_record_ids: tuple[Digest, ...] = ()
    request_record_count: NonNegativeInt
    task_record_count: NonNegativeInt
    round_record_count: NonNegativeInt

    @model_validator(mode="after")
    def headline_matches_run(self):
        if (
            self.total_running_time_seconds != self.run.timing.total_running_time_seconds
            or self.total_tokens != self.totals.total_tokens
            or self.usage_complete != self.totals.usage_complete
            or self.total_cost != self.totals.total_cost
            or self.cost_unit != self.totals.cost_unit
            or self.cost_complete != self.totals.cost_complete
        ):
            raise ValueError("run headline accounting mismatch")
        return self


__all__ = [
    "AccountingBreakdown", "AccountingTotals", "LogicalRequestAccountingInput",
    "PhysicalAttemptAccountingInput", "RequestMetricRecord", "RoundAccountingInput",
    "RoundMetricRecord", "RunAccountingInput", "RunMetricRecord", "ScopeTimingInput",
    "TaskAccountingInput", "TaskMetricRecord", "ToolAttemptAccountingInput",
]
