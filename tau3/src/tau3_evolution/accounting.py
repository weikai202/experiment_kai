"""Physical-attempt token accounting and substantive-effect Qwen cost."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class UsageAttempt:
    attempt_id: str
    logical_request_id: str
    provider_role: str
    input_tokens: int | None
    output_tokens: int | None
    causal_effect_id: str | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.logical_request_id:
            raise ValueError("usage attempt IDs are required")
        if self.provider_role not in {"qwen", "embedding", "user_simulator"}:
            raise ValueError("unknown provider role")
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("token counts must be non-negative integers or unavailable")


@dataclass(frozen=True)
class RoundAccounting:
    round_index: int
    total_running_time_seconds: float | None
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_tokens: int | None
    usage_complete: bool
    total_cost: int | None
    cost_unit: str
    cost_complete: bool
    physical_attempt_count: int
    latency_clock: str | None = None


def summarize_round(
    *,
    round_index: int,
    direct_latency_seconds: float | None,
    attempts: tuple[UsageAttempt, ...],
    committed_substantive_effect_ids: frozenset[str],
) -> RoundAccounting:
    if round_index not in (0, 1, 2):
        raise ValueError("round index must be 0, 1, or 2")
    if direct_latency_seconds is not None and (
        type(direct_latency_seconds) is not float
        or not math.isfinite(direct_latency_seconds)
        or direct_latency_seconds < 0
    ):
        raise ValueError("direct latency must be a finite non-negative float")
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("duplicate physical attempt ID")
    effect_bindings = [
        (attempt.logical_request_id, attempt.causal_effect_id)
        for attempt in attempts
        if attempt.provider_role == "qwen" and attempt.causal_effect_id is not None
    ]
    if len({logical_id for logical_id, _ in effect_bindings}) != len(effect_bindings):
        raise ValueError(
            "one logical Qwen request cannot bind multiple physical outputs to effects"
        )
    usage_complete = all(
        attempt.input_tokens is not None and attempt.output_tokens is not None
        for attempt in attempts
    )
    total_input = sum(attempt.input_tokens or 0 for attempt in attempts) if usage_complete else None
    total_output = (
        sum(attempt.output_tokens or 0 for attempt in attempts) if usage_complete else None
    )
    effective = [
        attempt
        for attempt in attempts
        if attempt.provider_role == "qwen"
        and attempt.causal_effect_id in committed_substantive_effect_ids
    ]
    bound_effects = {attempt.causal_effect_id for attempt in effective}
    unknown_effects = committed_substantive_effect_ids - bound_effects
    cost_complete = not unknown_effects and all(
        attempt.output_tokens is not None for attempt in effective
    )
    total_cost = sum(attempt.output_tokens or 0 for attempt in effective) if cost_complete else None
    return RoundAccounting(
        round_index=round_index,
        total_running_time_seconds=direct_latency_seconds,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=(total_input + total_output) if usage_complete else None,
        usage_complete=usage_complete,
        total_cost=total_cost,
        cost_unit="qwen_effective_output_tokens",
        cost_complete=cost_complete,
        physical_attempt_count=len(attempts),
    )
