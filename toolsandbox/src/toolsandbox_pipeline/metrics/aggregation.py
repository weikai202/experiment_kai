"""Pure attempt aggregation and substantive-effect Qwen accounting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from toolsandbox_pipeline.schemas.accounting import (
    AccountingBreakdown,
    AccountingTotals,
    LogicalRequestAccountingInput,
    PhysicalAttemptAccountingInput,
    RequestMetricRecord,
)

if TYPE_CHECKING:
    from toolsandbox_pipeline.schemas.checkpoint import QwenEffectiveEffect


_TOKEN_FIELDS = (
    "input_tokens", "uncached_input_tokens", "cache_read_input_tokens",
    "cache_write_input_tokens", "output_tokens", "total_tokens",
)


def _field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        if name not in row:
            raise ValueError(f"missing effect field: {name}")
        return row[name]
    if not hasattr(row, name):
        raise ValueError(f"missing effect field: {name}")
    return getattr(row, name)


def _deduplicate(rows: Iterable[Any], key_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        key = _field(row, key_name)
        if key in result and result[key] != row:
            raise ValueError(f"conflicting {key_name}")
        result[key] = row
    return result


def _validated_effects(
    rows: Iterable[QwenEffectiveEffect],
) -> tuple[QwenEffectiveEffect, ...]:
    """Require exact, canonically identity-valid Task 011 effect records."""
    from toolsandbox_pipeline.schemas.checkpoint import QwenEffectiveEffect

    result: dict[str, QwenEffectiveEffect] = {}
    for row in rows:
        if type(row) is not QwenEffectiveEffect:
            raise TypeError("exact QwenEffectiveEffect required")
        validated = QwenEffectiveEffect.model_validate(
            row.model_dump(mode="python"), strict=True
        )
        if validated != row:
            raise ValueError("invalid effective effect")
        existing = result.get(row.effect_id)
        if existing is not None and existing != row:
            raise ValueError("conflicting effect_id")
        result[row.effect_id] = row
    return tuple(result.values())


class MetricsAggregator:
    """Aggregate a validated ledger projection without opening the ledger."""

    def __init__(self, *, qwen_provider: str, qwen_model: str) -> None:
        if not qwen_provider or not qwen_model:
            raise ValueError("pinned Qwen provider and model are required")
        self.qwen_provider = qwen_provider
        self.qwen_model = qwen_model

    def aggregate(
        self,
        attempts: Iterable[PhysicalAttemptAccountingInput],
        logical_requests: Iterable[LogicalRequestAccountingInput] = (),
        effective_effects: Iterable[QwenEffectiveEffect] = (),
    ) -> AccountingTotals:
        attempt_by_id = _deduplicate(attempts, "attempt_id")
        logical_by_id = _deduplicate(logical_requests, "logical_request_id")
        effects = _validated_effects(effective_effects)
        return self._aggregate_projection(
            attempt_by_id, logical_by_id, effects,
        )

    def _aggregate_projection(
        self,
        attempt_by_id: dict[str, PhysicalAttemptAccountingInput],
        logical_by_id: dict[str, LogicalRequestAccountingInput],
        effective_effects: tuple[QwenEffectiveEffect, ...],
        *,
        included_application_ids: set[str] | None = None,
    ) -> AccountingTotals:
        self._validate_projection(attempt_by_id, logical_by_id)
        effective_attempt_ids = self._effective_attempt_ids(
            logical_by_id,
            attempt_by_id,
            effective_effects,
            included_application_ids=included_application_ids,
        )
        dispatched = [row for row in attempt_by_id.values() if row.dispatched]
        usage_complete = all(
            row.usage is not None and row.usage.usage_complete for row in dispatched
        )
        if usage_complete:
            token_values: dict[str, int | None] = {
                name: sum(getattr(row.usage, name) for row in dispatched)
                for name in _TOKEN_FIELDS
            }
        else:
            token_values = {name: None for name in _TOKEN_FIELDS}

        cost_complete = all(
            attempt_by_id[attempt_id].usage is not None
            and attempt_by_id[attempt_id].usage.output_tokens is not None
            for attempt_id in effective_attempt_ids
        )
        total_cost = (
            sum(attempt_by_id[attempt_id].usage.output_tokens for attempt_id in effective_attempt_ids)
            if cost_complete else None
        )
        dispatched_logical_ids = {row.logical_request_id for row in dispatched}
        return AccountingTotals(
            **token_values,
            usage_complete=usage_complete,
            total_cost=total_cost,
            cost_complete=cost_complete,
            prepared_logical_request_count=len(logical_by_id),
            dispatched_logical_request_count=len(dispatched_logical_ids),
            applied_logical_response_count=sum(
                row.status == "applied" for row in logical_by_id.values()
            ),
            physical_dispatch_count=len(dispatched),
            recovery_dispatch_count=sum(row.replayed_after_unknown_outcome for row in dispatched),
            abandoned_before_dispatch_count=sum(
                row.status in {"abandoned_before_dispatch", "rejected_before_dispatch"}
                for row in attempt_by_id.values()
            ),
            completed_attempt_count=sum(row.status == "completed" for row in dispatched),
            failed_attempt_count=sum(row.status == "failed" for row in dispatched),
            timeout_attempt_count=sum(
                row.unknown_outcome_kind == "timeout" for row in dispatched
            ),
            unknown_outcome_attempt_count=sum(row.status == "unknown_outcome" for row in dispatched),
            incomplete_usage_attempt_count=sum(
                row.usage is None or not row.usage.usage_complete for row in dispatched
            ),
        )

    def _validate_projection(
        self,
        attempts: dict[str, PhysicalAttemptAccountingInput],
        logical: dict[str, LogicalRequestAccountingInput],
    ) -> None:
        run_ids = {row.run_id for row in attempts.values()} | {row.run_id for row in logical.values()}
        if len(run_ids) > 1:
            raise ValueError("mixed run identities")
        by_logical: dict[str, list[PhysicalAttemptAccountingInput]] = defaultdict(list)
        for attempt in attempts.values():
            by_logical[attempt.logical_request_id].append(attempt)
        for logical_id, request in logical.items():
            rows = by_logical.get(logical_id, [])
            for row in rows:
                if (
                    row.run_id != request.run_id or row.role != request.role
                    or row.phase != request.phase or row.provider != request.provider
                    or row.model != request.model
                ):
                    raise ValueError("logical/physical identity mismatch")
            if request.status == "applied":
                source = attempts.get(request.source_attempt_id)
                if source is None or source.logical_request_id != logical_id or source.status != "completed":
                    raise ValueError("missing completed source attempt")

    def _effective_attempt_ids(
        self,
        logical: dict[str, LogicalRequestAccountingInput],
        attempts: dict[str, PhysicalAttemptAccountingInput],
        effects: tuple[QwenEffectiveEffect, ...],
        *,
        included_application_ids: set[str] | None = None,
    ) -> set[str]:
        applied_by_application = {
            row.application_id: row for row in logical.values() if row.status == "applied"
        }
        effective_attempt_ids: set[str] = set()
        for effect in effects:
            for application_id in effect.ordered_application_ids:
                if (
                    included_application_ids is not None
                    and application_id not in included_application_ids
                ):
                    continue
                request = applied_by_application.get(application_id)
                if request is None:
                    raise ValueError("effect references missing applied response")
                attempt = attempts[request.source_attempt_id]
                if attempt.provider != self.qwen_provider or attempt.model != self.qwen_model:
                    raise ValueError("effective effect must reference pinned Qwen")
                effective_attempt_ids.add(attempt.attempt_id)
        return effective_attempt_ids

    def breakdowns(
        self,
        attempts: Iterable[PhysicalAttemptAccountingInput],
        logical_requests: Iterable[LogicalRequestAccountingInput] = (),
        effective_effects: Iterable[QwenEffectiveEffect] = (),
    ) -> tuple[AccountingBreakdown, ...]:
        attempt_by_id = _deduplicate(attempts, "attempt_id")
        logical_by_id = _deduplicate(logical_requests, "logical_request_id")
        effects = _validated_effects(effective_effects)
        self._validate_projection(attempt_by_id, logical_by_id)
        self._effective_attempt_ids(logical_by_id, attempt_by_id, effects)
        attempts = tuple(attempt_by_id.values())
        logical = tuple(logical_by_id.values())
        dimensions = {
            "provider": lambda row: (row.provider,),
            "model": lambda row: (row.model,),
            "role": lambda row: (row.role,),
            "phase": lambda row: (row.phase,),
            "system_variant": lambda row: (row.system_variant or "none",),
            "provider_model": lambda row: (row.provider, row.model),
            "role_phase": lambda row: (row.role, row.phase),
        }
        output: list[AccountingBreakdown] = []
        for dimension, get_key in dimensions.items():
            keys = sorted({get_key(row) for row in attempts})
            for key in keys:
                selected_attempts = tuple(row for row in attempts if get_key(row) == key)
                logical_ids = {row.logical_request_id for row in selected_attempts}
                selected_logical = tuple(row for row in logical if row.logical_request_id in logical_ids)
                selected_apps = {row.application_id for row in selected_logical if row.application_id}
                output.append(AccountingBreakdown(
                    dimension=dimension,
                    key=key,
                    totals=self._aggregate_projection(
                        _deduplicate(selected_attempts, "attempt_id"),
                        _deduplicate(selected_logical, "logical_request_id"),
                        effects,
                        included_application_ids=selected_apps,
                    ),
                ))
        return tuple(output)

    def request_records(
        self,
        attempts: Iterable[PhysicalAttemptAccountingInput],
        logical_requests: Iterable[LogicalRequestAccountingInput],
        effective_effects: Iterable[QwenEffectiveEffect],
        *,
        recorded_at_utc: datetime,
    ) -> tuple[RequestMetricRecord, ...]:
        attempt_by_id = _deduplicate(attempts, "attempt_id")
        logical = _deduplicate(logical_requests, "logical_request_id")
        self._validate_projection(attempt_by_id, logical)
        effects = _validated_effects(effective_effects)
        effective_ids = self._effective_attempt_ids(logical, attempt_by_id, effects)
        applied_source_ids = {
            row.source_attempt_id for row in logical.values() if row.status == "applied"
        }
        records = []
        for attempt in sorted(attempt_by_id.values(), key=lambda row: row.attempt_id):
            usage = attempt.usage
            if attempt.attempt_id in effective_ids:
                contribution = usage.output_tokens if usage is not None else None
            else:
                contribution = 0
            records.append(RequestMetricRecord.build(
                recorded_at_utc=recorded_at_utc,
                run_id=attempt.run_id,
                record_kind="request",
                attempt=attempt,
                source_of_applied_response=attempt.attempt_id in applied_source_ids,
                effective_output_tokens=contribution,
            ))
        return tuple(records)


__all__ = ["MetricsAggregator"]
