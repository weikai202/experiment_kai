"""Direct monotonic timers for experimental scopes."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal

from toolsandbox_pipeline.schemas.accounting import ScopeTimingInput


class ScopeTimer:
    """A single-start/single-close timer with explicit same-boot recovery."""

    def __init__(
        self,
        scope_kind: str,
        scope_id: str,
        boot_id: str,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        time_ns: Callable[[], int] = time.time_ns,
        stored: ScopeTimingInput | None = None,
    ) -> None:
        self._monotonic_ns = monotonic_ns
        self._time_ns = time_ns
        self._scope_kind = scope_kind
        self._scope_id = scope_id
        self._boot_id = boot_id
        self._closed: ScopeTimingInput | None = None
        if stored is None:
            start_mono = monotonic_ns()
            start_utc = self._utc(time_ns())
            self._start_boot_id = boot_id
        else:
            if stored.scope_kind != scope_kind or stored.scope_id != scope_id:
                raise ValueError("scope identity mismatch")
            if stored.end_monotonic_ns is not None:
                self._closed = stored
            start_mono = stored.start_monotonic_ns
            start_utc = stored.started_at_utc
            self._start_boot_id = stored.boot_id
        self._start_monotonic_ns = start_mono
        self._started_at_utc = start_utc

    @staticmethod
    def _utc(epoch_ns: int) -> datetime:
        if type(epoch_ns) is not int or epoch_ns < 0:
            raise ValueError("time_ns must return a non-negative integer")
        return datetime.fromtimestamp(epoch_ns / 1_000_000_000, tz=timezone.utc)

    def snapshot(self) -> ScopeTimingInput:
        if self._closed is not None:
            return self._closed
        return ScopeTimingInput(
            scope_kind=self._scope_kind,
            scope_id=self._scope_id,
            boot_id=self._start_boot_id,
            started_at_utc=self._started_at_utc,
            start_monotonic_ns=self._start_monotonic_ns,
            timing_complete=False,
        )

    def close(self) -> ScopeTimingInput:
        if self._closed is not None:
            return self._closed
        end_monotonic_ns = self._monotonic_ns()
        ended_at_utc = self._utc(self._time_ns())
        if type(end_monotonic_ns) is not int or end_monotonic_ns < 0:
            raise ValueError("monotonic_ns must return a non-negative integer")
        same_boot = self._boot_id == self._start_boot_id
        if same_boot and end_monotonic_ns < self._start_monotonic_ns:
            raise ValueError("negative monotonic elapsed time")
        elapsed = (
            Decimal(end_monotonic_ns - self._start_monotonic_ns)
            / Decimal(1_000_000_000)
            if same_boot else None
        )
        result = ScopeTimingInput(
            scope_kind=self._scope_kind,
            scope_id=self._scope_id,
            boot_id=self._start_boot_id,
            ended_boot_id=self._boot_id,
            started_at_utc=self._started_at_utc,
            ended_at_utc=ended_at_utc,
            start_monotonic_ns=self._start_monotonic_ns,
            end_monotonic_ns=end_monotonic_ns,
            timing_complete=same_boot,
            total_running_time_seconds=elapsed,
        )
        self._closed = result
        return result


__all__ = ["ScopeTimer"]
