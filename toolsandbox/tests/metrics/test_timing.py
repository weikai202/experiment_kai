from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.metrics import ScopeTimer
from toolsandbox_pipeline.schemas.accounting import ScopeTimingInput


class Values:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_direct_monotonic_time_and_idempotent_close():
    timer = ScopeTimer(
        "round_total", "round-0", "boot-a",
        monotonic_ns=Values(1_000_000_000, 4_500_000_000),
        time_ns=Values(2_000_000_000, 20_000_000_000),
    )
    assert not timer.snapshot().timing_complete
    closed = timer.close()
    assert closed.total_running_time_seconds == Decimal("3.5")
    assert timer.close() is closed


def test_same_boot_resume_includes_elapsed_gap():
    started = ScopeTimingInput(
        scope_kind="training_run", scope_id="run", boot_id="boot-a",
        started_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        start_monotonic_ns=10, timing_complete=False,
    )
    resumed = ScopeTimer(
        "training_run", "run", "boot-a", stored=started,
        monotonic_ns=Values(2_000_000_010), time_ns=Values(2_000_000_000),
    ).close()
    assert resumed.timing_complete
    assert resumed.total_running_time_seconds == Decimal("2")


def test_boot_change_fails_closed_without_utc_fallback():
    started = ScopeTimingInput(
        scope_kind="evaluation_run", scope_id="run", boot_id="boot-a",
        started_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        start_monotonic_ns=10, timing_complete=False,
    )
    resumed = ScopeTimer(
        "evaluation_run", "run", "boot-b", stored=started,
        monotonic_ns=Values(1), time_ns=Values(9_000_000_000),
    ).close()
    assert resumed.ended_at_utc != resumed.started_at_utc
    assert not resumed.timing_complete
    assert resumed.total_running_time_seconds is None


def test_scope_and_negative_time_denied():
    started = ScopeTimingInput(
        scope_kind="scenario_task", scope_id="task", boot_id="boot",
        started_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        start_monotonic_ns=10, timing_complete=False,
    )
    with pytest.raises(ValueError, match="identity"):
        ScopeTimer("scenario_task", "other", "boot", stored=started)
    with pytest.raises(ValueError, match="negative"):
        ScopeTimer(
            "scenario_task", "task", "boot", stored=started,
            monotonic_ns=Values(9), time_ns=Values(1),
        ).close()
    with pytest.raises(ValidationError):
        ScopeTimingInput(
            scope_kind="scenario_task", scope_id="task", boot_id="boot",
            started_at_utc=datetime(2026, 1, 1), start_monotonic_ns=0,
            timing_complete=False,
        )
