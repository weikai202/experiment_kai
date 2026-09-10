import json
import os
import stat
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.metrics import MetricsArtifactWriter
from toolsandbox_pipeline.schemas.accounting import (
    AccountingTotals,
    PhysicalAttemptAccountingInput,
    RequestMetricRecord,
    RoundAccountingInput,
    RoundMetricRecord,
    RunAccountingInput,
    RunMetricRecord,
    ScopeTimingInput,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
HASH = "sha256:" + "b" * 64


def totals(tokens=0, cost=0):
    return AccountingTotals(
        input_tokens=tokens, uncached_input_tokens=tokens,
        cache_read_input_tokens=0, cache_write_input_tokens=0,
        output_tokens=0, total_tokens=tokens, usage_complete=True,
        total_cost=cost, cost_complete=True,
    )


def closed_timing(kind, scope, seconds=1):
    nanoseconds = seconds * 1_000_000_000
    return ScopeTimingInput(
        scope_kind=kind, scope_id=scope, boot_id="boot", ended_boot_id="boot",
        started_at_utc=NOW, ended_at_utc=NOW,
        start_monotonic_ns=0, end_monotonic_ns=nanoseconds,
        timing_complete=True,
        total_running_time_seconds=Decimal(seconds),
    )


def request_record(attempt_id):
    attempt = PhysicalAttemptAccountingInput(
        run_id="run", logical_request_id=f"logical-{attempt_id}",
        attempt_id=attempt_id, attempt_ordinal=0, role="policy", phase="online",
        provider="vllm", model="Qwen/Qwen3-32B", endpoint_kind="chat",
        dispatched=False, replayed_after_unknown_outcome=False,
        status="abandoned_before_dispatch",
    )
    return RequestMetricRecord.build(
        recorded_at_utc=NOW, run_id="run", attempt=attempt,
        source_of_applied_response=False, effective_output_tokens=0,
    )


def round_record(index):
    timing = closed_timing("round_total", f"round-{index}", index + 1)
    data = RoundAccountingInput(
        run_id="run", round_index=index, input_generation_id=f"g{index:03d}",
        train_shard_id=f"shard-{index}", published_generation_id=f"g{index + 1:03d}",
        total_timing=timing, completion_status="complete",
    )
    aggregate = totals(index + 1, index)
    return RoundMetricRecord.build(
        recorded_at_utc=NOW, run_id="run", round=data, totals=aggregate,
        started_at_utc=NOW, ended_at_utc=NOW,
        total_running_time_seconds=Decimal(index + 1),
        total_tokens=index + 1, usage_complete=True,
        total_cost=index, cost_complete=True, scenario_count=1,
    )


def run_record(rounds):
    timing = closed_timing("training_run", "run", 10)
    data = RunAccountingInput(
        run_id="run", run_kind="training", timing=timing, profile="strict_replay",
        manifest_sha256=HASH, config_sha256=HASH, completion_status="complete",
    )
    aggregate = totals(6, 3)
    return RunMetricRecord.build(
        recorded_at_utc=NOW, run_id="run", run=data, totals=aggregate,
        total_running_time_seconds=Decimal(10), total_tokens=6,
        usage_complete=True, total_cost=3, cost_complete=True,
        round_record_ids=tuple(record.record_id for record in rounds),
        request_record_count=0, task_record_count=0, round_record_count=3,
    )


def materialize(writer, requests=(), rounds=(), run=None, high=0):
    return writer.materialize(
        run_id="run", request_records=requests, task_records=(),
        round_records=rounds, run_record=run,
        ledger_high_water_marks={"attempts": high},
        pinned_qwen_provider="vllm", pinned_qwen_model="Qwen/Qwen3-32B",
    )


def test_exact_prefix_append_order_modes_manifest_and_noop(tmp_path):
    writer = MetricsArtifactWriter(tmp_path)
    first = request_record("a")
    second = request_record("b")
    materialize(writer, (first,), high=1)
    old = (tmp_path / "metrics/request_metrics.jsonl").read_bytes()
    materialize(writer, (second, first), high=2)
    new = (tmp_path / "metrics/request_metrics.jsonl").read_bytes()
    assert new.startswith(old)
    assert [json.loads(line)["attempt"]["attempt_id"] for line in new.splitlines()] == ["a", "b"]
    for path in (tmp_path / "metrics").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    before = {path.name: path.read_bytes() for path in (tmp_path / "metrics").iterdir()}
    materialize(writer, (first, second), high=2)
    assert before == {path.name: path.read_bytes() for path in (tmp_path / "metrics").iterdir()}


def test_conflicting_prefix_and_decreasing_high_water_fail(tmp_path):
    writer = MetricsArtifactWriter(tmp_path)
    materialize(writer, (request_record("a"),), high=2)
    with pytest.raises(ValueError, match="exact prefix"):
        materialize(writer, (request_record("b"),), high=3)
    with pytest.raises(ValueError, match="decreasing"):
        materialize(writer, (request_record("a"),), high=1)


@pytest.mark.parametrize("point", ["before_replace", "after_replace"])
def test_crash_injection_recovers_from_authoritative_projection(tmp_path, point):
    fired = False

    def crash(at, _path):
        nonlocal fired
        if at == point and not fired:
            fired = True
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        materialize(MetricsArtifactWriter(tmp_path, crash_hook=crash), (request_record("a"),), high=1)
    materialize(MetricsArtifactWriter(tmp_path), (request_record("a"),), high=1)
    assert len((tmp_path / "metrics/request_metrics.jsonl").read_bytes().splitlines()) == 1


def test_closed_training_requires_and_writes_three_direct_round_rows(tmp_path):
    rounds = tuple(round_record(index) for index in range(3))
    run = run_record(rounds)
    materialize(MetricsArtifactWriter(tmp_path), rounds=rounds, run=run, high=3)
    rows = [json.loads(line) for line in (tmp_path / "metrics/round_metrics.jsonl").read_bytes().splitlines()]
    assert [row["round"]["round_index"] for row in rows] == [0, 1, 2]
    assert [row["total_running_time_seconds"] for row in rows] == ["1", "2", "3"]
    assert [row["total_cost"] for row in rows] == [0, 1, 2]
    assert (tmp_path / "metrics/run_metrics.json").exists()


def test_run_metrics_with_missing_round_is_denied(tmp_path):
    rounds = (round_record(0), round_record(1))
    with pytest.raises(ValueError, match="round records 0, 1, and 2"):
        materialize(MetricsArtifactWriter(tmp_path), rounds=rounds, run=run_record(rounds), high=2)


def test_record_identity_tampering_and_symlink_root_are_denied(tmp_path):
    record = request_record("a")
    with pytest.raises(ValidationError, match="identity"):
        RequestMetricRecord.model_validate(
            {**record.model_dump(), "effective_output_tokens": 9}, strict=True,
        )
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    with pytest.raises(ValueError, match="symlink"):
        MetricsArtifactWriter(link)


def test_request_record_preserves_unknown_linked_output_usage():
    record = request_record("unknown").model_copy(
        update={"effective_output_tokens": None}
    )
    rebuilt = RequestMetricRecord.build(
        recorded_at_utc=record.recorded_at_utc,
        run_id=record.run_id,
        attempt=record.attempt,
        source_of_applied_response=True,
        effective_output_tokens=None,
    )
    assert rebuilt.effective_output_tokens is None


def test_construction_has_no_filesystem_side_effect(tmp_path):
    MetricsArtifactWriter(tmp_path)
    assert list(tmp_path.iterdir()) == []
