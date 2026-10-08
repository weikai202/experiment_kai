from __future__ import annotations

import pytest

from bfcl_pipeline.accounting import AttemptUsage, CommittedEffect, LogicalRequest, ResponseApplication, UsageLedger
from bfcl_pipeline.checkpoint import CheckpointStore
from bfcl_pipeline.evolution import ThreeRoundCoordinator, seed_generation
from bfcl_pipeline.providers import (
    MaxTokenCalibrationReceipt,
    MaxTokenObservation,
    QwenRequestConfig,
    calibration_from_payload,
    calibration_to_payload,
    verify_nonthinking_request,
)
from bfcl_pipeline.resources import GenerationResource, GenerationResourceManifest
from bfcl_pipeline.splits import build_split_manifest


def add_request_chain(ledger: UsageLedger, *, run_id: str, request_suffix: str, provider: str = "qwen", output_tokens: int | None = 4, effect_kind: str = "SKILL_UPDATE") -> tuple[LogicalRequest, AttemptUsage, ResponseApplication, CommittedEffect]:
    request = LogicalRequest.build(run_id, 0, f"case-{request_suffix}", 0, provider, "Qwen/Qwen3-32B" if provider == "qwen" else "text-embedding-3-small", "skill_rewrite", "i" * 64, "c" * 64)
    ledger.add_logical_request(request)
    attempt = AttemptUsage.build(request.logical_request_id, 0, None if output_tokens is None else 10, output_tokens, f"response-{request_suffix}")
    ledger.add_attempt(attempt)
    application = ResponseApplication.build(request.logical_request_id, attempt.attempt_id, attempt.response_sha256, "offline_update", True)
    ledger.apply_response(application)
    effect = CommittedEffect.build(effect_kind, "before", "after" if effect_kind != "NONE" else "before", (application.application_id,))
    ledger.commit_effect(effect)
    return request, attempt, application, effect


def test_effective_cost_counts_only_accepted_qwen_substantive_outputs():
    ledger = UsageLedger()
    add_request_chain(ledger, run_id="run", request_suffix="1", output_tokens=4)
    add_request_chain(ledger, run_id="run", request_suffix="2", output_tokens=3, effect_kind="NONE")
    add_request_chain(ledger, run_id="run", request_suffix="3", provider="embedding", output_tokens=0)
    snapshot = ledger.snapshot(2.5)
    assert snapshot.total_tokens == 37
    assert snapshot.qwen_effective_output_tokens == 4
    assert len(snapshot.committed_effect_ids) == 2
    assert UsageLedger.from_payload(ledger.to_payload()).ledger_sha256 == ledger.ledger_sha256


def test_qwen_is_explicitly_nonthinking():
    calibration = MaxTokenCalibrationReceipt.build(
        evidence_source="synthetic_test",
        dataset_manifest_sha256="d" * 64,
        train_case_ids=("train-case",),
        observations=(MaxTokenObservation("train-case", 32, "stop"),),
    )
    kwargs = QwenRequestConfig(max_tokens=64).as_openai_kwargs(calibration)
    verify_nonthinking_request(kwargs)
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert calibration_from_payload(calibration_to_payload(calibration)) == calibration


def test_calibration_loader_rejects_self_hashed_wrong_base_configuration():
    calibration = MaxTokenCalibrationReceipt.build(
        evidence_source="synthetic_test",
        dataset_manifest_sha256="d" * 64,
        train_case_ids=("train-case",),
        observations=(MaxTokenObservation("train-case", 32, "stop"),),
    )
    payload = calibration_to_payload(calibration)
    payload["base_request_config_sha256"] = "0" * 64
    core = dict(payload)
    core.pop("receipt_sha256")
    from bfcl_pipeline.canonical import sha256_json
    payload["receipt_sha256"] = sha256_json(core)
    with pytest.raises(ValueError, match="base request configuration"):
        calibration_from_payload(payload)


def test_missing_usage_is_explicit_and_blocks_exact_effective_cost():
    ledger = UsageLedger()
    add_request_chain(ledger, run_id="run", request_suffix="missing", output_tokens=None)
    snapshot = ledger.snapshot(1.0)
    assert not snapshot.usage_complete and snapshot.total_tokens is None
    assert snapshot.qwen_effective_output_tokens is None


def seed():
    resource = GenerationResource.build("seed@1", "g000", "skill", "seed", ["read"])
    return seed_generation("memory0", GenerationResourceManifest.build("g000", (resource,)))




class CounterClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 1.0
        return self.value


def next_state():
    return {
        "memory_sha256": "memory1",
        "skill_library_sha256": "skills1",
        "accepted_skill_versions": ("skill@1",),
    }


class RecordingOperation:
    def __init__(self, run_id, calls):
        self.run_id = run_id
        self.calls = calls

    def execute(self, dispatch, current, ledger, state):
        self.calls.append(dispatch.case_id)
        add_request_chain(ledger, run_id=self.run_id, request_suffix=dispatch.case_id, output_tokens=7)
        return next_state()

    def recover(self, dispatch, current, ledger, state):
        raise AssertionError("No in-flight dispatch should require reconciliation")


def operation_for(run_id, calls):
    return RecordingOperation(run_id, calls)


class NoDispatchOperation:
    def execute(self, *args):
        raise AssertionError("redispatched")

    def recover(self, *args):
        raise AssertionError("reconciled after finalized accounting")


def test_three_round_contract_publishes_next_generation(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    clock = CounterClock()
    calls = []
    coordinator = ThreeRoundCoordinator(manifest, CheckpointStore(tmp_path), clock=clock)
    generation = seed()
    product = coordinator.run_round("run", 0, generation, operation_for("run", calls))
    assert product.generation.generation_id == "g001"
    assert product.accounting.qwen_effective_output_tokens == 160 * 7
    assert CheckpointStore(tmp_path).recovery_stage("run", 0) == "accounting_finalized"
    complete = CheckpointStore(tmp_path).load("run", 0, "complete")
    finalized = CheckpointStore(tmp_path).load("run", 0, "accounting_finalized")
    assert complete is not None and finalized is not None
    assert finalized["material"]["complete_payload_sha256"] == complete["payload_sha256"]
    recovered = coordinator.run_round("run", 0, generation, NoDispatchOperation())
    assert recovered == product and tuple(calls) == manifest.rounds[0].case_ids


def test_finalized_accounting_rejects_a_validly_hashed_wrong_complete_binding(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    store = CheckpointStore(tmp_path)
    generation = seed()
    ThreeRoundCoordinator(manifest, store, clock=CounterClock()).run_round(
        "run", 0, generation, operation_for("run", [])
    )
    finalized = store.load("run", 0, "accounting_finalized")
    assert finalized is not None
    material = dict(finalized["material"])
    material["complete_payload_sha256"] = "0" * 64
    store.persist(
        "run",
        0,
        "accounting_finalized",
        finalized["generation_id"],
        tuple(finalized["completed_case_ids"]),
        material,
    )
    with pytest.raises(ValueError, match="bind the complete checkpoint"):
        ThreeRoundCoordinator(manifest, store, clock=CounterClock()).run_round(
            "run", 0, generation, NoDispatchOperation()
        )


@pytest.mark.parametrize("crash_stage", ["round_started", "effects_committed", "generation_published", "durable_completion", "complete", "accounting_finalized"])
def test_crash_after_every_durable_stage_resumes_without_redispatching_completed_work(families, tmp_path, crash_stage):
    manifest = build_split_manifest(families, "revision")
    store = CheckpointStore(tmp_path / crash_stage)
    generation = seed()
    clock = CounterClock()
    calls = []

    def crash(stage):
        if stage == crash_stage:
            raise RuntimeError("simulated crash")

    first = ThreeRoundCoordinator(manifest, store, clock=clock, after_stage=crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        first.run_round(f"run-{crash_stage}", 0, generation, operation_for(f"run-{crash_stage}", calls))
    resumed = ThreeRoundCoordinator(manifest, store, clock=clock).run_round(
        f"run-{crash_stage}", 0, generation, operation_for(f"run-{crash_stage}", calls)
    )
    assert resumed.generation.generation_id == "g001"
    assert tuple(calls) == manifest.rounds[0].case_ids


def test_crash_after_intermediate_case_resumes_from_durable_prefix(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    store = CheckpointStore(tmp_path / "mid-case")
    generation = seed()
    calls = []
    crashed = False

    def crash_after_tenth_case(stage):
        nonlocal crashed
        progress = store.load("run-mid-case", 0, "episodes_progress")
        if stage == "case_completed" and not crashed and progress is not None and len(progress["completed_case_ids"]) == 10:
            crashed = True
            raise RuntimeError("simulated intermediate crash")

    coordinator = ThreeRoundCoordinator(manifest, store, clock=CounterClock(), after_stage=crash_after_tenth_case)
    with pytest.raises(RuntimeError, match="intermediate crash"):
        coordinator.run_round("run-mid-case", 0, generation, operation_for("run-mid-case", calls))
    assert tuple(calls) == manifest.rounds[0].case_ids[:10]

    resumed = ThreeRoundCoordinator(manifest, store, clock=CounterClock()).run_round(
        "run-mid-case", 0, generation, operation_for("run-mid-case", calls)
    )
    assert resumed.completed_case_ids == manifest.rounds[0].case_ids
    assert tuple(calls) == manifest.rounds[0].case_ids


class UnknownOutcomeOperation(RecordingOperation):
    def __init__(self, run_id, calls, external_outcomes):
        super().__init__(run_id, calls)
        self.external_outcomes = external_outcomes
        self.reconciled = []

    def execute(self, dispatch, current, ledger, state):
        self.calls.append(dispatch.case_id)
        if not self.external_outcomes:
            self.external_outcomes[dispatch.dispatch_id] = dispatch.case_id
            raise RuntimeError("crash after external side effect")
        add_request_chain(ledger, run_id=self.run_id, request_suffix=dispatch.case_id, output_tokens=7)
        return next_state()

    def recover(self, dispatch, current, ledger, state):
        case_id = self.external_outcomes.get(dispatch.dispatch_id)
        if case_id != dispatch.case_id:
            raise RuntimeError("unknown outcome cannot be reconciled")
        self.reconciled.append(dispatch.case_id)
        add_request_chain(ledger, run_id=self.run_id, request_suffix=dispatch.case_id, output_tokens=7)
        return next_state()


def test_inflight_unknown_outcome_is_reconciled_not_redispatched(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    store = CheckpointStore(tmp_path / "unknown-outcome")
    generation = seed()
    calls = []
    external_outcomes = {}
    operation = UnknownOutcomeOperation("run-unknown", calls, external_outcomes)

    with pytest.raises(RuntimeError, match="external side effect"):
        ThreeRoundCoordinator(manifest, store, clock=CounterClock()).run_round(
            "run-unknown", 0, generation, operation
        )
    claim = store.load("run-unknown", 0, "case_dispatch")
    assert claim is not None and claim["completed_case_ids"] == []
    first_case = manifest.rounds[0].case_ids[0]
    assert calls == [first_case]

    product = ThreeRoundCoordinator(manifest, store, clock=CounterClock()).run_round(
        "run-unknown", 0, generation, operation
    )
    assert product.completed_case_ids == manifest.rounds[0].case_ids
    assert operation.reconciled == [first_case]
    assert calls.count(first_case) == 1
    assert tuple(calls) == manifest.rounds[0].case_ids


def test_latency_stops_after_complete_state_is_durable(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    store = CheckpointStore(tmp_path / "latency")
    generation = seed()
    clock = CounterClock()

    def observe(stage):
        if stage == "complete":
            assert store.load("run-latency", 0, "complete") is not None
            clock.value = 100.0

    product = ThreeRoundCoordinator(manifest, store, clock=clock, after_stage=observe).run_round(
        "run-latency", 0, generation, operation_for("run-latency", [])
    )
    assert product.accounting.direct_latency_seconds == 100.0
