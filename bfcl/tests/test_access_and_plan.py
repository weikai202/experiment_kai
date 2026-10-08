from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from bfcl_pipeline.access import DatasetGate, FinalTestAuthority
from bfcl_pipeline.accounting import AttemptUsage, LogicalRequest, UsageLedger
from bfcl_pipeline.canonical import sha256_json
from bfcl_pipeline.evaluation import (
    FinalExecutionOutcome,
    FinalCaseReceipt,
    FinalUnitArtifact,
    FinalUnitDispatch,
    NativeExecutionReceipt,
    build_final_plan,
    execute_final_plan,
)
from bfcl_pipeline.evolution import Generation, seed_generation
from bfcl_pipeline.providers import MaxTokenCalibrationReceipt, MaxTokenObservation, QwenRequestConfig
from bfcl_pipeline.reporting import artifact_accounting_summary, system_results_from_artifacts
from bfcl_pipeline.resources import GenerationResource, GenerationResourceManifest
from bfcl_pipeline.splits import build_split_manifest


def _lookup(families):
    return {case.case_id: {"id": case.case_id} for family in families for case in family.variants}


def _seed_generation():
    resource = GenerationResource.build("seed@1", "g000", "skill", "seed", ["read"])
    return seed_generation("memory-g000", GenerationResourceManifest.build("g000", (resource,)))


def _generation(generation_id: str, parent: str | None, skill_library_sha256: str) -> Generation:
    core = {
        "generation_id": generation_id,
        "parent_generation_id": parent,
        "memory_sha256": f"memory-{generation_id}",
        "skill_library_sha256": skill_library_sha256,
        "accepted_skill_versions": ("skill@1",),
    }
    return Generation(**core, generation_sha256=sha256_json(core))


def _plan(manifest, *, g000=None, g003=None):
    resource_manifest_sha256 = g003.skill_library_sha256 if g003 is not None else sha256_json("resource-manifest")
    g000 = g000 or _seed_generation()
    g003 = g003 or _generation("g003", "g002", resource_manifest_sha256)
    train_ids = manifest.rounds[0].case_ids[:2]
    # Synthetic fixture shaped like a formal live receipt; no model call occurs in this test.
    calibration = MaxTokenCalibrationReceipt.build(
        evidence_source="live_train_smoke",
        dataset_manifest_sha256=manifest.manifest_sha256,
        train_case_ids=train_ids,
        observations=tuple(MaxTokenObservation(case_id, 800, "stop") for case_id in train_ids),
    )
    return build_final_plan(
        manifest,
        g000,
        g003,
        qwen_config=QwenRequestConfig(max_tokens=calibration.chosen_max_tokens),
        calibration_receipt=calibration,
        embedding_client_config_sha256=sha256_json("embedding-config"),
        bfcl_revision="6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        environment_sha256=sha256_json("environment"),
        evaluator_sha256=sha256_json("evaluator"),
        resource_manifest_sha256=resource_manifest_sha256,
        experiment_config_sha256=sha256_json("experiment-config"),
    )


def _case_ids(plan):
    return tuple(
        f"multi_turn_{variant}_{int(fid.rsplit('_', 1)[1])}"
        for fid in plan.sealed_test_family_ids
        for variant in ("base", "miss_func", "miss_param", "long_context")
    )


def _outcome(dispatch, *, missing_usage=False):
    ledger = UsageLedger()
    request = LogicalRequest.build(
        "final-run",
        3,
        dispatch.case_id,
        0,
        "qwen",
        "Qwen/Qwen3-32B",
        dispatch.system,
        sha256_json(dispatch.case_id),
        sha256_json(dispatch.plan_sha256),
    )
    ledger.add_logical_request(request)
    attempt = AttemptUsage.build(
        request.logical_request_id,
        0,
        None if missing_usage else 10,
        None if missing_usage else 2,
        sha256_json({"case": dispatch.case_id, "system": dispatch.system}),
    )
    ledger.add_attempt(attempt)
    result = {"valid": True, "error_type": None}
    native_receipt = None if dispatch.system == "vanilla" else NativeExecutionReceipt.build(
        dispatch,
        sha256_json({"native_state": dispatch.case_id, "system": dispatch.system}),
        sha256_json(result),
    )
    return FinalExecutionOutcome(
        official_valid=True,
        failure_signature_sha256=None,
        sanitized_result=result,
        ledger=ledger,
        native_execution_receipt=native_receipt,
    )


class StepClock:
    def __init__(self, step=0.25):
        self.value = 0.0
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


def test_native_execution_receipt_strictly_binds_the_official_result(families):
    manifest = build_split_manifest(families, "revision")
    plan = _plan(manifest)
    case_id = _case_ids(plan)[0]
    dispatch = FinalUnitDispatch.build(plan, "owner", 160, "g000", "g000", case_id)
    outcome = _outcome(dispatch)
    assert outcome.native_execution_receipt is not None

    wrong_result_receipt = NativeExecutionReceipt.build(
        dispatch,
        sha256_json("native-state"),
        sha256_json({"valid": False, "error_type": "different"}),
    )
    with pytest.raises(ValueError, match="bind the official result"):
        FinalUnitArtifact.build(
            plan,
            dispatch,
            replace(outcome, native_execution_receipt=wrong_result_receipt),
            0.25,
        )
    with pytest.raises(ValueError, match="provenance hashes"):
        NativeExecutionReceipt.build(dispatch, "A" * 64, sha256_json(outcome.sanitized_result))

    artifact = FinalUnitArtifact.build(plan, dispatch, outcome, 0.25)
    wrong_native_core = asdict(outcome.native_execution_receipt)
    wrong_native_core.pop("receipt_sha256")
    wrong_native_core["official_result_sha256"] = sha256_json({"valid": False})
    wrong_native = NativeExecutionReceipt(
        **wrong_native_core,
        receipt_sha256=sha256_json(wrong_native_core),
    )
    case_receipt_core = asdict(artifact.receipt)
    case_receipt_core.pop("receipt_sha256")
    case_receipt_core["native_execution_receipt_sha256"] = wrong_native.receipt_sha256
    case_receipt = FinalCaseReceipt(
        **case_receipt_core,
        receipt_sha256=sha256_json(case_receipt_core),
    )
    artifact_core = asdict(artifact)
    artifact_core.pop("artifact_sha256")
    artifact_core.pop("receipt")
    artifact_core["native_execution_receipt"] = asdict(wrong_native)
    tampered = FinalUnitArtifact(
        **artifact_core,
        receipt=case_receipt,
        artifact_sha256=sha256_json(artifact_core),
    )
    with pytest.raises(ValueError, match="persisted official result"):
        tampered.validate(plan, dispatch)


class FakeExecutor:
    def __init__(self, missing_first=False):
        self.execute_calls = []
        self.recover_calls = []
        self.missing_first = missing_first

    def execute(self, dispatch):
        self.execute_calls.append(dispatch.dispatch_id)
        return _outcome(dispatch, missing_usage=self.missing_first and dispatch.unit_index == 0)

    def recover(self, dispatch):
        self.recover_calls.append(dispatch.dispatch_id)
        return _outcome(dispatch, missing_usage=self.missing_first and dispatch.unit_index == 0)


def _authority(tmp_path, plan, manifest, owner="owner"):
    return FinalTestAuthority(tmp_path / "test.authority", plan.plan_sha256, manifest.manifest_sha256, owner)


def test_access_gate_uses_same_owner_plan_bound_resume_lease(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    gate = DatasetGate(manifest, _lookup(families))
    with pytest.raises(PermissionError):
        gate.open("dev", "evolution")
    with pytest.raises(PermissionError, match="exactly one train round"):
        gate.open("train", "evolution")
    train_round, round_receipt = gate.open_train_round(0)
    assert len(train_round) == 160 and round_receipt.split == "train_round_0"
    dev, receipt = gate.open("dev", "skill_ab_validation")
    assert len(dev) == 160 and receipt.split == "dev"

    plan = _plan(manifest)
    authority = _authority(tmp_path, plan, manifest)
    test, _ = gate.open("test", "frozen_final_evaluation", authority=authority, plan_sha256=plan.plan_sha256)
    assert len(test) == 160
    resumed, _ = gate.open("test", "frozen_final_evaluation", authority=authority, plan_sha256=plan.plan_sha256)
    assert len(resumed) == 160
    other = _authority(tmp_path, plan, manifest, owner="other-owner")
    with pytest.raises(PermissionError, match="another plan or owner"):
        gate.open("test", "frozen_final_evaluation", authority=other, plan_sha256=plan.plan_sha256)


def test_durable_final_plan_produces_authoritative_artifacts_and_reporting_rows(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    plan = _plan(manifest)
    authority = _authority(tmp_path, plan, manifest)
    DatasetGate(manifest, _lookup(families)).open("test", "frozen_final_evaluation", authority=authority, plan_sha256=plan.plan_sha256)
    executor = FakeExecutor()
    rows = execute_final_plan(
        plan,
        _case_ids(plan),
        executor,
        state_root=tmp_path / "state",
        authority=authority,
        clock=StepClock(),
    )
    assert len(rows) == 3 and all(len(group) == 160 for group in rows)
    artifacts = tuple(artifact for group in rows for artifact in group)
    assert len(executor.execute_calls) == 480
    assert all(isinstance(artifact, FinalUnitArtifact) for artifact in artifacts)
    system_rows = system_results_from_artifacts(plan, authority.owner_id, artifacts)
    assert len(system_rows) == 480 and all(row.valid for row in system_rows)
    summary = artifact_accounting_summary(artifacts, rows.system_run_receipts)
    assert summary["usage_complete"] and summary["total_tokens"] == 480 * 12
    assert summary["direct_latency_seconds"] == sum(
        receipt.direct_latency_seconds for receipt in rows.system_run_receipts
    )
    assert set(summary["direct_latency_seconds_by_system"]) == {"vanilla", "g000", "g003"}
    assert summary["unit_latency_seconds_diagnostic"] == 480 * 0.25

    receipt_path = tmp_path / "state" / "system_runs" / "00.receipt.json"
    envelope = json.loads(receipt_path.read_text())
    envelope["payload"]["direct_latency_seconds"] = -1.0
    receipt_path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash mismatch"):
        execute_final_plan(plan, _case_ids(plan), executor, state_root=tmp_path / "state", authority=authority)


def test_final_runner_resumes_saved_prefix_without_rerunning_it(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    plan = _plan(manifest)
    authority = _authority(tmp_path, plan, manifest)
    DatasetGate(manifest, _lookup(families)).open("test", "frozen_final_evaluation", authority=authority, plan_sha256=plan.plan_sha256)
    executor = FakeExecutor()

    def crash(index):
        if index == 4:
            raise RuntimeError("crash after saved prefix")

    with pytest.raises(RuntimeError, match="saved prefix"):
        execute_final_plan(plan, _case_ids(plan), executor, state_root=tmp_path / "state", authority=authority, after_unit=crash)
    assert len(executor.execute_calls) == 5
    rows = execute_final_plan(plan, _case_ids(plan), executor, state_root=tmp_path / "state", authority=authority)
    assert sum(len(group) for group in rows) == 480
    assert len(executor.execute_calls) == 480
    assert len(set(executor.execute_calls)) == 480


def test_system_run_recovers_after_last_unit_before_run_completion(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    plan = _plan(manifest)
    authority = _authority(tmp_path, plan, manifest)
    DatasetGate(manifest, _lookup(families)).open(
        "test", "frozen_final_evaluation", authority=authority, plan_sha256=plan.plan_sha256
    )
    executor = FakeExecutor()

    def crash(index):
        if index == 159:
            raise RuntimeError("crash before durable system completion")

    with pytest.raises(RuntimeError, match="system completion"):
        execute_final_plan(
            plan,
            _case_ids(plan),
            executor,
            state_root=tmp_path / "state",
            authority=authority,
            after_unit=crash,
            clock=StepClock(),
            wall_clock=lambda: 1000.0,
            boot_id="boot-a",
        )
    result = execute_final_plan(
        plan,
        _case_ids(plan),
        executor,
        state_root=tmp_path / "state",
        authority=authority,
        clock=StepClock(),
        wall_clock=lambda: 1010.0,
        boot_id="boot-b",
    )
    assert len(executor.execute_calls) == 480
    assert len(set(executor.execute_calls)) == 480
    assert result.system_run_receipts[0].direct_latency_seconds == 10.0
    assert (tmp_path / "state" / "system_runs" / "00.complete.json").exists()


class UnknownOutcomeExecutor(FakeExecutor):
    def __init__(self):
        super().__init__()
        self.external = {}

    def execute(self, dispatch):
        self.execute_calls.append(dispatch.dispatch_id)
        if dispatch.unit_index == 0 and dispatch.dispatch_id not in self.external:
            self.external[dispatch.dispatch_id] = _outcome(dispatch)
            raise RuntimeError("crash after official evaluator completion")
        return _outcome(dispatch)

    def recover(self, dispatch):
        self.recover_calls.append(dispatch.dispatch_id)
        return self.external[dispatch.dispatch_id]


def test_final_runner_reconciles_unknown_current_unit_without_redispatch(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    plan = _plan(manifest)
    authority = _authority(tmp_path, plan, manifest)
    DatasetGate(manifest, _lookup(families)).open("test", "frozen_final_evaluation", authority=authority, plan_sha256=plan.plan_sha256)
    executor = UnknownOutcomeExecutor()
    with pytest.raises(RuntimeError, match="official evaluator"):
        execute_final_plan(
            plan,
            _case_ids(plan),
            executor,
            state_root=tmp_path / "state",
            authority=authority,
            clock=StepClock(),
            wall_clock=lambda: 1000.0,
            boot_id="boot-a",
        )
    rows = execute_final_plan(
        plan,
        _case_ids(plan),
        executor,
        state_root=tmp_path / "state",
        authority=authority,
        clock=StepClock(),
        wall_clock=lambda: 1010.0,
        boot_id="boot-b",
    )
    assert sum(len(group) for group in rows) == 480
    assert executor.recover_calls == [executor.execute_calls[0]]
    assert executor.execute_calls.count(executor.execute_calls[0]) == 1
    assert rows[0][0].accounting["direct_latency_seconds"] == 10.0


def test_final_runner_rejects_tampered_durable_timing_anchor(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    plan = _plan(manifest)
    authority = _authority(tmp_path, plan, manifest)
    DatasetGate(manifest, _lookup(families)).open(
        "test", "frozen_final_evaluation", authority=authority, plan_sha256=plan.plan_sha256
    )
    executor = UnknownOutcomeExecutor()
    state_root = tmp_path / "state"
    with pytest.raises(RuntimeError, match="official evaluator"):
        execute_final_plan(
            plan,
            _case_ids(plan),
            executor,
            state_root=state_root,
            authority=authority,
            clock=StepClock(),
            wall_clock=lambda: 1000.0,
            boot_id="boot-a",
        )
    path = state_root / "active_dispatch.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["started_wall_time"] = 0.0
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash mismatch"):
        execute_final_plan(
            plan,
            _case_ids(plan),
            executor,
            state_root=state_root,
            authority=authority,
            boot_id="boot-b",
        )


def test_final_artifact_tampering_and_missing_usage_are_explicit(families, tmp_path):
    manifest = build_split_manifest(families, "revision")
    plan = _plan(manifest)
    authority = _authority(tmp_path, plan, manifest)
    DatasetGate(manifest, _lookup(families)).open("test", "frozen_final_evaluation", authority=authority, plan_sha256=plan.plan_sha256)
    rows = execute_final_plan(plan, _case_ids(plan), FakeExecutor(missing_first=True), state_root=tmp_path / "state", authority=authority)
    artifacts = tuple(artifact for group in rows for artifact in group)
    summary = artifact_accounting_summary(artifacts, rows.system_run_receipts)
    assert not summary["usage_complete"]
    assert summary["total_tokens"] is None

    path = tmp_path / "state" / "units" / "0000.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["official_valid"] = False
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash mismatch"):
        execute_final_plan(plan, _case_ids(plan), FakeExecutor(), state_root=tmp_path / "state", authority=authority)


def test_final_plan_rejects_unbound_config_and_synthetic_calibration(families):
    manifest = build_split_manifest(families, "revision")
    plan = _plan(manifest)
    bad_plan = replace(plan, evaluator_sha256="changed")
    executor = FakeExecutor()
    with pytest.raises(ValueError, match="plan identity"):
        execute_final_plan(
            bad_plan,
            _case_ids(plan),
            executor,
            state_root=pytest.Path if False else None,  # type: ignore[arg-type]
            authority=None,  # type: ignore[arg-type]
        )

    train_id = manifest.rounds[0].case_ids[0]
    synthetic = MaxTokenCalibrationReceipt.build(
        evidence_source="synthetic_test",
        dataset_manifest_sha256=manifest.manifest_sha256,
        train_case_ids=(train_id,),
        observations=(MaxTokenObservation(train_id, 10, "stop"),),
    )
    with pytest.raises(ValueError, match="live train-smoke"):
        build_final_plan(
            manifest,
            _seed_generation(),
            _generation("g003", "g002", sha256_json("resource-manifest")),
            qwen_config=QwenRequestConfig(max_tokens=synthetic.chosen_max_tokens),
            calibration_receipt=synthetic,
            embedding_client_config_sha256=sha256_json("embedding-config"),
            bfcl_revision="6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
            environment_sha256=sha256_json("environment"),
            evaluator_sha256=sha256_json("evaluator"),
            resource_manifest_sha256=sha256_json("resource-manifest"),
            experiment_config_sha256=sha256_json("experiment-config"),
        )

    seed = _seed_generation()
    with pytest.raises(ValueError, match="G000 root"):
        _plan(
            manifest,
            g000=_generation("g000", "g999", seed.skill_library_sha256),
        )
    with pytest.raises(ValueError, match="G000 root"):
        _plan(
            manifest,
            g003=_generation("g003", "g001", sha256_json("resource-manifest")),
        )
