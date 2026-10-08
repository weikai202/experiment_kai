import json

import pytest

from tau3_evolution.canonical import atomic_write_json, canonical_sha256
from tau3_evolution.failure_analysis import (
    EvaluationRecord,
    trusted_trace_from_native_receipt,
)
from tau3_evolution.final_evaluation import (
    ExecutionBinding,
    FinalEvaluationPlan,
    FinalTestOnceGuard,
    TaskTrialCluster,
)
from tau3_evolution.final_runner import FinalUnitMaterial, ResumableFinalRunner
from tau3_evolution.ledger import (
    DurableAccountingLedger,
    LogicalRequest,
    OutputApplication,
    PhysicalAttempt,
    SubstantiveEffect,
)


def trace_generation_document(generation_id, *, forged=False):
    if generation_id is None:
        payload = {"kind": "no_generation", "generation_id": None}
    else:
        payload = {
            "generation_id": generation_id,
            "parent_generation_id": None if generation_id == "g000" else "g002",
            "memories": [],
            "skills": [],
            "vectors": [],
            "seed_library_sha256": "sha256:seed-library",
        }
    if forged:
        payload["forged"] = True
    return {**payload, "state_sha256": canonical_sha256(payload)}


class Clock:
    def __init__(self):
        self.value = 0

    def monotonic_ns(self):
        self.value += 500_000_000
        return self.value

    def time_ns(self):
        return 1_000_000_000 + self.value


class Executor:
    def __init__(self, root, crash_after_effect=False, wrong_generation_state=False):
        self.root = root
        self.calls = []
        self.recover_calls = []
        self.crash_after_effect = crash_after_effect
        self.wrong_generation_state = wrong_generation_state
        self.crashed = False
        self.receipts = {}

    def execute(self, **kwargs):
        self.calls.append((kwargs["system_id"], kwargs["simulator_seed"]))
        artifact = f"sha256:{kwargs['system_id']}-{kwargs['simulator_seed']}"
        scope = kwargs["accounting_scope_id"]
        path = self.root / f"ledger-{kwargs['system_id']}-{kwargs['simulator_seed']}.json"
        ledger = DurableAccountingLedger(path)
        logical = LogicalRequest(
            f"logical:{scope}",
            scope,
            "qwen",
            "executed_action",
            "sha256:request",
            "Qwen/Qwen3-32B",
            plan().execution_binding.qwen_decoding_sha256,
        )
        attempt = PhysicalAttempt(
            f"attempt:{scope}", logical.logical_request_id, 2, 3, True, "sha256:response"
        )
        effect = SubstantiveEffect(
            f"effect:{scope}", scope, "executed_action", artifact, True, True
        )
        application = OutputApplication(
            f"application:{scope}",
            logical.logical_request_id,
            attempt.attempt_id,
            effect.effect_id,
            "executed_action",
            True,
            0,
        )
        ledger.record_request(logical)
        ledger.record_attempt(attempt)
        ledger.record_effect(effect)
        ledger.record_application(application)
        accounting = ledger.snapshot(scope_id=scope, checkpoint_material_sha256=artifact)
        generation_id = kwargs["generation_id"]
        generation_document = trace_generation_document(
            generation_id,
            forged=self.wrong_generation_state and kwargs["system_id"] == "generation_0",
        )
        evaluator_sha = canonical_sha256(
            {
                "system_id": kwargs["system_id"],
                "task_id": kwargs["cluster"].task_id,
                "seed": kwargs["simulator_seed"],
                "reward": 1.0,
            }
        )
        trace_receipt = {
            "episode_id": kwargs["unit_dispatch_id"],
            "domain": kwargs["cluster"].domain,
            "task_id": kwargs["cluster"].task_id,
            "simulator_seed": kwargs["simulator_seed"],
            "system_id": kwargs["system_id"],
            "generation_id": generation_id,
            "evaluator_record_sha256": evaluator_sha,
            "reward": 1.0,
            "actions": [],
            "generation_state_document": generation_document,
            "accounting_ledger_document": ledger.document,
        }
        trace = trusted_trace_from_native_receipt(trace_receipt)
        evaluation = EvaluationRecord(
            kwargs["system_id"],
            kwargs["cluster"].domain,
            kwargs["cluster"].task_id,
            kwargs["simulator_seed"],
            1.0,
            kwargs["manifest_position"],
            trace.trajectory_sha256,
            evaluator_sha,
        )
        material = FinalUnitMaterial(
            evaluation,
            accounting,
            artifact,
            str(path),
            kwargs["execution_binding_sha256"],
            trace_receipt,
        )
        self.receipts[kwargs["unit_dispatch_id"]] = material
        if self.crash_after_effect and not self.crashed:
            self.crashed = True
            raise RuntimeError("crash after external side effect")
        return material

    def recover(self, **kwargs):
        self.recover_calls.append(kwargs["unit_dispatch_id"])
        return self.receipts.get(kwargs["unit_dispatch_id"])


def plan():
    max_tokens = 192
    qwen_config = canonical_sha256(
        {
            "model": "Qwen/Qwen3-32B",
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    binding = ExecutionBinding(
        output_limit_calibration_sha256="sha256:calibration",
        seed_library_sha256="sha256:seed-library",
        qwen_model="Qwen/Qwen3-32B",
        qwen_max_tokens=max_tokens,
        qwen_decoding_sha256=qwen_config,
        user_simulator_model="gpt-4o-mini-2024-07-18",
        user_simulator_config_sha256="sha256:user-config",
        embedding_model="text-embedding-3-small",
        embedding_config_sha256=canonical_sha256(
            {"model": "text-embedding-3-small", "encoding_format": "float", "fallback": None}
        ),
        native_runtime_sha256="sha256:runtime",
        evaluator_sha256="sha256:evaluator",
        vanilla_generation_sha256=trace_generation_document(None)["state_sha256"],
        generation_0_sha256=trace_generation_document("g000")["state_sha256"],
        updated_generation_sha256=trace_generation_document("g003")["state_sha256"],
    )
    return FinalEvaluationPlan(
        "sha256:evolution",
        "sha256:test",
        "g000",
        "g003",
        ("vanilla", "generation_0", "updated"),
        (TaskTrialCluster("airline", "task", (0, 1)),),
        binding,
    )


def test_fixed_order_three_system_report_and_accounting(tmp_path):
    executor = Executor(tmp_path)
    clock = Clock()
    guard = FinalTestOnceGuard(tmp_path / "global-guard.json")
    report = ResumableFinalRunner(
        tmp_path / "final",
        executor,
        guard,
        time_ns=clock.time_ns,
        monotonic_ns=clock.monotonic_ns,
        boot_id="boot",
    ).run(plan())
    assert executor.calls == [
        ("vanilla", 0),
        ("vanilla", 1),
        ("generation_0", 0),
        ("generation_0", 1),
        ("updated", 0),
        ("updated", 1),
    ]
    assert [row.system_id for row in report.systems] == [
        "vanilla",
        "generation_0",
        "updated",
    ]
    assert all(row.total_tokens == 10 and row.total_cost == 6 for row in report.systems)
    assert all(row.total_running_time_seconds == 2.5 for row in report.systems)
    state = json.loads((tmp_path / "final" / "execution.json").read_text())
    assert all(
        row["latency_clock"] == "monotonic_same_boot" for row in state["system_timings"].values()
    )


def test_rejects_self_hashed_trace_generation_outside_execution_binding(tmp_path):
    executor = Executor(tmp_path, wrong_generation_state=True)
    runner = ResumableFinalRunner(
        tmp_path / "final", executor, FinalTestOnceGuard(tmp_path / "global-guard.json")
    )
    with pytest.raises(ValueError, match="generation state does not match execution binding"):
        runner.run(plan())
    assert executor.calls.count(("generation_0", 0)) == 1


def test_crash_after_external_effect_recovers_without_redispatch(tmp_path):
    executor = Executor(tmp_path, crash_after_effect=True)
    root = tmp_path / "final"
    guard = FinalTestOnceGuard(tmp_path / "global-guard.json")
    with pytest.raises(RuntimeError, match="external side effect"):
        ResumableFinalRunner(root, executor, guard).run(plan())
    report = ResumableFinalRunner(root, executor, guard).run(plan())
    assert executor.calls.count(("vanilla", 0)) == 1
    assert len(executor.calls) == 6
    assert len(executor.recover_calls) == 1
    assert report.report_sha256


def test_claimed_unknown_outcome_refuses_sealed_test_redispatch(tmp_path):
    executor = Executor(tmp_path)
    root = tmp_path / "final"
    guard = FinalTestOnceGuard(tmp_path / "global-guard.json")
    state = {
        "plan_sha256": plan().plan_sha256,
        "system_order": ["vanilla", "generation_0", "updated"],
        "completed_positions": [],
        "active_dispatch": {
            "position": 0,
            "unit_dispatch_id": canonical_sha256(
                {
                    "plan_sha256": plan().plan_sha256,
                    "execution_binding_sha256": plan().execution_binding.binding_sha256,
                    "position": 0,
                    "system_id": "vanilla",
                    "domain": "airline",
                    "task_id": "task",
                    "simulator_seed": 0,
                }
            ),
            "system_id": "vanilla",
            "domain": "airline",
            "task_id": "task",
            "simulator_seed": 0,
            "started_at_unix_ns": 1,
            "started_monotonic_ns": 1,
            "boot_id": "boot",
        },
        "status": "running",
    }
    from tau3_evolution.canonical import atomic_write_json

    atomic_write_json(root / "execution.json", state)
    with pytest.raises(RuntimeError, match="refusing sealed-test redispatch"):
        ResumableFinalRunner(root, executor, guard).run(plan())
    assert executor.calls == []


def test_completed_final_report_is_idempotently_loaded(tmp_path):
    executor = Executor(tmp_path)
    runner = ResumableFinalRunner(
        tmp_path / "final", executor, FinalTestOnceGuard(tmp_path / "global-guard.json")
    )
    first = runner.run(plan())
    second = runner.run(plan())
    assert first == second and len(executor.calls) == 6


def test_completed_recovery_rejects_deleted_unit_result(tmp_path):
    root = tmp_path / "final"
    guard = FinalTestOnceGuard(tmp_path / "global-guard.json")
    runner = ResumableFinalRunner(root, Executor(tmp_path / "ledgers"), guard)
    runner.run(plan())
    (root / "units" / "000000.json").unlink()
    with pytest.raises(ValueError, match="unit result set is incomplete"):
        runner.run(plan())


def test_completed_recovery_rejects_tampered_unit_result(tmp_path):
    root = tmp_path / "final"
    guard = FinalTestOnceGuard(tmp_path / "global-guard.json")
    runner = ResumableFinalRunner(root, Executor(tmp_path / "ledgers"), guard)
    runner.run(plan())
    unit_path = root / "units" / "000000.json"
    payload = json.loads(unit_path.read_text())
    payload["direct_latency_seconds"] += 1.0
    atomic_write_json(unit_path, payload)
    with pytest.raises(ValueError, match="unit content hash mismatch"):
        runner.run(plan())


def test_completed_recovery_rejects_guard_report_mismatch(tmp_path):
    root = tmp_path / "final"
    guard_path = tmp_path / "global-guard.json"
    guard = FinalTestOnceGuard(guard_path)
    runner = ResumableFinalRunner(root, Executor(tmp_path / "ledgers"), guard)
    runner.run(plan())
    payload = json.loads(guard_path.read_text())
    payload["report_sha256"] = "sha256:tampered"
    atomic_write_json(guard_path, payload)
    with pytest.raises(RuntimeError, match="guard report mismatch"):
        runner.run(plan())


def test_global_once_guard_denies_a_second_output_root(tmp_path):
    guard = FinalTestOnceGuard(tmp_path / "global-guard.json")
    ResumableFinalRunner(tmp_path / "first", Executor(tmp_path / "first-ledgers"), guard).run(
        plan()
    )
    with pytest.raises(RuntimeError, match="another execution"):
        ResumableFinalRunner(tmp_path / "second", Executor(tmp_path / "second-ledgers"), guard).run(
            plan()
        )


def test_final_execution_binding_rejects_decoding_drift():
    valid = plan().execution_binding
    values = dict(valid.__dict__)
    values["qwen_decoding_sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="decoding binding"):
        ExecutionBinding(**values)


def test_final_execution_binding_rejects_non_sentinel_vanilla_state():
    valid = plan().execution_binding
    values = dict(valid.__dict__)
    values["vanilla_generation_sha256"] = "sha256:forged"
    with pytest.raises(ValueError, match="no-generation sentinel"):
        ExecutionBinding(**values)
