import json

import pytest
from test_manifests import official_fixture

from tau3_evolution import TAU_COMMIT
from tau3_evolution.accounting import UsageAttempt
from tau3_evolution.canonical import canonical_sha256
from tau3_evolution.checkpoint import RoundCheckpointStore, RoundMaterial
from tau3_evolution.evolution import EvolutionRunner, RoundExecution
from tau3_evolution.final_evaluation import FinalTestOnceGuard, build_final_plan
from tau3_evolution.ledger import (
    DurableAccountingLedger,
    LogicalRequest,
    OutputApplication,
    PhysicalAttempt,
    SubstantiveEffect,
)
from tau3_evolution.manifests import build_manifests
from tau3_evolution.model_boundary import CalibrationSample, OutputLimitCalibration
from tau3_evolution.seed_library import compile_seed_library


class Engine:
    def __init__(self, root):
        self.root = root
        self.calls = []

    def execute_round(self, **kwargs):
        self.calls.append(kwargs)
        index = kwargs["round_index"]
        scope = f"run:round:{index}"
        path = self.root / f"ledger-{index}.json"
        ledger = DurableAccountingLedger(path)
        request = LogicalRequest(
            f"logical-{index}",
            scope,
            "qwen",
            "offline_skill_update",
            "sha256:request",
            "Qwen/Qwen3-32B",
            "sha256:config",
        )
        attempt = PhysicalAttempt(
            f"attempt-{index}", request.logical_request_id, 10, 4, True, "sha256:response"
        )
        effect = SubstantiveEffect(f"effect-{index}", scope, "skill", "sha256:artifact", True, True)
        application = OutputApplication(
            f"application-{index}",
            request.logical_request_id,
            attempt.attempt_id,
            effect.effect_id,
            "skill_update",
            True,
            0,
        )
        ledger.record_request(request)
        ledger.record_attempt(attempt)
        ledger.record_effect(effect)
        ledger.record_application(application)
        keys = tuple(f"{ref.domain}:{ref.task_id}" for ref in kwargs["train_tasks"])
        return RoundExecution(
            keys,
            (effect.effect_id,),
            (
                UsageAttempt(
                    attempt.attempt_id, request.logical_request_id, "qwen", 10, 4, effect.effect_id
                ),
            ),
            (f"sha256:artifact-{index}",),
            (("skill", index + 1),),
            (f"sha256:lineage-{index}",),
            str(path),
            scope,
            (index * 4 + 1) * 1_000_000_000,
        )


def test_exact_three_round_chain_ledger_accounting_and_recovery(tmp_path):
    manifest, _, _ = build_manifests(official_fixture())
    engine = Engine(tmp_path)
    completion_times = iter([3_000_000_000, 8_000_000_000, 13_000_000_000])
    observed_completion_times = iter([2_000_000_000, 6_000_000_000, 10_000_000_000])
    boundary_calls = []

    def durable_boundary(path):
        document = json.loads(path.read_text())
        assert document["status"] == "material_complete"
        boundary_calls.append(path)
        return next(completion_times)

    runner = EvolutionRunner(
        manifest=manifest,
        checkpoints=RoundCheckpointStore(
            tmp_path / "checkpoints", receipt_boundary_ns=durable_boundary
        ),
        engine=engine,
        wall_time_ns=lambda: next(observed_completion_times),
    )
    first = runner.run("run")
    assert [(row.input_generation_id, row.output_generation_id) for row in first] == [
        ("g000", "g001"),
        ("g001", "g002"),
        ("g002", "g003"),
    ]
    assert [row.accounting.total_running_time_seconds for row in first] == [2.0, 3.0, 4.0]
    assert [row.accounting.total_cost for row in first] == [4, 4, 4]
    assert [len(call["train_tasks"]) for call in engine.calls] == [48, 48, 48]
    assert len(boundary_calls) == 3
    recovered = runner.run("run")
    assert all(row.recovered for row in recovered)
    assert len(engine.calls) == 3


def test_staged_material_recovers_without_redispatch(tmp_path):
    manifest, _, _ = build_manifests(official_fixture())
    store = RoundCheckpointStore(
        tmp_path / "checkpoints", receipt_boundary_ns=lambda path: 3_000_000_000
    )
    ledger_path = tmp_path / "ledger.json"
    ledger = DurableAccountingLedger(ledger_path)
    scope = "run:round:0"
    request = LogicalRequest(
        "logical", scope, "qwen", "policy", "sha256:request", "Qwen/Qwen3-32B", "sha256:config"
    )
    attempt = PhysicalAttempt("attempt", "logical", 1, 2, True, "sha256:response")
    effect = SubstantiveEffect("effect", scope, "executed_action", "sha256:a", True, True)
    application = OutputApplication("app", "logical", "attempt", "effect", "action", True, 0)
    ledger.record_request(request)
    ledger.record_attempt(attempt)
    ledger.record_effect(effect)
    ledger.record_application(application)
    refs = tuple(
        f"{domain}:{task}"
        for domain, split in manifest["domains"].items()
        for task in split["rounds"][0]
    )
    material = RoundMaterial(
        "run",
        0,
        "g000",
        "g001",
        refs,
        ("effect",),
        str(ledger_path),
        ledger.ledger_sha256,
        (),
        1_000_000_000,
    )
    store.stage_material(material)
    engine = Engine(tmp_path)
    real_bind = store.bind_timing

    def crash_after_material_receipt(*args, **kwargs):
        raise RuntimeError("crash after material-complete receipt")

    store.bind_timing = crash_after_material_receipt
    runner = EvolutionRunner(
        manifest=manifest, checkpoints=store, engine=engine, wall_time_ns=lambda: 3_000_000_000
    )
    with pytest.raises(RuntimeError, match="material-complete"):
        runner._run_round("run", 0)
    _, receipt = store.load("run", 0)
    assert receipt["status"] == "material_complete"
    store.bind_timing = real_bind
    row = runner._run_round("run", 0)
    assert row.recovered and row.accounting.total_running_time_seconds == 2.0
    assert engine.calls == []


def seed_library():
    return compile_seed_library(
        {
            domain: {
                "source_path": f"src/tau2/domains/{domain}/tools.py",
                "source_sha256": f"sha256:{domain}",
                "schemas": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a record.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
            for domain in ("airline", "retail", "telecom")
        },
        source_commit=TAU_COMMIT,
    )


def generation_document(generation_id, parent_generation_id, seed_library_sha256):
    payload = {
        "generation_id": generation_id,
        "parent_generation_id": parent_generation_id,
        "memories": [],
        "skills": [],
        "vectors": [],
        "seed_library_sha256": seed_library_sha256,
    }
    return {**payload, "state_sha256": canonical_sha256(payload)}


def final_plan_kwargs(manifest):
    task_id = manifest["domains"]["airline"]["rounds"][0][0]
    calibration = OutputLimitCalibration(
        "real_train_smoke",
        manifest["manifest_sha256"],
        (("airline", task_id),),
        (CalibrationSample("airline", task_id, 96, "stop"),),
        192,
        "Qwen/Qwen3-32B",
        "sha256:runtime",
    )
    library = seed_library()
    return {
        "calibration": calibration,
        "seed_library": library,
        "user_simulator_config_sha256": "sha256:user-config",
        "native_runtime_sha256": "sha256:runtime",
        "evaluator_sha256": "sha256:evaluator",
        "generation_0": generation_document("g000", None, library["library_sha256"]),
        "generation_1": generation_document("g001", "g000", library["library_sha256"]),
        "generation_2": generation_document("g002", "g001", library["library_sha256"]),
        "updated_generation": generation_document("g003", "g002", library["library_sha256"]),
    }


def test_final_plan_preserves_task_clusters_and_trials():
    manifest, _, sealed = build_manifests(official_fixture())
    plan = build_final_plan(
        manifest, sealed, simulator_seeds=(0, 1, 2), **final_plan_kwargs(manifest)
    )
    assert plan.systems == ("vanilla", "generation_0", "updated")
    assert len(plan.task_clusters) == 100
    assert all(cluster.simulator_seeds == (0, 1, 2) for cluster in plan.task_clusters)


def test_final_plan_rejects_official178_compatibility():
    _, compatibility, sealed = build_manifests(official_fixture())
    with pytest.raises(ValueError, match="evolution144"):
        build_final_plan(compatibility, sealed, **final_plan_kwargs(compatibility))


def rehash_generation(document):
    payload = {key: value for key, value in document.items() if key != "state_sha256"}
    return {**payload, "state_sha256": canonical_sha256(payload)}


def test_final_plan_rejects_self_hashed_wrong_parent_chain():
    manifest, _, sealed = build_manifests(official_fixture())
    kwargs = final_plan_kwargs(manifest)
    forged = dict(kwargs["updated_generation"])
    forged["parent_generation_id"] = "g001"
    kwargs["updated_generation"] = rehash_generation(forged)
    with pytest.raises(ValueError, match="parent mismatch"):
        build_final_plan(manifest, sealed, **kwargs)


def test_final_plan_rejects_self_hashed_record_vector_mismatch():
    manifest, _, sealed = build_manifests(official_fixture())
    kwargs = final_plan_kwargs(manifest)
    forged = dict(kwargs["updated_generation"])
    forged["memories"] = [
        {
            "record_id": "forged",
            "version": 1,
            "memory_kind": "policy",
            "domains": ["airline"],
            "tool_dependencies": [],
            "content": "missing its required embedding",
        }
    ]
    kwargs["updated_generation"] = rehash_generation(forged)
    with pytest.raises(ValueError, match="content-bound embedding"):
        build_final_plan(manifest, sealed, **kwargs)


def test_final_test_guard_is_one_time(tmp_path):
    manifest, _, sealed = build_manifests(official_fixture())
    plan = build_final_plan(manifest, sealed, **final_plan_kwargs(manifest))
    guard = FinalTestOnceGuard(tmp_path / "ledger.json")
    guard.claim(plan)
    with pytest.raises(RuntimeError, match="already"):
        guard.claim(plan)
    guard.complete(plan, "sha256:report")
    assert json.loads((tmp_path / "ledger.json").read_text())["status"] == "complete"
