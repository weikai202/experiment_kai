from pathlib import Path

import pytest

from tau3_evolution.access import AccessAudit, TaskRef
from tau3_evolution.canonical import canonical_sha256, read_json
from tau3_evolution.generation_store import GenerationState, GenerationStore, SkillRecord
from tau3_evolution.ledger import (
    DurableAccountingLedger,
    LogicalRequest,
    OutputApplication,
    PhysicalAttempt,
    SubstantiveEffect,
)
from tau3_evolution.minibench import BranchResult, DevTaskView
from tau3_evolution.model_boundary import CalibrationSample, OutputLimitCalibration
from tau3_evolution.retrieval import EmbeddingResponse
from tau3_evolution.round_workflow import EpisodeEvidence, PipelineRoundEngine, ProposedUpdate

EVOLUTION_MANIFEST = read_json(
    Path(__file__).parents[1] / "configs" / "splits" / "evolution_seed0.json"
)
CALIBRATION_TASK = (
    "airline",
    EVOLUTION_MANIFEST["domains"]["airline"]["rounds"][0][0],
)
CALIBRATION = OutputLimitCalibration(
    "real_train_smoke",
    EVOLUTION_MANIFEST["manifest_sha256"],
    (CALIBRATION_TASK,),
    (CalibrationSample(*CALIBRATION_TASK, 96, "stop"),),
    192,
    "Qwen/Qwen3-32B",
    "sha256:runtime",
)
QWEN_CONFIG_SHA256 = CALIBRATION.qwen_config_sha256


class CalibratedPipeline:
    calibration = CALIBRATION


def dev_views():
    return tuple(
        DevTaskView(domain, task_id, ("lookup",))
        for domain in ("airline", "retail", "telecom")
        for task_id in EVOLUTION_MANIFEST["domains"][domain]["dev"]
    )


def round_tasks(round_index=0):
    return tuple(
        TaskRef(domain, task_id)
        for domain in ("airline", "retail", "telecom")
        for task_id in EVOLUTION_MANIFEST["domains"][domain]["rounds"][round_index]
    )


class Embeddings:
    def __init__(self, crash_content=None):
        self.receipts = {}
        self.calls = []
        self.crash_content = crash_content
        self.crashed = False

    def embed(self, text, *, model, logical_request_id):
        self.calls.append(logical_request_id)
        scalar = float(sum(text.encode("utf-8")) % 17 + 1)
        response = EmbeddingResponse(
            model,
            (scalar, 1.0),
            f"embedding:{logical_request_id}",
            1,
            0,
            canonical_sha256({"model": model, "text": text}),
        )
        self.receipts[logical_request_id] = response
        if text == self.crash_content and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated embedding unknown outcome")
        return response

    def recover(self, *, logical_request_id):
        return self.receipts.get(logical_request_id)


class NativeExecutor:
    def __init__(self, crash_on=None):
        self.crash_on = crash_on
        self.crashed = False
        self.execute_counts = {}
        self.recover_calls = []
        self.receipts = {}

    def tool_names_for_task(self, task):
        return ("lookup",)

    def execute(self, **kwargs):
        task = kwargs["task"]
        scope = kwargs["scope_id"]
        key = f"{task.domain}:{task.task_id}"
        logical = LogicalRequest(
            f"logical:{key}",
            scope,
            "qwen",
            "executed_policy_action",
            "sha256:request",
            "Qwen/Qwen3-32B",
            QWEN_CONFIG_SHA256,
        )
        attempt = PhysicalAttempt(
            f"attempt:{key}", logical.logical_request_id, 2, 3, True, "sha256:response"
        )
        effect = SubstantiveEffect(
            f"action:{key}", scope, "executed_action", f"sha256:{key}", True, True
        )
        application = OutputApplication(
            f"app:{key}",
            logical.logical_request_id,
            attempt.attempt_id,
            effect.effect_id,
            "executed_action",
            True,
            0,
        )
        result = EpisodeEvidence(
            key,
            True,
            0.0,
            (f"sha256:failure:{key}",),
            f"sha256:episode:{key}",
            (logical, attempt, effect, application),
        )
        dispatch_id = kwargs["dispatch_id"]
        self.execute_counts[key] = self.execute_counts.get(key, 0) + 1
        self.receipts[dispatch_id] = result
        if key == self.crash_on and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated crash after durable native result")
        return result

    def recover(self, *, task, scope_id, dispatch_id):
        self.recover_calls.append(dispatch_id)
        return self.receipts.get(dispatch_id)


class Updater:
    def __init__(self, crash_stage=None):
        self.crash_stage = crash_stage
        self.crashed = False
        self.propose_calls = 0
        self.review_calls = 0
        self.proposal_receipts = {}
        self.review_receipts = {}

    def propose(self, **kwargs):
        self.propose_calls += 1
        scope = kwargs["scope_id"]
        logical = LogicalRequest(
            "logical:update",
            scope,
            "qwen",
            "skill_candidate",
            "sha256:update-request",
            "Qwen/Qwen3-32B",
            QWEN_CONFIG_SHA256,
        )
        attempt = PhysicalAttempt(
            "attempt:update", logical.logical_request_id, 4, 5, True, "sha256:update-response"
        )
        result = (
            ProposedUpdate(
                "skill",
                "airline.lookup",
                1,
                None,
                ("airline",),
                ("lookup",),
                "Use lookup before answering.",
                logical,
                attempt,
                "application:update",
            ),
        )
        operation_id = kwargs["operation_id"]
        self.proposal_receipts[operation_id] = result
        if self.crash_stage == "propose" and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated propose unknown outcome")
        return result

    def recover_propose(self, *, operation_id):
        return self.proposal_receipts.get(operation_id)

    def review(self, proposal, generation, *, operation_id):
        self.review_calls += 1
        result = (True, ())
        self.review_receipts[operation_id] = result
        if self.crash_stage == "review" and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated review unknown outcome")
        return result

    def recover_review(self, *, operation_id):
        return self.review_receipts.get(operation_id)


class DevExecutor:
    def __init__(self, crash=False):
        self.crash = crash
        self.crashed = False
        self.calls = 0
        self.receipts = {}

    def run_paired(
        self,
        proposal,
        selected_tasks,
        generation,
        scope_id,
        host_shared_configuration_sha256,
        manifest_sha256,
        access_receipt_sha256,
        operation_id,
    ):
        self.calls += 1
        rows = []
        for index, (domain, task_id) in enumerate(selected_tasks):
            for branch, reward in (("previous", 0.0), ("candidate", 1.0)):
                rows.append(
                    BranchResult(
                        domain,
                        task_id,
                        branch,
                        proposal.record_id,
                        reward,
                        True,
                        f"{domain}:{task_id}:{branch}",
                        host_shared_configuration_sha256,
                        manifest_sha256,
                        access_receipt_sha256,
                        f"sha256:trajectory:{index}:{branch}",
                        f"sha256:evaluator:{index}:{branch}",
                    )
                )
        result = (tuple(rows), ())
        self.receipts[operation_id] = result
        if self.crash and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated Dev unknown outcome")
        return result

    def recover_paired(self, *, operation_id):
        return self.receipts.get(operation_id)


def test_concrete_round_composes_native_online_offline_dev_and_publication(tmp_path):
    embedding_provider = Embeddings()
    store = GenerationStore(tmp_path / "generations")
    initial_skill = SkillRecord("airline.lookup", 0, ("airline",), ("lookup",), "old", None, None)
    vector, _ = store.embed_record(
        record_kind="skill",
        record_id=initial_skill.skill_id,
        version=0,
        content=initial_skill.content,
        provider=embedding_provider,
        scope_id="bootstrap",
    )
    store.publish(
        GenerationState("g000", None, (), (initial_skill,), (vector,), "sha256:seed-library")
    )
    dev = dev_views()
    manifest_sha = EVOLUTION_MANIFEST["manifest_sha256"]
    dev_audit = AccessAudit(
        "dev",
        "skill_ab_validation",
        "skill",
        manifest_sha,
        canonical_sha256([{"domain": row.domain, "task_id": row.task_id} for row in dev]),
        len(dev),
    )
    engine = PipelineRoundEngine(
        run_id="run",
        generation_store=store,
        ledger_root=tmp_path / "ledger",
        online_pipeline=CalibratedPipeline(),
        native_executor=NativeExecutor(),
        updater=Updater(),
        dev_executor=DevExecutor(),
        dev_task_views=dev,
        dev_access_audit=dev_audit,
        embedding_provider=embedding_provider,
        evolution_manifest=EVOLUTION_MANIFEST,
    )
    tasks = round_tasks()
    result = engine.execute_round(
        round_index=0,
        input_generation_id="g000",
        output_generation_id="g001",
        train_tasks=tasks,
        manifest_sha256=manifest_sha,
    )
    assert len(result.completed_task_keys) == 48
    assert result.accepted_skill_versions == (("airline.lookup", 1),)
    published = store.load("g001")
    assert published.skills[0].content == "Use lookup before answering."
    assert published.skills[0].dev_evidence_sha256 in result.artifact_hashes
    snapshot = DurableAccountingLedger(result.accounting_ledger_path).snapshot(
        scope_id=result.accounting_scope_id, checkpoint_material_sha256="sha256:checkpoint"
    )
    assert snapshot.total_cost == 48 * 3 + 5
    assert snapshot.total_tokens == 48 * 5 + 9 + 49


def round_components(
    tmp_path,
    native,
    updater=None,
    dev_executor=None,
    embedding_provider=None,
    store_class=GenerationStore,
):
    embedding_provider = embedding_provider or Embeddings()
    store = store_class(tmp_path / "generations")
    initial_skill = SkillRecord("airline.lookup", 0, ("airline",), ("lookup",), "old", None, None)
    vector, _ = store.embed_record(
        record_kind="skill",
        record_id=initial_skill.skill_id,
        version=0,
        content=initial_skill.content,
        provider=embedding_provider,
        scope_id="bootstrap",
    )
    store.publish(
        GenerationState("g000", None, (), (initial_skill,), (vector,), "sha256:seed-library")
    )
    dev = dev_views()
    manifest_sha = EVOLUTION_MANIFEST["manifest_sha256"]
    dev_audit = AccessAudit(
        "dev",
        "skill_ab_validation",
        "skill",
        manifest_sha,
        canonical_sha256([{"domain": row.domain, "task_id": row.task_id} for row in dev]),
        len(dev),
    )
    engine = PipelineRoundEngine(
        run_id="run",
        generation_store=store,
        ledger_root=tmp_path / "ledger",
        online_pipeline=CalibratedPipeline(),
        native_executor=native,
        updater=updater or Updater(),
        dev_executor=dev_executor or DevExecutor(),
        dev_task_views=dev,
        dev_access_audit=dev_audit,
        evolution_manifest=EVOLUTION_MANIFEST,
        embedding_provider=embedding_provider,
        time_ns=lambda: 1_000_000_000,
    )
    tasks = round_tasks()
    return store, engine, tasks, manifest_sha


def test_mid_round_claimed_episode_recovers_without_redispatching_completed_tasks(tmp_path):
    expected_tasks = round_tasks()
    native = NativeExecutor(crash_on=f"{expected_tasks[1].domain}:{expected_tasks[1].task_id}")
    _, engine, tasks, manifest_sha = round_components(tmp_path, native)
    kwargs = dict(
        round_index=0,
        input_generation_id="g000",
        output_generation_id="g001",
        train_tasks=tasks,
        manifest_sha256=manifest_sha,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        engine.execute_round(**kwargs)
    result = engine.execute_round(**kwargs)
    assert len(result.completed_task_keys) == 48
    assert native.execute_counts[f"{tasks[0].domain}:{tasks[0].task_id}"] == 1
    assert native.execute_counts[f"{tasks[1].domain}:{tasks[1].task_id}"] == 1
    assert sum(native.execute_counts.values()) == 48
    assert len(native.recover_calls) == 1


class JumpUpdater(Updater):
    def propose(self, **kwargs):
        row = super().propose(**kwargs)[0]
        return (
            ProposedUpdate(
                row.update_kind,
                row.record_id,
                2,
                row.memory_kind,
                row.domains,
                row.tool_dependencies,
                row.content,
                row.logical_request,
                row.physical_attempt,
                row.application_id,
            ),
        )


def test_update_version_must_match_actual_input_generation(tmp_path):
    _, engine, tasks, manifest_sha = round_components(
        tmp_path, NativeExecutor(), updater=JumpUpdater()
    )
    with pytest.raises(ValueError, match="exactly current"):
        engine.execute_round(
            round_index=0,
            input_generation_id="g000",
            output_generation_id="g001",
            train_tasks=tasks,
            manifest_sha256=manifest_sha,
        )


class UnappliedRetryUpdater(Updater):
    def review(self, proposal, generation, *, operation_id):
        scope = proposal.logical_request.scope_id
        request = LogicalRequest(
            "logical:retry",
            scope,
            "qwen",
            "retry_only",
            "sha256:retry-request",
            "Qwen/Qwen3-32B",
            QWEN_CONFIG_SHA256,
        )
        attempt = PhysicalAttempt(
            "attempt:retry", request.logical_request_id, 7, 99, True, "sha256:retry-response"
        )
        return True, (request, attempt)


def test_unapplied_retry_tokens_are_total_usage_but_not_effective_cost(tmp_path):
    _, engine, tasks, manifest_sha = round_components(
        tmp_path, NativeExecutor(), updater=UnappliedRetryUpdater()
    )
    result = engine.execute_round(
        round_index=0,
        input_generation_id="g000",
        output_generation_id="g001",
        train_tasks=tasks,
        manifest_sha256=manifest_sha,
    )
    snapshot = DurableAccountingLedger(result.accounting_ledger_path).snapshot(
        scope_id=result.accounting_scope_id, checkpoint_material_sha256="sha256:checkpoint"
    )
    assert snapshot.total_tokens == 48 * 5 + 9 + 49 + 106
    assert snapshot.total_cost == 48 * 3 + 5


class CrashPublishStore(GenerationStore):
    def __init__(self, root):
        super().__init__(root)
        self.publish_calls = 0
        self.crashed = False

    def publish(self, state):
        result = super().publish(state)
        if state.generation_id == "g001":
            self.publish_calls += 1
            if not self.crashed:
                self.crashed = True
                raise RuntimeError("simulated publication unknown outcome")
        return result


@pytest.mark.parametrize("stage", ["propose", "review", "dev", "embedding", "publication"])
def test_post_episode_operations_recover_without_external_redispatch(tmp_path, stage):
    updater = Updater(crash_stage=stage if stage in {"propose", "review"} else None)
    dev = DevExecutor(crash=stage == "dev")
    embeddings = Embeddings(
        crash_content="Use lookup before answering." if stage == "embedding" else None
    )
    store_class = CrashPublishStore if stage == "publication" else GenerationStore
    native = NativeExecutor()
    store, engine, tasks, manifest_sha = round_components(
        tmp_path,
        native,
        updater=updater,
        dev_executor=dev,
        embedding_provider=embeddings,
        store_class=store_class,
    )
    kwargs = dict(
        round_index=0,
        input_generation_id="g000",
        output_generation_id="g001",
        train_tasks=tasks,
        manifest_sha256=manifest_sha,
    )
    with pytest.raises(RuntimeError, match="unknown outcome"):
        engine.execute_round(**kwargs)
    result = engine.execute_round(**kwargs)
    assert len(result.completed_task_keys) == 48
    assert sum(native.execute_counts.values()) == 48
    assert updater.propose_calls == 1
    assert updater.review_calls == 1
    assert dev.calls == 1
    if stage == "embedding":
        accepted_calls = [
            logical
            for logical in embeddings.calls
            if embeddings.receipts[logical].vector and logical in embeddings.receipts
        ]
        assert len(accepted_calls) == len(set(accepted_calls))
    if stage == "publication":
        assert store.publish_calls == 1
