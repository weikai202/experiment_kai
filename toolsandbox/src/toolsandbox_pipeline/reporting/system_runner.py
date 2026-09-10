"""Fixed-order, plan-bound execution of the three final-evaluation systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import ConfigDict, model_validator

from toolsandbox_pipeline.metrics.timing import ScopeTimer
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.metrics.aggregation import MetricsAggregator
from toolsandbox_pipeline.schemas.accounting import (
    AccountingBreakdown,
    AccountingTotals,
    LogicalRequestAccountingInput,
    PhysicalAttemptAccountingInput,
    ScopeTimingInput,
    TaskAccountingInput,
)
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.checkpoint import QwenEffectiveEffect, QwenEffectKind
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.critic import CriticVerdict
from toolsandbox_pipeline.schemas.fixtures import ExternalReadAttempt
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnDecision
from toolsandbox_pipeline.schemas.reporting import (
    SYSTEM_ORDER,
    FinalEvaluationPlan,
    ScenarioEvaluationRecord,
    SystemId,
    TestScenarioPlan,
)
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeExecutionStatus,
    EpisodeResult,
    TrustedEvaluatorRecord,
    TrustedTrajectory,
)
from toolsandbox_pipeline.schemas.ledger_accounting import Task011AccountingSnapshot

from toolsandbox_pipeline.checkpointing.blob_store import RestrictedBlobStore

from .evaluation_plan import (
    EvaluationPlanError,
    SystemCompletionReceipt,
    TestOnceGuard,
)
from .vanilla_responder import VanillaDecisionRecord


class SystemRunnerError(RuntimeError):
    """Sanitized final-evaluation execution or evidence failure."""


class TerminalEpisodeFailure(SystemRunnerError):
    """A provider, schema, configuration, or trusted-evidence failure."""


class ReconciliationRequired(SystemRunnerError):
    """An official-live external read has an unknown outcome."""


class SystemExecutionAttestation(StrictModel):
    """Frozen, content-free declaration of one constructed execution stack."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    system_id: SystemId
    generation_id: Literal["g000", "g003"]
    active_components: tuple[str, ...]
    prompt_manifest_sha256: str
    token_limit_config_sha256: str
    environment_sha256: str
    fixture_or_live_tool_config_sha256: str
    shared_pipeline_config_sha256: str
    qwen_identity_sha256: str
    embedding_identity_sha256: str
    user_simulator_identity_sha256: str
    qwen_provider: Literal["vllm_openai_compatible"]
    qwen_model: Literal["Qwen/Qwen3-32B"]
    frozen_execution_sha256: str
    fresh_context_per_episode: Literal[True] = True
    response_sharing: Literal[False] = False
    result_sharing: Literal[False] = False


class SystemAccountingProjection(StrictModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    task_inputs: tuple[TaskAccountingInput, ...]
    physical_attempts: tuple[PhysicalAttemptAccountingInput, ...]
    logical_requests: tuple[LogicalRequestAccountingInput, ...]
    substantive_effects: tuple[QwenEffectiveEffect, ...]
    totals: AccountingTotals
    breakdowns: tuple[AccountingBreakdown, ...]


class StagedSystemMaterial(StrictModel):
    """All final system evidence persisted before the direct timer is closed."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    plan_sha256: str
    system_id: SystemId
    records: tuple[ScenarioEvaluationRecord, ...]
    accounting: SystemAccountingProjection
    content_sha256: str

    @classmethod
    def build(cls, **values: Any) -> "StagedSystemMaterial":
        provisional = cls.model_construct(
            schema_version=1,
            content_sha256="sha256:" + "0" * 64,
            **values,
        )
        payload = provisional.model_dump(mode="json", exclude={"content_sha256"})
        values["content_sha256"] = canonical_sha256(
            ["staged-system-material-v1", payload]
        )
        return cls(schema_version=1, **values)

    @model_validator(mode="after")
    def exact_identity(self) -> "StagedSystemMaterial":
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        if self.content_sha256 != canonical_sha256(
            ["staged-system-material-v1", payload]
        ):
            raise ValueError("staged system material identity mismatch")
        return self


@runtime_checkable
class SystemEpisodeExecutor(Protocol):
    """Capability supplied only by the final-test launcher after approval.

    Implementations materialize a fresh native scenario context for every call.
    The runner validates all returned content-free identities against the frozen
    plan and never accepts scores or metrics supplied outside Task014 records.
    """

    def run_episode(
        self,
        *,
        system_id: SystemId,
        scenario: TestScenarioPlan,
        episode_id: str,
        isolation_id: str,
    ) -> EpisodeResult: ...

    def execution_attestation(self) -> SystemExecutionAttestation: ...

    def load_trajectory(self, result: EpisodeResult) -> TrustedTrajectory: ...

    def load_evaluator(self, result: EpisodeResult) -> TrustedEvaluatorRecord: ...

    def load_online_decision(self, decision_reference: object) -> object: ...

    def load_external_attempt(self, attempt_id: str) -> ExternalReadAttempt: ...

    def load_accounting_snapshot(
        self, result: EpisodeResult
    ) -> Task011AccountingSnapshot: ...


class PersistedSystemResult(StrictModel):
    """Canonical restricted payload recoverable after completion-marker crashes."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    plan_sha256: str
    system_id: SystemId
    records: tuple[ScenarioEvaluationRecord, ...]
    accounting: SystemAccountingProjection
    material_reference: BlobReference
    timing: ScopeTimingInput
    content_sha256: str

    @classmethod
    def build(cls, **values: Any) -> "PersistedSystemResult":
        provisional = cls.model_construct(
            schema_version=1,
            content_sha256="sha256:" + "0" * 64,
            **values,
        )
        payload = provisional.model_dump(mode="json", exclude={"content_sha256"})
        values["content_sha256"] = canonical_sha256(
            ["persisted-system-result-v1", payload]
        )
        return cls(schema_version=1, **values)

    @model_validator(mode="after")
    def exact_identity(self) -> "PersistedSystemResult":
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        if self.content_sha256 != canonical_sha256(
            ["persisted-system-result-v1", payload]
        ):
            raise ValueError("persisted system result identity mismatch")
        return self


@runtime_checkable
class SystemResultStore(Protocol):
    def persist_material(self, material: StagedSystemMaterial) -> BlobReference: ...

    def load_material(self, reference: BlobReference) -> StagedSystemMaterial: ...

    def persist(self, result: PersistedSystemResult) -> BlobReference: ...

    def load(self, reference: BlobReference) -> PersistedSystemResult: ...


class RestrictedSystemResultStore:
    """Task011 restricted blob adapter; no raw provider material is projected."""

    def __init__(self, blobs: RestrictedBlobStore) -> None:
        if type(blobs) is not RestrictedBlobStore:
            raise TypeError("RestrictedBlobStore required")
        self.blobs = blobs

    def persist(self, result: PersistedSystemResult) -> BlobReference:
        if type(result) is not PersistedSystemResult:
            raise TypeError("PersistedSystemResult required")
        return self.blobs.put(
            canonical_json_bytes(result.model_dump(mode="json")),
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="PersistedSystemResult",
            schema_version=1,
        )

    def persist_material(self, material: StagedSystemMaterial) -> BlobReference:
        if type(material) is not StagedSystemMaterial:
            raise TypeError("StagedSystemMaterial required")
        return self.blobs.put(
            canonical_json_bytes(material.model_dump(mode="json")),
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="StagedSystemMaterial",
            schema_version=1,
        )

    def load_material(self, reference: BlobReference) -> StagedSystemMaterial:
        if (
            type(reference) is not BlobReference
            or reference.schema_name != "StagedSystemMaterial"
            or reference.schema_version != 1
            or reference.media_type != "application/vnd.toolsandbox.canonical+json"
        ):
            raise SystemRunnerError("invalid staged system material reference")
        try:
            return StagedSystemMaterial.model_validate_json(
                self.blobs.read(reference), strict=True
            )
        except Exception:
            raise SystemRunnerError("invalid staged system material") from None

    def load(self, reference: BlobReference) -> PersistedSystemResult:
        if (
            type(reference) is not BlobReference
            or reference.schema_name != "PersistedSystemResult"
            or reference.schema_version != 1
            or reference.media_type != "application/vnd.toolsandbox.canonical+json"
        ):
            raise SystemRunnerError("invalid system result reference")
        try:
            result = PersistedSystemResult.model_validate_json(
                self.blobs.read(reference), strict=True
            )
        except Exception:
            raise SystemRunnerError("invalid persisted system result") from None
        return result


@dataclass(frozen=True)
class SystemRunResult:
    system_id: SystemId
    records: tuple[ScenarioEvaluationRecord, ...]
    accounting: SystemAccountingProjection
    timing: ScopeTimingInput
    result_reference: BlobReference


class SystemRunner:
    """Run exactly one frozen system in manifest order, then seal its guard."""

    def __init__(
        self,
        *,
        plan: FinalEvaluationPlan,
        guard: TestOnceGuard,
        executors: Mapping[SystemId, SystemEpisodeExecutor],
        boot_id: str,
        result_store: SystemResultStore,
        timer_factory=ScopeTimer,
    ) -> None:
        if type(plan) is not FinalEvaluationPlan:
            raise TypeError("FinalEvaluationPlan required")
        if type(guard) is not TestOnceGuard or guard.plan != plan:
            raise TypeError("plan-bound TestOnceGuard required")
        if tuple(executors) != SYSTEM_ORDER:
            raise ValueError("exact fixed-order executor mapping required")
        if any(not isinstance(item, SystemEpisodeExecutor) for item in executors.values()):
            raise TypeError("SystemEpisodeExecutor required")
        if not boot_id:
            raise ValueError("boot identity required")
        if not isinstance(result_store, SystemResultStore):
            raise TypeError("SystemResultStore required")
        self.plan = plan
        self.guard = guard
        self.executors = dict(executors)
        self.boot_id = boot_id
        self.result_store = result_store
        self.timer_factory = timer_factory
        self.last_timing: ScopeTimingInput | None = None
        self._episode_owners: dict[str, tuple[SystemId, str]] = {}
        self._request_owners: dict[str, tuple[SystemId, str]] = {}
        self._attempt_owners: dict[str, tuple[SystemId, str]] = {}
        self._application_owners: dict[str, tuple[SystemId, str]] = {}
        self._run_owners: dict[str, SystemId] = {}
        self._validate_attestations()

    def run_all(
        self,
        *,
        resume_timings: Mapping[SystemId, ScopeTimingInput] | None = None,
    ) -> tuple[SystemRunResult, SystemRunResult, SystemRunResult]:
        timings = resume_timings or {}
        unknown = set(timings).difference(SYSTEM_ORDER)
        if unknown:
            raise ValueError("unknown system timing")
        if any(self.guard.state(system_id) != "authorized_not_started" for system_id in SYSTEM_ORDER):
            raise EvaluationPlanError("fresh run requires all systems unstarted")
        results = tuple(
            self.run_system(system_id, resume_timing=timings.get(system_id))
            for system_id in SYSTEM_ORDER
        )
        return results  # type: ignore[return-value]

    def resume_incomplete(
        self,
        *,
        resume_timings: Mapping[SystemId, ScopeTimingInput] | None = None,
    ) -> tuple[SystemRunResult, ...]:
        """Continue the identical plan without dispatching completed systems."""

        timings = resume_timings or {}
        if set(timings).difference(SYSTEM_ORDER):
            raise ValueError("unknown system timing")
        results = []
        for system_id in SYSTEM_ORDER:
            state = self.guard.state(system_id)
            if state == "completed":
                continue
            if state in {"terminal_failure", "reconciliation_required"}:
                raise EvaluationPlanError("terminal system cannot resume automatically")
            results.append(
                self.run_system(system_id, resume_timing=timings.get(system_id))
            )
        return tuple(results)

    def run_system(
        self,
        system_id: SystemId,
        *,
        resume_timing: ScopeTimingInput | None = None,
    ) -> SystemRunResult:
        if system_id not in SYSTEM_ORDER:
            raise ValueError("unknown final-evaluation system")
        state = self.guard.state(system_id)
        staged = self.guard.completion_receipt(system_id)
        material_reference = self.guard.staged_material_reference(system_id)
        if state == "authorized_not_started":
            self.guard.start_system(system_id)
        elif state == "running":
            if staged is not None:
                return self._restore_staged_completion(system_id, staged)
            if resume_timing is None:
                raise EvaluationPlanError(
                    "persisted evaluation timing is required for resume"
                )
            self.guard.assert_identical_resume(system_id, self.plan.plan_sha256)
        else:
            raise EvaluationPlanError("completed or terminal system cannot execute")

        timer = self.timer_factory(
            "evaluation_run",
            self._scope_id(system_id),
            self.boot_id,
            stored=resume_timing,
        )
        executor = self.executors[system_id]
        records: list[ScenarioEvaluationRecord] = []
        task_inputs: list[TaskAccountingInput] = []
        physical_attempts: list[PhysicalAttemptAccountingInput] = []
        logical_requests: list[LogicalRequestAccountingInput] = []
        substantive_effects: list[QwenEffectiveEffect] = []
        try:
            if material_reference is not None:
                material = self._load_staged_material(
                    system_id, material_reference
                )
                return self._finalize_material(
                    timer, material, material_reference
                )
            for scenario in self.plan.scenarios:
                episode_id = self._episode_id(system_id, scenario.scenario_id)
                isolation_id = self._isolation_id(system_id, scenario.scenario_id)
                self.guard.authorize_episode(system_id, scenario.scenario_id, episode_id)
                result = executor.run_episode(
                    system_id=system_id,
                    scenario=scenario,
                    episode_id=episode_id,
                    isolation_id=isolation_id,
                )
                if type(result) is not EpisodeResult:
                    raise TerminalEpisodeFailure("invalid episode result")
                if result.status is EpisodeExecutionStatus.RECONCILIATION_REQUIRED:
                    raise ReconciliationRequired("system requires external reconciliation")
                if result.status is not EpisodeExecutionStatus.COMPLETED_EVALUATED:
                    raise TerminalEpisodeFailure("system terminated before evaluation")
                records.append(
                    self._project_record(
                        system_id=system_id,
                        scenario=scenario,
                        episode_id=episode_id,
                        executor=executor,
                        result=result,
                    )
                )
                projection = executor.load_accounting_snapshot(result)
                self._validate_accounting_snapshot(result, projection)
                task_inputs.append(result.task_accounting_input)
                physical_attempts.extend(projection.physical_attempts)
                logical_requests.extend(projection.logical_requests)
                substantive_effects.extend(projection.substantive_effects)
            accounting = self._aggregate_accounting(
                system_id,
                tuple(records),
                tuple(task_inputs),
                tuple(physical_attempts),
                tuple(logical_requests),
                tuple(substantive_effects),
            )
            material = StagedSystemMaterial.build(
                plan_sha256=self.plan.plan_sha256,
                system_id=system_id,
                records=tuple(records),
                accounting=accounting,
            )
            material_reference = self.result_store.persist_material(material)
            if material_reference.sha256 != canonical_sha256(
                material.model_dump(mode="json")
            ):
                raise TerminalEpisodeFailure(
                    "staged system material store identity mismatch"
                )
            self.guard.stage_system_material(system_id, material_reference)
            return self._finalize_material(timer, material, material_reference)
        except ReconciliationRequired:
            if self.plan.profile != "official_live":
                self.guard.mark_terminal_failure(system_id)
                self.last_timing = timer.close()
                raise TerminalEpisodeFailure("reconciliation is not valid for this profile") from None
            self.guard.mark_reconciliation_required(system_id)
            self.last_timing = timer.close()
            raise
        except (EvaluationPlanError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            if (
                self.guard.state(system_id) == "running"
                and self.guard.staged_material_reference(system_id) is None
            ):
                self.guard.mark_terminal_failure(system_id)
            self.last_timing = timer.close()
            if isinstance(error, TerminalEpisodeFailure):
                raise
            raise TerminalEpisodeFailure("system execution failed") from None

    def _finalize_material(
        self,
        timer: ScopeTimer,
        material: StagedSystemMaterial,
        material_reference: BlobReference,
    ) -> SystemRunResult:
        """Close timing only after material and its guard pointer are durable."""

        self.last_timing = timer.close()
        persisted = PersistedSystemResult.build(
            plan_sha256=self.plan.plan_sha256,
            system_id=material.system_id,
            records=material.records,
            accounting=material.accounting,
            material_reference=material_reference,
            timing=self.last_timing,
        )
        reference = self.result_store.persist(persisted)
        if reference.sha256 != canonical_sha256(
            persisted.model_dump(mode="json")
        ):
            raise TerminalEpisodeFailure("system result store identity mismatch")
        receipt = SystemCompletionReceipt(
            result_reference=reference,
            result_sha256=reference.sha256,
            timing_sha256=canonical_sha256(
                self.last_timing.model_dump(mode="json")
            ),
        )
        self.guard.stage_system_completion(material.system_id, receipt)
        self.guard.complete_system(material.system_id, receipt)
        return SystemRunResult(
            material.system_id,
            material.records,
            material.accounting,
            self.last_timing,
            reference,
        )

    def _load_staged_material(
        self, system_id: SystemId, reference: BlobReference
    ) -> StagedSystemMaterial:
        material = self.result_store.load_material(reference)
        if (
            material.plan_sha256 != self.plan.plan_sha256
            or material.system_id != system_id
            or reference.sha256
            != canonical_sha256(material.model_dump(mode="json"))
        ):
            raise EvaluationPlanError("staged system material identity mismatch")
        return material

    def _restore_staged_completion(
        self, system_id: SystemId, receipt: SystemCompletionReceipt
    ) -> SystemRunResult:
        persisted = self.result_store.load(receipt.result_reference)
        material_reference = self.guard.staged_material_reference(system_id)
        if material_reference is None:
            raise EvaluationPlanError("completed system material is missing")
        material = self._load_staged_material(system_id, material_reference)
        if (
            persisted.plan_sha256 != self.plan.plan_sha256
            or persisted.system_id != system_id
            or receipt.result_sha256
            != canonical_sha256(persisted.model_dump(mode="json"))
            or receipt.timing_sha256
            != canonical_sha256(persisted.timing.model_dump(mode="json"))
            or persisted.material_reference != material_reference
            or persisted.records != material.records
            or persisted.accounting != material.accounting
        ):
            raise EvaluationPlanError("staged system result identity mismatch")
        self.guard.complete_system(system_id, receipt)
        self.last_timing = persisted.timing
        return SystemRunResult(
            system_id,
            persisted.records,
            persisted.accounting,
            persisted.timing,
            receipt.result_reference,
        )

    def _project_record(
        self,
        *,
        system_id: SystemId,
        scenario: TestScenarioPlan,
        episode_id: str,
        executor: SystemEpisodeExecutor,
        result: EpisodeResult,
    ) -> ScenarioEvaluationRecord:
        self._validate_episode_identity(system_id, scenario, episode_id, result)
        trajectory = executor.load_trajectory(result)
        evaluator = executor.load_evaluator(result)
        if type(trajectory) is not TrustedTrajectory or type(evaluator) is not TrustedEvaluatorRecord:
            raise SystemRunnerError("untrusted episode evidence")
        if (
            trajectory.identity != result.identity
            or trajectory.eligible_for_train_offline_consumption
            or result.trusted_trajectory_reference is None
            or result.evaluator_record_reference is None
            or trajectory.evaluator_record_reference != result.evaluator_record_reference
            or trajectory.evaluator_record_sha256 != result.evaluator_record_reference.sha256
            or trajectory.ending_context_reference != result.ending_context_reference
            or trajectory.ending_context_sha256 != result.ending_context_sha256
            or evaluator.ending_context_sha256 != result.ending_context_sha256
            or evaluator.evaluation_definition_sha256
            != scenario.evaluation_definition_sha256
            or canonical_sha256(evaluator.model_dump(mode="json"))
            != result.evaluator_record_reference.sha256
            or canonical_sha256(trajectory.model_dump(mode="json"))
            != result.trusted_trajectory_reference.sha256
        ):
            raise SystemRunnerError("episode evidence identity mismatch")

        owner = (system_id, scenario.scenario_id)
        self._register(self._episode_owners, episode_id, owner)
        self._register_many(self._request_owners, trajectory.logical_request_ids, owner)
        self._register_many(self._attempt_owners, trajectory.physical_attempt_ids, owner)
        for turn in trajectory.online_turns:
            if not set(turn.source_attempt_ids).issubset(trajectory.physical_attempt_ids):
                raise SystemRunnerError("turn attempt missing from trajectory")
            self._register_many(self._application_owners, turn.application_ids, owner)

        critic_trigger = critic_accept = critic_revise = critic_uncertain = revisions = 0
        for turn in trajectory.online_turns:
            decision = executor.load_online_decision(turn.decision_reference)
            if canonical_sha256(decision.model_dump(mode="json")) != turn.decision_sha256:
                raise SystemRunnerError("online decision identity mismatch")
            if system_id == "vanilla":
                if type(decision) is not VanillaDecisionRecord:
                    raise SystemRunnerError("Vanilla decision contract mismatch")
                expected_requests = (decision.logical_request_id,)
                if decision.source_attempt_id not in turn.source_attempt_ids:
                    raise SystemRunnerError("Vanilla source attempt mismatch")
                if decision.application_id not in turn.application_ids:
                    raise SystemRunnerError("Vanilla application mismatch")
                action = decision.action
            else:
                if type(decision) is not OnlineTurnDecision:
                    raise SystemRunnerError("pipeline decision contract mismatch")
                expected_requests = tuple(
                    item
                    for item in (
                        decision.initial_policy_logical_request_id,
                        decision.critic_logical_request_id,
                        decision.revision_logical_request_id,
                    )
                    if item is not None
                )
                action = decision.final_action
                if decision.critic_verdict is not None:
                    critic_trigger += 1
                    if decision.critic_verdict.verdict is CriticVerdict.ACCEPT:
                        critic_accept += 1
                    elif decision.critic_verdict.verdict is CriticVerdict.REVISE:
                        critic_revise += 1
                    else:
                        critic_uncertain += 1
                revisions += decision.revision_count
            if (
                expected_requests != turn.logical_request_ids
                or decision.state_id != turn.state_id
                or canonical_sha256(action.model_dump(mode="json"))
                != turn.final_action_sha256
            ):
                raise SystemRunnerError("online turn identity mismatch")

        fixture_hits = fixture_misses = external_exceptions = 0
        external_ids = tuple(
            attempt_id
            for action in trajectory.tool_actions
            for attempt_id in action.external_attempt_ids
        )
        if len(set(external_ids)) != len(external_ids):
            raise SystemRunnerError("duplicate external attempt identity")
        for attempt_id in external_ids:
            attempt = executor.load_external_attempt(attempt_id)
            if (
                type(attempt) is not ExternalReadAttempt
                or attempt.attempt_id != attempt_id
                or attempt.context.run_id != result.identity.run_id
                or attempt.context.scenario_id != scenario.scenario_id
            ):
                raise SystemRunnerError("external attempt identity mismatch")
            fixture_hits += attempt.status == "fixture_hit"
            fixture_misses += attempt.status == "fixture_miss"
            external_exceptions += attempt.status in {
                "fixture_miss",
                "failed",
                "rejected_before_dispatch",
                "unknown_outcome",
            }

        return ScenarioEvaluationRecord(
            system_id=system_id,
            manifest_position=scenario.manifest_position,
            scenario_id=scenario.scenario_id,
            scenario_family_id=scenario.scenario_family_id,
            variant=scenario.variant,
            categories=scenario.categories,
            episode_id=episode_id,
            trajectory_sha256=result.trusted_trajectory_reference.sha256,
            evaluator_record_sha256=result.evaluator_record_reference.sha256,
            similarity=evaluator.similarity,
            milestone_similarity=evaluator.milestone_similarity,
            minefield_similarity=evaluator.minefield_similarity,
            fully_successful=evaluator.fully_successful,
            effective_turn_count=evaluator.turn_count,
            critic_trigger_count=critic_trigger,
            critic_accept_count=critic_accept,
            critic_revise_count=critic_revise,
            critic_uncertain_count=critic_uncertain,
            revision_count=revisions,
            fixture_hit_count=fixture_hits,
            fixture_miss_count=fixture_misses,
            external_tool_exception_count=external_exceptions,
        )

    def _validate_attestations(self) -> None:
        for system_id, executor in self.executors.items():
            attestation = executor.execution_attestation()
            if type(attestation) is not SystemExecutionAttestation:
                raise TypeError("SystemExecutionAttestation required")
            system = self.plan.systems[SYSTEM_ORDER.index(system_id)]
            expected = (
                system_id,
                system.generation_id,
                system.allowed_components,
                system.prompt_manifest_sha256,
                system.token_limit_config_sha256,
                self.plan.environment_sha256,
                self.plan.fixture_or_live_tool_config_sha256,
                self.plan.shared_pipeline_config_sha256,
                self.plan.qwen_identity_sha256,
                self.plan.embedding_identity_sha256,
                self.plan.user_simulator_identity_sha256,
            )
            actual = (
                attestation.system_id,
                attestation.generation_id,
                attestation.active_components,
                attestation.prompt_manifest_sha256,
                attestation.token_limit_config_sha256,
                attestation.environment_sha256,
                attestation.fixture_or_live_tool_config_sha256,
                attestation.shared_pipeline_config_sha256,
                attestation.qwen_identity_sha256,
                attestation.embedding_identity_sha256,
                attestation.user_simulator_identity_sha256,
            )
            if actual != expected:
                raise ValueError("system execution attestation does not match plan")
            if attestation.frozen_execution_sha256 != execution_attestation_sha256(
                self.plan, system_id
            ):
                raise ValueError("frozen execution attestation hash mismatch")

    def _validate_accounting_snapshot(
        self,
        result: EpisodeResult,
        projection: Task011AccountingSnapshot,
    ) -> None:
        if type(projection) is not Task011AccountingSnapshot:
            raise TerminalEpisodeFailure("authoritative accounting snapshot required")
        identity = result.identity
        population = projection.population
        scope = population.scope
        if (
            scope.run_id != identity.run_id
            or scope.round_index != identity.round_index
            or scope.task_id != identity.episode_id
            or scope.scenario_id != identity.scenario_id
            or scope.scenario_family_id != identity.family_id
            or scope.system_variant != identity.system_variant
        ):
            raise TerminalEpisodeFailure("accounting population scope mismatch")
        owner = (identity.system_variant, identity.scenario_id)
        for item in (*projection.physical_attempts, *projection.logical_requests):
            if (
                item.run_id != identity.run_id
                or item.task_id != identity.episode_id
                or item.scenario_id != identity.scenario_id
                or item.scenario_family_id != identity.family_id
                or item.system_variant != identity.system_variant
            ):
                raise TerminalEpisodeFailure("accounting scope identity mismatch")
        self._register_many(
            self._request_owners,
            tuple(item.logical_request_id for item in projection.physical_attempts)
            + tuple(item.logical_request_id for item in projection.logical_requests),
            owner,
        )
        self._register_many(
            self._attempt_owners,
            tuple(item.attempt_id for item in projection.physical_attempts),
            owner,
        )
        self._register_many(
            self._application_owners,
            tuple(
                item.application_id
                for item in projection.logical_requests
                if item.application_id is not None
            ),
            owner,
        )
        logical_by_application = {
            item.application_id: item
            for item in projection.logical_requests
            if item.application_id is not None
        }
        trajectory = self.executors[identity.system_variant].load_trajectory(result)
        online_roles = {"vanilla", "policy", "critic", "revision"}
        online_logical_ids = tuple(
            item.logical_request_id
            for item in projection.logical_requests
            if item.role in online_roles
        )
        online_logical_set = set(online_logical_ids)
        online_attempt_ids = tuple(
            item.attempt_id
            for item in projection.physical_attempts
            if item.logical_request_id in online_logical_set
        )
        turn_application_ids = {
            application_id
            for turn in trajectory.online_turns
            for application_id in turn.application_ids
        }
        online_application_ids = {
            item.application_id
            for item in projection.logical_requests
            if item.role in online_roles and item.application_id is not None
        }
        if (
            online_logical_ids != trajectory.logical_request_ids
            or online_attempt_ids != trajectory.physical_attempt_ids
            or online_application_ids != turn_application_ids
            or any(
                item.role not in online_roles | {"embedding", "user_simulator"}
                for item in projection.logical_requests
            )
        ):
            raise TerminalEpisodeFailure(
                "accounting snapshot does not match authoritative episode population"
            )
        final_action_hashes = {turn.final_action_sha256 for turn in trajectory.online_turns}
        effect_application_ids: list[str] = []
        effect_artifact_hashes: set[str] = set()
        for effect in projection.substantive_effects:
            if (
                type(effect) is not QwenEffectiveEffect
                or effect.effect_kind is not QwenEffectKind.COMMITTED_ONLINE_ACTION
            ):
                raise TerminalEpisodeFailure("invalid substantive effect")
            effect_application_ids.extend(effect.ordered_application_ids)
            effect_artifact_hashes.add(effect.effect_artifact_sha256)
        if (
            len(effect_application_ids) != len(set(effect_application_ids))
            or set(effect_application_ids) != turn_application_ids
            or effect_artifact_hashes != final_action_hashes
            or len(projection.substantive_effects) != len(trajectory.online_turns)
            or any(item not in logical_by_application for item in effect_application_ids)
        ):
            raise TerminalEpisodeFailure("effect does not match committed action chain")

    def _aggregate_accounting(
        self,
        system_id: SystemId,
        records: tuple[ScenarioEvaluationRecord, ...],
        task_inputs: tuple[TaskAccountingInput, ...],
        attempts: tuple[PhysicalAttemptAccountingInput, ...],
        logical: tuple[LogicalRequestAccountingInput, ...],
        effects: tuple[QwenEffectiveEffect, ...],
    ) -> SystemAccountingProjection:
        if len(records) != len(task_inputs) or tuple(
            item.task_id for item in task_inputs
        ) != tuple(item.episode_id for item in records):
            raise TerminalEpisodeFailure("task accounting does not match records")
        attestation = self.executors[system_id].execution_attestation()
        aggregator = MetricsAggregator(
            qwen_provider=attestation.qwen_provider,
            qwen_model=attestation.qwen_model,
        )
        try:
            totals = aggregator.aggregate(attempts, logical, effects)
            breakdowns = aggregator.breakdowns(attempts, logical, effects)
        except Exception:
            raise TerminalEpisodeFailure("accounting projection is inconsistent") from None
        return SystemAccountingProjection(
            task_inputs=task_inputs,
            physical_attempts=attempts,
            logical_requests=logical,
            substantive_effects=effects,
            totals=totals,
            breakdowns=breakdowns,
        )

    def _validate_episode_identity(
        self,
        system_id: SystemId,
        scenario: TestScenarioPlan,
        episode_id: str,
        result: EpisodeResult,
    ) -> None:
        identity = result.identity
        system = self.plan.systems[SYSTEM_ORDER.index(system_id)]
        expected = (
            self.plan.profile,
            "final_test",
            None,
            None,
            scenario.scenario_family_id,
            scenario.scenario_id,
            episode_id,
            scenario.manifest_position,
            system_id,
            system.generation_id,
            scenario.starting_context_sha256,
            scenario.evaluation_definition_sha256,
            scenario.agent_tool_schema_sha256,
            self.plan.test_dataset_manifest_sha256,
            self.plan.shared_pipeline_config_sha256,
            system.prompt_manifest_sha256,
            system.token_limit_config_sha256,
            self.plan.fixture_or_live_tool_config_sha256,
            self.plan.environment_sha256,
        )
        actual = (
            identity.profile,
            identity.phase,
            identity.round_index,
            identity.shard_id,
            identity.family_id,
            identity.scenario_id,
            identity.episode_id,
            identity.manifest_position,
            identity.system_variant,
            identity.generation_id,
            identity.starting_context_sha256,
            identity.evaluation_definition_sha256,
            identity.agent_tool_schema_sha256,
            identity.dataset_manifest_sha256,
            identity.runtime_config_sha256,
            identity.prompt_manifest_sha256,
            identity.token_limit_config_sha256,
            identity.fixture_manifest_sha256,
            identity.environment_sha256,
        )
        if actual != expected:
            raise SystemRunnerError("episode identity does not match frozen plan")
        prior = self._run_owners.get(identity.run_id)
        if prior is not None and prior != system_id:
            raise SystemRunnerError("system run identity was shared")
        self._run_owners[identity.run_id] = system_id

    @staticmethod
    def _register(
        registry: dict[str, tuple[SystemId, str]],
        identity: str,
        owner: tuple[SystemId, str],
    ) -> None:
        prior = registry.get(identity)
        if prior is not None and prior != owner:
            raise SystemRunnerError("cross-episode identity reuse")
        registry[identity] = owner

    @classmethod
    def _register_many(
        cls,
        registry: dict[str, tuple[SystemId, str]],
        identities: tuple[str, ...],
        owner: tuple[SystemId, str],
    ) -> None:
        for identity in identities:
            cls._register(registry, identity, owner)

    def _scope_id(self, system_id: SystemId) -> str:
        return evaluation_scope_id(self.plan, system_id)

    def _episode_id(self, system_id: SystemId, scenario_id: str) -> str:
        return "test-episode-" + canonical_sha256(
            [self.plan.plan_sha256, system_id, scenario_id]
        )[7:]

    def _isolation_id(self, system_id: SystemId, scenario_id: str) -> str:
        return "test-isolation-" + canonical_sha256(
            [self.plan.plan_sha256, system_id, scenario_id]
        )[7:]


def evaluation_scope_id(plan: FinalEvaluationPlan, system_id: SystemId) -> str:
    """Stable scope identity used to persist a pre-dispatch timing snapshot."""

    return "final-evaluation-" + canonical_sha256(
        [plan.plan_sha256, system_id]
    )[7:]


def execution_attestation_sha256(
    plan: FinalEvaluationPlan, system_id: SystemId
) -> str:
    """Bind a constructed executor to every runtime-relevant frozen plan hash."""

    system = plan.systems[SYSTEM_ORDER.index(system_id)]
    return canonical_sha256(
        {
            "system": system.model_dump(mode="json"),
            "profile": plan.profile,
            "test_dataset_manifest_sha256": plan.test_dataset_manifest_sha256,
            "family_membership_sha256": plan.family_membership_sha256,
            "category_membership_sha256": plan.category_membership_sha256,
            "toolsandbox_source_sha256": plan.toolsandbox_source_sha256,
            "dependency_lock_sha256": plan.dependency_lock_sha256,
            "container_image_sha256": plan.container_image_sha256,
            "environment_sha256": plan.environment_sha256,
            "world_clock_sha256": plan.world_clock_sha256,
            "fixture_or_live_tool_config_sha256": plan.fixture_or_live_tool_config_sha256,
            "qwen_identity_sha256": plan.qwen_identity_sha256,
            "qwen_decoding_sha256": plan.qwen_decoding_sha256,
            "embedding_identity_sha256": plan.embedding_identity_sha256,
            "user_simulator_identity_sha256": plan.user_simulator_identity_sha256,
            "user_prompt_sha256": plan.user_prompt_sha256,
            "user_few_shot_sha256": plan.user_few_shot_sha256,
            "user_tools_sha256": plan.user_tools_sha256,
            "user_stop_behavior_sha256": plan.user_stop_behavior_sha256,
            "prompt_registry_sha256": plan.prompt_registry_sha256,
            "output_schema_registry_sha256": plan.output_schema_registry_sha256,
            "calibrated_token_limit_registry_sha256": plan.calibrated_token_limit_registry_sha256,
            "shared_pipeline_config_sha256": plan.shared_pipeline_config_sha256,
            "checkpoint_schema_sha256": plan.checkpoint_schema_sha256,
            "metrics_schema_sha256": plan.metrics_schema_sha256,
            "reporting_schema_sha256": plan.reporting_schema_sha256,
            "process_count": plan.process_count,
            "process_order": plan.process_order,
            "seed_policy": plan.seed_policy,
        }
    )


__all__ = [
    "ReconciliationRequired",
    "PersistedSystemResult",
    "RestrictedSystemResultStore",
    "StagedSystemMaterial",
    "SystemAccountingProjection",
    "SystemEpisodeExecutor",
    "SystemExecutionAttestation",
    "SystemRunResult",
    "SystemResultStore",
    "SystemRunner",
    "SystemRunnerError",
    "TerminalEpisodeFailure",
    "evaluation_scope_id",
    "execution_attestation_sha256",
]
