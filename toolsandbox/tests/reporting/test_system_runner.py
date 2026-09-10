from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from toolsandbox_pipeline.checkpointing import effective_effect_id
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.reporting.evaluation_plan import (
    CoordinatorRegistryAuthority,
    TestOnceGuard as OnceGuard,
)
from toolsandbox_pipeline.reporting.system_runner import (
    PersistedSystemResult,
    StagedSystemMaterial,
    ReconciliationRequired,
    SystemExecutionAttestation,
    SystemRunner,
    TerminalEpisodeFailure,
    evaluation_scope_id,
    execution_attestation_sha256,
)
from toolsandbox_pipeline.schemas.accounting import (
    LogicalRequestAccountingInput,
    PhysicalAttemptAccountingInput,
    ScopeTimingInput,
    TaskAccountingInput,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.checkpoint import (
    BlobReference,
    LLMResponseApplication,
    QwenEffectiveEffect,
    QwenEffectKind,
)
from toolsandbox_pipeline.schemas.ledger_accounting import (
    AccountingScope,
    Task011AccountingSnapshot,
    Task011LedgerPopulation,
)
from toolsandbox_pipeline.schemas.reporting import (
    PIPELINE_COMPONENTS,
    SYSTEM_ORDER,
    VANILLA_COMPONENTS,
    FinalEvaluationPlan,
    SystemPlan,
    TestScenarioPlan as ScenarioPlan,
    category_membership_sha256,
    family_membership_sha256,
)
from toolsandbox_pipeline.schemas.dataset import VARIANTS
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeExecutionStatus,
    EpisodeIdentity,
    EpisodeResult,
    OnlineTurnRecord,
    TrustedEvaluatorRecord,
    TrustedTrajectory,
)
from toolsandbox_pipeline.schemas.usage import TokenUsage
from toolsandbox_pipeline.reporting.vanilla_responder import VanillaDecisionRecord


D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64


class FakeSystemResultStore:
    def __init__(self):
        self.values = {}

    def persist(self, result):
        assert type(result) is PersistedSystemResult
        payload = canonical_json_bytes(result.model_dump(mode="json"))
        reference = BlobReference(
            sha256=canonical_sha256(result.model_dump(mode="json")),
            byte_count=len(payload),
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="PersistedSystemResult",
            schema_version=1,
            content_visibility="restricted",
        )
        self.values[reference.sha256] = result
        return reference

    def load(self, reference):
        return self.values[reference.sha256]

    def persist_material(self, material):
        assert type(material) is StagedSystemMaterial
        payload = canonical_json_bytes(material.model_dump(mode="json"))
        reference = BlobReference(
            sha256=canonical_sha256(material.model_dump(mode="json")),
            byte_count=len(payload),
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="StagedSystemMaterial",
            schema_version=1,
            content_visibility="restricted",
        )
        self.values[reference.sha256] = material
        return reference

    def load_material(self, reference):
        return self.values[reference.sha256]


def build_plan(output_root: Path, *, profile="strict_replay") -> FinalEvaluationPlan:
    scenarios = tuple(
        ScenarioPlan(
            manifest_position=index,
            scenario_id=f"family-{index // 8:02d}-variant-{index % 8}",
            scenario_family_id=f"family-{index // 8:02d}",
            variant=VARIANTS[index % 8],
            categories=("category",),
            starting_context_sha256=D1,
            evaluation_definition_sha256=D1,
            agent_tool_schema_sha256=D1,
        )
        for index in range(200)
    )
    return FinalEvaluationPlan.build(
        protocol_version="three-system-final-evaluation-v1",
        user_approval_reference="approval-1",
        frozen_at_utc="2026-09-10T00:00:00Z",
        profile=profile,
        training_run_id="train-run",
        training_run_sha256=D1,
        g000_sha256=D1,
        g003_sha256=D2,
        checkpoint_observation_registry_sha256=D1,
        ordered_system_ids=SYSTEM_ORDER,
        systems=(
            SystemPlan(
                system_id="vanilla",
                generation_id="g000",
                allowed_components=VANILLA_COMPONENTS,
                prompt_manifest_sha256=D1,
                token_limit_config_sha256=D1,
                token_limit_status="calibrated",
            ),
            SystemPlan(
                system_id="generation_0",
                generation_id="g000",
                allowed_components=PIPELINE_COMPONENTS,
                prompt_manifest_sha256=D2,
                token_limit_config_sha256=D2,
                token_limit_status="calibrated",
            ),
            SystemPlan(
                system_id="updated",
                generation_id="g003",
                allowed_components=PIPELINE_COMPONENTS,
                prompt_manifest_sha256=D2,
                token_limit_config_sha256=D2,
                token_limit_status="calibrated",
            ),
        ),
        test_dataset_manifest_sha256=D1,
        scenarios=scenarios,
        family_membership_sha256=family_membership_sha256(scenarios),
        category_membership_sha256=category_membership_sha256(scenarios),
        toolsandbox_source_sha256=D1,
        dependency_lock_sha256=D1,
        container_image_sha256=D1,
        environment_sha256=D1,
        world_clock_sha256=D1,
        fixture_or_live_tool_config_sha256=D1,
        qwen_identity_sha256=D1,
        qwen_decoding_sha256=D1,
        embedding_identity_sha256=D1,
        user_simulator_identity_sha256=D1,
        user_prompt_sha256=D1,
        user_few_shot_sha256=D1,
        user_tools_sha256=D1,
        user_stop_behavior_sha256=D1,
        prompt_registry_sha256=D1,
        output_schema_registry_sha256=D1,
        calibrated_token_limit_registry_sha256=D1,
        shared_pipeline_config_sha256=D1,
        checkpoint_schema_sha256=D1,
        metrics_schema_sha256=D1,
        reporting_schema_sha256=D1,
        process_count=1,
        process_order="test_manifest_order",
        seed_policy="scenario_id_sha256_v1",
        output_root=str(output_root),
    )


def accounting_snapshot(identity, attempts=(), logical=(), effects=()):
    applications = tuple(
        LLMResponseApplication(
            application_id=item.application_id,
            logical_request_id=item.logical_request_id,
            source_attempt_id=item.source_attempt_id,
            application_artifact_id="artifact-" + item.application_id,
            application_artifact_sha256=D1,
            committed_checkpoint_id="checkpoint-" + item.application_id,
            created_at_utc=datetime(2026, 9, 10, tzinfo=timezone.utc),
        )
        for item in logical
        if item.application_id is not None
    )
    population = Task011LedgerPopulation.build(
        scope=AccountingScope(
            run_id=identity.run_id,
            round_index=identity.round_index,
            task_id=identity.episode_id,
            scenario_family_id=identity.family_id,
            scenario_id=identity.scenario_id,
            system_variant=identity.system_variant,
        ),
        high_water_identity=D1,
        logical_request_ids=tuple(item.logical_request_id for item in logical),
        physical_attempt_ids=tuple(item.attempt_id for item in attempts),
        application_ids=tuple(item.application_id for item in applications),
        effect_ids=tuple(item.effect_id for item in effects),
    )
    encoded = canonical_json_bytes(population.model_dump(mode="json"))
    reference = BlobReference(
        sha256=canonical_sha256(population.model_dump(mode="json")),
        byte_count=len(encoded),
        media_type="application/vnd.toolsandbox.canonical+json",
        schema_name="Task011LedgerPopulation",
        schema_version=1,
        content_visibility="restricted",
    )
    return Task011AccountingSnapshot(
        population=population,
        population_reference=reference,
        physical_attempts=attempts,
        logical_requests=logical,
        applications=applications,
        substantive_effects=effects,
    )


def blob(digest: str, schema_name: str) -> BlobReference:
    return BlobReference(
        sha256=digest,
        byte_count=1,
        media_type="application/vnd.toolsandbox.canonical+json",
        schema_name=schema_name,
        schema_version=1,
        content_visibility="restricted",
    )


def task_timing(episode_id: str) -> ScopeTimingInput:
    return ScopeTimingInput(
        scope_kind="scenario_task",
        scope_id=episode_id,
        boot_id="boot",
        started_at_utc=datetime(2026, 9, 10, tzinfo=timezone.utc),
        start_monotonic_ns=1,
        timing_complete=False,
    )


class FakeExecutor:
    def __init__(self, plan, system_id, order_log, *, crash_on_call=None, failure=None):
        self.plan = plan
        self.system_id = system_id
        self.order_log = order_log
        self.crash_on_call = crash_on_call
        self.failure = failure
        self.call_count = 0
        self.dispatch_count = 0
        self.results = {}
        self.trajectories = {}
        self.evaluators = {}
        self.isolation_ids = []

    def execution_attestation(self):
        system = self.plan.systems[SYSTEM_ORDER.index(self.system_id)]
        return SystemExecutionAttestation(
            system_id=self.system_id,
            generation_id=system.generation_id,
            active_components=system.allowed_components,
            prompt_manifest_sha256=system.prompt_manifest_sha256,
            token_limit_config_sha256=system.token_limit_config_sha256,
            environment_sha256=self.plan.environment_sha256,
            fixture_or_live_tool_config_sha256=self.plan.fixture_or_live_tool_config_sha256,
            shared_pipeline_config_sha256=self.plan.shared_pipeline_config_sha256,
            qwen_identity_sha256=self.plan.qwen_identity_sha256,
            embedding_identity_sha256=self.plan.embedding_identity_sha256,
            user_simulator_identity_sha256=self.plan.user_simulator_identity_sha256,
            qwen_provider="vllm_openai_compatible",
            qwen_model="Qwen/Qwen3-32B",
            frozen_execution_sha256=execution_attestation_sha256(
                self.plan, self.system_id
            ),
        )

    def run_episode(self, *, system_id, scenario, episode_id, isolation_id):
        self.call_count += 1
        self.order_log.append((system_id, scenario.manifest_position, episode_id))
        self.isolation_ids.append(isolation_id)
        if episode_id in self.results:
            return self.results[episode_id]
        if self.crash_on_call == self.call_count:
            self.crash_on_call = None
            raise SystemExit("synthetic process interruption")
        if self.failure is not None:
            raise self.failure("synthetic")
        self.dispatch_count += 1
        system = self.plan.systems[SYSTEM_ORDER.index(system_id)]
        identity = EpisodeIdentity(
            run_id=f"final-{system_id}",
            profile=self.plan.profile,
            phase="final_test",
            family_id=scenario.scenario_family_id,
            scenario_id=scenario.scenario_id,
            episode_id=episode_id,
            manifest_position=scenario.manifest_position,
            system_variant=system_id,
            generation_id=system.generation_id,
            starting_context_sha256=scenario.starting_context_sha256,
            evaluation_definition_sha256=scenario.evaluation_definition_sha256,
            agent_tool_schema_sha256=scenario.agent_tool_schema_sha256,
            dataset_manifest_sha256=self.plan.test_dataset_manifest_sha256,
            runtime_config_sha256=self.plan.shared_pipeline_config_sha256,
            prompt_manifest_sha256=system.prompt_manifest_sha256,
            token_limit_config_sha256=system.token_limit_config_sha256,
            fixture_manifest_sha256=self.plan.fixture_or_live_tool_config_sha256,
            environment_sha256=self.plan.environment_sha256,
            max_messages=4,
        )
        ending = blob(canonical_sha256([episode_id, "ending"]), "ExecutionContextEnvelope")
        evaluator = TrustedEvaluatorRecord(
            milestone_similarity=1.0,
            minefield_similarity=0.0,
            similarity=1.0,
            turn_count=1,
            milestone_mapping=(),
            minefield_mapping=(),
            fully_successful=True,
            evaluation_definition_sha256=scenario.evaluation_definition_sha256,
            ending_context_sha256=ending.sha256,
        )
        evaluator_ref = blob(
            canonical_sha256(evaluator.model_dump(mode="json")),
            "TrustedEvaluatorRecord",
        )
        trajectory = TrustedTrajectory.build(
            identity=identity,
            messages=(),
            online_turns=(),
            tool_actions=(),
            logical_request_ids=(),
            physical_attempt_ids=(),
            ending_context_reference=ending,
            ending_context_sha256=ending.sha256,
            evaluator_record_reference=evaluator_ref,
            evaluator_record_sha256=evaluator_ref.sha256,
            skill_attributions=(),
            eligible_for_train_offline_consumption=False,
        )
        trajectory_ref = blob(
            canonical_sha256(trajectory.model_dump(mode="json")), "TrustedTrajectory"
        )
        result = EpisodeResult(
            identity=identity,
            status=EpisodeExecutionStatus.COMPLETED_EVALUATED,
            ending_context_reference=ending,
            ending_context_sha256=ending.sha256,
            trusted_trajectory_reference=trajectory_ref,
            evaluator_record_reference=evaluator_ref,
            task_accounting_input=TaskAccountingInput(
                run_id=identity.run_id,
                task_id=episode_id,
                scenario_family_id=identity.family_id,
                scenario_id=identity.scenario_id,
                system_variant=system_id,
                timing=task_timing(episode_id),
                completion_status="complete",
                evaluator_result_sha256=evaluator_ref.sha256,
                final_context_sha256=ending.sha256,
            ),
            last_checkpoint_ordinal=1,
        )
        self.results[episode_id] = result
        self.trajectories[episode_id] = trajectory
        self.evaluators[episode_id] = evaluator
        return result

    def load_trajectory(self, result):
        return self.trajectories[result.identity.episode_id]

    def load_evaluator(self, result):
        return self.evaluators[result.identity.episode_id]

    def load_online_decision(self, decision_reference):
        raise AssertionError("empty synthetic trajectory has no decisions")

    def load_external_attempt(self, attempt_id):
        raise AssertionError("empty synthetic trajectory has no external attempts")

    def load_accounting_snapshot(self, result):
        return accounting_snapshot(result.identity)


class AccountingExecutor(FakeExecutor):
    def __init__(self, *args, mode="complete", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.projections = {}
        self.decisions = {}

    @staticmethod
    def _request_ids(episode_id, suffix):
        value = canonical_sha256([episode_id, suffix])[7:]
        return (
            "llm-" + value,
            "attempt-" + value + "-00000001",
            "application-" + value,
        )

    @staticmethod
    def _attempt(identity, logical_id, attempt_id, role, provider, model, usage):
        now = datetime(2026, 9, 10, tzinfo=timezone.utc)
        return PhysicalAttemptAccountingInput(
            run_id=identity.run_id,
            task_id=identity.episode_id,
            scenario_family_id=identity.family_id,
            scenario_id=identity.scenario_id,
            system_variant=identity.system_variant,
            logical_request_id=logical_id,
            attempt_id=attempt_id,
            attempt_ordinal=1,
            role=role,
            phase="final_test",
            provider=provider,
            model=model,
            endpoint_kind="chat",
            dispatched=True,
            replayed_after_unknown_outcome=False,
            status="completed",
            started_at_utc=now,
            completed_at_utc=now,
            latency_seconds=Decimal("0.01"),
            usage=usage,
            response_sha256=D1,
        )

    def run_episode(self, **kwargs):
        episode_id = kwargs["episode_id"]
        if episode_id in self.projections:
            return self.results[episode_id]
        result = super().run_episode(**kwargs)
        base = self.trajectories[episode_id]
        identity = result.identity
        action = ActionEnvelope.model_validate(
            {"action": {"type": "assistant_message", "content": "done"}}
        )
        action_sha = canonical_sha256(action.model_dump(mode="json"))
        logical, attempt, application = self._request_ids(episode_id, "action")
        noop_logical, noop_attempt, noop_application = self._request_ids(
            episode_id, "noop"
        )
        user_logical, user_attempt, _ = self._request_ids(episode_id, "user")
        effect_artifact_id = "action-" + action_sha[7:]
        effect_artifact_sha256 = D2 if self.mode == "wrong_effect" else action_sha
        committed_checkpoint_id = "checkpoint-" + episode_id
        effect_id = effective_effect_id(
            effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
            effect_artifact_id=effect_artifact_id,
            effect_artifact_sha256=effect_artifact_sha256,
            ordered_application_ids=(application,),
            committed_checkpoint_id=committed_checkpoint_id,
        )
        decision = VanillaDecisionRecord(
            state_id="state-" + canonical_sha256(episode_id)[7:],
            action=action,
            logical_request_id=logical,
            source_attempt_id=attempt,
            application_id=application,
            effect_id=effect_id,
        )
        decision_ref = blob(
            canonical_sha256(decision.model_dump(mode="json")),
            "VanillaDecisionRecord",
        )
        online_turn = OnlineTurnRecord(
            agent_turn_index=0,
            state_id=decision.state_id,
            decision_reference=decision_ref,
            decision_sha256=decision_ref.sha256,
            final_action_sha256=action_sha,
            logical_request_ids=(logical,),
            source_attempt_ids=(attempt,),
            application_ids=(application,),
        )
        trajectory = TrustedTrajectory.build(
            identity=base.identity,
            messages=base.messages,
            online_turns=(online_turn,),
            tool_actions=(),
            logical_request_ids=(logical,),
            physical_attempt_ids=(attempt,),
            ending_context_reference=base.ending_context_reference,
            ending_context_sha256=base.ending_context_sha256,
            evaluator_record_reference=base.evaluator_record_reference,
            evaluator_record_sha256=base.evaluator_record_sha256,
            skill_attributions=(),
            eligible_for_train_offline_consumption=False,
        )
        trajectory_ref = blob(
            canonical_sha256(trajectory.model_dump(mode="json")), "TrustedTrajectory"
        )
        result = result.model_copy(
            update={"trusted_trajectory_reference": trajectory_ref}
        )
        complete = TokenUsage(
            input_tokens=5,
            uncached_input_tokens=5,
            cache_read_input_tokens=0,
            cache_write_input_tokens=0,
            output_tokens=3,
            total_tokens=8,
            usage_complete=True,
        )
        noop_usage = TokenUsage(
            input_tokens=4,
            uncached_input_tokens=4,
            cache_read_input_tokens=0,
            cache_write_input_tokens=0,
            output_tokens=2,
            total_tokens=6,
            usage_complete=True,
        )
        user_usage = (
            TokenUsage()
            if self.mode == "missing_usage"
            else TokenUsage(
                input_tokens=2,
                uncached_input_tokens=2,
                cache_read_input_tokens=0,
                cache_write_input_tokens=0,
                output_tokens=2,
                total_tokens=4,
                usage_complete=True,
            )
        )
        attempts = (
            self._attempt(
                identity,
                logical,
                attempt,
                "vanilla",
                "vllm_openai_compatible",
                "Qwen/Qwen3-32B",
                complete,
            ),
            self._attempt(
                identity,
                noop_logical,
                noop_attempt,
                "user_simulator",
                "openai",
                "gpt-4o-mini-2024-07-18",
                noop_usage,
            ),
            self._attempt(
                identity,
                user_logical,
                user_attempt,
                "user_simulator",
                "openai",
                "gpt-4o-mini-2024-07-18",
                user_usage,
            ),
        )
        logicals = (
            LogicalRequestAccountingInput(
                run_id=identity.run_id,
                task_id=identity.episode_id,
                scenario_family_id=identity.family_id,
                scenario_id=identity.scenario_id,
                system_variant=identity.system_variant,
                logical_request_id=logical,
                role="vanilla",
                phase="final_test",
                provider="vllm_openai_compatible",
                model="Qwen/Qwen3-32B",
                status="applied",
                source_attempt_id=attempt,
                application_id=application,
            ),
            LogicalRequestAccountingInput(
                run_id=identity.run_id,
                task_id=identity.episode_id,
                scenario_family_id=identity.family_id,
                scenario_id=identity.scenario_id,
                system_variant=identity.system_variant,
                logical_request_id=noop_logical,
                role="user_simulator",
                phase="final_test",
                provider="openai",
                model="gpt-4o-mini-2024-07-18",
                status="applied",
                source_attempt_id=noop_attempt,
                application_id=noop_application,
            ),
            LogicalRequestAccountingInput(
                run_id=identity.run_id,
                task_id=identity.episode_id,
                scenario_family_id=identity.family_id,
                scenario_id=identity.scenario_id,
                system_variant=identity.system_variant,
                logical_request_id=user_logical,
                role="user_simulator",
                phase="final_test",
                provider="openai",
                model="gpt-4o-mini-2024-07-18",
                status="response_completed",
            ),
        )
        effect = QwenEffectiveEffect(
            effect_id=effect_id,
            effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
            effect_artifact_id=effect_artifact_id,
            effect_artifact_sha256=effect_artifact_sha256,
            ordered_application_ids=(application,),
            committed_checkpoint_id=committed_checkpoint_id,
        )
        effects = () if self.mode == "unlinked" else (effect,)
        self.results[episode_id] = result
        self.trajectories[episode_id] = trajectory
        self.decisions[decision_ref.sha256] = decision
        self.projections[episode_id] = accounting_snapshot(
            identity, attempts, logicals, effects
        )
        return result

    def load_online_decision(self, decision_reference):
        return self.decisions[decision_reference.sha256]

    def load_accounting_snapshot(self, result):
        return self.projections[result.identity.episode_id]


def make_runner(tmp_path, *, profile="strict_replay", crash_on_call=None, failure=None):
    plan = build_plan(tmp_path / "out", profile=profile)
    authority = CoordinatorRegistryAuthority.create(
        tmp_path / "registry", authority_id="coordinator-1"
    )
    guard = OnceGuard.create(authority, plan)
    order = []
    executors = {
        system_id: FakeExecutor(
            plan,
            system_id,
            order,
            crash_on_call=crash_on_call if system_id == "vanilla" else None,
            failure=failure if system_id == "vanilla" else None,
        )
        for system_id in SYSTEM_ORDER
    }
    runner = SystemRunner(
        plan=plan,
        guard=guard,
        executors=executors,
        boot_id="boot",
        result_store=FakeSystemResultStore(),
    )
    return plan, guard, executors, order, runner


def test_runs_fixed_system_and_manifest_order_with_fresh_identities(tmp_path):
    _, guard, executors, order, runner = make_runner(tmp_path)
    results = runner.run_all()
    assert tuple(item.system_id for item in results) == SYSTEM_ORDER
    assert [item.records[0].manifest_position for item in results] == [0, 0, 0]
    assert [item.records[-1].manifest_position for item in results] == [199, 199, 199]
    assert [system for system, position, _ in order if position == 0] == list(SYSTEM_ORDER)
    assert len({episode for _, _, episode in order}) == 600
    assert len({item for executor in executors.values() for item in executor.isolation_ids}) == 600
    assert all(executor.dispatch_count == 200 for executor in executors.values())
    assert all(result.accounting.totals.total_tokens == 0 for result in results)
    assert all(result.accounting.totals.total_cost == 0 for result in results)
    assert all(result.timing.timing_complete for result in results)
    assert all(guard.state(system_id) == "completed" for system_id in SYSTEM_ORDER)
    with pytest.raises(Exception, match="fresh run|cannot execute"):
        runner.run_all()


def test_process_crash_resumes_same_episode_without_redispatch(tmp_path):
    plan, guard, executors, order, runner = make_runner(tmp_path, crash_on_call=2)
    with pytest.raises(SystemExit):
        runner.run_system("vanilla")
    assert guard.state("vanilla") == "running"
    assert executors["vanilla"].dispatch_count == 1
    first_episode = order[0][2]
    reopened = OnceGuard.open(guard.authority, plan)
    with pytest.raises(Exception, match="persisted evaluation timing"):
        SystemRunner(
            plan=plan,
            guard=reopened,
            executors=executors,
            boot_id="boot-resume",
            result_store=runner.result_store,
        ).resume_incomplete()
    resumed = SystemRunner(
        plan=plan,
        guard=reopened,
        executors=executors,
        boot_id="boot-resume",
        result_store=runner.result_store,
    ).resume_incomplete(
        resume_timings={
            "vanilla": ScopeTimingInput(
                scope_kind="evaluation_run",
                scope_id=evaluation_scope_id(plan, "vanilla"),
                boot_id="boot",
                started_at_utc=datetime(2026, 9, 10, tzinfo=timezone.utc),
                start_monotonic_ns=1,
                timing_complete=False,
            )
        }
    )
    assert tuple(item.system_id for item in resumed) == SYSTEM_ORDER
    assert executors["vanilla"].dispatch_count == 200
    assert order[2][2] == first_episode
    assert all(reopened.state(system_id) == "completed" for system_id in SYSTEM_ORDER)


def test_staged_completion_crash_restores_result_without_episode_dispatch(
    tmp_path, monkeypatch
):
    plan, guard, executors, _, runner = make_runner(tmp_path)
    original_complete = guard.complete_system
    crashed = False

    def crash_after_stage(system_id, receipt):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise SystemExit("synthetic completion-marker crash")
        return original_complete(system_id, receipt)

    monkeypatch.setattr(guard, "complete_system", crash_after_stage)
    with pytest.raises(SystemExit, match="completion-marker"):
        runner.run_system("vanilla")
    assert guard.state("vanilla") == "running"
    assert guard.completion_receipt("vanilla") is not None
    dispatches = executors["vanilla"].dispatch_count
    monkeypatch.setattr(guard, "complete_system", original_complete)
    restored = SystemRunner(
        plan=plan,
        guard=guard,
        executors=executors,
        boot_id="new-boot",
        result_store=runner.result_store,
    ).run_system("vanilla")
    assert guard.state("vanilla") == "completed"
    assert executors["vanilla"].dispatch_count == dispatches
    assert restored.result_reference == guard.completion_receipt(
        "vanilla"
    ).result_reference
    assert restored.timing.timing_complete


def test_material_checkpoint_crash_resumes_without_dispatch_and_marks_cross_boot_timing_incomplete(
    tmp_path, monkeypatch
):
    plan, guard, executors, _, runner = make_runner(tmp_path)
    original_stage = guard.stage_system_material

    def crash_after_material(system_id, reference):
        original_stage(system_id, reference)
        raise SystemExit("synthetic material-checkpoint crash")

    monkeypatch.setattr(guard, "stage_system_material", crash_after_material)
    with pytest.raises(SystemExit, match="material-checkpoint"):
        runner.run_system("vanilla")
    assert guard.staged_material_reference("vanilla") is not None
    assert guard.completion_receipt("vanilla") is None
    dispatches = executors["vanilla"].dispatch_count
    monkeypatch.setattr(guard, "stage_system_material", original_stage)
    resumed = SystemRunner(
        plan=plan,
        guard=guard,
        executors=executors,
        boot_id="new-boot",
        result_store=runner.result_store,
    ).run_system(
        "vanilla",
        resume_timing=ScopeTimingInput(
            scope_kind="evaluation_run",
            scope_id=evaluation_scope_id(plan, "vanilla"),
            boot_id="old-boot",
            started_at_utc=datetime(2026, 9, 10, tzinfo=timezone.utc),
            start_monotonic_ns=1,
            timing_complete=False,
        ),
    )
    assert executors["vanilla"].dispatch_count == dispatches
    assert resumed.timing.timing_complete is False
    assert guard.state("vanilla") == "completed"


@pytest.mark.parametrize(
    "mode,expected_tokens,usage_complete",
    [("complete", 3600, True), ("missing_usage", None, False)],
)
def test_accounting_includes_all_attempts_but_costs_only_linked_qwen(
    tmp_path, mode, expected_tokens, usage_complete
):
    plan = build_plan(tmp_path / mode / "out")
    authority = CoordinatorRegistryAuthority.create(
        tmp_path / mode / "registry", authority_id="coordinator-1"
    )
    guard = OnceGuard.create(authority, plan)
    order = []
    executors = {
        system_id: (
            AccountingExecutor(plan, system_id, order, mode=mode)
            if system_id == "vanilla"
            else FakeExecutor(plan, system_id, order)
        )
        for system_id in SYSTEM_ORDER
    }
    result = SystemRunner(
        plan=plan,
        guard=guard,
        executors=executors,
        boot_id="boot",
        result_store=FakeSystemResultStore(),
    ).run_system("vanilla")
    totals = result.accounting.totals
    assert totals.total_tokens == expected_tokens
    assert totals.usage_complete is usage_complete
    assert totals.total_cost == 600
    assert totals.cost_complete is True
    assert totals.physical_dispatch_count == 600
    assert totals.applied_logical_response_count == 400


@pytest.mark.parametrize("mode", ["unlinked", "wrong_effect"])
def test_unlinked_or_mismatched_action_effect_is_terminal(tmp_path, mode):
    plan = build_plan(tmp_path / mode / "out")
    authority = CoordinatorRegistryAuthority.create(
        tmp_path / mode / "registry", authority_id="coordinator-1"
    )
    guard = OnceGuard.create(authority, plan)
    order = []
    executors = {
        system_id: (
            AccountingExecutor(plan, system_id, order, mode=mode)
            if system_id == "vanilla"
            else FakeExecutor(plan, system_id, order)
        )
        for system_id in SYSTEM_ORDER
    }
    with pytest.raises(TerminalEpisodeFailure, match="effect"):
        SystemRunner(
            plan=plan,
            guard=guard,
            executors=executors,
            boot_id="boot",
            result_store=FakeSystemResultStore(),
        ).run_system("vanilla")
    assert guard.state("vanilla") == "terminal_failure"


def test_terminal_and_official_live_reconciliation_are_durable(tmp_path):
    _, guard, _, _, runner = make_runner(tmp_path / "terminal", failure=RuntimeError)
    with pytest.raises(TerminalEpisodeFailure):
        runner.run_system("vanilla")
    assert guard.state("vanilla") == "terminal_failure"

    _, live_guard, _, _, live = make_runner(
        tmp_path / "live", profile="official_live", failure=ReconciliationRequired
    )
    with pytest.raises(ReconciliationRequired):
        live.run_system("vanilla")
    assert live_guard.state("vanilla") == "reconciliation_required"


def test_attestation_drift_is_rejected_before_any_episode(tmp_path):
    plan, guard, executors, _, _ = make_runner(tmp_path)
    original = executors["vanilla"].execution_attestation

    def drifted():
        return original().model_copy(update={"active_components": ("critic",)})

    executors["vanilla"].execution_attestation = drifted
    with pytest.raises(ValueError, match="attestation"):
        SystemRunner(
            plan=plan,
            guard=guard,
            executors=executors,
            boot_id="boot",
            result_store=FakeSystemResultStore(),
        )
    assert executors["vanilla"].dispatch_count == 0
    executors["vanilla"].execution_attestation = original

    def runtime_hash_drifted():
        return original().model_copy(update={"frozen_execution_sha256": D2})

    executors["vanilla"].execution_attestation = runtime_hash_drifted
    with pytest.raises(ValueError, match="attestation hash"):
        SystemRunner(
            plan=plan,
            guard=guard,
            executors=executors,
            boot_id="boot",
            result_store=FakeSystemResultStore(),
        )
