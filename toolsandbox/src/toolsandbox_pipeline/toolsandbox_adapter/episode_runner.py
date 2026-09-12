"""Recoverable execution of one explicitly supplied ToolSandbox scenario."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
)
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.metrics.timing import ScopeTimer
from toolsandbox_pipeline.reproducibility.scenario_hashes import context_sha256
from toolsandbox_pipeline.schemas.accounting import ScopeTimingInput, TaskAccountingInput
from toolsandbox_pipeline.schemas.dataset import ScenarioRecord
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeExecutionStatus,
    EpisodeIdentity,
    EpisodeResult,
    EpisodeResumeInput,
    NativeMessageRecord,
    OnlineTurnRecord,
    SkillUseAttribution,
    ToolActionRecord,
    TrustedEvaluatorRecord,
    TrustedTrajectory,
)

from .native_evaluator import NativeEvaluator
from .native_loop import NativeLoop, RoleBoundary
from .trajectory_store import StoredContext, TrajectoryStore
from .transactional_roles import (
    ToolReconciliationRequired,
    EpisodeAgentRole,
    TransactionalAgentRole,
    TransactionalExecutionEnvironment,
    TransactionalUserRole,
)


class EpisodeRunnerError(RuntimeError):
    """Sanitized episode orchestration failure."""


PhysicalAttemptProvider = Callable[[tuple[str, ...]], tuple[str, ...]]


@dataclass(frozen=True)
class EpisodeRunInput:
    scenario: object
    manifest_record: ScenarioRecord
    identity: EpisodeIdentity
    roles: Mapping[RoleType, BaseRole]
    resume: EpisodeResumeInput | None = None
    resume_timing: ScopeTimingInput | None = None
    skill_versions: Mapping[str, str] | None = None
    eligible_for_train_offline_consumption: bool = False


class EpisodeRunner:
    def __init__(
        self,
        *,
        trajectory_store: TrajectoryStore,
        native_evaluator: NativeEvaluator,
        physical_attempt_provider: PhysicalAttemptProvider,
        boot_id: str,
        native_loop: NativeLoop | None = None,
        timer_factory: Callable[..., ScopeTimer] = ScopeTimer,
    ) -> None:
        if type(trajectory_store) is not TrajectoryStore:
            raise TypeError("TrajectoryStore required")
        if type(native_evaluator) is not NativeEvaluator:
            raise TypeError("NativeEvaluator required")
        if not boot_id:
            raise ValueError("boot identity required")
        self.trajectory_store = trajectory_store
        self.native_evaluator = native_evaluator
        self.physical_attempt_provider = physical_attempt_provider
        self.boot_id = boot_id
        self.native_loop = native_loop or NativeLoop()
        self.timer_factory = timer_factory

    def run(self, request: EpisodeRunInput) -> EpisodeResult:
        if type(request) is not EpisodeRunInput:
            raise TypeError("EpisodeRunInput required")
        identity = request.identity
        identity.validate_manifest_record(request.manifest_record)
        self._validate_roles(request.roles)
        scenario = request.scenario
        if getattr(scenario, "max_messages", None) != identity.max_messages:
            raise EpisodeRunnerError("scenario message limit mismatch")
        starting_context = getattr(scenario, "starting_context", None)
        if context_sha256(starting_context) != identity.starting_context_sha256:
            raise EpisodeRunnerError("starting context identity mismatch")
        timer = self.timer_factory(
            "scenario_task",
            identity.episode_id,
            self.boot_id,
            stored=request.resume_timing,
        )
        boundary_ordinal = 0
        installed_context: object | None = None
        latest_committed_context: StoredContext | None = None
        last_checkpoint_ordinal = (
            0 if request.resume is None else request.resume.last_checkpoint_ordinal
        )

        def boundary_hook(boundary: RoleBoundary, context: object) -> None:
            nonlocal boundary_ordinal, last_checkpoint_ordinal, installed_context
            nonlocal latest_committed_context
            boundary_ordinal += 1
            stored = self.trajectory_store.persist_context(context)
            event = self.trajectory_store.commit_context_checkpoint(
                identity,
                event_kind=(
                    "native_initial_system_boundary_committed"
                    if boundary.initial_system_setup
                    else "native_role_boundary_committed"
                ),
                stored=stored,
                recipient=boundary.recipient.value,
                boundary_ordinal=boundary_ordinal,
            )
            installed_context = context
            latest_committed_context = stored
            last_checkpoint_ordinal = max(last_checkpoint_ordinal, event.event_ordinal)

        def context_install_hook(context: object) -> None:
            nonlocal installed_context, latest_committed_context
            nonlocal last_checkpoint_ordinal
            installed_context = context
            stored = self.trajectory_store.persist_context(context)
            event = self.trajectory_store.commit_context_checkpoint(
                identity,
                event_kind="episode_context_installed",
                stored=stored,
                recipient=None,
                boundary_ordinal=0,
            )
            latest_committed_context = stored
            last_checkpoint_ordinal = max(last_checkpoint_ordinal, event.event_ordinal)

        try:
            if request.resume is None:
                loop_result = self.native_loop.run_fresh(
                    scenario,
                    request.roles,
                    boundary_hook=boundary_hook,
                    context_install_hook=context_install_hook,
                )
            else:
                resume = request.resume
                if resume.identity != identity:
                    raise EpisodeRunnerError("resume episode identity mismatch")
                restored = self.trajectory_store.load_context(
                    StoredContext(
                        resume.committed_context_reference,
                        resume.committed_context_sha256,
                    )
                )
                self._validate_resume_recipient(restored, resume.next_recipient)
                loop_result = self.native_loop.run_resumed(
                    restored,
                    request.roles,
                    max_messages=identity.max_messages,
                    initial_max_message_index=resume.initial_max_message_index,
                    boundary_hook=boundary_hook,
                    context_install_hook=context_install_hook,
                )
            if not loop_result.normally_terminated:
                raise EpisodeRunnerError("native loop did not terminate normally")
            ending = self.trajectory_store.persist_context(loop_result.context)
            evaluation = self.native_evaluator.evaluate(
                identity=identity,
                scenario=scenario,
                ending_context=loop_result.context,
            )
            last_checkpoint_ordinal = max(
                last_checkpoint_ordinal, evaluation.checkpoint_ordinal
            )
            messages = self._messages(loop_result.context)
            agent = request.roles[RoleType.AGENT]
            environment = request.roles[RoleType.EXECUTION_ENVIRONMENT]
            assert isinstance(agent, EpisodeAgentRole)
            assert isinstance(environment, TransactionalExecutionEnvironment)
            online_turns = tuple(agent.records)
            tool_actions = tuple(environment.records)
            online_turns = bind_executed_calls_to_turns(
                online_turns, tool_actions
            )
            logical_request_ids = tuple(
                request_id
                for turn in online_turns
                for request_id in turn.logical_request_ids
            )
            physical_attempt_ids = self.physical_attempt_provider(
                logical_request_ids
            )
            if len(set(physical_attempt_ids)) != len(physical_attempt_ids):
                raise EpisodeRunnerError("duplicate physical attempt identity")
            evaluator_sha256 = evaluation.reference.sha256
            skill_attributions = build_skill_attributions(
                tool_actions,
                evaluator_record=evaluation.record,
                evaluator_record_sha256=evaluator_sha256,
                generation_id=identity.generation_id,
                skill_versions=request.skill_versions or {},
            )
            trajectory = TrustedTrajectory.build(
                identity=identity,
                messages=messages,
                online_turns=online_turns,
                tool_actions=tool_actions,
                logical_request_ids=logical_request_ids,
                physical_attempt_ids=physical_attempt_ids,
                ending_context_reference=ending.reference,
                ending_context_sha256=ending.context_sha256,
                evaluator_record_reference=evaluation.reference,
                evaluator_record_sha256=evaluator_sha256,
                skill_attributions=skill_attributions,
                eligible_for_train_offline_consumption=(
                    request.eligible_for_train_offline_consumption
                ),
            )
            trajectory_reference, final_event = self.trajectory_store.persist_trajectory(
                identity, trajectory
            )
            last_checkpoint_ordinal = max(
                last_checkpoint_ordinal, final_event.event_ordinal
            )
            timing = timer.close()
            accounting = self._accounting(
                identity,
                timing,
                completion_status="complete",
                evaluator_sha256=evaluator_sha256,
                final_context_sha256=ending.context_sha256,
            )
            return EpisodeResult(
                identity=identity,
                status=EpisodeExecutionStatus.COMPLETED_EVALUATED,
                ending_context_reference=ending.reference,
                ending_context_sha256=ending.context_sha256,
                trusted_trajectory_reference=trajectory_reference,
                evaluator_record_reference=evaluation.reference,
                task_accounting_input=accounting,
                last_checkpoint_ordinal=last_checkpoint_ordinal,
            )
        except Exception as error:
            if installed_context is None or latest_committed_context is None:
                raise
            status = (
                EpisodeExecutionStatus.RECONCILIATION_REQUIRED
                if isinstance(error, ToolReconciliationRequired)
                else EpisodeExecutionStatus.TERMINAL_FAILURE_BEFORE_EVALUATION
            )
            failure_class = (
                "ToolReconciliationRequired"
                if status is EpisodeExecutionStatus.RECONCILIATION_REQUIRED
                else type(error).__name__
            )
            ending = latest_committed_context
            boundary_ordinal += 1
            event = self.trajectory_store.commit_context_checkpoint(
                identity,
                event_kind=status.value,
                stored=ending,
                recipient=None,
                boundary_ordinal=boundary_ordinal,
            )
            last_checkpoint_ordinal = max(last_checkpoint_ordinal, event.event_ordinal)
            timing = timer.close()
            accounting = self._accounting(
                identity,
                timing,
                completion_status=(
                    "partial"
                    if status is EpisodeExecutionStatus.RECONCILIATION_REQUIRED
                    else "failed"
                ),
                evaluator_sha256=None,
                final_context_sha256=ending.context_sha256,
            )
            return EpisodeResult(
                identity=identity,
                status=status,
                ending_context_reference=ending.reference,
                ending_context_sha256=ending.context_sha256,
                task_accounting_input=accounting,
                last_checkpoint_ordinal=last_checkpoint_ordinal,
                sanitized_failure_class=failure_class,
            )

    @staticmethod
    def _validate_roles(roles: Mapping[RoleType, BaseRole]) -> None:
        expected = {
            RoleType.AGENT: EpisodeAgentRole,
            RoleType.USER: TransactionalUserRole,
            RoleType.EXECUTION_ENVIRONMENT: TransactionalExecutionEnvironment,
        }
        for role_type, required in expected.items():
            if not isinstance(roles.get(role_type), required):
                raise TypeError(f"{required.__name__} required")

    @staticmethod
    def _validate_resume_recipient(context: object, expected: str) -> None:
        sandbox = context.get_database(
            DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False
        )
        recipient = sandbox["recipient"][-1]
        value = recipient.value if isinstance(recipient, RoleType) else str(recipient)
        if value != expected:
            raise EpisodeRunnerError("resume recipient mismatch")

    @staticmethod
    def _messages(context: object) -> tuple[NativeMessageRecord, ...]:
        rows = context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        ).to_dicts()
        records = []
        all_roles = tuple(role.value for role in RoleType)
        for row in rows:
            visible = row.get("visible_to")
            if visible is None:
                visible_to = all_roles
            else:
                visible_to = tuple(
                    item.value if isinstance(item, RoleType) else str(item)
                    for item in visible
                )
            sender = row["sender"]
            recipient = row["recipient"]
            records.append(
                NativeMessageRecord(
                    sandbox_message_index=int(row["sandbox_message_index"]),
                    sender=sender.value if isinstance(sender, RoleType) else str(sender),
                    recipient=(
                        recipient.value
                        if isinstance(recipient, RoleType)
                        else str(recipient)
                    ),
                    content=row["content"],
                    conversation_active=row.get("conversation_active"),
                    openai_tool_call_id=row.get("openai_tool_call_id"),
                    openai_function_name=row.get("openai_function_name"),
                    tool_call_exception=row.get("tool_call_exception"),
                    visible_to=visible_to,
                )
            )
        return tuple(records)

    @staticmethod
    def _accounting(
        identity: EpisodeIdentity,
        timing: ScopeTimingInput,
        *,
        completion_status: str,
        evaluator_sha256: str | None,
        final_context_sha256: str,
    ) -> TaskAccountingInput:
        return TaskAccountingInput(
            run_id=identity.run_id,
            task_id=identity.episode_id,
            scenario_family_id=identity.family_id,
            scenario_id=identity.scenario_id,
            system_variant=identity.system_variant,
            timing=timing,
            completion_status=completion_status,
            evaluator_result_sha256=evaluator_sha256,
            final_context_sha256=final_context_sha256,
        )


def build_skill_attributions(
    tool_actions: tuple[ToolActionRecord, ...],
    *,
    evaluator_record: TrustedEvaluatorRecord,
    evaluator_record_sha256: str,
    generation_id: str,
    skill_versions: Mapping[str, str],
) -> tuple[SkillUseAttribution, ...]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for action in tool_actions:
        if not action.executed or not action.committed or action.rolled_back:
            continue
        for call_id, skill_id, tool_id in zip(
            action.call_ids, action.selected_skill_ids, action.canonical_tool_ids
        ):
            if skill_id is not None:
                grouped.setdefault(skill_id, []).append((call_id, tool_id))
    records = []
    for skill_id in sorted(grouped):
        if skill_id not in skill_versions:
            raise EpisodeRunnerError("executed Skill version is missing")
        calls = grouped[skill_id]
        records.append(
            SkillUseAttribution(
                skill_id=skill_id,
                skill_version=skill_versions[skill_id],
                generation_id=generation_id,
                executed_call_ids=tuple(call for call, _ in calls),
                canonical_tool_ids=tuple(tool for _, tool in calls),
                evaluator_record_sha256=evaluator_record_sha256,
                fully_successful=evaluator_record.fully_successful,
            )
        )
    return tuple(records)


def bind_executed_calls_to_turns(
    online_turns: tuple[OnlineTurnRecord, ...],
    tool_actions: tuple[ToolActionRecord, ...],
) -> tuple[OnlineTurnRecord, ...]:
    """Attach each committed native tool batch to its originating Agent turn."""

    if any(type(turn) is not OnlineTurnRecord for turn in online_turns):
        raise TypeError("OnlineTurnRecord required")
    bound = list(online_turns)
    used_turns: set[int] = set()
    seen_calls: set[str] = set()
    cursor = 0
    for action in tool_actions:
        if type(action) is not ToolActionRecord:
            raise TypeError("ToolActionRecord required")
        if not action.executed or not action.committed:
            continue
        if seen_calls.intersection(action.call_ids):
            raise EpisodeRunnerError("tool call identity reused across turns")
        # Native conversation termination belongs to the User role. Retain its
        # transaction in the trajectory without inventing an Agent decision.
        if action.is_user_conversation_control:
            seen_calls.update(action.call_ids)
            continue
        match = next(
            (
                index
                for index in range(cursor, len(bound))
                if index not in used_turns
                and bound[index].final_action_sha256 == action.action_sha256
            ),
            None,
        )
        if match is None:
            raise EpisodeRunnerError("tool action has no originating Agent turn")
        if bound[match].executed_call_ids:
            raise EpisodeRunnerError("Agent turn tool calls were already bound")
        bound[match] = bound[match].model_copy(
            update={"executed_call_ids": action.call_ids}
        )
        used_turns.add(match)
        seen_calls.update(action.call_ids)
        cursor = match + 1
    return tuple(bound)


__all__ = [
    "EpisodeRunInput",
    "EpisodeRunner",
    "EpisodeRunnerError",
    "PhysicalAttemptProvider",
    "build_skill_attributions",
    "bind_executed_calls_to_turns",
]
