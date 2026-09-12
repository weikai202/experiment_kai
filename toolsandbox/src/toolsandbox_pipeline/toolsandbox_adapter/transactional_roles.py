"""Durable wrappers around unchanged native User and environment roles."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

import polars as pl
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    get_current_context,
    set_current_context,
)
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.checkpointing.tool_ledger import ToolLedger
from toolsandbox_pipeline.providers.user_simulator import InstrumentedGPT4oMiniUser
from toolsandbox_pipeline.online.turn_responder import DurableTurnResponder
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.checkpoint import (
    ToolActionIdentity,
    ToolRecoveryAction,
    ToolTransactionStatus,
)
from toolsandbox_pipeline.schemas.fixtures import ExternalReadAttempt
from toolsandbox_pipeline.schemas.online_turn import (
    OnlineTurnAuditRecord,
    OnlineTurnDecision,
)
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeIdentity,
    OnlineTurnRecord,
    ToolActionRecord,
)

from .pipeline_agent import PipelineAgent
from .call_identity import execution_action
from .trajectory_store import StoredContext, TrajectoryStore


class TransactionalRoleError(RuntimeError):
    """Sanitized transactional role failure."""


class ToolReconciliationRequired(TransactionalRoleError):
    pass


@dataclass(frozen=True)
class ToolActionBinding:
    action_sha256: str
    call_ids: tuple[str, ...]
    selected_skill_ids: tuple[str | None, ...]
    canonical_tool_ids: tuple[str, ...]
    effect_classes: tuple[str, ...]
    action_ordinal: int

    def __post_init__(self) -> None:
        width = len(self.call_ids)
        if width == 0 or any(
            len(items) != width
            for items in (
                self.selected_skill_ids,
                self.canonical_tool_ids,
                self.effect_classes,
            )
        ):
            raise ValueError("tool action binding columns must align")
        if len(set(self.call_ids)) != width or self.action_ordinal < 1:
            raise ValueError("invalid tool action binding identity")
        allowed = {
            "sandbox_read",
            "sandbox_write",
            "external_read",
            "conversation_control",
        }
        if not set(self.effect_classes) <= allowed:
            raise ValueError("unsupported transactional tool effect")


class ActionBindingProvider(Protocol):
    def __call__(self, messages: tuple[object, ...]) -> ToolActionBinding: ...


class ExternalBoundaryFactory(Protocol):
    def __call__(
        self, attempt_sink: Callable[[ExternalReadAttempt], None]
    ) -> AbstractContextManager[object]: ...


class OnlineAuditLoader(Protocol):
    def __call__(self, decision: OnlineTurnDecision) -> OnlineTurnAuditRecord: ...


class EpisodeAgentRole(BaseRole):
    """Explicit trusted marker for an episode-owned durable Agent boundary."""

    role_type = RoleType.AGENT

    @property
    def records(self) -> tuple[OnlineTurnRecord, ...]:
        raise NotImplementedError


class _CapturingResponder:
    def __init__(self, responder: DurableTurnResponder) -> None:
        self.responder = responder
        self.decision: OnlineTurnDecision | None = None

    def respond(self, turn: object):
        self.decision = self.responder.respond_decision(turn)
        return execution_action(self.decision)


class TransactionalAgentRole(EpisodeAgentRole):
    """Construct one Task013 responder per immutable native Agent turn."""

    role_type = RoleType.AGENT

    def __init__(
        self,
        *,
        identity: EpisodeIdentity,
        trajectory_store: TrajectoryStore,
        responder_factory: Callable[[int], DurableTurnResponder],
        audit_loader: OnlineAuditLoader,
        starting_turn_index: int = 0,
        starting_boundary_ordinal: int = 1,
    ) -> None:
        self.identity = identity
        self.trajectory_store = trajectory_store
        self.responder_factory = responder_factory
        self.audit_loader = audit_loader
        self._turn_index = starting_turn_index
        self._boundary_ordinal = starting_boundary_ordinal
        self._records: list[OnlineTurnRecord] = []
        self.decisions: list[OnlineTurnDecision] = []
        self.last_checkpoint_ordinal = 0

    @property
    def records(self) -> tuple[OnlineTurnRecord, ...]:
        return tuple(self._records)

    def respond(self, ending_index: int | None = None) -> None:
        if ending_index is not None:
            raise TransactionalRoleError("Agent does not process initial setup")
        responder = self.responder_factory(self._turn_index)
        if type(responder) is not DurableTurnResponder:
            raise TypeError("DurableTurnResponder required")
        capture = _CapturingResponder(responder)
        delegate = PipelineAgent(capture)
        before = get_current_context().max_sandbox_message_index
        delegate.respond()
        decision = capture.decision
        if decision is None:
            raise TransactionalRoleError("Agent decision missing")
        context = get_current_context()
        if context.max_sandbox_message_index <= before:
            raise TransactionalRoleError("Agent response appended no message")
        audit = self.audit_loader(decision)
        if type(audit) is not OnlineTurnAuditRecord:
            raise TypeError("OnlineTurnAuditRecord required")
        if audit.identity != decision.identity:
            raise TransactionalRoleError("online audit identity mismatch")
        if canonical_sha256(audit.model_dump(mode="json")) != decision.audit_record_sha256:
            raise TransactionalRoleError("online audit digest mismatch")
        decision_ref = self.trajectory_store.persist_online_decision(decision)
        record = OnlineTurnRecord(
            agent_turn_index=self._turn_index,
            state_id=decision.state_id,
            decision_reference=decision_ref,
            decision_sha256=decision_ref.sha256,
            final_action_sha256=canonical_sha256(
                decision.final_action.model_dump(mode="json")
            ),
            logical_request_ids=audit.logical_request_ids,
            source_attempt_ids=audit.source_attempt_ids,
            application_ids=audit.application_ids,
        )
        stored = self.trajectory_store.persist_context(context)
        event = self.trajectory_store.commit_context_checkpoint(
            self.identity,
            event_kind="agent_message_committed",
            stored=stored,
            recipient=self.role_type.value,
            boundary_ordinal=self._boundary_ordinal,
        )
        self.last_checkpoint_ordinal = event.event_ordinal
        self._records.append(record)
        self.decisions.append(decision)
        self._turn_index += 1
        self._boundary_ordinal += 1


def _json(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    raise TypeError("unsupported visible result identity value")


class TransactionalExecutionEnvironment(BaseRole):
    role_type = RoleType.EXECUTION_ENVIRONMENT

    def __init__(
        self,
        delegate: BaseRole,
        *,
        identity: EpisodeIdentity,
        tool_ledger: ToolLedger,
        trajectory_store: TrajectoryStore,
        binding_provider: ActionBindingProvider,
        backend_manifest_sha256: str,
        external_boundary_factory: ExternalBoundaryFactory | None = None,
    ) -> None:
        if getattr(delegate, "role_type", None) is not self.role_type:
            raise TypeError("ExecutionEnvironment delegate required")
        if type(tool_ledger) is not ToolLedger:
            raise TypeError("ToolLedger required")
        if type(trajectory_store) is not TrajectoryStore:
            raise TypeError("TrajectoryStore required")
        self.delegate = delegate
        self.identity = identity
        self.tool_ledger = tool_ledger
        self.trajectory_store = trajectory_store
        self.binding_provider = binding_provider
        self.backend_manifest_sha256 = backend_manifest_sha256
        self.external_boundary_factory = external_boundary_factory
        self.records: list[ToolActionRecord] = []
        self.last_checkpoint_ordinal = 0

    def respond(self, ending_index: int | None = None) -> None:
        messages = tuple(self.delegate.get_messages(ending_index))
        to_process = self._messages_to_process(messages)
        if to_process and all(getattr(item, "sender", None) is RoleType.SYSTEM for item in to_process):
            self.delegate.respond(ending_index=ending_index)
            return
        binding = self.binding_provider(to_process)
        if (all(message.sender is RoleType.AGENT for message in to_process)
                and tuple(message.openai_tool_call_id for message in to_process) != binding.call_ids):
            raise TransactionalRoleError("native message and binding execution IDs differ")
        if self.identity.profile == "offline" and "external_read" in binding.effect_classes:
            raise TransactionalRoleError("external reads are forbidden offline")
        pre_context = get_current_context()
        pre = self.trajectory_store.persist_context(pre_context)
        action_identity = ToolActionIdentity(
            run_id=self.identity.run_id,
            scenario_id=self.identity.scenario_id,
            pre_action_context_sha256=pre.context_sha256,
            action_fingerprint=binding.action_sha256,
            ordered_call_ids=binding.call_ids,
            action_ordinal=binding.action_ordinal,
            profile=self.identity.profile,
            effect_classes=binding.effect_classes,
            fixture_manifest_sha256=self.identity.fixture_manifest_sha256,
            backend_manifest_sha256=self.backend_manifest_sha256,
        )
        transaction = self.tool_ledger.prepare_transaction(
            action_identity, pre_context_reference=pre.reference
        )
        pre_event = self.trajectory_store.commit_context_checkpoint(
            self.identity,
            event_kind="tool_action_prepared",
            stored=pre,
            recipient=self.role_type.value,
            boundary_ordinal=binding.action_ordinal,
        )
        self.last_checkpoint_ordinal = pre_event.event_ordinal
        plan = self.tool_ledger.plan_recovery(transaction.transaction_id)
        if plan.action is ToolRecoveryAction.RESTORE_COMMITTED_CONTEXT:
            restored_ref = self.tool_ledger.restore_committed_context_reference(
                transaction.transaction_id
            )
            restored = StoredContext(
                reference=restored_ref,
                context_sha256=transaction.post_context_sha256 or "",
            )
            context = self.trajectory_store.load_context(restored)
            set_current_context(context)
            self.records.append(
                self._record(
                    transaction.transaction_id,
                    binding,
                    pre,
                    restored,
                    before_index=pre_context.max_sandbox_message_index,
                    external_attempt_ids=(),
                    failed=transaction.status is ToolTransactionStatus.FAILED_COMMITTED,
                    executed=True,
                )
            )
            return
        if plan.action is ToolRecoveryAction.RECONCILIATION_REQUIRED:
            try:
                self.tool_ledger.reconcile_and_allocate_attempt(transaction.transaction_id)
            except Exception:
                pass
            raise ToolReconciliationRequired("tool reconciliation required")
        if plan.prior_attempt_id is None:
            attempt_id = self.tool_ledger.allocate_attempt(transaction.transaction_id)
        else:
            attempt_id = self.tool_ledger.reconcile_and_allocate_attempt(
                transaction.transaction_id
            )
        # Every replay begins from the authenticated stored pre-context.
        replay_context = self.trajectory_store.load_context(pre)
        set_current_context(replay_context)
        self.tool_ledger.mark_in_flight(attempt_id)
        external_attempt_ids: list[str] = []

        def persist_external(attempt: ExternalReadAttempt) -> None:
            self.trajectory_store.persist_external_attempt(self.identity, attempt)
            external_attempt_ids.append(attempt.attempt_id)

        has_external = "external_read" in binding.effect_classes
        if has_external and self.external_boundary_factory is None:
            raise TransactionalRoleError("external boundary is required")
        boundary = (
            self.external_boundary_factory(persist_external)
            if has_external and self.external_boundary_factory is not None
            else nullcontext()
        )
        before_index = replay_context.max_sandbox_message_index
        with boundary:
            self.delegate.respond(ending_index=ending_index)
        post_context = get_current_context()
        post = self.trajectory_store.persist_context(post_context)
        result_rows = self._result_rows(post_context, before_index, binding)
        failed = any(row.get("tool_call_exception") is not None for row in result_rows)
        visible_identity = canonical_sha256(result_rows)
        committed = self.tool_ledger.commit_attempt(
            attempt_id,
            post_context_reference=post.reference,
            post_context_sha256=post.context_sha256,
            visible_result_identity=visible_identity,
            external_read_attempt_ids=tuple(external_attempt_ids),
            failed=failed,
        )
        post_event = self.trajectory_store.commit_context_checkpoint(
            self.identity,
            event_kind="tool_action_committed",
            stored=post,
            recipient=self.role_type.value,
            boundary_ordinal=binding.action_ordinal,
        )
        self.last_checkpoint_ordinal = post_event.event_ordinal
        self.records.append(
            self._record(
                committed.transaction_id,
                binding,
                pre,
                post,
                before_index=before_index,
                external_attempt_ids=tuple(external_attempt_ids),
                failed=failed,
                executed=True,
            )
        )

    @staticmethod
    def _messages_to_process(messages: tuple[object, ...]) -> tuple[object, ...]:
        selected: list[object] = []
        for message in reversed(messages):
            if getattr(message, "recipient", None) is not RoleType.EXECUTION_ENVIRONMENT:
                break
            selected.append(message)
        selected.reverse()
        if not selected:
            raise TransactionalRoleError("no environment message set")
        return tuple(selected)

    @staticmethod
    def _result_rows(context: object, before_index: int, binding: ToolActionBinding) -> list[dict[str, object]]:
        sandbox = context.get_database(
            DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        user_control = all(effect == "conversation_control" for effect in binding.effect_classes)
        recipient = RoleType.USER if user_control else RoleType.AGENT
        selected = sandbox.filter(
            (pl.col("sandbox_message_index") > before_index)
            & (pl.col("sender") == RoleType.EXECUTION_ENVIRONMENT)
            & (pl.col("recipient") == recipient)
        )
        # Native User control messages may omit their model call label. Agent
        # tool results must always reference their exact host execution IDs.
        matched = pl.col("openai_tool_call_id").is_in(binding.call_ids)
        if user_control:
            matched = matched | pl.col("openai_tool_call_id").is_null()
        if selected.filter(~matched.fill_null(False)).height:
            raise TransactionalRoleError("native result execution ID mismatch")
        rows = selected.filter(matched).to_dicts()
        visible = []
        for row in rows:
            visible.append(
                {
                    key: _json(row.get(key))
                    for key in (
                        "sandbox_message_index",
                        "sender",
                        "recipient",
                        "content",
                        "conversation_active",
                        "openai_tool_call_id",
                        "openai_function_name",
                        "tool_call_exception",
                    )
                }
            )
        return visible

    @staticmethod
    def _record(
        transaction_id: str,
        binding: ToolActionBinding,
        pre: StoredContext,
        post: StoredContext,
        *,
        before_index: int,
        external_attempt_ids: tuple[str, ...],
        failed: bool,
        executed: bool,
    ) -> ToolActionRecord:
        from_context = get_current_context()
        indices = tuple(
            int(row["sandbox_message_index"])
            for row in TransactionalExecutionEnvironment._result_rows(from_context, before_index, binding)
        )
        return ToolActionRecord(
            transaction_id=transaction_id,
            action_sha256=binding.action_sha256,
            call_ids=binding.call_ids,
            selected_skill_ids=binding.selected_skill_ids,
            canonical_tool_ids=binding.canonical_tool_ids,
            effect_classes=binding.effect_classes,
            pre_context_reference=pre.reference,
            pre_context_sha256=pre.context_sha256,
            post_context_reference=post.reference,
            post_context_sha256=post.context_sha256,
            result_message_indices=indices,
            external_attempt_ids=external_attempt_ids,
            executed=executed,
            committed=True,
            rolled_back=failed,
            failed=failed,
        )


class TransactionalUserRole(BaseRole):
    role_type = RoleType.USER

    def __init__(
        self,
        delegate: BaseRole,
        *,
        identity: EpisodeIdentity,
        trajectory_store: TrajectoryStore,
        starting_boundary_ordinal: int = 1,
    ) -> None:
        if getattr(delegate, "role_type", None) is not self.role_type:
            raise TypeError("User delegate required")
        if identity.profile == "official_live":
            if (
                type(delegate) is not InstrumentedGPT4oMiniUser
                or getattr(delegate, "_durability_seam", None) is None
            ):
                raise TransactionalRoleError(
                    "official User requires the Task006 durability seam"
                )
        elif type(delegate) is InstrumentedGPT4oMiniUser:
            raise TransactionalRoleError("remote User is forbidden in this profile")
        self.delegate = delegate
        self.identity = identity
        self.trajectory_store = trajectory_store
        self._boundary_ordinal = starting_boundary_ordinal
        self.last_checkpoint_ordinal = 0

    def respond(self, ending_index: int | None = None) -> None:
        before = get_current_context().max_sandbox_message_index
        self.delegate.respond(ending_index=ending_index)
        context = get_current_context()
        if context.max_sandbox_message_index <= before:
            raise TransactionalRoleError("User response appended no message")
        stored = self.trajectory_store.persist_context(context)
        event = self.trajectory_store.commit_context_checkpoint(
            self.identity,
            event_kind="user_message_committed",
            stored=stored,
            recipient=self.role_type.value,
            boundary_ordinal=self._boundary_ordinal,
        )
        self.last_checkpoint_ordinal = event.event_ordinal
        self._boundary_ordinal += 1

    def reset(self) -> None:
        reset = getattr(self.delegate, "reset", None)
        if reset is not None:
            reset()

    def teardown(self) -> None:
        teardown = getattr(self.delegate, "teardown", None)
        if teardown is not None:
            teardown()


__all__ = [
    "ActionBindingProvider",
    "ExternalBoundaryFactory",
    "EpisodeAgentRole",
    "OnlineAuditLoader",
    "ToolActionBinding",
    "ToolReconciliationRequired",
    "TransactionalAgentRole",
    "TransactionalExecutionEnvironment",
    "TransactionalRoleError",
    "TransactionalUserRole",
]
