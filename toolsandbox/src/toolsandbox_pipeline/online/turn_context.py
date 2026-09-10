"""Explicit, side-effect-free dependencies for one durable online turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from toolsandbox_pipeline.online.controller import Controller
from toolsandbox_pipeline.online.controller_inputs import ActionHistoryEntry
from toolsandbox_pipeline.online.prompt_contracts import (
    CriticContext,
    InitialPolicyContext,
    PreparedRoleRequest,
)
from toolsandbox_pipeline.online.state_builder import StateBuilder
from toolsandbox_pipeline.retrieval.service import (
    PolicySkillRetrievalBundle,
    RetrievalService,
    WorldRetrievalBundle,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope, AssistantMessageAction
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.critic import CriticOutput
from toolsandbox_pipeline.schemas.generation import GenerationSnapshot
from toolsandbox_pipeline.schemas.online_turn import (
    OnlineTurnIdentity,
    OnlineTurnStage,
)
from toolsandbox_pipeline.schemas.state import (
    CommittedToolOutcomeInput,
    CompactVerifiedState,
    PendingDependencyInput,
)
from toolsandbox_pipeline.schemas.tool_metadata import ControllerToolMetadata


SAFE_CLARIFICATION_VERSION = "safe-clarification-v1"
SAFE_CLARIFICATION_TEXT = (
    "I cannot take a safe action with the information currently available. "
    "Please clarify the missing details."
)


def default_safe_failure_action() -> ActionEnvelope:
    return ActionEnvelope(
        action=AssistantMessageAction(
            type="assistant_message",
            content=SAFE_CLARIFICATION_TEXT,
        )
    )


@dataclass(frozen=True)
class PreparedInitialPolicy:
    context: InitialPolicyContext
    request: PreparedRoleRequest

    def __post_init__(self) -> None:
        if self.request.role != "policy" or self.request.state_id != self.context.state_id:
            raise ValueError("prepared Initial Policy identity mismatch")


@dataclass(frozen=True)
class PreparedCritic:
    context: CriticContext
    request: PreparedRoleRequest

    def __post_init__(self) -> None:
        if self.request.role != "critic" or self.request.state_id != self.context.initial.state_id:
            raise ValueError("prepared Critic identity mismatch")


class InitialPolicyRequestBuilder(Protocol):
    def __call__(
        self,
        state: CompactVerifiedState,
        retrieval: PolicySkillRetrievalBundle,
    ) -> PreparedInitialPolicy: ...


class CriticRequestBuilder(Protocol):
    def __call__(
        self,
        initial: InitialPolicyContext,
        action: ActionEnvelope,
        controller_decision: ControllerDecision,
        retrieval: WorldRetrievalBundle,
    ) -> PreparedCritic: ...


class RevisionRequestBuilder(Protocol):
    def __call__(
        self,
        initial: InitialPolicyContext,
        critic: CriticContext,
        critic_output: CriticOutput,
    ) -> PreparedRoleRequest: ...


@dataclass(frozen=True)
class TurnRoleRequestBuilders:
    initial_policy: InitialPolicyRequestBuilder
    critic: CriticRequestBuilder
    revision: RevisionRequestBuilder


class CheckpointEventSink(Protocol):
    """Durably record one sanitized orchestrator component and return its ID."""

    def load_checkpoint(self, checkpoint_id: str) -> tuple[str, object] | None: ...

    def commit_stage(
        self,
        *,
        identity: OnlineTurnIdentity,
        state_id: str,
        stage: OnlineTurnStage,
        component_payload: object,
    ) -> str: ...


class TurnLifecycleHooks(Protocol):
    def before_stage(self, stage: OnlineTurnStage) -> None: ...

    def after_stage(self, stage: OnlineTurnStage) -> None: ...


class DurableRoleExecutorProtocol(Protocol):
    def execute(self, prepared: PreparedRoleRequest): ...

    def apply(
        self,
        execution: object,
        *,
        stage: OnlineTurnStage,
        state_id: str,
        application_payload: object,
    ): ...

    def commit_online_action(
        self,
        *,
        checkpoint_id: str,
        action: ActionEnvelope,
        application_ids: tuple[str, ...],
        checkpoint_payload: object,
    ): ...


@dataclass(frozen=True)
class TurnContext:
    identity: OnlineTurnIdentity
    generation: GenerationSnapshot
    retrieval: RetrievalService
    state_builder: StateBuilder
    controller: Controller
    controller_tool_metadata: tuple[ControllerToolMetadata, ...]
    role_request_builders: TurnRoleRequestBuilders
    durable_roles: DurableRoleExecutorProtocol
    checkpoint_sink: CheckpointEventSink
    committed_tool_outcomes: tuple[CommittedToolOutcomeInput, ...] = ()
    pending_dependencies: tuple[PendingDependencyInput, ...] = ()
    prior_visible_failed_action_history: tuple[ActionHistoryEntry, ...] = ()
    structured_constraint_tension: bool = False
    safe_failure_action_factory: Callable[[], ActionEnvelope] = default_safe_failure_action
    lifecycle_hooks: TurnLifecycleHooks | None = None

    def __post_init__(self) -> None:
        generation_id = self.generation.manifest.generation_id
        if generation_id != self.identity.expected_generation_id:
            raise ValueError("identity does not pin the supplied generation")
        if self.retrieval.snapshot is not self.generation:
            raise ValueError("RetrievalService must use the exact supplied generation snapshot")
        names = [item.canonical_tool_name for item in self.controller_tool_metadata]
        if len(names) != len(set(names)):
            raise ValueError("duplicate Controller tool metadata")
        if any(not item.visible for item in self.prior_visible_failed_action_history):
            raise ValueError("online action history must contain visible entries only")


__all__ = [
    "CheckpointEventSink",
    "CriticRequestBuilder",
    "DurableRoleExecutorProtocol",
    "InitialPolicyRequestBuilder",
    "PreparedCritic",
    "PreparedInitialPolicy",
    "RevisionRequestBuilder",
    "SAFE_CLARIFICATION_TEXT",
    "SAFE_CLARIFICATION_VERSION",
    "TurnContext",
    "TurnLifecycleHooks",
    "TurnRoleRequestBuilders",
    "default_safe_failure_action",
]
