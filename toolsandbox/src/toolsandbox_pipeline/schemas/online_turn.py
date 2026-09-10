"""Strict identities and audit contracts for one durable Agent turn."""

from __future__ import annotations

from enum import Enum
from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.online.controller_inputs import (
    ActionHistoryEntry,
    ReproducibilityProfile,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.critic import CriticOutput
from toolsandbox_pipeline.schemas.state import (
    CommittedToolOutcomeInput,
    PendingDependencyInput,
)
from toolsandbox_pipeline.toolsandbox_adapter.contracts import AdapterTurn


class _FrozenStrictModel(StrictModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class OnlineTurnIdentity(_FrozenStrictModel):
    run_id: str = Field(min_length=1)
    profile: ReproducibilityProfile
    phase: str = Field(min_length=1)
    round_index: int | None = Field(default=None, ge=0)
    shard_id: str | None = Field(default=None, min_length=1)
    family_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    agent_turn_index: int = Field(ge=0)
    expected_generation_id: str = Field(pattern=r"^g[0-9]{3}$")
    dataset_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_config_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    token_limit_config_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    fixture_manifest_sha256: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    environment_identity: str = Field(min_length=1)

    @model_validator(mode="after")
    def profile_fixture_identity(self) -> "OnlineTurnIdentity":
        if (
            self.profile is ReproducibilityProfile.STRICT_REPLAY
            and self.fixture_manifest_sha256 is None
        ):
            raise ValueError("strict_replay requires a fixture manifest identity")
        return self


class OnlineTurnInput(_FrozenStrictModel):
    """One Adapter turn plus only the validated projections needed to reduce it."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )

    identity: OnlineTurnIdentity
    adapter_turn: AdapterTurn
    committed_tool_outcomes: tuple[CommittedToolOutcomeInput, ...] = ()
    pending_dependencies: tuple[PendingDependencyInput, ...] = ()
    prior_visible_failed_action_history: tuple[ActionHistoryEntry, ...] = ()
    structured_constraint_tension: bool = False

    @model_validator(mode="after")
    def exact_adapter_type(self) -> "OnlineTurnInput":
        if type(self.adapter_turn) is not AdapterTurn:
            raise TypeError("adapter_turn must be an exact AdapterTurn")
        if any(not item.visible for item in self.prior_visible_failed_action_history):
            raise ValueError("online action history must contain visible entries only")
        return self


class OnlineTurnStage(str, Enum):
    STATE_BUILT = "state_built"
    POLICY_RETRIEVAL_COMPLETED = "policy_retrieval_completed"
    INITIAL_POLICY_COMPLETED = "initial_policy_completed"
    INITIAL_POLICY_APPLIED = "initial_policy_applied"
    INITIAL_CONTROLLER_COMPLETED = "initial_controller_completed"
    WORLD_RETRIEVAL_COMPLETED = "world_retrieval_completed"
    CRITIC_COMPLETED = "critic_completed"
    CRITIC_APPLIED = "critic_applied"
    REVISION_COMPLETED = "revision_completed"
    REVISION_APPLIED = "revision_applied"
    POST_REVISION_CONTROLLER_COMPLETED = "post_revision_controller_completed"
    FINAL_ACTION_COMMITTED = "final_action_committed"
    TERMINAL_FAILURE = "terminal_failure"


class RetrievalReference(_FrozenStrictModel):
    retrieval_kind: str = Field(pattern=r"^(policy|skill|world)$")
    record_id: str = Field(min_length=1)
    record_version: str | None = None
    generation_id: str = Field(pattern=r"^g[0-9]{3}$")
    rank: int = Field(ge=1, le=3)
    score: float = Field(ge=-1.0, le=1.0)
    document_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    query_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    cache_hit: bool
    source_attempt_id: str = Field(min_length=1)


class OnlineTurnStageRecord(_FrozenStrictModel):
    stage: OnlineTurnStage
    ordinal: int = Field(ge=0)
    component_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    checkpoint_id: str = Field(min_length=1)


class OnlineTurnDecision(_FrozenStrictModel):
    identity: OnlineTurnIdentity
    state_id: str = Field(min_length=1)
    generation_id: str = Field(pattern=r"^g[0-9]{3}$")
    final_action: ActionEnvelope
    initial_policy_logical_request_id: str = Field(min_length=1)
    critic_logical_request_id: str | None = Field(default=None, min_length=1)
    revision_logical_request_id: str | None = Field(default=None, min_length=1)
    initial_controller_decision: ControllerDecision
    post_revision_controller_decision: ControllerDecision | None = None
    critic_verdict: CriticOutput | None = None
    revision_count: int = Field(ge=0, le=1)
    retrieval_references: tuple[RetrievalReference, ...]
    audit_record_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def routing_shape(self) -> "OnlineTurnDecision":
        used_critic = self.critic_logical_request_id is not None
        used_revision = self.revision_logical_request_id is not None
        if used_critic != (self.critic_verdict is not None):
            raise ValueError("Critic request and verdict must occur together")
        if used_revision != (self.revision_count == 1):
            raise ValueError("Revision identity must match revision_count")
        if used_revision != (self.post_revision_controller_decision is not None):
            raise ValueError("Revision requires one post-Revision Controller decision")
        if used_revision and not used_critic:
            raise ValueError("Revision requires a Critic result")
        return self


class OnlineTurnAuditRecord(_FrozenStrictModel):
    identity: OnlineTurnIdentity
    state_id: str = Field(min_length=1)
    generation_id: str = Field(pattern=r"^g[0-9]{3}$")
    stages: tuple[OnlineTurnStageRecord, ...]
    initial_controller_decision: ControllerDecision
    post_revision_controller_decision: ControllerDecision | None = None
    critic_verdict: CriticOutput | None = None
    final_action_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    logical_request_ids: tuple[str, ...]
    source_attempt_ids: tuple[str, ...]
    application_ids: tuple[str, ...]
    retrieval_references: tuple[RetrievalReference, ...]

    @model_validator(mode="after")
    def aligned_requests(self) -> "OnlineTurnAuditRecord":
        lengths = {
            len(self.logical_request_ids),
            len(self.source_attempt_ids),
            len(self.application_ids),
        }
        if len(lengths) != 1:
            raise ValueError("request, source-attempt, and application IDs must align")
        if len(set(self.application_ids)) != len(self.application_ids):
            raise ValueError("application IDs must be unique")
        if tuple(item.ordinal for item in self.stages) != tuple(range(len(self.stages))):
            raise ValueError("stage ordinals must be contiguous")
        return self


class OnlineTurnFailureCode(str, Enum):
    IDENTITY_FAILURE = "identity_failure"
    STATE_FAILURE = "state_failure"
    RETRIEVAL_FAILURE = "retrieval_failure"
    CHECKPOINT_FAILURE = "checkpoint_failure"
    ROLE_FAILURE = "role_failure"
    CONTROLLER_FAILURE = "controller_failure"
    APPLICATION_FAILURE = "application_failure"


class OnlineTurnFailure(RuntimeError):
    """Sanitized terminal turn failure; never includes provider exception text."""

    def __init__(
        self,
        code: OnlineTurnFailureCode,
        *,
        stage: OnlineTurnStage | None,
        state_id: str | None,
    ) -> None:
        self.code = code
        self.stage = stage
        self.state_id = state_id
        super().__init__(code.value)

    def __repr__(self) -> str:
        return (
            f"OnlineTurnFailure(code={self.code.value!r}, "
            f"stage={None if self.stage is None else self.stage.value!r})"
        )


__all__ = [
    "OnlineTurnAuditRecord",
    "OnlineTurnDecision",
    "OnlineTurnFailure",
    "OnlineTurnFailureCode",
    "OnlineTurnIdentity",
    "OnlineTurnInput",
    "OnlineTurnStage",
    "OnlineTurnStageRecord",
    "RetrievalReference",
]
