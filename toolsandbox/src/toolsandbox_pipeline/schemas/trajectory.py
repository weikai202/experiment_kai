"""Strict restricted records for one native ToolSandbox episode."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.accounting import TaskAccountingInput
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.dataset import ScenarioRecord


Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
NativeRole = Literal["SYSTEM", "USER", "AGENT", "EXECUTION_ENVIRONMENT"]


class _FrozenStrictModel(StrictModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


class EpisodeExecutionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED_EVALUATED = "completed_evaluated"
    TERMINAL_FAILURE_BEFORE_EVALUATION = "terminal_failure_before_evaluation"
    RECONCILIATION_REQUIRED = "reconciliation_required"


class EpisodeIdentity(_FrozenStrictModel):
    run_id: Identifier
    profile: Literal["offline", "strict_replay", "official_live"]
    phase: Identifier
    round_index: Annotated[int, Field(ge=0, le=2)] | None = None
    shard_id: Identifier | None = None
    family_id: Identifier
    scenario_id: Identifier
    episode_id: Identifier
    manifest_position: Annotated[int, Field(ge=0)]
    system_variant: Literal["vanilla", "generation_0", "updated"]
    generation_id: Annotated[str, Field(pattern=r"^g[0-9]{3}$")]
    starting_context_sha256: Digest
    evaluation_definition_sha256: Digest
    agent_tool_schema_sha256: Digest
    dataset_manifest_sha256: Digest
    runtime_config_sha256: Digest
    prompt_manifest_sha256: Digest
    token_limit_config_sha256: Digest
    fixture_manifest_sha256: Digest
    environment_sha256: Digest
    max_messages: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def round_and_shard_are_paired(self) -> "EpisodeIdentity":
        if (self.round_index is None) != (self.shard_id is None):
            raise ValueError("round_index and shard_id must be supplied together")
        return self

    def validate_manifest_record(self, record: ScenarioRecord) -> None:
        """Fail before context installation if caller metadata is inconsistent."""

        if type(record) is not ScenarioRecord:
            raise TypeError("exact ScenarioRecord required")
        expected = (
            self.scenario_id,
            self.family_id,
            self.max_messages,
            self.starting_context_sha256,
            self.evaluation_definition_sha256,
            self.agent_tool_schema_sha256,
        )
        actual = (
            record.scenario_id,
            record.scenario_family_id,
            record.max_messages,
            record.starting_context_sha256,
            record.evaluation_definition_sha256,
            record.agent_facing_tool_schema_sha256,
        )
        if expected != actual:
            raise ValueError("episode identity does not match scenario manifest record")


class EpisodeResumeInput(_FrozenStrictModel):
    identity: EpisodeIdentity
    committed_context_reference: BlobReference
    committed_context_sha256: Digest
    initial_max_message_index: Annotated[int, Field(ge=0)]
    last_checkpoint_ordinal: Annotated[int, Field(ge=1)]
    next_recipient: NativeRole


class NativeMessageRecord(_FrozenStrictModel):
    sandbox_message_index: Annotated[int, Field(ge=0)]
    sender: NativeRole
    recipient: NativeRole
    content: str
    conversation_active: bool | None = None
    openai_tool_call_id: str | None = Field(default=None, min_length=1)
    openai_function_name: str | None = Field(default=None, min_length=1)
    tool_call_exception: str | None = None
    visible_to: tuple[NativeRole, ...]

    @model_validator(mode="after")
    def visibility_is_nonempty_and_unique(self) -> "NativeMessageRecord":
        if not self.visible_to or len(set(self.visible_to)) != len(self.visible_to):
            raise ValueError("native message visibility must be non-empty and unique")
        return self


class ToolActionRecord(_FrozenStrictModel):
    transaction_id: Identifier
    action_sha256: Digest
    call_ids: tuple[Identifier, ...]
    selected_skill_ids: tuple[Identifier | None, ...]
    canonical_tool_ids: tuple[Identifier, ...]
    effect_classes: tuple[Identifier, ...]
    pre_context_reference: BlobReference
    pre_context_sha256: Digest
    post_context_reference: BlobReference | None = None
    post_context_sha256: Digest | None = None
    result_message_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
    external_attempt_ids: tuple[Identifier, ...] = ()
    executed: bool
    committed: bool
    rolled_back: bool
    failed: bool

    @property
    def is_user_conversation_control(self) -> bool:
        """Native User termination has a transaction but no Agent decision."""
        return (
            bool(self.effect_classes)
            and all(effect == "conversation_control" for effect in self.effect_classes)
            and all(name == "end_conversation" for name in self.canonical_tool_ids)
            and all(skill is None for skill in self.selected_skill_ids)
        )

    @model_validator(mode="after")
    def coherent_tool_action(self) -> "ToolActionRecord":
        width = len(self.call_ids)
        if width == 0 or any(
            len(values) != width
            for values in (
                self.selected_skill_ids,
                self.canonical_tool_ids,
                self.effect_classes,
            )
        ):
            raise ValueError("tool action columns must be non-empty and aligned")
        if len(set(self.call_ids)) != width:
            raise ValueError("tool call IDs must be unique")
        if self.committed != (
            self.post_context_reference is not None
            and self.post_context_sha256 is not None
        ):
            raise ValueError("committed tool action requires a post-context")
        if self.rolled_back and not self.failed:
            raise ValueError("only failed actions may be rolled back")
        if not self.executed and (self.committed or self.result_message_indices):
            raise ValueError("unexecuted actions cannot have committed results")
        return self


class OnlineTurnRecord(_FrozenStrictModel):
    agent_turn_index: Annotated[int, Field(ge=0)]
    state_id: Identifier
    decision_reference: BlobReference
    decision_sha256: Digest
    final_action_sha256: Digest
    logical_request_ids: tuple[Identifier, ...]
    source_attempt_ids: tuple[Identifier, ...]
    application_ids: tuple[Identifier, ...]
    executed_call_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_ordered_references(self) -> "OnlineTurnRecord":
        for values in (
            self.logical_request_ids,
            self.source_attempt_ids,
            self.application_ids,
            self.executed_call_ids,
        ):
            if len(set(values)) != len(values):
                raise ValueError("duplicate online-turn identity")
        return self


class NativeEvaluationMatch(_FrozenStrictModel):
    node_index: Annotated[int, Field(ge=0)]
    snapshot_index: Annotated[int, Field(ge=0)]
    similarity: Annotated[float, Field(ge=0.0, le=1.0)]


class TrustedEvaluatorRecord(_FrozenStrictModel):
    milestone_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    minefield_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    turn_count: Annotated[int, Field(ge=0)]
    milestone_mapping: tuple[NativeEvaluationMatch, ...]
    minefield_mapping: tuple[NativeEvaluationMatch, ...]
    fully_successful: bool
    evaluation_definition_sha256: Digest
    ending_context_sha256: Digest

    @model_validator(mode="after")
    def native_score_rule(self) -> "TrustedEvaluatorRecord":
        expected = (
            0.0 if self.minefield_similarity != 0.0 else self.milestone_similarity
        )
        if self.similarity != expected:
            raise ValueError("native evaluator total-score rule violated")
        if self.fully_successful != (self.similarity == 1.0):
            raise ValueError("fully_successful must use exact native similarity")
        for mapping in (self.milestone_mapping, self.minefield_mapping):
            indices = tuple(item.node_index for item in mapping)
            if len(set(indices)) != len(indices):
                raise ValueError("duplicate evaluator node mapping")
        return self


class SkillUseAttribution(_FrozenStrictModel):
    skill_id: Identifier
    skill_version: Identifier
    generation_id: Annotated[str, Field(pattern=r"^g[0-9]{3}$")]
    executed_call_ids: tuple[Identifier, ...]
    canonical_tool_ids: tuple[Identifier, ...]
    evaluator_record_sha256: Digest
    fully_successful: bool

    @model_validator(mode="after")
    def executed_calls_are_required(self) -> "SkillUseAttribution":
        if not self.executed_call_ids:
            raise ValueError("Skill attribution requires an executed call")
        if len(self.executed_call_ids) != len(self.canonical_tool_ids):
            raise ValueError("Skill attribution call/tool columns must align")
        if len(set(self.executed_call_ids)) != len(self.executed_call_ids):
            raise ValueError("duplicate attributed call")
        return self


def _record_hashes(values: tuple[_FrozenStrictModel, ...]) -> list[str]:
    return [canonical_sha256(value.model_dump(mode="json")) for value in values]


class TrustedTrajectory(_FrozenStrictModel):
    trajectory_id: Digest
    identity: EpisodeIdentity
    messages: tuple[NativeMessageRecord, ...]
    online_turns: tuple[OnlineTurnRecord, ...]
    tool_actions: tuple[ToolActionRecord, ...]
    logical_request_ids: tuple[Identifier, ...]
    physical_attempt_ids: tuple[Identifier, ...]
    ending_context_reference: BlobReference
    ending_context_sha256: Digest
    evaluator_record_reference: BlobReference
    evaluator_record_sha256: Digest
    skill_attributions: tuple[SkillUseAttribution, ...]
    eligible_for_train_offline_consumption: bool

    def identity_payload(self) -> list[object]:
        return [
            "trusted-trajectory-v1",
            self.identity.model_dump(mode="json"),
            _record_hashes(self.messages),
            _record_hashes(self.online_turns),
            _record_hashes(self.tool_actions),
            list(self.logical_request_ids),
            list(self.physical_attempt_ids),
            self.ending_context_sha256,
            self.evaluator_record_sha256,
            _record_hashes(self.skill_attributions),
            self.eligible_for_train_offline_consumption,
        ]

    @model_validator(mode="after")
    def canonical_identity_and_order(self) -> "TrustedTrajectory":
        if self.trajectory_id != canonical_sha256(self.identity_payload()):
            raise ValueError("trajectory identity mismatch")
        indices = tuple(message.sandbox_message_index for message in self.messages)
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise ValueError("native messages must be uniquely ordered")
        turns = tuple(turn.agent_turn_index for turn in self.online_turns)
        if turns != tuple(sorted(turns)) or len(set(turns)) != len(turns):
            raise ValueError("online turns must be uniquely ordered")
        for values in (self.logical_request_ids, self.physical_attempt_ids):
            if len(set(values)) != len(values):
                raise ValueError("duplicate trajectory request identity")
        executed_actions = tuple(
            action
            for action in self.tool_actions
            if action.executed and action.committed
        )
        all_call_ids = tuple(call_id for action in executed_actions for call_id in action.call_ids)
        if len(set(all_call_ids)) != len(all_call_ids):
            raise ValueError("tool call identity reused across turns")
        executed_actions = tuple(action for action in executed_actions if not action.is_user_conversation_control)
        action_call_ids = tuple(
            call_id for action in executed_actions for call_id in action.call_ids
        )
        turn_call_ids = tuple(
            call_id for turn in self.online_turns for call_id in turn.executed_call_ids
        )
        if (
            action_call_ids != turn_call_ids
            or any(
                not any(
                    action.action_sha256 == turn.final_action_sha256
                    and action.call_ids == turn.executed_call_ids
                    for action in executed_actions
                )
                for turn in self.online_turns
                if turn.executed_call_ids
            )
        ):
            raise ValueError("executed tool calls must bind to originating turns")
        skill_ids = tuple(item.skill_id for item in self.skill_attributions)
        if len(set(skill_ids)) != len(skill_ids):
            raise ValueError("at most one attribution per Skill per episode")
        if self.eligible_for_train_offline_consumption and self.identity.round_index is None:
            raise ValueError("only a train-round trajectory may enter offline updates")
        return self

    @classmethod
    def build(cls, **values: object) -> "TrustedTrajectory":
        provisional = cls.model_construct(trajectory_id="sha256:" + "0" * 64, **values)
        return cls(trajectory_id=canonical_sha256(provisional.identity_payload()), **values)


class EpisodeResult(_FrozenStrictModel):
    identity: EpisodeIdentity
    status: EpisodeExecutionStatus
    ending_context_reference: BlobReference
    ending_context_sha256: Digest
    trusted_trajectory_reference: BlobReference | None = None
    evaluator_record_reference: BlobReference | None = None
    task_accounting_input: TaskAccountingInput
    last_checkpoint_ordinal: Annotated[int, Field(ge=1)]
    sanitized_failure_class: Identifier | None = None

    @model_validator(mode="after")
    def terminal_shape(self) -> "EpisodeResult":
        completed = self.status is EpisodeExecutionStatus.COMPLETED_EVALUATED
        if completed != (
            self.trusted_trajectory_reference is not None
            and self.evaluator_record_reference is not None
        ):
            raise ValueError("completed episode requires trajectory and evaluator")
        if completed == (self.sanitized_failure_class is not None):
            raise ValueError("only incomplete episodes require a failure class")
        if self.task_accounting_input.run_id != self.identity.run_id:
            raise ValueError("task accounting run identity mismatch")
        if self.task_accounting_input.task_id != self.identity.episode_id:
            raise ValueError("task accounting episode identity mismatch")
        return self


__all__ = [
    "EpisodeExecutionStatus",
    "EpisodeIdentity",
    "EpisodeResult",
    "EpisodeResumeInput",
    "NativeEvaluationMatch",
    "NativeMessageRecord",
    "OnlineTurnRecord",
    "SkillUseAttribution",
    "ToolActionRecord",
    "TrustedEvaluatorRecord",
    "TrustedTrajectory",
]
