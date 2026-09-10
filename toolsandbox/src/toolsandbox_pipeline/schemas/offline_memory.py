"""Strict records for train-only Policy and World memory updates."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.base import JsonObject, StrictModel
from toolsandbox_pipeline.schemas.critic import CriticErrorCode, CriticOutput
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory


Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1, max_length=512)]
Reason = Annotated[str, Field(min_length=1, max_length=240)]
ShortList = Annotated[tuple[Text, ...], Field(max_length=5)]


class _FrozenStrict(StrictModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


def _unique(values: tuple[object, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name}")


class MemoryUpdateIdentity(_FrozenStrict):
    run_id: Identifier
    round_index: Annotated[int, Field(ge=0, le=2)]
    shard_id: Identifier
    current_generation_id: Annotated[str, Field(pattern=r"^g[0-9]{3}$")]
    next_generation_id: Annotated[str, Field(pattern=r"^g[0-9]{3}$")]
    dataset_manifest_sha256: Digest
    config_manifest_sha256: Digest
    prompt_manifest_sha256: Digest
    sealed_input_buffer_sha256: Digest

    @model_validator(mode="after")
    def consecutive_generation(self) -> "MemoryUpdateIdentity":
        expected = f"g{int(self.current_generation_id[1:]) + 1:03d}"
        if self.next_generation_id != expected:
            raise ValueError("next generation must be consecutive")
        return self


class PolicyTrajectoryProjection(_FrozenStrict):
    projection_version: Literal["policy-trajectory-v1"] = "policy-trajectory-v1"
    trajectory_id: Digest
    manifest_position: Annotated[int, Field(ge=0)]
    visible_states: tuple[JsonObject, ...]
    retrieved_policy_memory: tuple[JsonObject, ...]
    retrieved_skills: tuple[JsonObject, ...]
    proposed_actions: tuple[ActionEnvelope, ...]
    final_actions: tuple[ActionEnvelope, ...]
    controller_codes: tuple[Identifier, ...]
    visible_tool_outcomes: tuple[JsonObject, ...]
    native_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    fully_successful: bool
    host_attribution: Literal["successful", "unsuccessful"]

    @model_validator(mode="after")
    def trusted_label(self) -> "PolicyTrajectoryProjection":
        expected = "successful" if self.fully_successful else "unsuccessful"
        if self.host_attribution != expected:
            raise ValueError("Policy attribution must follow native success")
        return self


class WorldTrajectoryProjection(_FrozenStrict):
    projection_version: Literal["world-trajectory-v1"] = "world-trajectory-v1"
    trajectory_id: Digest
    manifest_position: Annotated[int, Field(ge=0)]
    visible_states: tuple[JsonObject, ...]
    draft_action: ActionEnvelope
    controller_codes: tuple[Identifier, ...]
    visible_tool_outcomes: tuple[JsonObject, ...]
    critic_output: CriticOutput
    revision_occurred: bool
    attributable_failure: bool
    attribution_kind: Literal[
        "real_tool_exception",
        "fixture_miss",
        "related_milestone_below_perfect",
        "minefield_hit",
        "verified_success",
    ]

    @model_validator(mode="after")
    def attribution_is_conservative(self) -> "WorldTrajectoryProjection":
        failure_kinds = {
            "real_tool_exception",
            "fixture_miss",
            "related_milestone_below_perfect",
            "minefield_hit",
        }
        if self.attributable_failure != (self.attribution_kind in failure_kinds):
            raise ValueError("World attribution label is inconsistent")
        return self


class PolicyMemoryCandidateContent(_FrozenStrict):
    scope: Text
    applicability: ShortList
    action_guidance: Text
    avoid: ShortList

    @field_validator("applicability", "avoid", mode="before")
    @classmethod
    def freeze_json_arrays(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        return value

    @model_validator(mode="after")
    def unique_lists(self) -> "PolicyMemoryCandidateContent":
        _unique(self.applicability, "applicability items")
        _unique(self.avoid, "avoid items")
        return self


class WorldMemoryCandidateContent(_FrozenStrict):
    action_pattern: Text
    state_conditions: ShortList
    schema_conditions: ShortList
    likely_error_codes: Annotated[tuple[CriticErrorCode, ...], Field(max_length=5)]
    outcome_calibration: Text
    correction_principle: Text

    @field_validator(
        "state_conditions", "schema_conditions", "likely_error_codes", mode="before"
    )
    @classmethod
    def freeze_json_arrays(cls, value: object) -> object:
        if type(value) is list:
            value = tuple(value)
        if cls is WorldMemoryCandidateContent and isinstance(value, tuple):
            return value
        return value

    @field_validator("likely_error_codes", mode="before")
    @classmethod
    def exact_error_enums(cls, value: object) -> object:
        if type(value) is list:
            value = tuple(value)
        if isinstance(value, tuple):
            try:
                return tuple(
                    item if type(item) is CriticErrorCode else CriticErrorCode(item)
                    for item in value
                )
            except (TypeError, ValueError) as error:
                raise ValueError("unknown Critic error code") from error
        return value

    @model_validator(mode="after")
    def unique_lists(self) -> "WorldMemoryCandidateContent":
        _unique(self.state_conditions, "state conditions")
        _unique(self.schema_conditions, "schema conditions")
        _unique(self.likely_error_codes, "error codes")
        return self


class PolicyMemoryCandidate(_FrozenStrict):
    result: Literal["CANDIDATE"]
    role: Literal["policy"]
    candidate: PolicyMemoryCandidateContent


class WorldMemoryCandidate(_FrozenStrict):
    result: Literal["CANDIDATE"]
    role: Literal["world"]
    candidate: WorldMemoryCandidateContent


class MemoryCandidateNone(_FrozenStrict):
    result: Literal["NONE"]


MemoryCandidateDecision = PolicyMemoryCandidate | WorldMemoryCandidate | MemoryCandidateNone


class PolicyMemoryCandidateDecision(_FrozenStrict):
    """One exact Policy candidate wire object or exact NONE object."""

    result: Literal["CANDIDATE", "NONE"]
    role: Literal["policy"] | None = None
    candidate: PolicyMemoryCandidateContent | None = None

    @model_validator(mode="after")
    def conditional_fields(self) -> "PolicyMemoryCandidateDecision":
        expected = {"result"} if self.result == "NONE" else {"result", "role", "candidate"}
        if self.model_fields_set != expected:
            raise ValueError("Policy candidate conditional fields mismatch")
        return self

    def decision(self) -> PolicyMemoryCandidate | MemoryCandidateNone:
        if self.result == "NONE":
            return MemoryCandidateNone(result="NONE")
        assert self.candidate is not None
        return PolicyMemoryCandidate(result="CANDIDATE", role="policy", candidate=self.candidate)


class WorldMemoryCandidateDecision(_FrozenStrict):
    """One exact World candidate wire object or exact NONE object."""

    result: Literal["CANDIDATE", "NONE"]
    role: Literal["world"] | None = None
    candidate: WorldMemoryCandidateContent | None = None

    @model_validator(mode="after")
    def conditional_fields(self) -> "WorldMemoryCandidateDecision":
        expected = {"result"} if self.result == "NONE" else {"result", "role", "candidate"}
        if self.model_fields_set != expected:
            raise ValueError("World candidate conditional fields mismatch")
        return self

    def decision(self) -> WorldMemoryCandidate | MemoryCandidateNone:
        if self.result == "NONE":
            return MemoryCandidateNone(result="NONE")
        assert self.candidate is not None
        return WorldMemoryCandidate(result="CANDIDATE", role="world", candidate=self.candidate)


class MemoryReviewAdd(_FrozenStrict):
    decision: Literal["ADD"]
    reason: Reason


class MemoryReviewMerge(_FrozenStrict):
    decision: Literal["MERGE"]
    target_memory_id: Identifier
    reason: Reason


class MemoryReviewSkip(_FrozenStrict):
    decision: Literal["SKIP"]
    reason: Reason


MemoryReviewDecision = Annotated[
    MemoryReviewAdd | MemoryReviewMerge | MemoryReviewSkip,
    Field(discriminator="decision"),
]


class MemoryReviewOutput(_FrozenStrict):
    """Gateway-compatible strict reviewer union with conditional field checks."""

    decision: Literal["ADD", "MERGE", "SKIP"]
    target_memory_id: Identifier | None = None
    reason: Reason

    @model_validator(mode="after")
    def conditional_target(self) -> "MemoryReviewOutput":
        expected = {"decision", "target_memory_id", "reason"} if self.decision == "MERGE" else {"decision", "reason"}
        if self.model_fields_set != expected:
            raise ValueError("review target is allowed exactly for MERGE")
        return self

    def review(self) -> MemoryReviewAdd | MemoryReviewMerge | MemoryReviewSkip:
        if self.decision == "ADD":
            return MemoryReviewAdd(decision="ADD", reason=self.reason)
        if self.decision == "SKIP":
            return MemoryReviewSkip(decision="SKIP", reason=self.reason)
        assert self.target_memory_id is not None
        return MemoryReviewMerge(
            decision="MERGE", target_memory_id=self.target_memory_id, reason=self.reason
        )


class StagedMemoryMutation(_FrozenStrict):
    mutation_id: Digest
    role: Literal["policy", "world"]
    operation: Literal["ADD", "MERGE"]
    trajectory_id: Digest
    source_memory_id: Identifier | None
    record: PolicyMemory | WorldMemory
    record_sha256: Digest

    @model_validator(mode="after")
    def identity_and_role(self) -> "StagedMemoryMutation":
        role_matches = (self.role == "policy") == isinstance(self.record, PolicyMemory)
        source_matches = (self.operation == "MERGE") == (self.source_memory_id is not None)
        digest = canonical_sha256(self.record.model_dump(mode="json"))
        payload = {
            "protocol": "memory-mutation-v1",
            "role": self.role,
            "operation": self.operation,
            "trajectory_id": self.trajectory_id,
            "source_memory_id": self.source_memory_id,
            "record_sha256": digest,
        }
        if not role_matches or not source_matches:
            raise ValueError("memory mutation shape mismatch")
        if self.record_sha256 != digest or self.mutation_id != canonical_sha256(payload):
            raise ValueError("memory mutation identity mismatch")
        return self

    @classmethod
    def build(
        cls,
        *,
        role: Literal["policy", "world"],
        operation: Literal["ADD", "MERGE"],
        trajectory_id: str,
        source_memory_id: str | None,
        record: PolicyMemory | WorldMemory,
    ) -> "StagedMemoryMutation":
        record_sha256 = canonical_sha256(record.model_dump(mode="json"))
        payload = {
            "protocol": "memory-mutation-v1",
            "role": role,
            "operation": operation,
            "trajectory_id": trajectory_id,
            "source_memory_id": source_memory_id,
            "record_sha256": record_sha256,
        }
        return cls(
            mutation_id=canonical_sha256(payload),
            role=role,
            operation=operation,
            trajectory_id=trajectory_id,
            source_memory_id=source_memory_id,
            record=record,
            record_sha256=record_sha256,
        )


class MemoryUpdateUnitResult(_FrozenStrict):
    unit_id: Identifier
    role: Literal["policy", "world"]
    trajectory_id: Digest
    candidate_logical_request_id: Identifier
    candidate_source_attempt_id: Identifier
    candidate_application_id: Identifier
    reviewer_logical_request_id: Identifier | None = None
    reviewer_source_attempt_id: Identifier | None = None
    reviewer_application_id: Identifier | None = None
    candidate_result: Literal["CANDIDATE", "NONE"]
    review_decision: Literal["ADD", "MERGE", "SKIP"] | None = None
    mutation: StagedMemoryMutation | None = None
    substantive_effect_id: Identifier | None = None
    last_checkpoint_id: Identifier

    @model_validator(mode="after")
    def result_shape(self) -> "MemoryUpdateUnitResult":
        has_review = self.candidate_result == "CANDIDATE"
        review_fields = (
            self.reviewer_logical_request_id,
            self.reviewer_source_attempt_id,
            self.reviewer_application_id,
            self.review_decision,
        )
        if has_review != all(value is not None for value in review_fields):
            raise ValueError("candidate result/reviewer linkage mismatch")
        mutated = self.review_decision in {"ADD", "MERGE"}
        if mutated != (self.mutation is not None and self.substantive_effect_id is not None):
            raise ValueError("only substantive mutations have effect links")
        return self


class MemoryRoundResult(_FrozenStrict):
    identity: MemoryUpdateIdentity
    selected_policy_trajectory_ids: tuple[Digest, ...]
    selected_world_trajectory_ids: tuple[Digest, ...]
    ordered_unit_ids: tuple[Identifier, ...]
    units: tuple[MemoryUpdateUnitResult, ...]
    policy_counts: dict[Literal["ADD", "MERGE", "SKIP", "NONE"], int]
    world_counts: dict[Literal["ADD", "MERGE", "SKIP", "NONE"], int]
    staged_policy_memory: tuple[PolicyMemory, ...]
    staged_world_memory: tuple[WorldMemory, ...]
    source_generation_sha256: Digest
    last_checkpoint_id: Identifier | None
    ledger_high_water_marks: dict[str, int]
    accounting_projections: tuple[JsonObject, ...]
    result_sha256: Digest

    def identity_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"result_sha256"})

    @model_validator(mode="after")
    def result_identity(self) -> "MemoryRoundResult":
        if self.result_sha256 != canonical_sha256(self.identity_payload()):
            raise ValueError("memory round result hash mismatch")
        if self.ordered_unit_ids != tuple(unit.unit_id for unit in self.units):
            raise ValueError("memory unit order mismatch")
        return self

    @classmethod
    def build(cls, **values: object) -> "MemoryRoundResult":
        provisional = cls.model_construct(result_sha256="sha256:" + "0" * 64, **values)
        return cls(result_sha256=canonical_sha256(provisional.identity_payload()), **values)


__all__ = [
    "MemoryCandidateDecision",
    "MemoryCandidateNone",
    "MemoryReviewAdd",
    "MemoryReviewDecision",
    "MemoryReviewMerge",
    "MemoryReviewOutput",
    "MemoryReviewSkip",
    "MemoryRoundResult",
    "MemoryUpdateIdentity",
    "MemoryUpdateUnitResult",
    "PolicyMemoryCandidate",
    "PolicyMemoryCandidateContent",
    "PolicyMemoryCandidateDecision",
    "PolicyTrajectoryProjection",
    "StagedMemoryMutation",
    "WorldMemoryCandidate",
    "WorldMemoryCandidateContent",
    "WorldMemoryCandidateDecision",
    "WorldTrajectoryProjection",
]
