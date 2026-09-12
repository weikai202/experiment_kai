"""Deterministic current-round Policy/World memory-update orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.offline.memory_projection import reject_sensitive_candidate
from toolsandbox_pipeline.offline.memory_prompts import MemoryPromptSet
from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
from toolsandbox_pipeline.offline.memory_roles import (
    OfflineMemoryTokenLimits,
    PreparedMemoryRequest,
    prepare_candidate_request,
    prepare_review_request,
)
from toolsandbox_pipeline.offline.memory_updates import apply_memory_review
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryCandidateNone,
    MemoryReviewOutput,
    MemoryRoundResult,
    MemoryUpdateIdentity,
    MemoryUpdateUnitResult,
    PolicyMemoryCandidate,
    PolicyMemoryCandidateDecision,
    PolicyTrajectoryProjection,
    StagedMemoryMutation,
    WorldMemoryCandidate,
    WorldMemoryCandidateDecision,
    WorldTrajectoryProjection,
)


class MemoryOrchestrationError(RuntimeError):
    pass


class MemoryTrajectoryReference(StrictModel):
    """Content-free sealed-buffer reference plus Task 015 allowlisted projections.

    A Task 014 adapter constructs this record; it is not a competing trajectory
    body and cannot contain raw responses, contexts, or evaluator definitions.
    """

    model_config = ConfigDict(frozen=True)
    trajectory_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    trajectory_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    run_id: str = Field(min_length=1, pattern=r"^\S+$")
    round_index: int = Field(ge=0, le=2)
    shard_id: str = Field(min_length=1, pattern=r"^\S+$")
    generation_id: str = Field(pattern=r"^g[0-9]{3}$")
    dataset_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split: Literal["train"]
    status: Literal["completed_evaluated"]
    eligible_for_train_offline_consumption: Literal[True]
    manifest_position: int = Field(ge=0)
    policy_projection: PolicyTrajectoryProjection
    world_projection: WorldTrajectoryProjection | None = None

    @model_validator(mode="after")
    def projection_identity(self) -> "MemoryTrajectoryReference":
        projections = (self.policy_projection, self.world_projection)
        if any(
            projection is not None
            and (
                projection.trajectory_id != self.trajectory_id
                or projection.manifest_position != self.manifest_position
            )
            for projection in projections
        ):
            raise ValueError("trajectory projection/reference mismatch")
        return self


class SealedMemoryTrajectoryBuffer(StrictModel):
    model_config = ConfigDict(frozen=True)
    identity: MemoryUpdateIdentity
    entries: tuple[MemoryTrajectoryReference, ...]

    def canonical_references(self) -> list[dict[str, object]]:
        return [
            {
                "trajectory_id": entry.trajectory_id,
                "trajectory_sha256": entry.trajectory_sha256,
                "manifest_position": entry.manifest_position,
            }
            for entry in self.entries
        ]

    @model_validator(mode="after")
    def sealed_identity_and_order(self) -> "SealedMemoryTrajectoryBuffer":
        positions = tuple(entry.manifest_position for entry in self.entries)
        trajectory_ids = tuple(entry.trajectory_id for entry in self.entries)
        if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
            raise ValueError("sealed trajectories must follow unique manifest order")
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("duplicate trajectory in sealed buffer")
        for entry in self.entries:
            if (
                entry.run_id != self.identity.run_id
                or entry.round_index != self.identity.round_index
                or entry.shard_id != self.identity.shard_id
                or entry.generation_id != self.identity.current_generation_id
                or entry.dataset_manifest_sha256 != self.identity.dataset_manifest_sha256
                or entry.config_manifest_sha256 != self.identity.config_manifest_sha256
            ):
                raise ValueError("trajectory does not belong to current train buffer")
        expected = canonical_sha256(
            {
                "protocol": "sealed-memory-buffer-v1",
                "run_id": self.identity.run_id,
                "round_index": self.identity.round_index,
                "shard_id": self.identity.shard_id,
                "generation_id": self.identity.current_generation_id,
                "entries": self.canonical_references(),
            }
        )
        if self.identity.sealed_input_buffer_sha256 != expected:
            raise ValueError("sealed trajectory buffer hash mismatch")
        return self


@dataclass(frozen=True)
class AppliedMemoryDecision:
    output: object = field(repr=False)
    logical_request_id: str
    source_attempt_id: str
    application_id: str
    checkpoint_id: str


class AppliedMemoryRoleExecutor(Protocol):
    def execute_and_apply(self, prepared: PreparedMemoryRequest) -> AppliedMemoryDecision: ...


class MemoryUpdateDurability(Protocol):
    def load_unit(self, unit_reference: str) -> MemoryUpdateUnitResult | None: ...

    def checkpoint(self, event_kind: str, payload: dict[str, object]) -> str: ...

    def commit_mutation(
        self,
        mutation: StagedMemoryMutation,
        ordered_application_ids: tuple[str, ...],
    ) -> tuple[str, str]: ...

    def record_non_substantive(
        self, outcome: Literal["none", "skip"], application_ids: tuple[str, ...]
    ) -> None: ...

    def commit_unit(self, result: MemoryUpdateUnitResult) -> MemoryUpdateUnitResult: ...

    def high_water_marks(self) -> dict[str, int]: ...


def selected_policy_entries(
    entries: tuple[MemoryTrajectoryReference, ...],
) -> tuple[MemoryTrajectoryReference, ...]:
    return tuple(sorted(entries[-50:], key=lambda entry: entry.manifest_position))


def selected_world_entries(
    entries: tuple[MemoryTrajectoryReference, ...],
) -> tuple[MemoryTrajectoryReference, ...]:
    return tuple(entry for entry in entries if entry.world_projection is not None)


class MemoryUpdateOrchestrator:
    def __init__(
        self,
        *,
        prompts: MemoryPromptSet,
        limits: OfflineMemoryTokenLimits,
        limits_sha256: str,
        structured_output_wire_mode: Literal["guided_json", "structured_outputs_json"],
        role_executor: AppliedMemoryRoleExecutor,
        retriever: MemoryCandidateRetriever,
        durability: MemoryUpdateDurability,
        source_generation_sha256: str,
        current_policy_memory: tuple[PolicyMemory, ...],
        current_world_memory: tuple[WorldMemory, ...],
        input_representation: Literal["v1", "packed-v2"] = "v1",
        world_input_representation: Literal["v1"] | None = None,
    ) -> None:
        if input_representation not in ("v1", "packed-v2"):
            raise ValueError("unknown reflection input representation")
        if world_input_representation not in (None, "v1"):
            raise ValueError("World reflection supports v1 only")
        self.world_input_representation = world_input_representation
        self.input_representation = input_representation
        self.prompts = prompts
        self.limits = limits
        self.limits_sha256 = limits_sha256
        self.structured_output_wire_mode = structured_output_wire_mode
        self.role_executor = role_executor
        self.retriever = retriever
        self.durability = durability
        self.source_generation_sha256 = source_generation_sha256
        self.current = {
            "policy": {record.memory_id: record for record in current_policy_memory},
            "world": {record.memory_id: record for record in current_world_memory},
        }
        self.staged: dict[str, dict[str, PolicyMemory | WorldMemory]] = {
            "policy": {}, "world": {}
        }

    def run(self, buffer: SealedMemoryTrajectoryBuffer) -> MemoryRoundResult:
        if type(buffer) is not SealedMemoryTrajectoryBuffer:
            raise TypeError("sealed Task 015 trajectory buffer required")
        if self.input_representation == "packed-v2" and self.world_input_representation is None and any(
            entry.world_projection is not None for entry in buffer.entries
        ):
            raise MemoryOrchestrationError("packed-v2 supports Policy-only buffers")
        policy_entries = selected_policy_entries(buffer.entries)
        world_entries = selected_world_entries(buffer.entries)
        policy_ids = {entry.trajectory_id for entry in policy_entries}
        world_ids = {entry.trajectory_id for entry in world_entries}
        units: list[MemoryUpdateUnitResult] = []
        for entry in buffer.entries:
            if entry.trajectory_id in policy_ids:
                units.append(self._unit(buffer.identity, entry, "policy"))
            if entry.trajectory_id in world_ids:
                units.append(self._unit(buffer.identity, entry, "world"))
        counts = {
            role: {decision: 0 for decision in ("ADD", "MERGE", "SKIP", "NONE")}
            for role in ("policy", "world")
        }
        for unit in units:
            decision = unit.review_decision or "NONE"
            counts[unit.role][decision] += 1
        last_checkpoint = units[-1].last_checkpoint_id if units else None
        return MemoryRoundResult.build(
            identity=buffer.identity,
            selected_policy_trajectory_ids=tuple(entry.trajectory_id for entry in policy_entries),
            selected_world_trajectory_ids=tuple(entry.trajectory_id for entry in world_entries),
            ordered_unit_ids=tuple(unit.unit_id for unit in units),
            units=tuple(units),
            policy_counts=counts["policy"],
            world_counts=counts["world"],
            staged_policy_memory=tuple(
                self.staged["policy"][key] for key in sorted(self.staged["policy"], key=lambda value: value.encode("utf-8"))
            ),
            staged_world_memory=tuple(
                self.staged["world"][key] for key in sorted(self.staged["world"], key=lambda value: value.encode("utf-8"))
            ),
            source_generation_sha256=self.source_generation_sha256,
            last_checkpoint_id=last_checkpoint,
            ledger_high_water_marks=self.durability.high_water_marks(),
            accounting_projections=tuple(
                {
                    "role": unit.role,
                    "candidate_application_id": unit.candidate_application_id,
                    "reviewer_application_id": unit.reviewer_application_id,
                    "substantive_effect_id": unit.substantive_effect_id,
                }
                for unit in units
            ),
        )

    def _unit(
        self,
        identity: MemoryUpdateIdentity,
        entry: MemoryTrajectoryReference,
        role: Literal["policy", "world"],
    ) -> MemoryUpdateUnitResult:
        unit_reference = "memory-unit-" + canonical_sha256(
            {
                "protocol": "memory-unit-v1",
                "run_id": identity.run_id,
                "round_index": identity.round_index,
                "trajectory_id": entry.trajectory_id,
                "memory_role": role,
                "buffer_sha256": identity.sealed_input_buffer_sha256,
            }
        )[7:]
        recovered = self.durability.load_unit(unit_reference)
        if recovered is not None:
            if recovered.unit_id != unit_reference or recovered.role != role or recovered.trajectory_id != entry.trajectory_id:
                raise MemoryOrchestrationError("recovered memory unit identity mismatch")
            if recovered.mutation is not None:
                self._remember_mutation(recovered.mutation, None)
            return recovered
        projection = entry.policy_projection if role == "policy" else entry.world_projection
        if projection is None:
            raise MemoryOrchestrationError("missing trusted role projection")
        candidate_request = prepare_candidate_request(
            projection,
            input_representation=(self.world_input_representation or self.input_representation) if role == "world" else self.input_representation,
            unit_reference=unit_reference,
            prompts=self.prompts,
            limits=self.limits,
            limits_sha256=self.limits_sha256,
            structured_output_wire_mode=self.structured_output_wire_mode,
        )
        candidate_applied = self.role_executor.execute_and_apply(candidate_request)
        wrapper = candidate_applied.output
        if type(wrapper) is PolicyMemoryCandidateDecision:
            candidate = wrapper.decision()
        elif type(wrapper) is WorldMemoryCandidateDecision:
            candidate = wrapper.decision()
        else:
            raise MemoryOrchestrationError("wrong candidate output type")
        reject_sensitive_candidate(candidate, projection)
        candidate_checkpoint = self.durability.checkpoint(
            "memory_candidate_applied",
            {
                "unit_reference": unit_reference,
                "application_id": candidate_applied.application_id,
                "candidate_result": candidate.result,
            },
        )
        if isinstance(candidate, MemoryCandidateNone):
            self.durability.record_non_substantive("none", (candidate_applied.application_id,))
            result = MemoryUpdateUnitResult(
                unit_id=unit_reference,
                role=role,
                trajectory_id=entry.trajectory_id,
                candidate_logical_request_id=candidate_applied.logical_request_id,
                candidate_source_attempt_id=candidate_applied.source_attempt_id,
                candidate_application_id=candidate_applied.application_id,
                candidate_result="NONE",
                last_checkpoint_id=candidate_checkpoint,
            )
            return self.durability.commit_unit(result)
        matches = self.retriever.retrieve(candidate)
        embedding_checkpoint = self.durability.checkpoint(
            "memory_candidate_embedding_completed",
            {
                "unit_reference": unit_reference,
                "embedding_attempt_id": matches.embedding.source_attempt_id,
                "candidate_input_sha256": matches.embedding.key.input_sha256,
            },
        )
        review_request = prepare_review_request(
            candidate,
            matches.records,
            unit_reference=unit_reference,
            prompts=self.prompts,
            limits=self.limits,
            limits_sha256=self.limits_sha256,
            structured_output_wire_mode=self.structured_output_wire_mode,
        )
        review_applied = self.role_executor.execute_and_apply(review_request)
        if type(review_applied.output) is not MemoryReviewOutput:
            raise MemoryOrchestrationError("wrong reviewer output type")
        review = review_applied.output.review()
        self.durability.checkpoint(
            "memory_review_applied",
            {
                "unit_reference": unit_reference,
                "application_id": review_applied.application_id,
                "decision": review.decision,
            },
        )
        visible = tuple(self.current[role].values()) + tuple(self.staged[role].values())
        binary_label = (
            projection.fully_successful
            if type(projection) is PolicyTrajectoryProjection
            else projection.attributable_failure
        )
        mutation = apply_memory_review(
            candidate,
            review,
            trajectory_id=entry.trajectory_id,
            next_generation_id=identity.next_generation_id,
            binary_label=binary_label,
            all_visible_records=visible,
            supplied_matches=matches.records,
        )
        applications = (candidate_applied.application_id, review_applied.application_id)
        if mutation is None:
            self.durability.record_non_substantive("skip", applications)
            final_checkpoint = self.durability.checkpoint(
                "memory_unit_no_change",
                {"unit_reference": unit_reference, "decision": "SKIP"},
            )
            effect_id = None
        else:
            effect_id, final_checkpoint = self.durability.commit_mutation(mutation, applications)
            self._remember_mutation(mutation, matches.embedding.vector)
        result = MemoryUpdateUnitResult(
            unit_id=unit_reference,
            role=role,
            trajectory_id=entry.trajectory_id,
            candidate_logical_request_id=candidate_applied.logical_request_id,
            candidate_source_attempt_id=candidate_applied.source_attempt_id,
            candidate_application_id=candidate_applied.application_id,
            reviewer_logical_request_id=review_applied.logical_request_id,
            reviewer_source_attempt_id=review_applied.source_attempt_id,
            reviewer_application_id=review_applied.application_id,
            candidate_result="CANDIDATE",
            review_decision=review.decision,
            mutation=mutation,
            substantive_effect_id=effect_id,
            last_checkpoint_id=final_checkpoint,
        )
        return self.durability.commit_unit(result)

    def _remember_mutation(
        self, mutation: StagedMemoryMutation, candidate_vector: tuple[float, ...] | None
    ) -> None:
        self.staged[mutation.role][mutation.record.memory_id] = mutation.record
        if mutation.operation == "ADD":
            if candidate_vector is None:
                self.retriever.remember_recovered(mutation.record)
                return
            vector = candidate_vector
        else:
            assert mutation.source_memory_id is not None
            vector = self.retriever.vector_for(mutation.source_memory_id)
        self.retriever.remember_staged(mutation.record, semantic_vector=vector)


__all__ = [
    "AppliedMemoryDecision",
    "AppliedMemoryRoleExecutor",
    "MemoryOrchestrationError",
    "MemoryTrajectoryReference",
    "MemoryUpdateDurability",
    "MemoryUpdateOrchestrator",
    "SealedMemoryTrajectoryBuffer",
    "selected_policy_entries",
    "selected_world_entries",
]
