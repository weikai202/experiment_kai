"""Strict train-only Skill update and Dev Mini-Bench contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.base import JsonObject, StrictModel
from toolsandbox_pipeline.schemas.skill import (
    SkillApplicability,
    SkillCostProfile,
    SkillFailureMode,
    SkillOnlineStatistics,
    SkillRecord,
    SkillRiskProfile,
    SkillStatePredicate,
    SkillValidation,
)

Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
GenerationId = Annotated[str, Field(pattern=r"^g[0-9]{3}$")]
Version = Annotated[str, Field(pattern=r"^v1\.(0|[1-9][0-9]*)$")]
ShortText = Annotated[str, Field(min_length=1, max_length=240)]
Text = Annotated[str, Field(min_length=1, max_length=512)]


class _FrozenStrict(StrictModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
    )


def _ordered_unique(values: tuple[str, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name}")
    if values != tuple(sorted(values, key=lambda item: item.encode("utf-8"))):
        raise ValueError(f"{name} must be in UTF-8 order")


class SkillRoundIdentity(_FrozenStrict):
    run_id: Identifier
    round_index: Annotated[int, Field(ge=0, le=2)]
    shard_id: Identifier
    current_generation_id: GenerationId
    next_generation_id: GenerationId
    dataset_manifest_sha256: Digest
    config_manifest_sha256: Digest
    prompt_manifest_sha256: Digest
    token_limit_config_sha256: Digest
    sealed_input_buffer_sha256: Digest
    data_seed: Literal[0] = 0

    @model_validator(mode="after")
    def consecutive_generation(self) -> "SkillRoundIdentity":
        if self.next_generation_id != f"g{int(self.current_generation_id[1:]) + 1:03d}":
            raise ValueError("next generation must be consecutive")
        return self


EvidenceKind = Literal[
    "visible_tool_exception",
    "fixture_miss",
    "controller_rejection",
    "related_milestone_shortfall",
    "minefield_hit",
    "visible_terminal_state_failure",
]


class FailureModeLineageRecord(_FrozenStrict):
    lineage_id: Annotated[str, Field(pattern=r"^lineage_[0-9a-f]{64}$")]
    failure_signature_sha256: Digest
    skill_id: Identifier
    evidence_kind: EvidenceKind
    canonical_tool_dependencies: tuple[Identifier, ...]
    sanitized_outcome_class: Identifier
    mode_id: Identifier
    source_evidence_sha256: Digest
    producing_round: Annotated[int, Field(ge=0, le=2)]
    failure_mode_effect_id: Identifier
    accepted_skill_version: Version | None = None
    accepted_skill_effect_id: Identifier | None = None

    @model_validator(mode="after")
    def exact_lineage(self) -> "FailureModeLineageRecord":
        _ordered_unique(
            self.canonical_tool_dependencies, "lineage tool dependencies"
        )
        signature = canonical_sha256(
            {
                "skill_id": self.skill_id,
                "evidence_kind": self.evidence_kind,
                "sorted_public_canonical_tool_dependencies": list(
                    self.canonical_tool_dependencies
                ),
                "sanitized_outcome_class": self.sanitized_outcome_class,
            }
        )
        if signature != self.failure_signature_sha256:
            raise ValueError("failure signature mismatch")
        expected_id = "lineage_" + canonical_sha256(
            {
                "failure_signature_sha256": signature,
                "mode_id": self.mode_id,
                "source_evidence_sha256": self.source_evidence_sha256,
                "producing_round": self.producing_round,
                "failure_mode_effect_id": self.failure_mode_effect_id,
            }
        )[7:]
        if self.lineage_id != expected_id:
            raise ValueError("lineage identity mismatch")
        if (self.accepted_skill_version is None) != (
            self.accepted_skill_effect_id is None
        ):
            raise ValueError("accepted Skill lineage fields must be paired")
        return self


class SkillFailureEvidence(_FrozenStrict):
    skill_id: Identifier
    skill_version: Version
    trajectory_id: Digest
    episode_id: Identifier
    manifest_position: Annotated[int, Field(ge=0)]
    evidence_kind: EvidenceKind
    canonical_tool_dependencies: tuple[Identifier, ...]
    sanitized_outcome_class: Identifier
    generalized_failure: Text
    source_evidence_sha256: Digest

    @model_validator(mode="after")
    def safe_ordered_evidence(self) -> "SkillFailureEvidence":
        _ordered_unique(self.canonical_tool_dependencies, "tool dependencies")
        expected = canonical_sha256(
            {
                "protocol": "skill-failure-evidence-v1",
                "skill_id": self.skill_id,
                "skill_version": self.skill_version,
                "trajectory_id": self.trajectory_id,
                "episode_id": self.episode_id,
                "manifest_position": self.manifest_position,
                "evidence_kind": self.evidence_kind,
                "canonical_tool_dependencies": list(self.canonical_tool_dependencies),
                "sanitized_outcome_class": self.sanitized_outcome_class,
                "generalized_failure": self.generalized_failure,
            }
        )
        if self.source_evidence_sha256 != expected:
            raise ValueError("failure evidence identity mismatch")
        return self


class FailureModeAdd(_FrozenStrict):
    decision: Literal["ADD"]
    task_condition: ShortText
    failure_mode: ShortText


class FailureModeMerge(_FrozenStrict):
    decision: Literal["MERGE"]
    mode_id: Identifier


class FailureModeSkip(_FrozenStrict):
    decision: Literal["SKIP"]
    reason: ShortText


FailureModeDecision = Annotated[
    FailureModeAdd | FailureModeMerge | FailureModeSkip,
    Field(discriminator="decision"),
]


class SkillContent(_FrozenStrict):
    skill_id: Identifier
    name: Identifier
    description: Text
    applicability: SkillApplicability
    required_inputs: tuple[SkillStatePredicate, ...]
    expected_outputs: tuple[Text, ...]
    tool_dependencies: tuple[Identifier, ...]
    success_criteria: tuple[Text, ...]
    cost_profile: SkillCostProfile
    risk_profile: SkillRiskProfile
    instruction: Text

    @model_validator(mode="after")
    def semantic_lists(self) -> "SkillContent":
        _ordered_unique(self.tool_dependencies, "tool dependencies")
        for values, name in (
            (self.expected_outputs, "expected outputs"),
            (self.success_criteria, "success criteria"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name}")
        return self

    @classmethod
    def from_record(cls, record: SkillRecord) -> "SkillContent":
        return cls(**{name: getattr(record, name) for name in cls.model_fields})


class SkillContentCandidate(_FrozenStrict):
    candidate: SkillContent


class PreparedSkillRewrite(_FrozenStrict):
    unit_id: Identifier
    skill_id: Identifier
    messages: tuple[JsonObject, JsonObject]
    output_schema_sha256: Digest
    canonical_input_fingerprint: Digest
    prompt_version: Literal["skill-candidate-v1"]
    prompt_sha256: Digest
    max_tokens: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def exact_messages(self) -> "PreparedSkillRewrite":
        if tuple(message.get("role") for message in self.messages) != ("system", "user"):
            raise ValueError("exact system/user messages required")
        return self


class DevScenarioSelection(_FrozenStrict):
    skill_id: Identifier
    dataset_manifest_sha256: Digest
    selector_input_sha256: Digest
    selected_scenario_ids: Annotated[tuple[Identifier, ...], Field(max_length=20)]
    selected_scenario_ids_sha256: Digest
    eligible_family_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def selected_identity(self) -> "DevScenarioSelection":
        if len(set(self.selected_scenario_ids)) != len(self.selected_scenario_ids):
            raise ValueError("duplicate selected dev scenario")
        expected = canonical_sha256(list(self.selected_scenario_ids))
        if self.selected_scenario_ids_sha256 != expected:
            raise ValueError("selected dev identity mismatch")
        return self


class DevBranchResult(_FrozenStrict):
    scenario_id: Identifier
    branch: Literal["previous", "candidate"]
    episode_id: Identifier
    evaluated_skill_id: Identifier
    evaluated_skill_version: Version
    shared_configuration_sha256: Digest
    complete: bool
    fully_successful: bool | None = None
    similarity: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    minefield_hit: bool | None = None
    evaluator_record_sha256: Digest | None = None

    @model_validator(mode="after")
    def complete_shape(self) -> "DevBranchResult":
        fields = (
            self.fully_successful,
            self.similarity,
            self.minefield_hit,
            self.evaluator_record_sha256,
        )
        if self.complete != all(value is not None for value in fields):
            raise ValueError("complete dev branch fields mismatch")
        return self


class DevMiniBenchResult(_FrozenStrict):
    skill_id: Identifier
    selection_sha256: Digest
    scenario_count: Annotated[int, Field(ge=0, le=20)]
    previous_full_success_count: Annotated[int, Field(ge=0)]
    candidate_full_success_count: Annotated[int, Field(ge=0)]
    previous_similarity_sum: Annotated[float, Field(ge=0.0)]
    candidate_similarity_sum: Annotated[float, Field(ge=0.0)]
    previous_minefield_hit_count: Annotated[int, Field(ge=0)]
    candidate_minefield_hit_count: Annotated[int, Field(ge=0)]
    complete: bool
    accepted: bool
    reason: Literal[
        "higher_full_success",
        "higher_similarity_without_more_minefields",
        "no_relevant_dev_scenarios",
        "incomplete_branch",
        "not_improved",
    ]
    branch_result_sha256: Digest

    @model_validator(mode="after")
    def decision_rule(self) -> "DevMiniBenchResult":
        counts = (
            self.previous_full_success_count,
            self.candidate_full_success_count,
            self.previous_minefield_hit_count,
            self.candidate_minefield_hit_count,
        )
        if any(value > self.scenario_count for value in counts):
            raise ValueError("dev aggregate exceeds scenario count")
        expected = self.candidate_full_success_count > self.previous_full_success_count
        if self.candidate_full_success_count == self.previous_full_success_count:
            expected = (
                self.candidate_similarity_sum > self.previous_similarity_sum
                and self.candidate_minefield_hit_count
                <= self.previous_minefield_hit_count
            )
        if not self.complete or self.scenario_count == 0:
            expected = False
        if self.accepted != expected:
            raise ValueError("Mini-Bench acceptance rule mismatch")
        return self


class StagedSkillMutation(_FrozenStrict):
    skill_id: Identifier
    previous_version: Version
    accepted: bool
    previous_record: SkillRecord
    staged_records: tuple[SkillRecord, ...]
    failure_buffer_sha256: Digest
    statistics_sha256: Digest
    mini_bench_result: DevMiniBenchResult | None = None
    mini_bench_result_sha256: Digest | None = None
    accepted_effect_id: Identifier | None = None

    @model_validator(mode="after")
    def mutation_shape(self) -> "StagedSkillMutation":
        if self.previous_record.skill_id != self.skill_id:
            raise ValueError("previous Skill identity mismatch")
        if self.accepted != (self.accepted_effect_id is not None):
            raise ValueError("accepted mutation requires one effect")
        if self.accepted and self.mini_bench_result_sha256 is None:
            raise ValueError("accepted mutation requires Mini-Bench evidence")
        if (self.mini_bench_result is None) != (self.mini_bench_result_sha256 is None):
            raise ValueError("Mini-Bench result and hash must be paired")
        if self.mini_bench_result is not None:
            if self.mini_bench_result.skill_id != self.skill_id:
                raise ValueError("Mini-Bench result Skill mismatch")
            if self.mini_bench_result.accepted != self.accepted:
                raise ValueError("Mini-Bench and mutation decisions disagree")
            if self.mini_bench_result_sha256 != canonical_sha256(
                self.mini_bench_result.model_dump(mode="json")
            ):
                raise ValueError("Mini-Bench result hash mismatch")
        if any(record.skill_id != self.skill_id for record in self.staged_records):
            raise ValueError("cross-Skill staged mutation")
        if self.accepted:
            if len(self.staged_records) != 2:
                raise ValueError("accepted mutation requires old and new versions")
            previous, active = self.staged_records
            if previous.version != self.previous_version or previous.status != "deprecated":
                raise ValueError("accepted mutation must deprecate previous version")
            if active.status != "active" or active.failure_mode_buffer:
                raise ValueError("accepted Skill must be active with empty failure buffer")
            if active.online_statistics.evaluated_uses != 0:
                raise ValueError("accepted Skill statistics must reset")
        elif len(self.staged_records) != 1:
            raise ValueError("non-accepted mutation stages one active Skill")
        return self


class SkillUpdateUnitResult(_FrozenStrict):
    unit_id: Identifier
    skill_id: Identifier
    status: Literal["unchanged", "rejected", "accepted", "terminal_failure"]
    failure_decision_application_ids: tuple[Identifier, ...]
    substantive_failure_application_ids: tuple[Identifier, ...]
    substantive_failure_effect_ids: tuple[Identifier, ...]
    candidate_application_id: Identifier | None = None
    staged_mutation_sha256: Digest | None = None
    total_cost_application_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def cost_links_are_substantive(self) -> "SkillUpdateUnitResult":
        allowed = set(self.substantive_failure_application_ids)
        if self.status == "accepted":
            if self.candidate_application_id is None or self.candidate_application_id not in self.total_cost_application_ids:
                raise ValueError("accepted candidate must be cost-linked")
        elif not set(self.total_cost_application_ids) <= allowed:
            raise ValueError("non-substantive application in total_cost")
        return self


class SkillRoundResult(_FrozenStrict):
    identity: SkillRoundIdentity
    ordered_skill_ids: tuple[Identifier, ...]
    ordered_unit_ids: tuple[Identifier, ...]
    unit_results: tuple[SkillUpdateUnitResult, ...]
    staged_mutations: tuple[StagedSkillMutation, ...]
    lineage_records: tuple[FailureModeLineageRecord, ...]
    lineage_record_ids: tuple[Identifier, ...]
    lineage_records_sha256: Digest
    accepted_skill_ids: tuple[Identifier, ...]
    rejected_skill_ids: tuple[Identifier, ...]
    completion_status: Literal["completed"] = "completed"
    result_sha256: Digest

    @model_validator(mode="after")
    def ordered_and_hashed(self) -> "SkillRoundResult":
        _ordered_unique(self.ordered_skill_ids, "Skill IDs")
        for values, name in (
            (self.ordered_unit_ids, "unit IDs"),
            (self.lineage_record_ids, "lineage IDs"),
            (self.accepted_skill_ids, "accepted Skill IDs"),
            (self.rejected_skill_ids, "rejected Skill IDs"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name}")
        if self.lineage_record_ids != tuple(
            record.lineage_id for record in self.lineage_records
        ):
            raise ValueError("lineage record IDs do not match typed records")
        if self.ordered_unit_ids != tuple(unit.unit_id for unit in self.unit_results):
            raise ValueError("unit result IDs do not match ordered unit IDs")
        if self.lineage_records_sha256 != canonical_sha256(
            [record.model_dump(mode="json") for record in self.lineage_records]
        ):
            raise ValueError("lineage record output hash mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if self.result_sha256 != canonical_sha256(payload):
            raise ValueError("Skill round result identity mismatch")
        return self


__all__ = [
    "DevBranchResult",
    "DevMiniBenchResult",
    "DevScenarioSelection",
    "EvidenceKind",
    "FailureModeLineageRecord",
    "FailureModeAdd",
    "FailureModeDecision",
    "FailureModeMerge",
    "FailureModeSkip",
    "PreparedSkillRewrite",
    "SkillContent",
    "SkillContentCandidate",
    "SkillFailureEvidence",
    "SkillRoundIdentity",
    "SkillRoundResult",
    "SkillUpdateUnitResult",
    "StagedSkillMutation",
]
