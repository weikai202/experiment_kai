"""Strict, content-free contracts for final three-system reporting."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.base import JsonObject, StrictModel
from toolsandbox_pipeline.schemas.dataset import VARIANTS

Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
SystemId = Literal["vanilla", "generation_0", "updated"]
SYSTEM_ORDER: tuple[SystemId, ...] = ("vanilla", "generation_0", "updated")
VANILLA_COMPONENTS = (
    "direct_qwen_responder", "adapter_action_conversion", "qwen_durability",
)
PIPELINE_COMPONENTS = (
    "state_builder", "policy_memory", "world_memory", "skill_retrieval",
    "initial_policy", "controller", "critic", "revision",
    "adapter_action_conversion", "qwen_durability",
)


class _FrozenStrict(StrictModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, allow_inf_nan=False,
        protected_namespaces=(),
    )


class SystemPlan(_FrozenStrict):
    system_id: SystemId
    generation_id: Literal["g000", "g003"]
    allowed_components: tuple[Identifier, ...]
    prompt_manifest_sha256: Digest
    token_limit_config_sha256: Digest
    token_limit_status: Literal["calibrated"]

    @model_validator(mode="after")
    def exact_system_shape(self) -> "SystemPlan":
        generation = "g000" if self.system_id != "updated" else "g003"
        if self.generation_id != generation:
            raise ValueError("system generation must be G000/G003")
        expected = VANILLA_COMPONENTS if self.system_id == "vanilla" else PIPELINE_COMPONENTS
        if self.allowed_components != expected:
            raise ValueError("system allowed components are frozen")
        return self


class TestScenarioPlan(_FrozenStrict):
    manifest_position: Annotated[int, Field(ge=0)]
    scenario_id: Identifier
    scenario_family_id: Identifier
    variant: Identifier
    categories: tuple[Identifier, ...]
    starting_context_sha256: Digest
    evaluation_definition_sha256: Digest
    agent_tool_schema_sha256: Digest

    @model_validator(mode="after")
    def finite_membership(self) -> "TestScenarioPlan":
        if self.variant not in VARIANTS:
            raise ValueError("unknown ToolSandbox variant")
        if not self.categories:
            raise ValueError("scenario requires native category membership")
        ordered = tuple(sorted(self.categories, key=lambda item: item.encode("utf-8")))
        if self.categories != ordered or len(set(ordered)) != len(ordered):
            raise ValueError("categories must be unique UTF-8 ordered")
        return self


def family_membership_sha256(scenarios: tuple[TestScenarioPlan, ...]) -> str:
    families: dict[str, list[dict[str, str]]] = {}
    for item in scenarios:
        families.setdefault(item.scenario_family_id, []).append({
            "scenario_id": item.scenario_id, "variant": item.variant,
        })
    return canonical_sha256([
        {"scenario_family_id": family_id, "ordered_variants": members}
        for family_id, members in sorted(families.items(), key=lambda pair: pair[0].encode("utf-8"))
    ])


def category_membership_sha256(scenarios: tuple[TestScenarioPlan, ...]) -> str:
    return canonical_sha256([
        {"scenario_id": item.scenario_id, "categories": list(item.categories)}
        for item in scenarios
    ])


class FinalEvaluationPlan(_FrozenStrict):
    schema_version: Literal[1] = 1
    protocol_version: Literal["three-system-final-evaluation-v1"]
    plan_sha256: Digest
    user_approval_reference: Identifier
    frozen_at_utc: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
    profile: Literal["strict_replay", "official_live"]
    training_run_id: Identifier
    training_run_sha256: Digest
    g000_sha256: Digest
    g003_sha256: Digest
    checkpoint_observation_registry_sha256: Digest
    ordered_system_ids: tuple[SystemId, SystemId, SystemId]
    systems: tuple[SystemPlan, SystemPlan, SystemPlan]
    test_dataset_manifest_sha256: Digest
    scenarios: tuple[TestScenarioPlan, ...]
    family_membership_sha256: Digest
    category_membership_sha256: Digest
    toolsandbox_source_sha256: Digest
    dependency_lock_sha256: Digest
    container_image_sha256: Digest
    environment_sha256: Digest
    world_clock_sha256: Digest
    fixture_or_live_tool_config_sha256: Digest
    qwen_identity_sha256: Digest
    qwen_decoding_sha256: Digest
    embedding_identity_sha256: Digest
    user_simulator_identity_sha256: Digest
    user_prompt_sha256: Digest
    user_few_shot_sha256: Digest
    user_tools_sha256: Digest
    user_stop_behavior_sha256: Digest
    prompt_registry_sha256: Digest
    output_schema_registry_sha256: Digest
    calibrated_token_limit_registry_sha256: Digest
    shared_pipeline_config_sha256: Digest
    checkpoint_schema_sha256: Digest
    metrics_schema_sha256: Digest
    reporting_schema_sha256: Digest
    process_count: Literal[1]
    process_order: Literal["test_manifest_order"]
    seed_policy: Literal["scenario_id_sha256_v1"]
    cluster_seed: Literal[0] = 0
    cluster_replicates: Literal[10000] = 10000
    output_root: str
    prior_test_result: bool = False

    @model_validator(mode="after")
    def frozen_identity(self) -> "FinalEvaluationPlan":
        from pathlib import Path

        if self.ordered_system_ids != SYSTEM_ORDER:
            raise ValueError("three-system order is fixed")
        if tuple(item.system_id for item in self.systems) != SYSTEM_ORDER:
            raise ValueError("system plans must use fixed order")
        positions = tuple(item.manifest_position for item in self.scenarios)
        if positions != tuple(range(len(self.scenarios))):
            raise ValueError("test scenarios must be in contiguous manifest order")
        if len({item.scenario_id for item in self.scenarios}) != len(self.scenarios):
            raise ValueError("duplicate test scenario")
        family_counts: dict[str, int] = {}
        family_variants: dict[str, set[str]] = {}
        for item in self.scenarios:
            family_counts[item.scenario_family_id] = family_counts.get(item.scenario_family_id, 0) + 1
            family_variants.setdefault(item.scenario_family_id, set()).add(item.variant)
        if (
            len(self.scenarios) != 200
            or len(family_counts) != 25
            or set(family_counts.values()) != {8}
            or any(variants != set(VARIANTS) for variants in family_variants.values())
        ):
            raise ValueError("final test membership must be 25 families of eight variants")
        if self.family_membership_sha256 != family_membership_sha256(self.scenarios):
            raise ValueError("family membership hash mismatch")
        if self.category_membership_sha256 != category_membership_sha256(self.scenarios):
            raise ValueError("category membership hash mismatch")
        generation_0, updated = self.systems[1], self.systems[2]
        if (
            generation_0.allowed_components != updated.allowed_components
            or generation_0.prompt_manifest_sha256 != updated.prompt_manifest_sha256
            or generation_0.token_limit_config_sha256 != updated.token_limit_config_sha256
        ):
            raise ValueError("Generation-0 and Updated pipeline configuration drift")
        if self.prior_test_result:
            raise ValueError("a final plan cannot include prior test results")
        path = Path(self.output_root)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("output root must be an absolute normalized path")
        forbidden_fragments = ("sk-", "https://", "http://", "api_key", "authorization")
        serialized = str(self.model_dump(mode="json", exclude={"plan_sha256"})).lower()
        if any(item in serialized for item in forbidden_fragments):
            raise ValueError("secret or raw endpoint in final plan")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"plan_sha256"}))
        if self.plan_sha256 != expected:
            raise ValueError("final plan hash mismatch")
        return self

    @classmethod
    def build(cls, **values: object) -> "FinalEvaluationPlan":
        zero = "sha256:" + "0" * 64
        provisional = cls.model_construct(plan_sha256=zero, **values)
        values["plan_sha256"] = canonical_sha256(
            provisional.model_dump(mode="json", exclude={"plan_sha256"})
        )
        return cls(**values)


class ScenarioEvaluationRecord(_FrozenStrict):
    schema_version: Literal[1] = 1
    system_id: SystemId
    manifest_position: Annotated[int, Field(ge=0)]
    scenario_id: Identifier
    scenario_family_id: Identifier
    variant: Identifier
    categories: tuple[Identifier, ...]
    episode_id: Identifier
    trajectory_sha256: Digest
    evaluator_record_sha256: Digest
    similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    milestone_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    minefield_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    fully_successful: bool
    effective_turn_count: Annotated[int, Field(ge=0)]
    critic_trigger_count: Annotated[int, Field(ge=0)] = 0
    critic_accept_count: Annotated[int, Field(ge=0)] = 0
    critic_revise_count: Annotated[int, Field(ge=0)] = 0
    critic_uncertain_count: Annotated[int, Field(ge=0)] = 0
    revision_count: Annotated[int, Field(ge=0)] = 0
    fixture_hit_count: Annotated[int, Field(ge=0)] = 0
    fixture_miss_count: Annotated[int, Field(ge=0)] = 0
    external_tool_exception_count: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def native_score(self) -> "ScenarioEvaluationRecord":
        if self.similarity != (0.0 if self.minefield_similarity != 0.0 else self.milestone_similarity):
            raise ValueError("native similarity rule violated")
        if self.fully_successful != (self.similarity == 1.0):
            raise ValueError("fully_successful must use native similarity")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("duplicate native category")
        if self.system_id == "vanilla" and any((self.critic_trigger_count, self.critic_accept_count, self.critic_revise_count, self.critic_uncertain_count, self.revision_count)):
            raise ValueError("Vanilla cannot emit Critic/Revision metrics")
        if (
            self.critic_accept_count + self.critic_revise_count + self.critic_uncertain_count
            != self.critic_trigger_count
            or self.critic_trigger_count > self.effective_turn_count
            or self.revision_count > self.critic_revise_count
        ):
            raise ValueError("Critic/Revision decision counts do not reconcile")
        return self


class FailureSignatureObservation(_FrozenStrict):
    scenario_id: Identifier
    skill_id: Identifier | None
    evidence_kind: Identifier | None
    canonical_tool_dependencies: tuple[Identifier, ...] | None
    sanitized_outcome_class: Identifier | None
    observation_sha256: Digest

    @model_validator(mode="after")
    def signature_shape(self) -> "FailureSignatureObservation":
        values = (self.skill_id, self.evidence_kind, self.canonical_tool_dependencies, self.sanitized_outcome_class)
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError("failure signature is complete or wholly unavailable")
        if self.canonical_tool_dependencies is not None:
            ordered = tuple(sorted(self.canonical_tool_dependencies, key=lambda item: item.encode("utf-8")))
            if self.canonical_tool_dependencies != ordered or len(set(ordered)) != len(ordered):
                raise ValueError("tool dependencies must be unique UTF-8 ordered")
        expected = canonical_sha256({
            "protocol": "test-failure-signature-observation-v1",
            "scenario_id": self.scenario_id,
            "skill_id": self.skill_id,
            "evidence_kind": self.evidence_kind,
            "canonical_tool_dependencies": list(self.canonical_tool_dependencies) if self.canonical_tool_dependencies is not None else None,
            "sanitized_outcome_class": self.sanitized_outcome_class,
        })
        if self.observation_sha256 != expected:
            raise ValueError("failure observation hash mismatch")
        return self

    @classmethod
    def build(cls, **values: object) -> "FailureSignatureObservation":
        values["observation_sha256"] = canonical_sha256({
            "protocol": "test-failure-signature-observation-v1",
            "scenario_id": values["scenario_id"],
            "skill_id": values.get("skill_id"),
            "evidence_kind": values.get("evidence_kind"),
            "canonical_tool_dependencies": list(values["canonical_tool_dependencies"]) if values.get("canonical_tool_dependencies") is not None else None,
            "sanitized_outcome_class": values.get("sanitized_outcome_class"),
        })
        return cls(**values)

    @property
    def complete(self) -> bool:
        return self.skill_id is not None

    @property
    def signature_sha256(self) -> str | None:
        if not self.complete:
            return None
        return canonical_sha256({
            "skill_id": self.skill_id,
            "evidence_kind": self.evidence_kind,
            "sorted_public_canonical_tool_dependencies": list(self.canonical_tool_dependencies or ()),
            "sanitized_outcome_class": self.sanitized_outcome_class,
        })


FailureClassification = Literal[
    "not_generation_0_failure", "unmatched", "ambiguous", "related_unrepaired", "related_repaired", "incomplete_evidence"
]


class FailureModeAttributionRow(_FrozenStrict):
    schema_version: Literal[1] = 1
    system_id: SystemId
    manifest_position: Annotated[int, Field(ge=0)]
    scenario_id: Identifier
    scenario_family_id: Identifier
    failure_signature_sha256: Digest | None
    matched_lineage_ids: tuple[Identifier, ...]
    matched_mode_ids: tuple[Identifier, ...]
    matched_skill_versions: tuple[Identifier, ...]
    generation_0_fully_successful: bool
    updated_fully_successful: bool
    generation_0_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    updated_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    vanilla_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    evolved_skill_use_proof_sha256: Digest | None
    classification: FailureClassification
    reason_code: Identifier


class FailureModeRepairSummary(_FrozenStrict):
    schema_version: Literal[1] = 1
    generation_0_failure_case_count: Annotated[int, Field(ge=0)]
    failure_mode_related_case_count: Annotated[int, Field(ge=0)]
    failure_mode_repaired_case_count: Annotated[int, Field(ge=0)]
    failure_mode_repair_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    repair_rate_complete: bool
    failure_mode_unmatched_case_count: Annotated[int, Field(ge=0)]
    failure_mode_ambiguous_case_count: Annotated[int, Field(ge=0)]
    incomplete_evidence_case_count: Annotated[int, Field(ge=0)]
    updated_minus_generation_0_similarity_points_overall: float
    updated_minus_generation_0_similarity_points_related_subset: float | None
    updated_minus_vanilla_similarity_points_overall: float
    observational_not_causal: Literal[True] = True


class FamilyStabilityRecord(_FrozenStrict):
    schema_version: Literal[1] = 1
    system_id: SystemId
    scenario_family_id: Identifier
    variant_count: Literal[8]
    all_variants_fully_successful: bool
    minimum_native_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    similarity_range: Annotated[float, Field(ge=0.0, le=1.0)]
    related_generation_0_failure_variant_count: Annotated[int, Field(ge=0, le=8)]
    repaired_variant_count: Annotated[int, Field(ge=0, le=8)]
    all_related_failures_repaired: bool | None


class FamilyStabilitySummary(_FrozenStrict):
    schema_version: Literal[1] = 1
    system_id: SystemId
    family_count: Annotated[int, Field(ge=0)]
    all_variants_success_family_count: Annotated[int, Field(ge=0)]
    all_variants_success_family_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    mean_family_minimum_similarity: Annotated[float, Field(ge=0.0, le=1.0)] | None
    mean_within_family_similarity_range: Annotated[float, Field(ge=0.0, le=1.0)] | None
    related_family_count: Annotated[int, Field(ge=0)]
    fully_repaired_related_family_count: Annotated[int, Field(ge=0)]
    fully_repaired_related_family_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None


class CategoryMetric(_FrozenStrict):
    category: Identifier
    scenario_count: Annotated[int, Field(gt=0)]
    mean_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    fully_successful_rate: Annotated[float, Field(ge=0.0, le=1.0)]


class CategoryResultRecord(_FrozenStrict):
    schema_version: Literal[1] = 1
    system_id: SystemId
    metric: CategoryMetric


class SystemAggregate(_FrozenStrict):
    schema_version: Literal[1] = 1
    system_id: SystemId
    scenario_count: Annotated[int, Field(ge=0)]
    family_count: Annotated[int, Field(ge=0)]
    mean_similarity: Annotated[float, Field(ge=0.0, le=1.0)] | None
    mean_milestone_similarity: Annotated[float, Field(ge=0.0, le=1.0)] | None
    mean_minefield_similarity: Annotated[float, Field(ge=0.0, le=1.0)] | None
    fully_successful_count: Annotated[int, Field(ge=0)]
    fully_successful_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    mean_effective_turn_count: float | None
    median_effective_turn_count: float | None
    critic_trigger_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    critic_accept_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    critic_revise_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    critic_uncertain_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    revision_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    mean_post_revision_similarity: Annotated[float, Field(ge=0.0, le=1.0)] | None
    fixture_hit_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    fixture_miss_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    external_tool_exception_rate: Annotated[float, Field(ge=0.0, le=1.0)] | None
    failure_count: Annotated[int, Field(ge=0)] = 0
    timeout_count: Annotated[int, Field(ge=0)] = 0
    reconciliation_count: Annotated[int, Field(ge=0)] = 0
    category_metrics: tuple[CategoryMetric, ...]
    macro_category_mean_similarity: float | None
    micro_category_mean_similarity: float | None
    total_running_time_seconds: Annotated[float, Field(ge=0.0)] | None
    total_tokens: Annotated[int, Field(ge=0)] | None
    usage_complete: bool
    total_cost: Annotated[int, Field(ge=0)] | None
    cost_unit: Literal["qwen_effective_output_tokens"] = "qwen_effective_output_tokens"
    cost_complete: bool


class TrainingRoundHeadline(_FrozenStrict):
    schema_version: Literal[1] = 1
    round_index: Annotated[int, Field(ge=0, le=2)]
    total_running_time_seconds: Annotated[float, Field(ge=0.0)]
    total_tokens: Annotated[int, Field(ge=0)] | None
    usage_complete: bool
    total_cost: Annotated[int, Field(ge=0)] | None
    cost_unit: Literal["qwen_effective_output_tokens"] = "qwen_effective_output_tokens"
    cost_complete: bool
    immutable_round_record_sha256: Digest

    @model_validator(mode="after")
    def completeness(self) -> "TrainingRoundHeadline":
        if self.usage_complete != (self.total_tokens is not None):
            raise ValueError("round usage completeness mismatch")
        if self.cost_complete != (self.total_cost is not None):
            raise ValueError("round cost completeness mismatch")
        return self


class MainTable(_FrozenStrict):
    schema_version: Literal[1] = 1
    plan_sha256: Digest
    system_aggregates: tuple[SystemAggregate, SystemAggregate, SystemAggregate]
    training_round_headlines: tuple[
        TrainingRoundHeadline, TrainingRoundHeadline, TrainingRoundHeadline
    ]
    failure_mode_repair_summary: FailureModeRepairSummary
    family_stability_summaries: tuple[
        FamilyStabilitySummary, FamilyStabilitySummary, FamilyStabilitySummary
    ]
    qwen_weights_modified: Literal[False] = False

    @model_validator(mode="after")
    def fixed_orders(self) -> "MainTable":
        if tuple(item.system_id for item in self.system_aggregates) != SYSTEM_ORDER:
            raise ValueError("main table system order mismatch")
        if tuple(item.round_index for item in self.training_round_headlines) != (0, 1, 2):
            raise ValueError("training round headline order mismatch")
        if tuple(item.system_id for item in self.family_stability_summaries) != SYSTEM_ORDER:
            raise ValueError("family summary system order mismatch")
        return self


class PairwiseBootstrapResult(_FrozenStrict):
    left_system: SystemId
    right_system: SystemId
    observed_difference: float
    percentile_95_lower: float
    percentile_95_upper: float


class ClusterBootstrapReport(_FrozenStrict):
    schema_version: Literal[1] = 1
    seed: Literal[0] = 0
    replicates: Literal[10000] = 10000
    family_count: Literal[25] = 25
    variants_per_family: Literal[8] = 8
    index_stream_sha256: Digest
    pairs: tuple[PairwiseBootstrapResult, PairwiseBootstrapResult, PairwiseBootstrapResult]


ChecklistStatus = Literal["pass", "fail", "not_applicable_with_reason", "not_run"]


class ReproducibilityCheck(_FrozenStrict):
    item_id: Identifier
    required: bool
    status: ChecklistStatus
    evidence_sha256: Digest | None
    reason_code: Identifier | None

    @model_validator(mode="after")
    def evidence_rule(self) -> "ReproducibilityCheck":
        if self.status in {"pass", "fail"} and self.evidence_sha256 is None:
            raise ValueError("performed check requires immutable evidence")
        if self.status == "not_applicable_with_reason" and self.reason_code is None:
            raise ValueError("not-applicable check requires reason")
        return self


class StrictReplaySmokeEvidence(_FrozenStrict):
    first_final_context_hashes_sha256: Digest
    second_final_context_hashes_sha256: Digest
    first_native_scores_sha256: Digest
    second_native_scores_sha256: Digest
    first_retrieval_ids_sha256: Digest
    second_retrieval_ids_sha256: Digest
    first_request_output_hashes_sha256: Digest
    second_request_output_hashes_sha256: Digest
    evidence_sha256: Digest

    @model_validator(mode="after")
    def evidence_identity(self) -> "StrictReplaySmokeEvidence":
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"evidence_sha256"}))
        if self.evidence_sha256 != expected:
            raise ValueError("strict-replay evidence hash mismatch")
        return self

    @property
    def identical(self) -> bool:
        return (
            self.first_final_context_hashes_sha256 == self.second_final_context_hashes_sha256
            and self.first_native_scores_sha256 == self.second_native_scores_sha256
            and self.first_retrieval_ids_sha256 == self.second_retrieval_ids_sha256
            and self.first_request_output_hashes_sha256 == self.second_request_output_hashes_sha256
        )


class ReproducibilityChecklist(_FrozenStrict):
    schema_version: Literal[1] = 1
    profile: Literal["strict_replay", "official_live"]
    checks: tuple[ReproducibilityCheck, ...]
    conclusion: Literal["reproducible", "partially_reproducible"]
    claim_scope: Literal["trajectory_reproducible", "configuration_traceable_or_statistically_reproducible"]
    fixed_user_model_deviation_disclosed: bool
    checklist_sha256: Digest

    @model_validator(mode="after")
    def exact_conclusion_and_hash(self) -> "ReproducibilityChecklist":
        if len({item.item_id for item in self.checks}) != len(self.checks):
            raise ValueError("duplicate reproducibility check")
        if tuple(item.item_id for item in self.checks) != tuple(
            f"section24_item_{index:02d}" for index in range(1, 15)
        ):
            raise ValueError("exact ordered Section 24 checklist required")
        for index, item in enumerate(self.checks, start=1):
            expected_required = index != 13 or self.profile == "strict_replay"
            if item.required != expected_required:
                raise ValueError("Section 24 requiredness cannot be downgraded")
        item_13 = self.checks[12]
        if self.profile == "official_live" and (
            item_13.status != "not_applicable_with_reason"
            or item_13.reason_code != "official_live_profile"
            or item_13.evidence_sha256 is not None
        ):
            raise ValueError("official-live item 13 has one fixed N/A reason")
        expected_conclusion = (
            "partially_reproducible"
            if any(item.required and item.status in {"fail", "not_run"} for item in self.checks)
            else "reproducible"
        )
        if self.conclusion != expected_conclusion:
            raise ValueError("reproducibility conclusion mismatch")
        expected_scope = (
            "trajectory_reproducible"
            if self.profile == "strict_replay"
            else "configuration_traceable_or_statistically_reproducible"
        )
        if self.claim_scope != expected_scope:
            raise ValueError("reproducibility claim scope mismatch")
        if self.fixed_user_model_deviation_disclosed != (self.profile == "official_live"):
            raise ValueError("official-live User-model deviation disclosure mismatch")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"checklist_sha256"}))
        if self.checklist_sha256 != expected_hash:
            raise ValueError("reproducibility checklist hash mismatch")
        return self


class ArtifactManifestEntry(_FrozenStrict):
    path: str
    sha256: Digest
    byte_count: Annotated[int, Field(ge=0)]
    mode: Literal["0600"] = "0600"


class ReportManifest(_FrozenStrict):
    schema_version: Literal[1] = 1
    plan_sha256: Digest
    entries: tuple[ArtifactManifestEntry, ...]
    manifest_sha256: Digest


__all__ = [name for name in tuple(globals()) if not name.startswith("_")]
