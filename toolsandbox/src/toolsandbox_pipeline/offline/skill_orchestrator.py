"""Deterministic Task 016 orchestration over sealed synthetic/train references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.offline_skill import (
    DevMiniBenchResult,
    FailureModeAdd,
    FailureModeDecision,
    FailureModeMerge,
    SkillContent,
    SkillFailureEvidence,
    SkillRoundIdentity,
    SkillRoundResult,
    SkillUpdateUnitResult,
)
from toolsandbox_pipeline.schemas.skill import SkillRecord
from toolsandbox_pipeline.schemas.trajectory import TrustedTrajectory

from .failure_lineage import (
    FailureModeLineageRecord,
    build_lineage,
    link_accepted_skill,
)
from .failure_modes import apply_failure_mode_decision, failure_mode_id
from .skill_rewrite import stage_skill_rewrite, validate_skill_candidate
from .skill_statistics import (
    ManifestSkillUse,
    mark_rewrite_attempt,
    rewrite_triggered,
    stage_skill_statistics,
)


@dataclass(frozen=True)
class SkillTrajectoryEntry:
    trajectory: TrustedTrajectory
    failure_evidence: tuple[SkillFailureEvidence, ...] = ()


class SealedSkillTrajectoryBuffer:
    """Content-validated current-round train input; no dev/test loader exists here."""

    def __init__(self, identity: SkillRoundIdentity, entries: tuple[SkillTrajectoryEntry, ...]):
        if type(identity) is not SkillRoundIdentity:
            raise TypeError("strict SkillRoundIdentity required")
        self.identity = identity
        self.entries = entries
        positions = tuple(entry.trajectory.identity.manifest_position for entry in entries)
        if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
            raise ValueError("sealed train trajectories must use unique manifest order")
        for entry in entries:
            trajectory = entry.trajectory
            episode = trajectory.identity
            if (
                not trajectory.eligible_for_train_offline_consumption
                or episode.run_id != identity.run_id
                or episode.round_index != identity.round_index
                or episode.shard_id != identity.shard_id
                or episode.generation_id != identity.current_generation_id
                or episode.dataset_manifest_sha256 != identity.dataset_manifest_sha256
                or episode.runtime_config_sha256 != identity.config_manifest_sha256
            ):
                raise ValueError("ineligible or identity-mismatched train trajectory")
            attributions = {item.skill_id: item for item in trajectory.skill_attributions}
            for evidence in entry.failure_evidence:
                attribution = attributions.get(evidence.skill_id)
                if (
                    evidence.trajectory_id != trajectory.trajectory_id
                    or evidence.episode_id != episode.episode_id
                    or evidence.manifest_position != episode.manifest_position
                    or attribution is None
                    or attribution.fully_successful
                    or evidence.skill_version != attribution.skill_version
                ):
                    raise ValueError("failure evidence lacks one trusted train failure")
        expected = canonical_sha256(
            {
                "protocol": "sealed-skill-buffer-v1",
                "run_id": identity.run_id,
                "round_index": identity.round_index,
                "shard_id": identity.shard_id,
                "generation_id": identity.current_generation_id,
                "entries": [
                    [
                        entry.trajectory.trajectory_id,
                        canonical_sha256(entry.trajectory.model_dump(mode="json")),
                        entry.trajectory.identity.manifest_position,
                    ]
                    for entry in entries
                ],
            }
        )
        if identity.sealed_input_buffer_sha256 != expected:
            raise ValueError("sealed Skill trajectory buffer hash mismatch")


@dataclass(frozen=True)
class AppliedFailureDecision:
    decision: FailureModeDecision
    application_id: str


@dataclass(frozen=True)
class AppliedSkillCandidate:
    candidate: SkillContent
    application_id: str


@dataclass(frozen=True)
class MiniBenchExecution:
    result: DevMiniBenchResult
    ordered_qwen_application_ids: tuple[str, ...] = ()


class FailureDecisionExecutor(Protocol):
    def decide(
        self,
        *,
        evidence: SkillFailureEvidence,
        current_buffer: tuple,
        unit_id: str,
    ) -> AppliedFailureDecision: ...


class CandidateExecutor(Protocol):
    def rewrite(
        self,
        *,
        skill: SkillRecord,
        evidence: tuple[SkillFailureEvidence, ...],
        unit_id: str,
    ) -> AppliedSkillCandidate: ...


class DevMiniBenchExecutor(Protocol):
    def evaluate(
        self,
        *,
        current_skill: SkillRecord,
        candidate: SkillContent,
        earlier_accepted_skills: tuple[SkillRecord, ...],
        unit_id: str,
    ) -> MiniBenchExecution: ...


class SkillEffectSink(Protocol):
    def checkpoint(self, *, event_kind: str, unit_id: str, payload: object) -> None: ...

    def commit_failure_mutation(
        self,
        *,
        unit_id: str,
        application_id: str,
        mutation_artifact_sha256: str,
    ) -> str: ...

    def record_non_substantive(self, *, unit_id: str, application_ids: tuple[str, ...]) -> None: ...

    def commit_accepted_skill(
        self,
        *,
        unit_id: str,
        ordered_application_ids: tuple[str, ...],
        candidate: SkillContent,
        mini_bench: DevMiniBenchResult,
    ) -> str: ...


class SkillUpdateOrchestrator:
    def __init__(
        self,
        *,
        failure_executor: FailureDecisionExecutor,
        candidate_executor: CandidateExecutor,
        mini_bench_executor: DevMiniBenchExecutor,
        effect_sink: SkillEffectSink,
        public_tool_inventory: tuple[str, ...],
    ) -> None:
        self.failure_executor = failure_executor
        self.candidate_executor = candidate_executor
        self.mini_bench_executor = mini_bench_executor
        self.effect_sink = effect_sink
        self.public_tool_inventory = public_tool_inventory

    def run(
        self,
        *,
        buffer: SealedSkillTrajectoryBuffer,
        current_skills: tuple[SkillRecord, ...],
    ) -> SkillRoundResult:
        active = {skill.skill_id: skill for skill in current_skills if skill.status == "active"}
        if len(active) != len(current_skills) or len(active) != len({item.skill_id for item in current_skills}):
            raise ValueError("exactly one active current version per Skill required")
        ordered_skill_ids = tuple(sorted(active, key=lambda item: item.encode("utf-8")))
        uses = tuple(
            ManifestSkillUse(
                manifest_position=entry.trajectory.identity.manifest_position,
                episode_id=entry.trajectory.identity.episode_id,
                attribution=attribution,
            )
            for entry in buffer.entries
            for attribution in sorted(
                entry.trajectory.skill_attributions,
                key=lambda item: item.skill_id.encode("utf-8"),
            )
        )
        units: list[SkillUpdateUnitResult] = []
        mutations = []
        lineages: list[FailureModeLineageRecord] = []
        accepted_records: list[SkillRecord] = []
        global_seq = max(
            (mode.last_observed_seq for skill in active.values() for mode in skill.failure_mode_buffer),
            default=0,
        )
        for skill_id in ordered_skill_ids:
            current = active[skill_id]
            statistics = stage_skill_statistics(current.online_statistics, uses, skill_id=skill_id)
            staged_buffer = current.failure_mode_buffer
            applications: list[str] = []
            substantive_apps: list[str] = []
            effect_ids: list[str] = []
            skill_lineages: list[FailureModeLineageRecord] = []
            failures = tuple(
                evidence
                for entry in buffer.entries
                for evidence in entry.failure_evidence
                if evidence.skill_id == skill_id
            )
            for evidence_index, evidence in enumerate(failures):
                unit_id = f"skill-failure-{buffer.identity.round_index}-{skill_id}-{evidence_index:06d}"
                self.effect_sink.checkpoint(event_kind="before_failure_request", unit_id=unit_id, payload=evidence.source_evidence_sha256)
                applied = self.failure_executor.decide(
                    evidence=evidence,
                    current_buffer=staged_buffer,
                    unit_id=unit_id,
                )
                applications.append(applied.application_id)
                next_seq = None
                if isinstance(applied.decision, (FailureModeAdd, FailureModeMerge)):
                    global_seq += 1
                    next_seq = global_seq
                mutation = apply_failure_mode_decision(
                    skill_id=skill_id,
                    buffer=staged_buffer,
                    decision=applied.decision,
                    next_observed_seq=next_seq,
                )
                self.effect_sink.checkpoint(event_kind="failure_decision_applied", unit_id=unit_id, payload=mutation.after_sha256)
                if mutation.substantive:
                    effect_id = self.effect_sink.commit_failure_mutation(
                        unit_id=unit_id,
                        application_id=applied.application_id,
                        mutation_artifact_sha256=mutation.after_sha256,
                    )
                    substantive_apps.append(applied.application_id)
                    effect_ids.append(effect_id)
                    mode_id = (
                        failure_mode_id(skill_id, applied.decision.task_condition, applied.decision.failure_mode)
                        if isinstance(applied.decision, FailureModeAdd)
                        else applied.decision.mode_id
                    )
                    lineage = build_lineage(
                        skill_id=skill_id,
                        evidence_kind=evidence.evidence_kind,
                        canonical_tool_dependencies=evidence.canonical_tool_dependencies,
                        sanitized_outcome_class=evidence.sanitized_outcome_class,
                        mode_id=mode_id,
                        source_evidence_sha256=evidence.source_evidence_sha256,
                        producing_round=buffer.identity.round_index,
                        failure_mode_effect_id=effect_id,
                    )
                    skill_lineages.append(lineage)
                else:
                    self.effect_sink.record_non_substantive(
                        unit_id=unit_id,
                        application_ids=(applied.application_id,),
                    )
                staged_buffer = mutation.after
            staged_skill = SkillRecord(
                **current.model_dump(
                    mode="python",
                    exclude={"online_statistics", "failure_mode_buffer"},
                ),
                online_statistics=statistics,
                failure_mode_buffer=staged_buffer,
            )
            status = "unchanged"
            candidate_application_id = None
            total_cost_apps = list(substantive_apps)
            staged_mutation = None
            if rewrite_triggered(statistics):
                statistics = mark_rewrite_attempt(statistics)
                staged_skill = SkillRecord(
                    **staged_skill.model_dump(mode="python", exclude={"online_statistics"}),
                    online_statistics=statistics,
                )
                rewrite_unit_id = f"skill-rewrite-{buffer.identity.round_index}-{skill_id}"
                self.effect_sink.checkpoint(event_kind="rewrite_attempt_staged", unit_id=rewrite_unit_id, payload=statistics.model_dump(mode="json"))
                candidate_result = self.candidate_executor.rewrite(
                    skill=staged_skill,
                    evidence=failures,
                    unit_id=rewrite_unit_id,
                )
                candidate_application_id = candidate_result.application_id
                validate_skill_candidate(
                    current=staged_skill,
                    candidate=candidate_result.candidate,
                    public_tool_inventory=self.public_tool_inventory,
                )
                mini = self.mini_bench_executor.evaluate(
                    current_skill=staged_skill,
                    candidate=candidate_result.candidate,
                    earlier_accepted_skills=tuple(accepted_records),
                    unit_id=rewrite_unit_id,
                )
                chain = (candidate_application_id, *mini.ordered_qwen_application_ids)
                accepted_effect_id = None
                if mini.result.accepted:
                    accepted_effect_id = self.effect_sink.commit_accepted_skill(
                        unit_id=rewrite_unit_id,
                        ordered_application_ids=chain,
                        candidate=candidate_result.candidate,
                        mini_bench=mini.result,
                    )
                    total_cost_apps.extend(chain)
                    status = "accepted"
                else:
                    self.effect_sink.record_non_substantive(unit_id=rewrite_unit_id, application_ids=chain)
                    status = "rejected"
                staged_mutation = stage_skill_rewrite(
                    current=current,
                    staged_statistics=statistics,
                    staged_failure_modes=staged_buffer,
                    candidate=candidate_result.candidate,
                    mini_bench=mini.result,
                    accepted_effect_id=accepted_effect_id,
                )
                if mini.result.accepted:
                    accepted = staged_mutation.staged_records[-1]
                    accepted_records.append(accepted)
                    skill_lineages = [
                        link_accepted_skill(
                            record,
                            accepted_skill_version=accepted.version,
                            accepted_skill_effect_id=accepted_effect_id,
                        )
                        for record in skill_lineages
                    ]
            if staged_mutation is None:
                from toolsandbox_pipeline.schemas.offline_skill import StagedSkillMutation

                staged_mutation = StagedSkillMutation(
                    skill_id=skill_id,
                    previous_version=current.version,
                    accepted=False,
                    previous_record=current,
                    staged_records=(staged_skill,),
                    failure_buffer_sha256=canonical_sha256([item.model_dump(mode="json") for item in staged_buffer]),
                    statistics_sha256=canonical_sha256(statistics.model_dump(mode="json")),
                )
            mutations.append(staged_mutation)
            lineages.extend(skill_lineages)
            units.append(
                SkillUpdateUnitResult(
                    unit_id=f"skill-update-{buffer.identity.round_index}-{skill_id}",
                    skill_id=skill_id,
                    status=status,
                    failure_decision_application_ids=tuple(applications),
                    substantive_failure_application_ids=tuple(substantive_apps),
                    substantive_failure_effect_ids=tuple(effect_ids),
                    candidate_application_id=candidate_application_id,
                    staged_mutation_sha256=canonical_sha256(staged_mutation.model_dump(mode="json")),
                    total_cost_application_ids=tuple(total_cost_apps),
                )
            )
        result_payload = {
            "identity": buffer.identity.model_dump(mode="json"),
            "ordered_skill_ids": list(ordered_skill_ids),
            "ordered_unit_ids": [unit.unit_id for unit in units],
            "unit_results": [unit.model_dump(mode="json") for unit in units],
            "staged_mutations": [mutation.model_dump(mode="json") for mutation in mutations],
            "lineage_records": [record.model_dump(mode="json") for record in lineages],
            "lineage_record_ids": [record.lineage_id for record in lineages],
            "lineage_records_sha256": canonical_sha256(
                [record.model_dump(mode="json") for record in lineages]
            ),
            "accepted_skill_ids": [unit.skill_id for unit in units if unit.status == "accepted"],
            "rejected_skill_ids": [unit.skill_id for unit in units if unit.status == "rejected"],
            "completion_status": "completed",
        }
        return SkillRoundResult(
            identity=buffer.identity,
            ordered_skill_ids=ordered_skill_ids,
            ordered_unit_ids=tuple(unit.unit_id for unit in units),
            unit_results=tuple(units),
            staged_mutations=tuple(mutations),
            lineage_records=tuple(lineages),
            lineage_record_ids=tuple(record.lineage_id for record in lineages),
            lineage_records_sha256=result_payload["lineage_records_sha256"],
            accepted_skill_ids=tuple(
                unit.skill_id for unit in units if unit.status == "accepted"
            ),
            rejected_skill_ids=tuple(
                unit.skill_id for unit in units if unit.status == "rejected"
            ),
            completion_status="completed",
            result_sha256=canonical_sha256(result_payload),
        )


__all__ = [
    "AppliedFailureDecision",
    "AppliedSkillCandidate",
    "MiniBenchExecution",
    "SealedSkillTrajectoryBuffer",
    "SkillTrajectoryEntry",
    "SkillUpdateOrchestrator",
]
