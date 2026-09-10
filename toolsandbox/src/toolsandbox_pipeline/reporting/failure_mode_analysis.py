"""Exact failure-mode attribution and eight-variant family stability."""

from __future__ import annotations

from collections import defaultdict
from math import fsum

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.fixtures import ExternalReadAttempt
from toolsandbox_pipeline.schemas.offline_skill import FailureModeLineageRecord
from toolsandbox_pipeline.schemas.reporting import (
    SYSTEM_ORDER,
    FailureModeAttributionRow,
    FailureModeRepairSummary,
    FailureSignatureObservation,
    FamilyStabilityRecord,
    FamilyStabilitySummary,
    ScenarioEvaluationRecord,
)
from toolsandbox_pipeline.schemas.trajectory import TrustedTrajectory


class FailureModeAnalysisError(ValueError):
    """Sanitized failure-analysis contract violation."""


def _scenario_matrix(records: tuple[ScenarioEvaluationRecord, ...]):
    matrix: dict[tuple[str, str], ScenarioEvaluationRecord] = {}
    positions: dict[int, str] = {}
    for record in records:
        key = (record.system_id, record.scenario_id)
        if key in matrix:
            raise FailureModeAnalysisError("duplicate system/scenario result")
        matrix[key] = record
        if record.system_id == "vanilla":
            if record.manifest_position in positions:
                raise FailureModeAnalysisError("duplicate manifest position")
            positions[record.manifest_position] = record.scenario_id
    try:
        ordered_ids = tuple(positions[index] for index in range(len(positions)))
    except KeyError as error:
        raise FailureModeAnalysisError("non-contiguous manifest positions") from error
    expected = {
        (system_id, scenario_id)
        for system_id in SYSTEM_ORDER
        for scenario_id in ordered_ids
    }
    if set(matrix) != expected:
        raise FailureModeAnalysisError("incomplete three-system scenario matrix")
    for scenario_id in ordered_ids:
        shared = [matrix[(system_id, scenario_id)] for system_id in SYSTEM_ORDER]
        reference = shared[0]
        identity = (
            reference.manifest_position,
            reference.scenario_family_id,
            reference.variant,
            reference.categories,
        )
        if any(
            (item.manifest_position, item.scenario_family_id, item.variant, item.categories)
            != identity
            for item in shared[1:]
        ):
            raise FailureModeAnalysisError("shared scenario identity drift")
    return ordered_ids, matrix


def analyze_failure_modes(
    *,
    lineage_records: tuple[FailureModeLineageRecord, ...],
    scenario_records: tuple[ScenarioEvaluationRecord, ...],
    generation_0_trajectories: tuple[TrustedTrajectory, ...],
    generation_0_external_attempts: tuple[ExternalReadAttempt, ...],
    updated_trajectories: tuple[TrustedTrajectory, ...],
) -> tuple[tuple[FailureModeAttributionRow, ...], FailureModeRepairSummary]:
    """Classify failures using signatures derived from trusted host records."""

    ordered_ids, matrix = _scenario_matrix(scenario_records)
    observations = derive_failure_signature_observations(
        generation_0_trajectories, generation_0_external_attempts, matrix
    )
    observation_by_id: dict[str, list[FailureSignatureObservation]] = defaultdict(list)
    for observation in observations:
        observation_by_id[observation.scenario_id].append(observation)
    proof_by_key = _derive_updated_skill_use_proofs(updated_trajectories, matrix)
    lineage_by_signature: dict[str, list[FailureModeLineageRecord]] = defaultdict(list)
    lineage_by_id: dict[str, FailureModeLineageRecord] = {}
    for lineage in lineage_records:
        existing = lineage_by_id.get(lineage.lineage_id)
        if existing is not None:
            if existing != lineage:
                raise FailureModeAnalysisError("conflicting failure lineage identity")
            continue
        lineage_by_id[lineage.lineage_id] = lineage
        lineage_by_signature[lineage.failure_signature_sha256].append(lineage)
    for values in lineage_by_signature.values():
        values.sort(key=lambda row: row.lineage_id.encode("utf-8"))

    rows: list[FailureModeAttributionRow] = []
    classifications: dict[str, str] = {}
    related_differences: list[float] = []
    overall_differences: list[float] = []
    vanilla_differences: list[float] = []
    failure_count = 0
    for scenario_id in ordered_ids:
        vanilla = matrix[("vanilla", scenario_id)]
        generation_0 = matrix[("generation_0", scenario_id)]
        updated = matrix[("updated", scenario_id)]
        overall_differences.append(updated.similarity - generation_0.similarity)
        vanilla_differences.append(updated.similarity - vanilla.similarity)
        scenario_observations = tuple(observation_by_id.get(scenario_id, ()))
        signatures = tuple(
            value
            for value in (item.signature_sha256 for item in scenario_observations)
            if value is not None
        )
        matches_by_id = {
            lineage.lineage_id: lineage
            for signature in signatures
            for lineage in lineage_by_signature.get(signature, ())
        }
        matches = tuple(
            matches_by_id[key]
            for key in sorted(matches_by_id, key=lambda item: item.encode("utf-8"))
        )
        signature = (
            matches[0].failure_signature_sha256
            if len(matches) == 1
            else signatures[0] if len(signatures) == 1 else None
        )
        proof_hash = None
        if generation_0.fully_successful:
            classification = "not_generation_0_failure"
            reason = "generation_0_fully_successful"
            matches = ()
            signature = None
        else:
            failure_count += 1
            if not signatures:
                classification = "incomplete_evidence"
                reason = "missing_complete_host_signature"
            elif not matches:
                classification = "unmatched"
                reason = "zero_exact_lineage_matches"
            elif len(matches) > 1:
                classification = "ambiguous"
                reason = "multiple_exact_lineage_matches"
            else:
                lineage = matches[0]
                proof = None
                if (
                    lineage.accepted_skill_version is not None
                    and lineage.accepted_skill_effect_id is not None
                ):
                    proof = proof_by_key.get(
                        (scenario_id, lineage.skill_id, lineage.accepted_skill_version)
                    )
                if updated.fully_successful and proof is not None:
                    classification = "related_repaired"
                    reason = "updated_success_and_linked_skill_version_used"
                    proof_hash = proof
                else:
                    classification = "related_unrepaired"
                    reason = (
                        "updated_not_fully_successful"
                        if not updated.fully_successful
                        else "linked_evolved_skill_use_not_proven"
                    )
                related_differences.append(updated.similarity - generation_0.similarity)
        classifications[scenario_id] = classification
        common = dict(
            manifest_position=vanilla.manifest_position,
            scenario_id=scenario_id,
            scenario_family_id=vanilla.scenario_family_id,
            failure_signature_sha256=signature,
            matched_lineage_ids=tuple(item.lineage_id for item in matches),
            matched_mode_ids=tuple(item.mode_id for item in matches),
            matched_skill_versions=tuple(
                item.accepted_skill_version
                for item in matches
                if item.accepted_skill_version is not None
            ),
            generation_0_fully_successful=generation_0.fully_successful,
            updated_fully_successful=updated.fully_successful,
            generation_0_similarity=generation_0.similarity,
            updated_similarity=updated.similarity,
            vanilla_similarity=vanilla.similarity,
            evolved_skill_use_proof_sha256=proof_hash,
            classification=classification,
            reason_code=reason,
        )
        for system_id in SYSTEM_ORDER:
            rows.append(FailureModeAttributionRow(system_id=system_id, **common))

    related = sum(
        classifications[item] in {"related_repaired", "related_unrepaired"}
        for item in ordered_ids
    )
    repaired = sum(
        classifications[item] == "related_repaired" for item in ordered_ids
    )
    count = len(ordered_ids)
    summary = FailureModeRepairSummary(
        generation_0_failure_case_count=failure_count,
        failure_mode_related_case_count=related,
        failure_mode_repaired_case_count=repaired,
        failure_mode_repair_rate=None if related == 0 else repaired / related,
        repair_rate_complete=related > 0,
        failure_mode_unmatched_case_count=sum(
            classifications[item] == "unmatched" for item in ordered_ids
        ),
        failure_mode_ambiguous_case_count=sum(
            classifications[item] == "ambiguous" for item in ordered_ids
        ),
        incomplete_evidence_case_count=sum(
            classifications[item] == "incomplete_evidence" for item in ordered_ids
        ),
        updated_minus_generation_0_similarity_points_overall=(
            0.0 if count == 0 else fsum(overall_differences) / count
        ),
        updated_minus_generation_0_similarity_points_related_subset=(
            None if related == 0 else fsum(related_differences) / related
        ),
        updated_minus_vanilla_similarity_points_overall=(
            0.0 if count == 0 else fsum(vanilla_differences) / count
        ),
    )
    ordered_rows = tuple(sorted(
        rows,
        key=lambda row: (SYSTEM_ORDER.index(row.system_id), row.manifest_position),
    ))
    return ordered_rows, summary


def derive_failure_signature_observations(
    trajectories: tuple[TrustedTrajectory, ...],
    external_attempts: tuple[ExternalReadAttempt, ...],
    scenario_records_or_matrix: (
        tuple[ScenarioEvaluationRecord, ...]
        | dict[tuple[str, str], ScenarioEvaluationRecord]
    ),
) -> tuple[FailureSignatureObservation, ...]:
    """Derive zero or more exact signatures per failed Generation-0 case.

    Only committed tool-action columns, evaluated Skill attributions, and finite
    host counters/headlines are consumed. Unsupported or incomplete evidence is
    represented as a wholly unavailable observation and can never match lineage.
    """

    if isinstance(scenario_records_or_matrix, tuple):
        ordered_ids, matrix = _scenario_matrix(scenario_records_or_matrix)
    else:
        matrix = scenario_records_or_matrix
        ordered_ids = tuple(
            item.scenario_id
            for key, item in sorted(
                matrix.items(), key=lambda pair: pair[1].manifest_position
            )
            if key[0] == "vanilla"
        )
    failed_ids = tuple(
        scenario_id
        for scenario_id in ordered_ids
        if not matrix[("generation_0", scenario_id)].fully_successful
    )
    by_scenario: dict[str, TrustedTrajectory] = {}
    attempts_by_id: dict[str, ExternalReadAttempt] = {}
    for attempt in external_attempts:
        if type(attempt) is not ExternalReadAttempt:
            raise FailureModeAnalysisError("exact ExternalReadAttempt required")
        if attempt.attempt_id in attempts_by_id:
            raise FailureModeAnalysisError("duplicate external attempt")
        attempts_by_id[attempt.attempt_id] = attempt
    referenced_attempt_ids: set[str] = set()
    for trajectory in trajectories:
        if type(trajectory) is not TrustedTrajectory:
            raise FailureModeAnalysisError("exact TrustedTrajectory required")
        identity = trajectory.identity
        if identity.system_variant != "generation_0" or identity.generation_id != "g000":
            raise FailureModeAnalysisError(
                "failure signatures require Generation-0 G000 trajectories"
            )
        if identity.scenario_id in by_scenario:
            raise FailureModeAnalysisError("duplicate Generation-0 trajectory")
        scenario = matrix.get(("generation_0", identity.scenario_id))
        if scenario is None or scenario.fully_successful:
            raise FailureModeAnalysisError(
                "Generation-0 trajectory is outside failed result set"
            )
        if (
            trajectory.trajectory_id != scenario.trajectory_sha256
            or trajectory.evaluator_record_sha256
            != scenario.evaluator_record_sha256
            or identity.family_id != scenario.scenario_family_id
            or identity.manifest_position != scenario.manifest_position
        ):
            raise FailureModeAnalysisError(
                "Generation-0 trajectory/result identity mismatch"
            )
        by_scenario[identity.scenario_id] = trajectory
        referenced_attempt_ids.update(
            attempt_id
            for action in trajectory.tool_actions
            for attempt_id in action.external_attempt_ids
        )
    if set(attempts_by_id) != referenced_attempt_ids:
        raise FailureModeAnalysisError(
            "external attempts must exactly match trusted trajectory references"
        )

    output: list[FailureSignatureObservation] = []
    for scenario_id in failed_ids:
        scenario = matrix[("generation_0", scenario_id)]
        trajectory = by_scenario.get(scenario_id)
        complete = () if trajectory is None else _complete_host_signatures(
            trajectory, attempts_by_id
        )
        if not complete:
            output.append(FailureSignatureObservation.build(
                scenario_id=scenario_id,
                skill_id=None,
                evidence_kind=None,
                canonical_tool_dependencies=None,
                sanitized_outcome_class=None,
            ))
        else:
            output.extend(
                FailureSignatureObservation.build(scenario_id=scenario_id, **values)
                for values in complete
            )
    return tuple(output)


def _complete_host_signatures(
    trajectory: TrustedTrajectory,
    attempts_by_id: dict[str, ExternalReadAttempt],
) -> tuple[dict[str, object], ...]:
    calls: dict[str, tuple[str | None, str, tuple[str, ...]]] = {}
    for action in trajectory.tool_actions:
        if not action.executed or not action.committed:
            continue
        for call_id, skill_id, tool_id in zip(
            action.call_ids,
            action.selected_skill_ids,
            action.canonical_tool_ids,
            strict=True,
        ):
            if call_id in calls:
                raise FailureModeAnalysisError("duplicate committed tool call")
            calls[call_id] = (skill_id, tool_id, action.external_attempt_ids)
    values: list[dict[str, object]] = []
    for attribution in trajectory.skill_attributions:
        if (
            attribution.generation_id != "g000"
            or attribution.evaluator_record_sha256
            != trajectory.evaluator_record_sha256
            or attribution.fully_successful
        ):
            continue
        if any(
            call_id not in calls
            or calls[call_id][0] != attribution.skill_id
            or calls[call_id][1] != tool_id
            for call_id, tool_id in zip(
                attribution.executed_call_ids,
                attribution.canonical_tool_ids,
                strict=True,
            )
        ):
            continue
        failure_attempts: list[ExternalReadAttempt] = []
        for call_id in attribution.executed_call_ids:
            for attempt_id in calls[call_id][2]:
                attempt = attempts_by_id.get(attempt_id)
                if (
                    attempt is None
                    or attempt.context.run_id != trajectory.identity.run_id
                    or attempt.context.scenario_id != trajectory.identity.scenario_id
                    or attempt.context.logical_tool_call_id != call_id
                    or attempt.canonical_tool_name != calls[call_id][1]
                ):
                    continue
                if attempt.status in {
                    "fixture_miss", "failed", "rejected_before_dispatch",
                    "unknown_outcome",
                }:
                    failure_attempts.append(attempt)
        if len(failure_attempts) != 1:
            continue
        attempt = failure_attempts[0]
        evidence_kind = (
            "fixture_miss"
            if attempt.status == "fixture_miss"
            else "visible_tool_exception"
        )
        outcome = attempt.sanitized_exception_class
        if outcome is None:
            continue
        values.append({
            "skill_id": attribution.skill_id,
            "evidence_kind": evidence_kind,
            "canonical_tool_dependencies": tuple(sorted(
                set(attribution.canonical_tool_ids),
                key=lambda item: item.encode("utf-8"),
            )),
            "sanitized_outcome_class": outcome,
        })
    return tuple(values)


def _derive_updated_skill_use_proofs(
    trajectories: tuple[TrustedTrajectory, ...],
    matrix: dict[tuple[str, str], ScenarioEvaluationRecord],
) -> dict[tuple[str, str, str], str]:
    proofs: dict[tuple[str, str, str], str] = {}
    for trajectory in trajectories:
        if type(trajectory) is not TrustedTrajectory:
            raise FailureModeAnalysisError("exact TrustedTrajectory required")
        identity = trajectory.identity
        if identity.system_variant != "updated" or identity.generation_id != "g003":
            raise FailureModeAnalysisError(
                "Skill-use proof requires Updated G003 trajectory"
            )
        scenario = matrix.get(("updated", identity.scenario_id))
        if scenario is None:
            raise FailureModeAnalysisError("Updated trajectory is outside result matrix")
        if (
            trajectory.trajectory_id != scenario.trajectory_sha256
            or trajectory.evaluator_record_sha256 != scenario.evaluator_record_sha256
            or identity.family_id != scenario.scenario_family_id
            or identity.manifest_position != scenario.manifest_position
        ):
            raise FailureModeAnalysisError("Updated trajectory/result identity mismatch")
        turns_by_call = {
            call_id: turn
            for turn in trajectory.online_turns
            for call_id in turn.executed_call_ids
        }
        for attribution in trajectory.skill_attributions:
            if (
                attribution.generation_id != "g003"
                or attribution.evaluator_record_sha256
                != trajectory.evaluator_record_sha256
                or attribution.fully_successful != scenario.fully_successful
            ):
                raise FailureModeAnalysisError("Skill attribution/evaluator mismatch")
            if any(
                call_id not in turns_by_call
                for call_id in attribution.executed_call_ids
            ):
                raise FailureModeAnalysisError(
                    "attributed call is absent from committed online turns"
                )
            linked_turns = [
                turn
                for turn in trajectory.online_turns
                if set(turn.executed_call_ids).intersection(
                    attribution.executed_call_ids
                )
            ]
            if any(not turn.application_ids for turn in linked_turns):
                raise FailureModeAnalysisError(
                    "attributed call lacks applied decision chain"
                )
            application_ids = tuple(
                application_id
                for turn in linked_turns
                for application_id in turn.application_ids
            )
            if not application_ids or len(set(application_ids)) != len(application_ids):
                raise FailureModeAnalysisError(
                    "invalid attributed application lineage"
                )
            key = (
                identity.scenario_id,
                attribution.skill_id,
                attribution.skill_version,
            )
            if key in proofs:
                raise FailureModeAnalysisError("duplicate derived Skill-use proof")
            proofs[key] = canonical_sha256({
                "protocol": "updated-g003-skill-use-proof-v1",
                "scenario_id": identity.scenario_id,
                "trajectory_id": trajectory.trajectory_id,
                "evaluator_record_sha256": trajectory.evaluator_record_sha256,
                "skill_id": attribution.skill_id,
                "skill_version": attribution.skill_version,
                "executed_call_ids": list(attribution.executed_call_ids),
                "online_turn_application_ids": list(application_ids),
            })
    return proofs


def compute_family_stability(
    scenario_records: tuple[ScenarioEvaluationRecord, ...],
    attribution_rows: tuple[FailureModeAttributionRow, ...],
) -> tuple[tuple[FamilyStabilityRecord, ...], tuple[FamilyStabilitySummary, ...]]:
    """Compute stability over exactly eight variants per family."""

    ordered_ids, matrix = _scenario_matrix(scenario_records)
    attribution = {
        row.scenario_id: row.classification
        for row in attribution_rows
        if row.system_id == "generation_0"
    }
    if set(attribution) != set(ordered_ids):
        raise FailureModeAnalysisError("attribution coverage mismatch")
    family_ids = sorted(
        {matrix[("vanilla", item)].scenario_family_id for item in ordered_ids},
        key=lambda item: item.encode("utf-8"),
    )
    if len(family_ids) != 25:
        raise FailureModeAnalysisError("formal stability analysis requires exactly 25 families")
    output: list[FamilyStabilityRecord] = []
    summaries: list[FamilyStabilitySummary] = []
    for system_id in SYSTEM_ORDER:
        system_rows: list[FamilyStabilityRecord] = []
        for family_id in family_ids:
            variants = [
                matrix[(system_id, item)]
                for item in ordered_ids
                if matrix[(system_id, item)].scenario_family_id == family_id
            ]
            if len(variants) != 8:
                raise FailureModeAnalysisError(
                    "each family must contain exactly eight variants"
                )
            related_ids = [
                item.scenario_id
                for item in variants
                if attribution[item.scenario_id]
                in {"related_repaired", "related_unrepaired"}
            ]
            repaired = sum(
                attribution[item] == "related_repaired" for item in related_ids
            )
            scores = [item.similarity for item in variants]
            row = FamilyStabilityRecord(
                system_id=system_id,
                scenario_family_id=family_id,
                variant_count=8,
                all_variants_fully_successful=all(
                    item.fully_successful for item in variants
                ),
                minimum_native_similarity=min(scores),
                similarity_range=max(scores) - min(scores),
                related_generation_0_failure_variant_count=len(related_ids),
                repaired_variant_count=repaired,
                all_related_failures_repaired=(
                    None if not related_ids else repaired == len(related_ids)
                ),
            )
            output.append(row)
            system_rows.append(row)
        count = len(system_rows)
        related_families = [
            item
            for item in system_rows
            if item.related_generation_0_failure_variant_count > 0
        ]
        fully_repaired = sum(
            item.all_related_failures_repaired is True for item in related_families
        )
        all_success = sum(item.all_variants_fully_successful for item in system_rows)
        summaries.append(
            FamilyStabilitySummary(
                system_id=system_id,
                family_count=count,
                all_variants_success_family_count=all_success,
                all_variants_success_family_rate=(
                    None if count == 0 else all_success / count
                ),
                mean_family_minimum_similarity=(
                    None
                    if count == 0
                    else fsum(item.minimum_native_similarity for item in system_rows)
                    / count
                ),
                mean_within_family_similarity_range=(
                    None
                    if count == 0
                    else fsum(item.similarity_range for item in system_rows) / count
                ),
                related_family_count=len(related_families),
                fully_repaired_related_family_count=fully_repaired,
                fully_repaired_related_family_rate=(
                    None
                    if not related_families
                    else fully_repaired / len(related_families)
                ),
            )
        )
    return tuple(output), tuple(summaries)


__all__ = [
    "FailureModeAnalysisError",
    "analyze_failure_modes",
    "compute_family_stability",
    "derive_failure_signature_observations",
]
