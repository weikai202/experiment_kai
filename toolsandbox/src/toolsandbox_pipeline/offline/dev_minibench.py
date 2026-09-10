"""Pure paired Dev Mini-Bench aggregation and exact acceptance rule."""

from __future__ import annotations

import math

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.offline_skill import (
    DevBranchResult,
    DevMiniBenchResult,
    DevScenarioSelection,
)


def evaluate_dev_branches(
    selection: DevScenarioSelection,
    branch_results: tuple[DevBranchResult, ...],
) -> DevMiniBenchResult:
    expected = tuple(
        (scenario_id, branch)
        for scenario_id in selection.selected_scenario_ids
        for branch in ("previous", "candidate")
    )
    actual = tuple((result.scenario_id, result.branch) for result in branch_results)
    if actual != expected:
        raise ValueError("Dev branches must be complete, paired, and selected-order exact")
    if any(result.evaluated_skill_id != selection.skill_id for result in branch_results):
        raise ValueError("Dev branch Skill identity mismatch")
    for index in range(0, len(branch_results), 2):
        previous, candidate = branch_results[index : index + 2]
        if previous.shared_configuration_sha256 != candidate.shared_configuration_sha256:
            raise ValueError("A/B branches differ beyond evaluated Skill")
        if previous.episode_id == candidate.episode_id:
            raise ValueError("A/B branches require distinct episode IDs")
    complete = all(result.complete for result in branch_results)
    previous = branch_results[0::2]
    candidate = branch_results[1::2]
    p_success = sum(result.fully_successful is True for result in previous)
    c_success = sum(result.fully_successful is True for result in candidate)
    p_similarity = math.fsum(result.similarity or 0.0 for result in previous)
    c_similarity = math.fsum(result.similarity or 0.0 for result in candidate)
    p_minefield = sum(result.minefield_hit is True for result in previous)
    c_minefield = sum(result.minefield_hit is True for result in candidate)
    if not selection.selected_scenario_ids:
        accepted, reason = False, "no_relevant_dev_scenarios"
    elif not complete:
        accepted, reason = False, "incomplete_branch"
    elif c_success > p_success:
        accepted, reason = True, "higher_full_success"
    elif c_success == p_success and c_similarity > p_similarity and c_minefield <= p_minefield:
        accepted, reason = True, "higher_similarity_without_more_minefields"
    else:
        accepted, reason = False, "not_improved"
    return DevMiniBenchResult(
        skill_id=selection.skill_id,
        selection_sha256=canonical_sha256(selection.model_dump(mode="json")),
        scenario_count=len(selection.selected_scenario_ids),
        previous_full_success_count=p_success,
        candidate_full_success_count=c_success,
        previous_similarity_sum=p_similarity,
        candidate_similarity_sum=c_similarity,
        previous_minefield_hit_count=p_minefield,
        candidate_minefield_hit_count=c_minefield,
        complete=complete,
        accepted=accepted,
        reason=reason,
        branch_result_sha256=canonical_sha256(
            [result.model_dump(mode="json") for result in branch_results]
        ),
    )


__all__ = ["evaluate_dev_branches"]
