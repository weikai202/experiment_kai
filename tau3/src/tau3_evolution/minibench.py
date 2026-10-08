"""Deterministic, relevance-filtered paired Dev Mini-Bench for tau3."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

from .access import AccessAudit
from .canonical import canonical_sha256


@dataclass(frozen=True)
class SkillCandidate:
    skill_id: str
    previous_version: int
    candidate_version: int
    domains: tuple[str, ...]
    tool_dependencies: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.skill_id or self.previous_version < 0:
            raise ValueError("invalid skill candidate")
        if self.candidate_version != self.previous_version + 1:
            raise ValueError("candidate version must increment by one")
        if not self.domains and not self.tool_dependencies:
            raise ValueError("skill must declare domain or tool relevance")


@dataclass(frozen=True)
class DevTaskView:
    domain: str
    task_id: str
    available_tool_names: tuple[str, ...]


@dataclass(frozen=True)
class DevSelection:
    skill_id: str
    manifest_sha256: str
    access_receipt_sha256: str
    dev_task_refs_sha256: str
    host_shared_configuration_sha256: str
    selected: tuple[tuple[str, str], ...]
    selector_input_sha256: str


def _selection_key(seed: int, skill_id: str, domain: str, task_id: str) -> tuple[str, bytes, bytes]:
    raw = f"{seed}\0{skill_id}\0{domain}\0{task_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), domain.encode("utf-8"), task_id.encode("utf-8")


def select_relevant_dev_tasks(
    *,
    candidate: SkillCandidate,
    manifest_sha256: str,
    task_views: tuple[DevTaskView, ...],
    access_audit: AccessAudit,
    seed: int = 0,
    maximum: int = 20,
) -> DevSelection:
    if type(seed) is not int or seed != 0 or maximum != 20:
        raise ValueError("frozen Mini-Bench requires seed=0 and maximum=20")
    identities = [(view.domain, view.task_id) for view in task_views]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate dev task view")
    task_refs_sha = canonical_sha256(
        [{"domain": domain, "task_id": task_id} for domain, task_id in identities]
    )
    if (
        access_audit.split != "dev"
        or access_audit.purpose != "skill_ab_validation"
        or access_audit.manifest_sha256 != manifest_sha256
        or access_audit.task_refs_sha256 != task_refs_sha
        or access_audit.task_count != len(task_views)
    ):
        raise ValueError("Dev views are not bound to the exact audited manifest selection")
    domains = set(candidate.domains)
    dependencies = set(candidate.tool_dependencies)
    eligible = [
        view
        for view in task_views
        if (not domains or view.domain in domains)
        and (not dependencies or dependencies <= set(view.available_tool_names))
    ]
    selected = tuple(
        (view.domain, view.task_id)
        for view in sorted(
            eligible,
            key=lambda view: _selection_key(seed, candidate.skill_id, view.domain, view.task_id),
        )[:maximum]
    )
    selector_input = [
        {
            "domain": view.domain,
            "task_id": view.task_id,
            "available_tool_names": list(view.available_tool_names),
        }
        for view in task_views
    ]
    selector_input_sha = canonical_sha256(selector_input)
    receipt_sha = canonical_sha256(asdict(access_audit))
    return DevSelection(
        skill_id=candidate.skill_id,
        manifest_sha256=manifest_sha256,
        access_receipt_sha256=receipt_sha,
        dev_task_refs_sha256=task_refs_sha,
        host_shared_configuration_sha256=canonical_sha256(
            {
                "manifest_sha256": manifest_sha256,
                "access_receipt_sha256": receipt_sha,
                "selector_input_sha256": selector_input_sha,
                "seed": seed,
                "maximum": maximum,
            }
        ),
        selected=selected,
        selector_input_sha256=selector_input_sha,
    )


@dataclass(frozen=True)
class BranchResult:
    domain: str
    task_id: str
    branch: str
    evaluated_skill_id: str
    reward: float
    complete: bool
    episode_id: str
    shared_configuration_sha256: str
    manifest_sha256: str
    access_receipt_sha256: str
    trajectory_sha256: str
    evaluator_record_sha256: str

    def __post_init__(self) -> None:
        if self.branch not in {"previous", "candidate"}:
            raise ValueError("invalid Mini-Bench branch")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise ValueError("finite float reward required")
        for value in (
            self.shared_configuration_sha256,
            self.manifest_sha256,
            self.access_receipt_sha256,
            self.trajectory_sha256,
            self.evaluator_record_sha256,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("Mini-Bench evidence requires content hashes")


@dataclass(frozen=True)
class MiniBenchResult:
    skill_id: str
    task_count: int
    previous_successes: int
    candidate_successes: int
    previous_reward_sum: float
    candidate_reward_sum: float
    complete: bool
    accepted: bool
    reason: str
    evidence_sha256: str


def evaluate_paired_branches(
    selection: DevSelection, results: tuple[BranchResult, ...]
) -> MiniBenchResult:
    expected = tuple(
        (domain, task_id, branch)
        for domain, task_id in selection.selected
        for branch in ("previous", "candidate")
    )
    actual = tuple((row.domain, row.task_id, row.branch) for row in results)
    if actual != expected:
        raise ValueError("branches must be complete, paired, and in selected order")
    if any(row.evaluated_skill_id != selection.skill_id for row in results):
        raise ValueError("evaluated Skill identity mismatch")
    for index in range(0, len(results), 2):
        previous, candidate = results[index : index + 2]
        for row in (previous, candidate):
            if (
                row.shared_configuration_sha256 != selection.host_shared_configuration_sha256
                or row.manifest_sha256 != selection.manifest_sha256
                or row.access_receipt_sha256 != selection.access_receipt_sha256
            ):
                raise ValueError("A/B branch evidence is not host/config/manifest bound")
        if previous.episode_id == candidate.episode_id:
            raise ValueError("A/B branches require distinct episode IDs")
    previous_rows = results[0::2]
    candidate_rows = results[1::2]
    complete = all(row.complete for row in results)
    previous_successes = sum(row.reward == 1.0 for row in previous_rows)
    candidate_successes = sum(row.reward == 1.0 for row in candidate_rows)
    previous_sum = math.fsum(row.reward for row in previous_rows)
    candidate_sum = math.fsum(row.reward for row in candidate_rows)
    if not selection.selected:
        accepted, reason = False, "no_relevant_dev_tasks"
    elif not complete:
        accepted, reason = False, "incomplete_branch"
    elif candidate_successes > previous_successes:
        accepted, reason = True, "higher_full_success"
    elif candidate_successes == previous_successes and candidate_sum > previous_sum:
        accepted, reason = True, "higher_native_reward_without_success_regression"
    else:
        accepted, reason = False, "not_improved"
    return MiniBenchResult(
        skill_id=selection.skill_id,
        task_count=len(selection.selected),
        previous_successes=previous_successes,
        candidate_successes=candidate_successes,
        previous_reward_sum=previous_sum,
        candidate_reward_sum=candidate_sum,
        complete=complete,
        accepted=accepted,
        reason=reason,
        evidence_sha256=canonical_sha256(
            [
                {
                    "domain": row.domain,
                    "task_id": row.task_id,
                    "branch": row.branch,
                    "reward": row.reward,
                    "complete": row.complete,
                    "episode_id": row.episode_id,
                    "shared_configuration_sha256": row.shared_configuration_sha256,
                    "manifest_sha256": row.manifest_sha256,
                    "access_receipt_sha256": row.access_receipt_sha256,
                    "trajectory_sha256": row.trajectory_sha256,
                    "evaluator_record_sha256": row.evaluator_record_sha256,
                }
                for row in results
            ]
        ),
    )
