from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

from .dataset import Family
from .canonical import sha256_json


@dataclass(frozen=True)
class SkillCandidate:
    skill_id: str
    version: str
    canonical_tool_dependencies: tuple[str, ...]
    content_sha256: str


@dataclass(frozen=True)
class SharedMiniBenchConfig:
    model_id: str
    generation_id: str
    max_steps: int
    environment_sha256: str
    config_sha256: str

    @classmethod
    def build(cls, model_id: str, generation_id: str, max_steps: int, environment_sha256: str) -> "SharedMiniBenchConfig":
        core = {"model_id": model_id, "generation_id": generation_id, "max_steps": max_steps, "environment_sha256": environment_sha256}
        return cls(**core, config_sha256=sha256_json(core))


@dataclass(frozen=True)
class MiniBenchOutcome:
    family_id: str
    case_id: str
    current_valid: bool
    candidate_valid: bool
    current_episode_id: str
    candidate_episode_id: str
    shared_config_sha256: str
    current_trajectory_sha256: str
    candidate_trajectory_sha256: str
    current_complete: bool = True
    candidate_complete: bool = True


@dataclass(frozen=True)
class MiniBenchDecision:
    accepted: bool
    reason: str
    selected_family_ids: tuple[str, ...]
    selected_case_ids: tuple[str, ...]
    current_successes: int
    candidate_successes: int


def select_relevant_families(candidate: SkillCandidate, dev_families: Iterable[Family], seed: int = 0, limit: int = 5) -> tuple[Family, ...]:
    if limit != 5:
        raise ValueError("The frozen BFCL Mini-Bench limit is five families")
    dependencies = set(candidate.canonical_tool_dependencies)
    relevant = [family for family in dev_families if dependencies <= family.tool_dependencies]
    relevant.sort(key=lambda f: hashlib.sha256(f"{seed}\0{candidate.skill_id}\0{f.family_id}".encode()).digest())
    return tuple(relevant[:limit])


def decide(candidate: SkillCandidate, selected: tuple[Family, ...], outcomes: Iterable[MiniBenchOutcome], shared_config: SharedMiniBenchConfig) -> MiniBenchDecision:
    expected = tuple(case.case_id for family in selected for case in family.variants)
    rows = tuple(outcomes)
    if len(selected) > 5 or len(expected) > 20:
        raise ValueError("Mini-Bench exceeds the family-preserving limit")
    if tuple(row.case_id for row in rows) != expected:
        raise ValueError("Mini-Bench results must cover complete families in canonical order")
    expected_families = tuple(f.family_id for f in selected for _ in f.variants)
    if tuple(row.family_id for row in rows) != expected_families:
        raise ValueError("Mini-Bench family identities do not match selected cases")
    for row in rows:
        if not row.current_complete or not row.candidate_complete:
            raise ValueError("Both Mini-Bench branches must be complete")
        if row.current_episode_id == row.candidate_episode_id:
            raise ValueError("Paired branches require distinct episode IDs")
        if row.shared_config_sha256 != shared_config.config_sha256:
            raise ValueError("Paired branches do not share the frozen configuration")
        if len(row.current_trajectory_sha256) != 64 or len(row.candidate_trajectory_sha256) != 64:
            raise ValueError("Paired branches require trusted trajectory evidence")
    current = sum(row.current_valid for row in rows)
    proposed = sum(row.candidate_valid for row in rows)
    accepted = proposed > current
    return MiniBenchDecision(
        accepted=accepted,
        reason="more_officially_valid_cases" if accepted else "no_strict_official_accuracy_improvement",
        selected_family_ids=tuple(f.family_id for f in selected),
        selected_case_ids=expected,
        current_successes=current,
        candidate_successes=proposed,
    )
