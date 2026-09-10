"""Exact one-use-per-Skill/episode train statistics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from toolsandbox_pipeline.schemas.skill import SkillOnlineStatistics
from toolsandbox_pipeline.schemas.trajectory import SkillUseAttribution


@dataclass(frozen=True)
class ManifestSkillUse:
    manifest_position: int
    episode_id: str
    attribution: SkillUseAttribution

    def __post_init__(self) -> None:
        if type(self.manifest_position) is not int or self.manifest_position < 0:
            raise ValueError("nonnegative manifest position required")
        if type(self.episode_id) is not str or not self.episode_id:
            raise ValueError("episode identity required")
        if type(self.attribution) is not SkillUseAttribution:
            raise TypeError("strict SkillUseAttribution required")


def ordered_unique_uses(values: Iterable[ManifestSkillUse]) -> tuple[ManifestSkillUse, ...]:
    uses = tuple(values)
    expected = tuple(
        sorted(
            uses,
            key=lambda item: (
                item.manifest_position,
                item.attribution.skill_id.encode("utf-8"),
            ),
        )
    )
    if uses != expected:
        raise ValueError("Skill uses must follow manifest position then Skill ID")
    identities = tuple((item.episode_id, item.attribution.skill_id) for item in uses)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate Skill/episode evaluated use")
    return uses


def stage_skill_statistics(
    current: SkillOnlineStatistics,
    uses: Iterable[ManifestSkillUse],
    *,
    skill_id: str,
) -> SkillOnlineStatistics:
    ordered = ordered_unique_uses(uses)
    relevant = tuple(item for item in ordered if item.attribution.skill_id == skill_id)
    successes = current.successes + sum(item.attribution.fully_successful for item in relevant)
    failures = current.failures + sum(not item.attribution.fully_successful for item in relevant)
    evaluated = successes + failures
    return SkillOnlineStatistics(
        evaluated_uses=evaluated,
        successes=successes,
        failures=failures,
        success_rate=successes / evaluated if evaluated else 0.0,
        last_update_attempt_at_use_count=current.last_update_attempt_at_use_count,
    )


def rewrite_triggered(statistics: SkillOnlineStatistics) -> bool:
    return (
        statistics.evaluated_uses >= 10
        and statistics.failures * 4 > statistics.evaluated_uses
        and statistics.evaluated_uses > statistics.last_update_attempt_at_use_count
    )


def mark_rewrite_attempt(statistics: SkillOnlineStatistics) -> SkillOnlineStatistics:
    if not rewrite_triggered(statistics):
        raise ValueError("Skill rewrite is not triggered")
    return statistics.model_copy(
        update={"last_update_attempt_at_use_count": statistics.evaluated_uses}
    )


__all__ = [
    "ManifestSkillUse",
    "mark_rewrite_attempt",
    "ordered_unique_uses",
    "rewrite_triggered",
    "stage_skill_statistics",
]
