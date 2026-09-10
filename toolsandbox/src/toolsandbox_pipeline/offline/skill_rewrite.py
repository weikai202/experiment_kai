"""Host validation and accepted/rejected Skill staging."""

from __future__ import annotations

import re

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.offline_skill import (
    DevMiniBenchResult,
    SkillContent,
    StagedSkillMutation,
)
from toolsandbox_pipeline.schemas.skill import (
    SkillOnlineStatistics,
    SkillRecord,
)


def validate_skill_candidate(
    *,
    current: SkillRecord,
    candidate: SkillContent,
    public_tool_inventory: tuple[str, ...],
    protected_literals: tuple[str, ...] = (),
) -> None:
    if candidate.skill_id != current.skill_id:
        raise ValueError("candidate changed skill_id")
    if not set(candidate.tool_dependencies) <= set(public_tool_inventory):
        raise ValueError("candidate introduced unknown dependency")
    if not set(candidate.tool_dependencies) <= set(current.tool_dependencies):
        raise ValueError("candidate expanded dependency scope")
    if candidate == SkillContent.from_record(current):
        raise ValueError("candidate is a semantic no-op")
    prose = (
        candidate.name,
        candidate.description,
        candidate.instruction,
        *candidate.applicability.best_used_when,
        *candidate.expected_outputs,
        *candidate.success_criteria,
    )
    for text in prose:
        if any(
            re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text)
            for name in public_tool_inventory
        ):
            raise ValueError("canonical tool identifier in candidate prose")
        normalized = text.casefold()
        if any(len(value.strip()) >= 4 and value.strip().casefold() in normalized for value in protected_literals):
            raise ValueError("candidate contains protected scenario literal")


def next_skill_version(version: str) -> str:
    match = re.fullmatch(r"v1\.(0|[1-9][0-9]*)", version)
    if match is None:
        raise ValueError("unsupported Skill version")
    return f"v1.{int(match.group(1)) + 1}"


def stage_skill_rewrite(
    *,
    current: SkillRecord,
    staged_statistics: SkillOnlineStatistics,
    staged_failure_modes,
    candidate: SkillContent,
    mini_bench: DevMiniBenchResult,
    accepted_effect_id: str | None,
) -> StagedSkillMutation:
    validation_hash = canonical_sha256(mini_bench.model_dump(mode="json"))
    if mini_bench.skill_id != current.skill_id:
        raise ValueError("Mini-Bench Skill identity mismatch")
    if mini_bench.accepted:
        if not accepted_effect_id:
            raise ValueError("accepted Skill requires atomic effect identity")
        validation = {
            name: getattr(mini_bench, name)
            for name in (
                "scenario_count",
                "previous_full_success_count",
                "candidate_full_success_count",
                "previous_similarity_sum",
                "candidate_similarity_sum",
                "previous_minefield_hit_count",
                "candidate_minefield_hit_count",
            )
        }
        deprecated = current.model_copy(update={"status": "deprecated"})
        active = SkillRecord(
            **candidate.model_dump(mode="python"),
            failure_mode_buffer=(),
            online_statistics=SkillOnlineStatistics(
                evaluated_uses=0,
                successes=0,
                failures=0,
                success_rate=0.0,
                last_update_attempt_at_use_count=0,
            ),
            validation=validation,
            version=next_skill_version(current.version),
            status="active",
        )
        staged_records = (deprecated, active)
    else:
        if accepted_effect_id is not None:
            raise ValueError("rejected candidate cannot receive cost effect")
        staged_current = current.model_copy(
            update={
                "failure_mode_buffer": staged_failure_modes,
                "online_statistics": staged_statistics,
            }
        )
        staged_records = (staged_current,)
    return StagedSkillMutation(
        skill_id=current.skill_id,
        previous_version=current.version,
        accepted=mini_bench.accepted,
        previous_record=current,
        staged_records=staged_records,
        failure_buffer_sha256=canonical_sha256(
            [item.model_dump(mode="json") for item in staged_failure_modes]
        ),
        statistics_sha256=canonical_sha256(staged_statistics.model_dump(mode="json")),
        mini_bench_result=mini_bench,
        mini_bench_result_sha256=validation_hash,
        accepted_effect_id=accepted_effect_id,
    )


__all__ = [
    "next_skill_version",
    "stage_skill_rewrite",
    "validate_skill_candidate",
]
