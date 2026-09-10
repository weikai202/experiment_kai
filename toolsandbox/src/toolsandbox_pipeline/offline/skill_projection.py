"""Train-only projections for failure updates and one-Skill rewrites."""

from __future__ import annotations

from collections.abc import Mapping

from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.offline_skill import SkillContent, SkillFailureEvidence
from toolsandbox_pipeline.schemas.skill import SkillFailureMode, SkillOnlineStatistics, SkillRecord


_FORBIDDEN_KEYS = frozenset(
    {
        "authorization",
        "dev",
        "endpoint",
        "evaluator_definition",
        "headers",
        "hidden_database",
        "milestone_mapping",
        "minefield_mapping",
        "raw_response",
        "target_dataframe",
        "test",
        "vector",
    }
)


def _audit(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS:
                raise ValueError("forbidden field in Skill update projection")
            _audit(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _audit(child)


def failure_update_projection(
    evidence: SkillFailureEvidence,
    buffer: tuple[SkillFailureMode, ...],
) -> dict[str, object]:
    projection = {
        "failure_evidence": {
            "evidence_kind": evidence.evidence_kind,
            "canonical_tool_dependencies": list(evidence.canonical_tool_dependencies),
            "sanitized_outcome_class": evidence.sanitized_outcome_class,
            "generalized_failure": evidence.generalized_failure,
        },
        "current_failure_modes": [item.model_dump(mode="json") for item in buffer],
    }
    _audit(projection)
    canonical_json_bytes(projection)
    return projection


def skill_rewrite_projection(
    *,
    skill: SkillRecord,
    statistics: SkillOnlineStatistics,
    failure_modes: tuple[SkillFailureMode, ...],
    public_tool_schemas: tuple[dict[str, object], ...],
    generalized_train_trajectories: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if skill.online_statistics != statistics or skill.failure_mode_buffer != failure_modes:
        raise ValueError("rewrite projection must use currently staged Skill state")
    if len(failure_modes) > 5:
        raise ValueError("failure mode capacity exceeded")
    projection = {
        "current_skill": SkillContent.from_record(skill).model_dump(mode="json"),
        "statistics": statistics.model_dump(mode="json"),
        "failure_modes": [item.model_dump(mode="json") for item in failure_modes],
        "public_tool_schemas": list(public_tool_schemas),
        "generalized_train_trajectories": list(generalized_train_trajectories),
    }
    _audit(projection)
    canonical_json_bytes(projection)
    return projection


__all__ = ["failure_update_projection", "skill_rewrite_projection"]
