"""Allowlisted one-trajectory projections and conservative literal guards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryCandidateDecision,
    MemoryCandidateNone,
    PolicyMemoryCandidate,
    PolicyTrajectoryProjection,
    WorldMemoryCandidate,
    WorldTrajectoryProjection,
)


class MemoryProjectionError(ValueError):
    pass


_FORBIDDEN_KEYS = frozenset(
    {
        "authorization",
        "canonical_name",
        "canonical_tool_ids",
        "database",
        "embedding",
        "endpoint",
        "environment",
        "evaluator_definition",
        "headers",
        "hidden_database",
        "mapping",
        "milestone_mapping",
        "minefield_mapping",
        "raw_response",
        "source_ref",
        "target_dataframe",
        "vector",
    }
)
_PUBLIC_VOCABULARY = frozenset(
    {
        "agent",
        "argument",
        "arguments",
        "assistant",
        "call",
        "dependency",
        "function",
        "result",
        "schema",
        "tool",
        "user",
    }
)


def _walk(value: object) -> Iterable[tuple[str | None, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield None, child
            yield from _walk(child)


def validate_projection_visibility(
    projection: PolicyTrajectoryProjection | WorldTrajectoryProjection,
) -> None:
    if type(projection) not in (PolicyTrajectoryProjection, WorldTrajectoryProjection):
        raise TypeError("strict Task 015 trajectory projection required")
    payload = projection.model_dump(mode="json")
    for key, _ in _walk(payload):
        if key is not None and key.casefold() in _FORBIDDEN_KEYS:
            raise MemoryProjectionError("forbidden field in offline projection")
    canonical_json_bytes(payload)


def projection_json(
    projection: PolicyTrajectoryProjection | WorldTrajectoryProjection,
) -> str:
    validate_projection_visibility(projection)
    return canonical_json_bytes(projection.model_dump(mode="json")).decode("utf-8")


def sensitive_literals(
    projection: PolicyTrajectoryProjection | WorldTrajectoryProjection,
) -> frozenset[str]:
    """Use only concrete values already present in the permitted projection."""

    validate_projection_visibility(projection)
    payload = projection.model_dump(mode="json")
    protected: set[str] = set()
    for key, value in _walk(payload):
        if not isinstance(value, str) or len(value.strip()) < 4:
            continue
        normalized = value.strip().casefold()
        if normalized in _PUBLIC_VOCABULARY:
            continue
        if key in {
            "content",
            "arguments",
            "result",
            "call_id",
            "openai_tool_call_id",
            "value",
        } or (key is None and len(normalized.split()) > 1):
            protected.add(normalized)
    return frozenset(protected)


def _candidate_strings(decision: MemoryCandidateDecision) -> tuple[str, ...]:
    if isinstance(decision, MemoryCandidateNone):
        return ()
    if isinstance(decision, PolicyMemoryCandidate):
        content = decision.candidate
        return (
            content.scope,
            *content.applicability,
            content.action_guidance,
            *content.avoid,
        )
    if isinstance(decision, WorldMemoryCandidate):
        content = decision.candidate
        return (
            content.action_pattern,
            *content.state_conditions,
            *content.schema_conditions,
            content.outcome_calibration,
            content.correction_principle,
        )
    raise TypeError("strict candidate decision required")


def reject_sensitive_candidate(
    decision: MemoryCandidateDecision,
    projection: PolicyTrajectoryProjection | WorldTrajectoryProjection,
) -> None:
    protected = sensitive_literals(projection)
    for text in _candidate_strings(decision):
        normalized = text.strip().casefold()
        if normalized in _PUBLIC_VOCABULARY:
            continue
        for literal in protected:
            if normalized == literal or (len(literal) >= 8 and literal in normalized):
                raise MemoryProjectionError("candidate preserves a protected literal")


__all__ = [
    "MemoryProjectionError",
    "projection_json",
    "reject_sensitive_candidate",
    "sensitive_literals",
    "validate_projection_visibility",
]
