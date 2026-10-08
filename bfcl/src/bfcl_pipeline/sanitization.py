from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SanitizedEpisodeEvidence:
    official_valid: bool
    official_outcome_class: str
    visible_execution_failure_classes: tuple[str, ...]


def _official_class(valid: bool, error_type: Any) -> str:
    if valid:
        return "valid"
    value = str(error_type or "unknown").lower()
    if "turn" in value and ("count" in value or "number" in value):
        return "turn_count_mismatch"
    if "irrelev" in value or "miss" in value and ("func" in value or "param" in value):
        return "irrelevance_failure"
    if "state" in value:
        return "state_mismatch"
    if "response" in value:
        return "response_mismatch"
    return "official_checker_failure"


def sanitize_episode_evidence(official_result: dict[str, Any], visible_execution_failure_classes: Iterable[str]) -> SanitizedEpisodeEvidence:
    """Project the only evaluator fields allowed to offline update prompts.

    Callers provide already host-classified visible execution failures. Raw
    observations, exception strings, possible answers, and checker details are
    intentionally not represented in the return type.
    """
    valid = official_result.get("valid")
    if not isinstance(valid, bool):
        raise ValueError("Official result must provide boolean valid")
    allowed_visible = {"literal_parse_error", "tool_exception", "unknown_function", "argument_error"}
    visible = tuple(sorted(set(visible_execution_failure_classes)))
    if not set(visible) <= allowed_visible:
        raise ValueError("Visible execution failure class is not host-defined")
    return SanitizedEpisodeEvidence(valid, _official_class(valid, official_result.get("error_type")), visible)
