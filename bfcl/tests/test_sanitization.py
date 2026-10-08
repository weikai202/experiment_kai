from __future__ import annotations

import pytest

from bfcl_pipeline.sanitization import sanitize_episode_evidence


def test_sanitizer_drops_restricted_checker_fields_and_raw_details():
    result = sanitize_episode_evidence(
        {"valid": False, "error_type": "state mismatch: secret entity", "possible_answer": "restricted", "trace": {"raw": "restricted"}},
        ["tool_exception"],
    )
    assert result.official_outcome_class == "state_mismatch"
    assert "secret" not in repr(result) and "restricted" not in repr(result)


def test_sanitizer_rejects_caller_defined_failure_classes():
    with pytest.raises(ValueError, match="host-defined"):
        sanitize_episode_evidence({"valid": False, "error_type": "x"}, ["raw exception text"])
