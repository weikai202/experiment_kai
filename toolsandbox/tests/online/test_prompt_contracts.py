import pytest
from toolsandbox_pipeline.online.prompt_contracts import PromptSafeControllerDecision
from toolsandbox_pipeline.online.prompt_builder import safe_controller
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256


def test_safe_evidence_hash():
    data = dict(code="INVALID_FUNCTION", source_kind="schema", source_ref="private_tool")
    decision = ControllerDecision.model_validate(dict(blocking_codes=["INVALID_FUNCTION"], critic_trigger_codes=[], evidence=[data]))
    safe = safe_controller(decision)
    assert safe.evidence[0].evidence_ref == canonical_sha256(data)
    assert "private_tool" not in safe.model_dump_json()
    assert "source_ref" not in safe.model_dump_json()
    with pytest.raises(ValueError):
        PromptSafeControllerDecision(blocking_codes=("INVALID_FUNCTION",), critic_trigger_codes=(), evidence=())
