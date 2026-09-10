import json
import pytest
from toolsandbox_pipeline.online.prompt_builder import audit_envelope, prepare_request
from tests.online.test_prompt_builder import contexts, prepared, MAPPING


def test_verbatim_user_text_not_substring_filtered(tmp_path):
    initial, _, _ = contexts(tmp_path, content="I typed search_contacts, evaluator, and source_ref")
    request = prepared(initial, 0)
    assert "I typed search_contacts, evaluator, and source_ref" in request.messages[1].content


def test_generated_canonical_and_structural_fields_fail(tmp_path):
    initial, _, _ = contexts(tmp_path)
    envelope = json.loads(initial.user_envelope)
    envelope["skills"][0]["instruction"] = "Use search_contacts"
    with pytest.raises(ValueError):
        audit_envelope(envelope, MAPPING)
    envelope = json.loads(initial.user_envelope)
    envelope["evaluator"] = {}
    with pytest.raises(ValueError):
        audit_envelope(envelope, MAPPING)
