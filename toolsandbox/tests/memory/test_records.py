import json
import pytest
from pydantic import ValidationError
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory


def policy_data(**updates):
    return dict(memory_id="pm_" + "a" * 64, scope="Visible lookup", applicability=["Before use"],
                action_guidance="Verify the result", avoid=["Guessing"], evidence_trajectory_ids=["a", "b"],
                support_count=2, success_rate=0.5, confidence=0.5, created_version="g000", status="active", **updates)


def policy(**updates):
    data = policy_data()
    data.update(updates)
    return PolicyMemory.model_validate_json(json.dumps(data))


def test_roundtrip_frozen():
    record = policy()
    assert PolicyMemory.model_validate_json(record.model_dump_json()) == record
    with pytest.raises(ValidationError):
        record.scope = "changed"


@pytest.mark.parametrize("updates", [
    {"memory_id": "wm_" + "a" * 64}, {"scope": ""}, {"scope": "x" * 513},
    {"applicability": ["x"] * 2}, {"avoid": [str(i) for i in range(6)]},
    {"support_count": True}, {"support_count": "2"}, {"support_count": -1},
    {"success_rate": 0.3}, {"confidence": 0.8}, {"success_rate": float("nan")},
    {"evidence_trajectory_ids": ["b", "a"]}, {"evidence_trajectory_ids": ["a", "a"]},
    {"created_version": "latest"}, {"status": "draft"}, {"replacement_action": {}},
])
def test_invalid(updates):
    with pytest.raises(ValidationError):
        policy(**updates)


def test_world():
    data = dict(memory_id="wm_" + "a" * 64, action_pattern="Read", state_conditions=[], schema_conditions=[],
                likely_error_codes=["INSUFFICIENT_CONTEXT"], outcome_calibration="Uncertain", correction_principle="Verify",
                evidence_trajectory_ids=[], support_count=0, empirical_failure_rate=0.0, confidence=0.0,
                created_version="g000", status="active")
    assert WorldMemory.model_validate_json(json.dumps(data)).support_count == 0
    data["likely_error_codes"] = ["made_up"]
    with pytest.raises(ValidationError):
        WorldMemory.model_validate_json(json.dumps(data))
