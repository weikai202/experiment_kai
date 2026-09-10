import json
import pytest
from pydantic import ValidationError
from toolsandbox_pipeline.schemas.skill import SkillRecord, SkillStatePredicate, validate_skill_inventory
from toolsandbox_pipeline.skills.store import validate_skills


def skill(**updates):
    data = dict(skill_id="lookup", name="Lookup", description="Find a visible value",
                applicability=dict(required_state=[], forbidden_state=[], best_used_when=["Before a change"]),
                required_inputs=[], expected_outputs=["A result"], tool_dependencies=["search_contacts"],
                success_criteria=["Result verified"], failure_mode_buffer=[],
                cost_profile=dict(expected_tool_calls=1, latency="low", token_cost="low"),
                risk_profile=dict(risk_if_skipped="low", risk_if_wrong="medium"), instruction="Check the visible result",
                online_statistics=dict(evaluated_uses=0, successes=0, failures=0, success_rate=0.0, last_update_attempt_at_use_count=0),
                validation=None, version="v1.0", status="active")
    data.update(updates)
    return SkillRecord.model_validate_json(json.dumps(data))


def test_skill_roundtrip_and_inventory():
    record = skill()
    assert SkillRecord.model_validate_json(record.model_dump_json()) == record
    validate_skill_inventory(record, ("search_contacts",))
    with pytest.raises(ValueError):
        validate_skill_inventory(skill(instruction="Use search_contacts now"), ("search_contacts",))
    with pytest.raises(ValueError):
        validate_skill_inventory(record, ())


@pytest.mark.parametrize("value", [
    {"path": "bad", "op": "exists"}, {"path": "/bad~2", "op": "exists"},
    {"path": "/x", "op": "exists", "value": None}, {"path": "/x", "op": "eq"},
    {"path": "/x", "op": "in", "value": 1}, {"path": "/x", "op": "eq", "value": float("inf")},
])
def test_invalid_predicates(value):
    with pytest.raises(ValidationError):
        SkillStatePredicate.model_validate_json(json.dumps(value))


def test_predicate_deep_immutability_and_wire():
    p = SkillStatePredicate.model_validate_json('{"path":"/x","op":"in","value":[{"x":[1]}]}')
    with pytest.raises(TypeError):
        p.value[0]["x"].append(2)
    p = SkillStatePredicate.model_validate_json('{"path":"/x","op":"exists"}')
    assert "value" not in p.model_dump(mode="json")


@pytest.mark.parametrize("updates", [
    {"version": "v1.-1"}, {"version": "v1.01"}, {"expected_outputs": ["a", "a"]},
    {"tool_dependencies": ["z", "a"]}, {"name": ""}, {"native_score": 1},
    {"online_statistics": dict(evaluated_uses=1, successes=0, failures=0, success_rate=0.0, last_update_attempt_at_use_count=0)},
    {"validation": dict(scenario_count=1, previous_full_success_count=1, candidate_full_success_count=0,
                        previous_similarity_sum=0.0, candidate_similarity_sum=0.0, previous_minefield_hit_count=0, candidate_minefield_hit_count=0)},
])
def test_invalid_skill(updates):
    with pytest.raises(ValidationError):
        skill(**updates)


def test_active_version():
    old, new = skill(status="deprecated"), skill(version="v1.10")
    assert validate_skills((old, new), ("search_contacts",)) == (new,)
    for records in ((new, old), (new, new), (old,), (skill(), new)):
        with pytest.raises(ValueError):
            validate_skills(records, ("search_contacts",))
