import pytest
from toolsandbox_pipeline.skills.views import skill_views
from tests.skills.test_records import skill
from tests.retrieval.test_queries import state


def test_scrambled_policy_and_controller():
    policy, controller = skill_views(skill(), "g000", state(), {"search_contacts": "scrambled"})
    assert policy.tool_dependencies == ("scrambled",)
    assert controller.tool_dependencies == ("search_contacts",)
    assert "search_contacts" not in policy.model_dump_json()
    assert skill_views(skill(), "g000", state(), {}) is None


def test_strict_predicate_prefilter():
    for value, expected in (("agent_turn", True), (True, False), (1, False)):
        record = skill(required_inputs=[{"path": "/conversation_status", "op": "eq", "value": value}])
        result = skill_views(record, "g000", state(), {"search_contacts": "scrambled"})
        assert (result is not None) == expected
        if result:
            assert result[1].required_inputs[0].code.startswith("sha256:")
