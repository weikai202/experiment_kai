import json
import pytest
from toolsandbox_pipeline.online.state_builder import StateBuilder
from toolsandbox_pipeline.schemas.state import StateBuildInput, VisibleMessageInput, VisibleRole, AgentFacingToolInput
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.retrieval.queries import policy_query, world_query, document, validate_input
from tests.memory.test_records import policy
from tests.skills.test_records import skill


def state(content="Please look up Alice"):
    data = StateBuildInput(episode_id="episode-secret", scenario_id="scenario-secret", scenario_family_id="family-secret",
        visible_messages=(VisibleMessageInput(source_message_index=0, sender=VisibleRole.USER, recipient=VisibleRole.AGENT, content=content),),
        available_tools=(AgentFacingToolInput(name="scrambled", schema={"type": "function", "function": {"name": "scrambled", "parameters": {"type": "object"}}}),))
    return StateBuilder({}).build(data).state


def action():
    return ActionEnvelope.model_validate_json('{"action":{"type":"function_call","call_id":"call-secret","selected_skill_id":"skill-secret","name":"scrambled","arguments":{"name":"Alice"}}}')


def test_queries():
    query = policy_query(state())
    data = json.loads(query)
    assert data["visible_messages"] == [{"sender": "USER", "recipient": "AGENT", "content": "Please look up Alice"}]
    assert data["available_tool_names"] == ["scrambled"]
    for excluded in ("episode-secret", "scenario-secret", "family-secret", "message_id", "parameters", "state_id"):
        assert excluded not in query
    world = world_query(state(), action())
    assert json.loads(world)["action"] == {"type": "function_call", "name": "scrambled", "arguments": {"name": "Alice"}}
    assert "call-secret" not in world and "skill-secret" not in world
    with pytest.raises(TypeError):
        policy_query({"hidden": "database"})


def test_documents():
    assert set(json.loads(document(policy()))) == {"scope", "applicability", "action_guidance", "avoid"}
    text = document(skill())
    for excluded in ("search_contacts", "online_statistics", "failure_mode_buffer", "validation", "version", "skill_id"):
        assert excluded not in text


def test_byte_limits():
    assert len(validate_input("\u00e9" * 4000).encode()) == 8000
    with pytest.raises(ValueError):
        validate_input("\u00e9" * 4000 + "a")
    with pytest.raises(ValueError):
        policy_query(state("x" * 8000))


def test_nested_projection_rejects_sidecars_and_preserves_batch_order():
    from toolsandbox_pipeline.retrieval.queries import state_projection, RetrievalStateProjectionV1
    data = state_projection(state()).model_dump(mode="json")
    data["visible_messages"][0]["canonical_tool_name"] = "private"
    with pytest.raises(ValueError):
        RetrievalStateProjectionV1.model_validate_json(json.dumps(data))
    batch = ActionEnvelope.model_validate_json('{"action":{"type":"parallel_batch","calls":[{"call_id":"first","selected_skill_id":null,"name":"a","arguments":{}},{"call_id":"second","selected_skill_id":null,"name":"b","arguments":{}}]}}')
    assert [c["name"] for c in json.loads(world_query(state(), batch))["action"]["calls"]] == ["a", "b"]
