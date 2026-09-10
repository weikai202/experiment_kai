from toolsandbox_pipeline.online.grounding import TransformSpec, ground_arguments, load_grounding_transforms
from toolsandbox_pipeline.schemas.state import AgentFacingTool, CompactVerifiedState, CurrentObservation, VerifiedFact, VisibleMessage, VisibleRole


def state():
    return CompactVerifiedState(episode_id="e", scenario_id="s", scenario_family_id="f", state_id="sha256:x", agent_turn_index=0, visible_messages=(VisibleMessage(message_id="m1", sender=VisibleRole.USER, recipient=VisibleRole.AGENT, content="Send to Alice at +123"),), current_observation=CurrentObservation(message_id="m1", content="Send to Alice at +123"), verified_facts={"/result/id": VerifiedFact(value=7, source_message_id="m0", call_id="old", result_pointer="/id", agent_facing_tool_name="lookup")}, completed_tool_calls=(), failed_actions=(), pending_dependencies=(), available_tools=(), conversation_status="agent_turn")


def test_grounding_fact_span_schema_nested_and_strict_types():
    schema = {"type":"object","properties":{"id":{"type":"integer"},"name":{"type":"string"},"flag":{"const":True},"nested":{"type":"array","items":{"type":"object","properties":{"kind":{"enum":["x"]}}}}}}
    result = ground_arguments({"id":7,"name":"Alice","flag":True,"nested":[{"kind":"x"}]}, schema, state())
    assert not result.ungrounded_pointers
    span = next(x for x in result.evidence if x.argument_pointer == "/name" and x.source_kind.value == "user_message_span")
    assert state().visible_messages[0].content[span.start_offset:span.end_offset] == "Alice"
    assert ground_arguments({"id":True}, schema, state()).ungrounded_pointers == ("/id",)


def test_empty_containers_and_safe_injected_transform():
    schema = {"type":"object","properties":{"empty":{"type":"array"},"value":{"type":"string"}}}
    result = ground_arguments({"empty":[],"value":"SEND TO ALICE AT +123"}, schema, state(), transforms=(TransformSpec(name="upper", version="1"),), transform_registry={("upper","1"): lambda x: x.upper() if isinstance(x,str) else x})
    assert not result.ungrounded_pointers
    assert any(x.transform_name == "upper" for x in result.evidence)


def test_removed_schema_information_is_not_recovered():
    result = ground_arguments({"value":"not visible"}, {"type":"object","properties":{"value":{}}}, state())
    assert result.ungrounded_pointers == ("/value",)


def test_production_transform_registry_starts_empty():
    assert load_grounding_transforms() == ()
