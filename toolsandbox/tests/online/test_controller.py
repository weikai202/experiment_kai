from toolsandbox_pipeline.online.controller import Controller, action_fingerprint
from toolsandbox_pipeline.online.controller_inputs import ActionHistoryEntry, ControllerInput, ReproducibilityProfile, RetrievedSkillControllerView, SkillStatus
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.controller import BlockingCode, CriticTriggerCode
from toolsandbox_pipeline.schemas.state import AgentFacingTool, CompactVerifiedState, ControllerProvenanceSidecar, CurrentObservation, VerifiedFact, VisibleMessage, VisibleRole
from toolsandbox_pipeline.schemas.tool_metadata import ControllerToolMetadata, MetadataPredicate, PredicateOp, ToolEffect, ToolRisk


def make_state(schema=None):
    schema = schema or {"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":False}
    tool = AgentFacingTool(name="public", schema={"type":"function","function":{"name":"public","parameters":schema}})
    return CompactVerifiedState(episode_id="e",scenario_id="s",scenario_family_id="f",state_id="sha256:s",agent_turn_index=0,visible_messages=(VisibleMessage(message_id="m",sender=VisibleRole.USER,recipient=VisibleRole.AGENT,content="answer 1"),),current_observation=CurrentObservation(message_id="m",content="answer 1"),verified_facts={"/x":VerifiedFact(value=1,source_message_id="m0",call_id="old",result_pointer="/x",agent_facing_tool_name="old")},completed_tool_calls=(),failed_actions=(),pending_dependencies=(),available_tools=(tool,),conversation_status="agent_turn")


def metadata(effect=ToolEffect.SANDBOX_READ, risk=ToolRisk.LOW, **kwargs):
    return ControllerToolMetadata(canonical_tool_name="canonical",effect=effect,risk=risk,prerequisites=kwargs.pop("prerequisites",()),parallel_safe=kwargs.pop("parallel_safe",True),read_resources=kwargs.pop("read_resources",()),write_resources=kwargs.pop("write_resources",()),critic_required=kwargs.pop("critic_required",False),agent_forbidden=kwargs.pop("agent_forbidden",False))


def data(action, *, state=None, meta=None, mapping=None, skills=(), history=(), tension=False):
    state = state or make_state()
    return ControllerInput(state=state,provenance_sidecar=ControllerProvenanceSidecar(state_id=state.state_id,fact_provenance=(),dependency_provenance=()),action=ActionEnvelope.model_validate(action,strict=True),tool_metadata=(meta or metadata(),),agent_to_canonical_name=mapping or {"public":"canonical"},mapping_manifest_hash="sha256:m",retrieved_skills=skills,action_history=history,generation_id="g0",reproducibility_profile=ReproducibilityProfile.STRICT_REPLAY,structured_constraint_tension=tension)


def function(arguments, skill=None, name="public", call_id="c"):
    return {"action":{"type":"function_call","call_id":call_id,"selected_skill_id":skill,"name":name,"arguments":arguments}}


def test_single_valid_and_deterministic_immutable():
    value = data(function({"value":1}))
    before = value.model_dump_json()
    first = Controller().decide(value)
    assert first == Controller()(value)
    assert value.model_dump_json() == before
    assert first.blocking_codes == [] and first.critic_trigger_codes == []


def test_schema_grounding_dependency_history_and_skill_codes():
    prerequisite = MetadataPredicate(code="ready",path="/verified_facts/missing",op=PredicateOp.EXISTS)
    fp = action_fingerprint("canonical", {"value":2,"extra":"x"})
    result = Controller()(data(function({"value":2,"extra":"x"},skill="missing"),meta=metadata(prerequisites=(prerequisite,)),history=(ActionHistoryEntry(action_fingerprint=fp,failed=True,visible=True,source_ref="h"),)))
    assert result.blocking_codes == [BlockingCode.INVALID_PARAMETER,BlockingCode.UNGROUNDED_ARGUMENT,BlockingCode.MISSING_DEPENDENCY,BlockingCode.CONSTRAINT_VIOLATION,BlockingCode.REPEATED_FAILED_ACTION]
    assert {x.code for x in result.evidence} == set(result.blocking_codes)


def test_risk_external_and_assistant_triggers():
    result = Controller()(data(function({"value":1}),meta=metadata(effect=ToolEffect.EXTERNAL_READ,risk=ToolRisk.MEDIUM,critic_required=True)))
    assert result.critic_trigger_codes == [CriticTriggerCode.CRITIC_REQUIRED_TOOL,CriticTriggerCode.MEDIUM_OR_HIGH_RISK,CriticTriggerCode.EXTERNAL_READ_REVIEW]
    assistant = Controller()(data({"action":{"type":"assistant_message","content":"answer"}},tension=True))
    assert assistant.critic_trigger_codes == [CriticTriggerCode.ASSISTANT_MESSAGE_REVIEW,CriticTriggerCode.STRUCTURED_CONSTRAINT_TENSION]


def test_unknown_forbidden_external_write_and_parallel():
    unknown = Controller()(data(function({},name="missing"),mapping={"missing":"absent"}))
    assert BlockingCode.INVALID_FUNCTION in unknown.blocking_codes
    forbidden_meta = ControllerToolMetadata(canonical_tool_name="end_conversation",effect=ToolEffect.CONVERSATION_CONTROL,risk=ToolRisk.HIGH,prerequisites=(),parallel_safe=False,read_resources=(),write_resources=(),critic_required=True,agent_forbidden=True)
    forbidden = Controller()(data(function({},name="end_conversation"),meta=forbidden_meta,mapping={"end_conversation":"end_conversation"}))
    assert BlockingCode.AGENT_FORBIDDEN_TOOL in forbidden.blocking_codes
    external = metadata(effect=ToolEffect.EXTERNAL_WRITE,risk=ToolRisk.HIGH,parallel_safe=False,critic_required=True)
    assert BlockingCode.UNAUTHORIZED_EXTERNAL_SIDE_EFFECT in Controller()(data(function({"value":1}),meta=external)).blocking_codes
    batch={"action":{"type":"parallel_batch","calls":[{"call_id":"a","selected_skill_id":None,"name":"public","arguments":{"value":1}},{"call_id":"b","selected_skill_id":None,"name":"public","arguments":{"value":1}}]}}
    parallel=Controller()(data(batch,meta=metadata(parallel_safe=False)))
    assert BlockingCode.DEPENDENT_PARALLEL_CALLS in parallel.blocking_codes
    assert CriticTriggerCode.PARALLEL_BATCH_REVIEW in parallel.critic_trigger_codes


def test_active_skill_success_and_generation_failure():
    skill=RetrievedSkillControllerView(skill_id="sk",version="v1",status=SkillStatus.ACTIVE,generation_id="g0",tool_dependencies=("canonical",))
    assert BlockingCode.CONSTRAINT_VIOLATION not in Controller()(data(function({"value":1},skill="sk"),skills=(skill,))).blocking_codes
    stale=skill.model_copy(update={"generation_id":"old"})
    assert BlockingCode.CONSTRAINT_VIOLATION in Controller()(data(function({"value":1},skill="sk"),skills=(stale,))).blocking_codes
