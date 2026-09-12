"""Role-specific, canonical prompt assembly over validated visible data."""
import json
import re
from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256, verify_state_id
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.critic import CriticOutput
from toolsandbox_pipeline.schemas.state import CompactVerifiedState
from toolsandbox_pipeline.retrieval.service import PolicySkillRetrievalBundle, WorldRetrievalBundle
from .prompt_contracts import (InitialPolicyContext, CriticContext, RevisionContext, PolicyMemoryPromptView,
    WorldMemoryPromptView, PromptSafeControllerDecision, PromptSafeControllerEvidence, PromptMessage, PreparedRoleRequest, visible_argument_locations, mismatch_locations, feedback_skill_bindings)
from .token_limits import select_limit


def safe_controller(decision, proposed_action=None, *, retrieved_skills=()):
    if type(decision) is not ControllerDecision:
        raise TypeError("validated ControllerDecision required")
    decision = ControllerDecision.model_validate_json(decision.model_dump_json())
    if proposed_action is not None and type(proposed_action) is not ActionEnvelope:
        raise TypeError("validated ActionEnvelope required")
    if proposed_action is not None:
        proposed_action = ActionEnvelope.model_validate_json(proposed_action.model_dump_json())
    locations = visible_argument_locations(proposed_action) if proposed_action is not None else {}
    mismatches = mismatch_locations(proposed_action, retrieved_skills) if proposed_action is not None else {}
    return PromptSafeControllerDecision(
        blocking_codes=tuple(c.value for c in decision.blocking_codes),
        critic_trigger_codes=tuple(c.value for c in decision.critic_trigger_codes),
        evidence=tuple(PromptSafeControllerEvidence(code=e.code.value, source_kind=e.source_kind.value,
            evidence_ref=canonical_sha256(e.model_dump(mode="json")),
            action_pointer=(locations.get(e.source_ref) if e.source_kind.value in ("state", "schema") else
                            mismatches.get(e.source_ref) if e.code.value == "CONSTRAINT_VIOLATION" and e.source_kind.value == "skill" else None),
            reason="selected_skill_tool_mismatch" if (e.code.value == "CONSTRAINT_VIOLATION" and e.source_kind.value == "skill" and e.source_ref in mismatches) else None)
            for e in decision.evidence),
    )


def _memory_views(hits, records, generation, model, query_hash):
    from toolsandbox_pipeline.retrieval.queries import document, input_hash
    if len(hits) != len(records) or len(hits) > 3:
        raise ValueError("retrieval count mismatch")
    if len({r.memory_id for r in records}) != len(records):
        raise ValueError("duplicate retrieved records")
    output = []
    for index, (hit, record) in enumerate(zip(hits, records), 1):
        if hit.rank != index or hit.generation_id != generation or hit.record_id != record.memory_id or record.status != "active":
            raise ValueError("retrieval identity/rank mismatch")
        if hit.query_sha256 != query_hash or hit.document_sha256 != input_hash(document(record)):
            raise ValueError("retrieved query/document hash mismatch")
        data = record.model_dump(mode="json")
        output.append(model.model_validate_json(json.dumps({"rank": index, **{k: data[k] for k in model.model_fields if k != "rank"}})))
    return tuple(output)


def initial_context(state, retrieval, prompt):
    if type(state) is not CompactVerifiedState or type(retrieval) is not PolicySkillRetrievalBundle:
        raise TypeError("validated state and Policy retrieval bundle required")
    data = state.model_dump(mode="json", by_alias=True)
    if not verify_state_id(data) or retrieval.state_id != state.state_id or prompt.entry.role != "policy":
        raise ValueError("state/retrieval/prompt mismatch")
    from toolsandbox_pipeline.retrieval.queries import input_hash, policy_query
    query_hash = input_hash(policy_query(state))
    if retrieval.query.key.input_sha256 != query_hash:
        raise ValueError("retrieval query/state mismatch")
    memories = _memory_views(retrieval.policy_hits, retrieval.policy_memory, retrieval.generation_id, PolicyMemoryPromptView, query_hash)
    if len(retrieval.skill_hits) != len(retrieval.skills) or len(retrieval.skills) > 3:
        raise ValueError("Skill hit count mismatch")
    for i, (hit, skill) in enumerate(zip(retrieval.skill_hits, retrieval.skills), 1):
        if (hit.rank, skill.rank, hit.record_id, hit.record_version, hit.generation_id) != (i, i, skill.skill_id, skill.version, retrieval.generation_id):
            raise ValueError("Skill hit identity mismatch")
        semantic = skill.model_dump(mode="json", exclude={"rank", "skill_id", "version", "tool_dependencies"})
        if hit.query_sha256 != query_hash or hit.document_sha256 != input_hash(canonical_json_bytes(semantic).decode()):
            raise ValueError("Skill document/query identity mismatch")
    envelope = dict(state=data, policy_memory=[m.model_dump(mode="json") for m in memories], skills=[s.model_dump(mode="json") for s in retrieval.skills])
    identities = tuple((h.record_id, h.record_version) for h in (*retrieval.policy_hits, *retrieval.skill_hits))
    payload = dict(state_id=state.state_id, generation_id=retrieval.generation_id,
                   ordered_record_identities=[list(v) for v in identities], prompt_sha256=prompt.entry.sha256, user_envelope=envelope)
    return InitialPolicyContext(state_id=state.state_id, generation_id=retrieval.generation_id,
        prompt_sha256=prompt.entry.sha256, ordered_record_identities=identities,
        user_envelope=canonical_json_bytes(envelope).decode(), policy_context_hash=canonical_sha256(payload))


def critic_context(initial, action, decision, retrieval):
    if type(initial) is not InitialPolicyContext or type(action) is not ActionEnvelope or type(retrieval) is not WorldRetrievalBundle:
        raise TypeError("validated Initial context, action, and World bundle required")
    if (retrieval.state_id, retrieval.generation_id) != (initial.state_id, initial.generation_id):
        raise ValueError("World retrieval state/generation mismatch")
    from toolsandbox_pipeline.retrieval.queries import world_query, input_hash
    state = CompactVerifiedState.model_validate_json(json.dumps(json.loads(initial.user_envelope)["state"]))
    if retrieval.query.key.input_sha256 != input_hash(world_query(state, action)):
        raise ValueError("World query does not match proposed action")
    feedback = safe_controller(decision, action, retrieved_skills=json.loads(initial.user_envelope)["skills"])
    return CriticContext(initial=initial, proposed_action_json=action.model_dump_json(),
        proposed_action_sha256=canonical_sha256(json.loads(action.model_dump_json())), controller_feedback=feedback,
        controller_feedback_sha256=canonical_sha256(feedback.model_dump(mode="json")),
        world_memory=_memory_views(retrieval.hits, retrieval.world_memory, initial.generation_id, WorldMemoryPromptView, retrieval.query.key.input_sha256))


def audit_envelope(envelope, canonical_to_agent):
    state = envelope["state"]
    available = {tool["name"] for tool in state["available_tools"]}
    if set(canonical_to_agent.values()) != available or len(set(canonical_to_agent.values())) != len(canonical_to_agent):
        raise ValueError("complete bijective host mapping required")
    hidden_names = {k for k, v in canonical_to_agent.items() if k != v}
    for tool in state["available_tools"]:
        function = tool["schema"].get("function", {})
        if function.get("name") != tool["name"] or tool["name"] in hidden_names:
            raise ValueError("augmented schema tool identity mismatch")
    for skill in (*envelope.get("skills", []), *envelope.get("retrieved_skill_bindings", [])):
        if not set(skill["tool_dependencies"]) <= available:
            raise ValueError("unmapped Skill dependency")
    forbidden = {"source_ref", "sidecar", "controller_provenance_sidecar", "canonical_tool_name", "mapping",
                 "evaluator", "milestones", "minefields", "target_dataframe", "hidden_database", "vector",
                 "authorization", "headers", "api_key"}
    # Verbatim visible values and user-defined argument/schema property names are data.
    def structural(value):
        if isinstance(value, dict):
            if forbidden & value.keys():
                raise ValueError("forbidden prompt field")
            for key, child in value.items():
                if key not in ("content", "value", "arguments", "result", "schema", "properties"):
                    structural(child)
        elif isinstance(value, list):
            for child in value:
                structural(child)
    structural(envelope)
    for record in (*envelope.get("policy_memory", []), *envelope.get("world_memory", []), *envelope.get("skills", [])):
        text = canonical_json_bytes(record).decode()
        if any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text) for name in hidden_names):
            raise ValueError("canonical identifier in generated guidance")
    action = envelope.get("proposed_action", {}).get("action", {})
    calls = action.get("calls", [action] if action.get("type") == "function_call" else [])
    allowed_invalid = set(envelope.get("controller_feedback", {}).get("blocking_codes", [])) & {"INVALID_FUNCTION", "AGENT_FORBIDDEN_TOOL"}
    if any(call["name"] in hidden_names or (call["name"] not in available and not allowed_invalid) for call in calls):
        raise ValueError("non-agent-facing proposed action")


def fingerprint(role, messages, schema_hash, max_tokens, qwen):
    payload = dict(provider=qwen.provider, model=qwen.model, role=role,
        messages=[m.model_dump(mode="json") for m in messages], output_schema_sha256=schema_hash,
        max_tokens=max_tokens, temperature=0.0, seed=0, top_p="omitted", enable_thinking=False,
        structured_output_wire_mode=qwen.structured_output_wire_mode)
    if role == "critic" and qwen.critic_structured_output_mode != "json_schema":
        payload.update(critic_structured_output_mode=qwen.critic_structured_output_mode,
                       critic_grammar_sha256=qwen.critic_grammar_sha256)
    return canonical_sha256(payload)


def prepare_request(context, *, prompt, token_limits, token_limit_config_sha256, qwen_config,
                    canonical_to_agent, mode="offline", count_prompt_tokens=None, runtime_inputs_sha256=None):
    if mode == "formal":
        qwen_config.validate_external()
        if runtime_inputs_sha256 is None or getattr(token_limits, "runtime_inputs_sha256", None) != runtime_inputs_sha256:
            raise ValueError("formal calibration runtime bindings mismatch")
    if type(context) is InitialPolicyContext:
        role, initial = "policy", InitialPolicyContext.model_validate(context)
        envelope = json.loads(initial.user_envelope)
        if prompt.entry.sha256 != initial.prompt_sha256:
            raise ValueError("Initial prompt changed")
    elif type(context) is CriticContext:
        role, context = "critic", CriticContext.model_validate(context)
        initial = context.initial
        envelope = dict(state=json.loads(initial.user_envelope)["state"], proposed_action=json.loads(context.proposed_action_json),
            controller_feedback=context.controller_feedback.model_dump(mode="json"), world_memory=[m.model_dump(mode="json") for m in context.world_memory])
    elif type(context) is RevisionContext:
        role, context = "revision", RevisionContext.model_validate(context)
        initial = context.initial
        envelope = {**json.loads(initial.user_envelope), "proposed_action": json.loads(context.critic.proposed_action_json),
                    "controller_feedback": context.critic.controller_feedback.model_dump(mode="json"), "critic_feedback": json.loads(context.critic_feedback_json)}
    else:
        raise TypeError("strict role context required")
    if role in ("critic", "revision"):
        critic = context if role == "critic" else context.critic
        bindings = feedback_skill_bindings(ActionEnvelope.model_validate_json(critic.proposed_action_json),
                                          critic.controller_feedback, json.loads(initial.user_envelope)["skills"])
        if bindings:
            envelope["retrieved_skill_bindings"] = bindings
    if prompt.entry.role != role:
        raise ValueError("role/prompt mismatch")
    audit_envelope(envelope, canonical_to_agent)
    output = CriticOutput if role == "critic" else ActionEnvelope
    schema_hash = canonical_sha256(output.model_json_schema())
    selected = select_limit(token_limits, role, mode=mode, qwen_config_sha256=canonical_sha256(qwen_config.model_dump(mode="json")),
                            prompt_sha256=prompt.entry.sha256, schema_sha256=schema_hash)
    messages = (PromptMessage(role="system", content=prompt.text), PromptMessage(role="user", content=canonical_json_bytes(envelope).decode()))
    if qwen_config.context_limit is not None:
        if count_prompt_tokens is None:
            raise ValueError("manifest-pinned chat tokenizer required for context check")
        count = count_prompt_tokens(messages)
        if type(count) is not int or count < 0 or count + selected.max_tokens > qwen_config.context_limit:
            raise ValueError("Qwen context limit exceeded")
    return PreparedRoleRequest(role=role, messages=messages, output_model_name=output.__name__, output_schema_sha256=schema_hash,
        max_tokens=selected.max_tokens, token_limit_config_status=token_limits.status, token_limit_config_sha256=token_limit_config_sha256,
        selected_limit=selected, state_id=initial.state_id, generation_id=initial.generation_id,
        prompt_version=prompt.entry.prompt_version, prompt_sha256=prompt.entry.sha256, user_envelope_sha256=canonical_sha256(envelope),
        canonical_input_fingerprint=fingerprint(role, messages, schema_hash, selected.max_tokens, qwen_config))
