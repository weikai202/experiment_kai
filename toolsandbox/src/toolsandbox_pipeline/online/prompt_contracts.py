"""Immutable prompt contexts store exact canonical envelopes, not mutable state."""
import json
from typing import Annotated, Literal
from pydantic import Field, model_validator, model_serializer
from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256, verify_state_id
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.critic import CriticOutput, CriticErrorCode
from toolsandbox_pipeline.schemas.memory import Digest, FrozenRecord, GenerationId, Identifier, Text
from toolsandbox_pipeline.providers.contracts import PhysicalAttemptResult
from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig


def visible_argument_locations(action):
    """Map only exact Controller argument refs to existing visible JSON Pointers.

    The Controller uses a doubled slash for a single call, including a one-call
    batch. Enumerating the action avoids interpreting arbitrary host references.
    """
    payload = action.model_dump(mode="json")["action"]
    if payload["type"] == "assistant_message":
        return {}
    calls = payload["calls"] if payload["type"] == "parallel_batch" else [payload]
    locations = {}

    def visit(value, host, pointer):
        locations[host] = pointer
        if isinstance(value, dict):
            for key, child in value.items():
                token = key.replace("~", "~0").replace("/", "~1")
                visit(child, host + "/" + token, pointer + "/" + token)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, host + f"/{index}", pointer + f"/{index}")

    for index, call in enumerate(calls):
        host = f"action:/calls/{index}" if len(calls) > 1 else "action:/"
        pointer = f"/action/calls/{index}" if payload["type"] == "parallel_batch" else "/action"
        visit(call["arguments"], host + "/arguments", pointer + "/arguments")
    return locations


class PromptSafeControllerEvidence(FrozenRecord):
    code: str
    source_kind: Literal["state", "schema", "skill", "tool_metadata", "action_history"]
    evidence_ref: Digest
    action_pointer: str | None = None
    reason: Literal["selected_skill_tool_mismatch"] | None = None

    @model_serializer(mode="wrap")
    def omit_absent_location(self, handler):
        # Preserve exact old serialized feedback and its hash on replay.
        data = handler(self)
        if self.action_pointer is None:
            data.pop("action_pointer", None)
        if self.reason is None:
            data.pop("reason", None)
        return data


class PromptSafeControllerDecision(FrozenRecord):
    blocking_codes: tuple[str, ...]
    critic_trigger_codes: tuple[str, ...]
    evidence: tuple[PromptSafeControllerEvidence, ...]

    @model_validator(mode="after")
    def validate_decision(self):
        ControllerDecision.model_validate_json(json.dumps(dict(
            blocking_codes=self.blocking_codes, critic_trigger_codes=self.critic_trigger_codes,
            evidence=[dict(code=e.code, source_kind=e.source_kind, source_ref=e.evidence_ref) for e in self.evidence])))
        return self


class PolicyMemoryPromptView(FrozenRecord):
    rank: Annotated[int, Field(ge=1, le=3)]
    scope: Text
    applicability: tuple[Text, ...]
    action_guidance: Text
    avoid: tuple[Text, ...]


class WorldMemoryPromptView(FrozenRecord):
    rank: Annotated[int, Field(ge=1, le=3)]
    action_pattern: Text
    state_conditions: tuple[Text, ...]
    schema_conditions: tuple[Text, ...]
    likely_error_codes: tuple[CriticErrorCode, ...]
    outcome_calibration: Text
    correction_principle: Text


def canonical_envelope(text, keys):
    data = json.loads(text)
    if type(data) is not dict or set(data) != set(keys) or canonical_json_bytes(data).decode() != text:
        raise ValueError("noncanonical or wrong role envelope")
    from toolsandbox_pipeline.schemas.state import CompactVerifiedState
    CompactVerifiedState.model_validate_json(json.dumps(data["state"]))
    if not verify_state_id(data["state"]):
        raise ValueError("state identity mismatch")
    return data


class InitialPolicyContext(FrozenRecord):
    state_id: Identifier
    generation_id: GenerationId
    prompt_sha256: Digest
    ordered_record_identities: tuple[tuple[str, str | None], ...]
    user_envelope: str
    policy_context_hash: Digest

    def identity_payload(self):
        return dict(state_id=self.state_id, generation_id=self.generation_id,
                    ordered_record_identities=[list(v) for v in self.ordered_record_identities],
                    prompt_sha256=self.prompt_sha256, user_envelope=json.loads(self.user_envelope))

    @model_validator(mode="after")
    def invariants(self):
        data = canonical_envelope(self.user_envelope, ("state", "policy_memory", "skills"))
        if data["state"]["state_id"] != self.state_id or canonical_sha256(self.identity_payload()) != self.policy_context_hash:
            raise ValueError("Initial Policy context identity mismatch")
        if len(set(self.ordered_record_identities)) != len(self.ordered_record_identities):
            raise ValueError("duplicate retrieved identities")
        from toolsandbox_pipeline.skills.views import RetrievedSkillPolicyView
        for key, model in (("policy_memory", PolicyMemoryPromptView), ("skills", RetrievedSkillPolicyView)):
            if len(data[key]) > 3 or [v["rank"] for v in data[key]] != list(range(1, len(data[key]) + 1)):
                raise ValueError("retrieval rank mismatch")
            for value in data[key]:
                model.model_validate_json(json.dumps(value))
        return self


class VisibleSkillBinding(FrozenRecord):
    skill_id: Identifier
    tool_dependencies: tuple[Identifier, ...]


def visible_skill_calls(action):
    payload = action.model_dump(mode="json")["action"]
    if payload["type"] == "assistant_message":
        return {}
    if payload["type"] == "function_call":
        return {"/action/selected_skill_id": payload}
    return {f"/action/calls/{index}/selected_skill_id": call
            for index, call in enumerate(payload["calls"])}


def mismatch_locations(action, skills):
    """Only exact retrieved Skill refs, proven visible tool mismatch, unique call."""
    from toolsandbox_pipeline.skills.views import RetrievedSkillPolicyView
    views = [RetrievedSkillPolicyView.model_validate_json(json.dumps(skill)) for skill in skills]
    if len({skill.skill_id for skill in views}) != len(views):
        raise ValueError("duplicate visible Skill identity")
    locations = {}
    for skill in views:
        bound = [(pointer, call) for pointer, call in visible_skill_calls(action).items()
                 if call["selected_skill_id"] == skill.skill_id]
        if len(bound) == 1 and skill.tool_dependencies and bound[0][1]["name"] not in skill.tool_dependencies:
            locations[f"skill:{skill.skill_id}@{skill.version}"] = bound[0][0]
    return locations


def validate_feedback_locations(action, feedback, skill_bindings=(), available_tools=None):
    visible_pointers = set(visible_argument_locations(action).values())
    calls = visible_skill_calls(action)
    bindings = [VisibleSkillBinding.model_validate_json(json.dumps(value)) for value in skill_bindings]
    by_id = {binding.skill_id: binding for binding in bindings}
    if len(by_id) != len(bindings):
        raise ValueError("duplicate visible Skill binding")
    used = set()
    for evidence in feedback.evidence:
        if evidence.reason is None:
            if evidence.action_pointer is not None and evidence.action_pointer not in visible_pointers:
                raise ValueError("Controller location is not a visible proposed argument")
            continue
        call = calls.get(evidence.action_pointer)
        binding = by_id.get(call["selected_skill_id"]) if call is not None else None
        if (evidence.code != "CONSTRAINT_VIOLATION" or evidence.source_kind != "skill"
                or binding is None or not binding.tool_dependencies
                or call["name"] in binding.tool_dependencies
                or (available_tools is not None and not set(binding.tool_dependencies) <= set(available_tools))):
            raise ValueError("unproven visible Skill tool mismatch")
        used.add(binding.skill_id)
    if used != set(by_id):
        raise ValueError("unused visible Skill binding")


def feedback_skill_bindings(action, feedback, skills):
    """Minimal source projection from the immutable Initial Policy skill views."""
    calls = visible_skill_calls(action)
    selected = {calls[e.action_pointer]["selected_skill_id"] for e in feedback.evidence
                if e.reason is not None and e.action_pointer in calls}
    bindings = [{"skill_id": skill["skill_id"], "tool_dependencies": skill["tool_dependencies"]}
                for skill in skills if skill["skill_id"] in selected]
    validate_feedback_locations(action, feedback, bindings)
    return bindings


class CriticContext(FrozenRecord):
    initial: InitialPolicyContext
    proposed_action_json: str
    proposed_action_sha256: Digest
    controller_feedback: PromptSafeControllerDecision
    controller_feedback_sha256: Digest
    world_memory: tuple[WorldMemoryPromptView, ...]

    @model_validator(mode="after")
    def invariants(self):
        action = ActionEnvelope.model_validate_json(self.proposed_action_json)
        if canonical_sha256(json.loads(action.model_dump_json())) != self.proposed_action_sha256:
            raise ValueError("proposed action identity mismatch")
        initial_data = json.loads(self.initial.user_envelope)
        bindings = feedback_skill_bindings(action, self.controller_feedback, initial_data["skills"])
        validate_feedback_locations(action, self.controller_feedback, bindings,
                                    [tool["name"] for tool in initial_data["state"]["available_tools"]])
        if canonical_sha256(self.controller_feedback.model_dump(mode="json")) != self.controller_feedback_sha256:
            raise ValueError("Controller projection identity mismatch")
        if not (self.controller_feedback.blocking_codes or self.controller_feedback.critic_trigger_codes):
            raise ValueError("Critic requires Controller feedback")
        if len(self.world_memory) > 3 or [v.rank for v in self.world_memory] != list(range(1, len(self.world_memory) + 1)):
            raise ValueError("World retrieval ranks mismatch")
        return self


class RevisionContext(FrozenRecord):
    initial: InitialPolicyContext
    critic: CriticContext
    critic_feedback_json: str

    @model_validator(mode="after")
    def same_initial(self):
        if self.initial != self.critic.initial:
            raise ValueError("Revision must reuse the original Initial context")
        CriticOutput.model_validate_json(self.critic_feedback_json)
        return self


class PromptMessage(FrozenRecord):
    role: Literal["system", "user"]
    content: Identifier


class PreparedRoleRequest(FrozenRecord):
    role: Literal["policy", "critic", "revision"]
    messages: tuple[PromptMessage, PromptMessage]
    output_model_name: Literal["ActionEnvelope", "CriticOutput"]
    output_schema_sha256: Digest
    max_tokens: Annotated[int, Field(gt=0)]
    token_limit_config_status: Literal["provisional", "calibration", "calibrated"]
    token_limit_config_sha256: Digest
    selected_limit: RoleTokenLimitConfig
    state_id: Identifier
    generation_id: GenerationId
    prompt_version: Literal["v1", "v2", "v3", "v4"]
    prompt_sha256: Digest
    user_envelope_sha256: Digest
    canonical_input_fingerprint: Digest

    @model_validator(mode="after")
    def invariants(self):
        from toolsandbox_pipeline.retrieval.index import file_hash
        if self.prompt_version == "v2" and self.role != "critic":
            raise ValueError("prompt v2 is restricted to Critic")
        if tuple(m.role for m in self.messages) != ("system", "user"):
            raise ValueError("exactly system/user messages required")
        if file_hash(self.messages[0].content.encode()) != self.prompt_sha256 or canonical_sha256(json.loads(self.messages[1].content)) != self.user_envelope_sha256:
            raise ValueError("prepared prompt/envelope identity mismatch")
        if self.selected_limit.max_tokens != self.max_tokens or self.selected_limit.role != self.role:
            raise ValueError("selected role limit mismatch")
        keys = {"policy": ("state", "policy_memory", "skills"),
                "critic": ("state", "proposed_action", "controller_feedback", "world_memory"),
                "revision": ("state", "policy_memory", "skills", "proposed_action", "controller_feedback", "critic_feedback")}[self.role]
        raw_envelope = json.loads(self.messages[1].content)
        if self.role != "policy" and "retrieved_skill_bindings" in raw_envelope:
            keys = (*keys, "retrieved_skill_bindings")
        envelope = canonical_envelope(self.messages[1].content, keys)
        if envelope["state"]["state_id"] != self.state_id:
            raise ValueError("prepared state identity mismatch")
        if "proposed_action" in envelope:
            action = ActionEnvelope.model_validate_json(json.dumps(envelope["proposed_action"]))
            feedback = PromptSafeControllerDecision.model_validate_json(json.dumps(envelope["controller_feedback"]))
            bindings = envelope.get("retrieved_skill_bindings", [])
            validate_feedback_locations(action, feedback, bindings,
                                        [tool["name"] for tool in envelope["state"]["available_tools"]])
            if "skills" in envelope and bindings != feedback_skill_bindings(action, feedback, envelope["skills"]):
                raise ValueError("Skill binding differs from Initial Policy skills")
        if "critic_feedback" in envelope:
            CriticOutput.model_validate_json(json.dumps(envelope["critic_feedback"]))
        from toolsandbox_pipeline.skills.views import RetrievedSkillPolicyView
        for key, model in (("policy_memory", PolicyMemoryPromptView), ("world_memory", WorldMemoryPromptView), ("skills", RetrievedSkillPolicyView)):
            if key not in envelope:
                continue
            records = envelope[key]
            if type(records) is not list or len(records) > 3 or [r.get("rank") for r in records] != list(range(1, len(records) + 1)):
                raise ValueError("prepared retrieval ranks mismatch")
            for record in records:
                model.model_validate_json(json.dumps(record))
        return self


class RoleCallResult(FrozenRecord):
    prepared_request: PreparedRoleRequest
    attempt: PhysicalAttemptResult
    output_json: str | None
    truncated: bool = False

    @property
    def output(self):
        if self.output_json is None:
            return None
        model = CriticOutput if self.prepared_request.role == "critic" else ActionEnvelope
        return model.model_validate_json(self.output_json)

    @model_validator(mode="after")
    def invariants(self):
        if self.truncated != (self.attempt.finish_reason == "length") or self.truncated != (self.output_json is None):
            raise ValueError("truncation/output mismatch")
        self.output
        return self
