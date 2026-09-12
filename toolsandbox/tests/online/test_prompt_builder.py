import json
from pathlib import Path
import pytest
from toolsandbox_pipeline.online.prompt_loader import load_prompts
from toolsandbox_pipeline.online.prompt_builder import initial_context, critic_context, prepare_request, safe_controller
from toolsandbox_pipeline.online.prompt_contracts import RevisionContext
from toolsandbox_pipeline.online.token_limits import ProvisionalTokenLimits
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
from toolsandbox_pipeline.retrieval import RetrievalService
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.schemas.runtime import QwenConfig
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from tests.memory.test_generation_store import generation, load
from tests.retrieval.test_embedding_cache import dependencies
from tests.retrieval.test_queries import state, action

ROOT = Path(__file__).resolve().parents[2]
MAPPING = {"search_contacts": "scrambled"}


def contexts(tmp_path, content="Please look up Alice", proposed=None, decision=None):
    snapshot = load(generation(tmp_path))
    _, _, kwargs = dependencies()
    prompts = load_prompts(ROOT)
    current = state(content)
    proposed = proposed if proposed is not None else action()
    decision = decision if decision is not None else ControllerDecision.model_validate_json(json.dumps(dict(blocking_codes=[], critic_trigger_codes=["MEDIUM_OR_HIGH_RISK"],
        evidence=[dict(code="MEDIUM_OR_HIGH_RISK", source_kind="tool_metadata", source_ref="search_contacts:private")])))
    with EmbeddingCache(tmp_path / "queries", EmbeddingIdentity(), 2) as cache:
        service = RetrievalService(snapshot, cache=cache, **kwargs)
        initial = initial_context(current, service.retrieve_policy_skills(current, canonical_to_agent=MAPPING), prompts[0])
        world = service.retrieve_world(current, proposed, controller_decision=decision)
        critic = critic_context(initial, proposed, decision, world)
    revision = RevisionContext(initial=initial, critic=critic, critic_feedback_json=json.dumps(dict(verdict="accept", predicted_outcome="success", predicted_effect="A visible result", error_codes=[], correction="")))
    return initial, critic, revision


def prepared(context, role):
    prompts = load_prompts(ROOT)
    raw = (ROOT / "configs/online_token_limits.provisional.json").read_bytes()
    return prepare_request(context, prompt=prompts[role], token_limits=ProvisionalTokenLimits.model_validate_json(raw),
        token_limit_config_sha256=file_hash(raw), qwen_config=QwenConfig(structured_output_wire_mode="guided_json"),
        canonical_to_agent=MAPPING)


def test_envelope_shapes_and_reuse(tmp_path):
    contexts_ = contexts(tmp_path)
    requests = [prepared(context, i) for i, context in enumerate(contexts_)]
    data = [json.loads(r.messages[1].content) for r in requests]
    assert set(data[0]) == {"state", "policy_memory", "skills"}
    assert set(data[1]) == {"state", "proposed_action", "controller_feedback", "world_memory"}
    assert set(data[2]) == {"state", "policy_memory", "skills", "proposed_action", "controller_feedback", "critic_feedback"}
    for key in ("state", "policy_memory", "skills"):
        assert data[0][key] == data[2][key]
    assert data[1]["controller_feedback"] == data[2]["controller_feedback"]
    assert [r.max_tokens for r in requests] == [256, 384, 256]
    assert all(len(r.messages) == 2 for r in requests)
    assert "search_contacts:private" not in requests[1].messages[1].content


def test_context_mutation_rejected(tmp_path):
    initial, critic, revision = contexts(tmp_path)
    with pytest.raises(ValueError):
        prepared(initial.model_copy(update={"generation_id": "g001"}), 0)
    with pytest.raises(ValueError):
        prepared(revision.model_copy(update={"initial": initial.model_copy(update={"state_id": "wrong"})}), 2)


def test_critic_and_revision_receive_same_grounding_location(tmp_path):
    from toolsandbox_pipeline.schemas.action import ActionEnvelope
    from toolsandbox_pipeline.online.prompt_contracts import CriticContext
    from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
    proposed = ActionEnvelope.model_validate(dict(action=dict(type="function_call", call_id="c", selected_skill_id=None,
        name="scrambled", arguments=dict(timestamp=123, days=-1))))
    decision = ControllerDecision.model_validate(dict(blocking_codes=["UNGROUNDED_ARGUMENT"], critic_trigger_codes=[],
        evidence=[dict(code="UNGROUNDED_ARGUMENT", source_kind="state", source_ref="action://arguments/days")]))
    initial, critic, _ = contexts(tmp_path, proposed=proposed, decision=decision)
    feedback = critic.controller_feedback
    revision = RevisionContext(initial=initial, critic=critic, critic_feedback_json=json.dumps(dict(
        verdict="revise", predicted_outcome="failure", predicted_effect="The days value is unsupported.",
        error_codes=["UNGROUNDED_ARGUMENT"], correction="Ground days in visible evidence.")))
    critic_data = json.loads(prepared(critic, 1).messages[1].content)
    revision_data = json.loads(prepared(revision, 2).messages[1].content)
    assert critic_data["controller_feedback"] == revision_data["controller_feedback"]
    assert critic_data["controller_feedback"]["evidence"][0]["action_pointer"] == "/action/arguments/days"
    assert critic_data["proposed_action"] == revision_data["proposed_action"] == proposed.model_dump(mode="json")
    # A forged but correctly hashed pointer must fail before request preparation.
    bad_feedback = feedback.model_copy(update={"evidence": (feedback.evidence[0].model_copy(update={"action_pointer": "/action/arguments/private_hidden"}),)})
    with pytest.raises(ValueError, match="visible proposed argument"):
        CriticContext(initial=initial, proposed_action_json=proposed.model_dump_json(),
            proposed_action_sha256=canonical_sha256(proposed.model_dump(mode="json")), controller_feedback=bad_feedback,
            controller_feedback_sha256=canonical_sha256(bad_feedback.model_dump(mode="json")), world_memory=())


@pytest.mark.parametrize("role", [1, 2])
@pytest.mark.parametrize("pointer", ["/action/arguments/missing", "/action/name", "/controller_sidecar/canonical_tool_name"])
def test_direct_prepared_request_rejects_forged_location(tmp_path, role, pointer):
    from toolsandbox_pipeline.online.prompt_contracts import PreparedRoleRequest
    from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
    request = prepared(contexts(tmp_path)[role], role)
    data = request.model_dump(mode="json")
    envelope = json.loads(data["messages"][1]["content"])
    envelope["controller_feedback"]["evidence"][0]["action_pointer"] = pointer
    # Repair the envelope hash: semantic validation must also hold on direct
    # construction / checkpoint restoration, independently of CriticContext.
    data["messages"][1]["content"] = canonical_json_bytes(envelope).decode()
    data["user_envelope_sha256"] = canonical_sha256(envelope)
    with pytest.raises(ValueError, match="visible proposed argument"):
        PreparedRoleRequest.model_validate_json(json.dumps(data))
    assert PreparedRoleRequest.model_validate_json(request.model_dump_json()) == request


def test_v3_version_propagation_and_legacy_version_acceptance(tmp_path):
    from toolsandbox_pipeline.online.prompt_loader import PromptEntry
    from toolsandbox_pipeline.online.prompt_contracts import PreparedRoleRequest
    prompts = load_prompts(ROOT)
    assert [p.entry.prompt_version for p in prompts] == ['v3', 'v3', 'v3']
    requests = [prepared(context, i) for i, context in enumerate(contexts(tmp_path))]
    assert [request.prompt_version for request in requests] == ['v3', 'v3', 'v3']
    legacy = requests[1].model_dump(mode='json')
    legacy['prompt_version'] = 'v1'
    assert PreparedRoleRequest.model_validate_json(json.dumps(legacy)).prompt_version == 'v1'
    for index in (0, 2):
        entry = prompts[index].entry.model_dump(mode='json')
        entry['prompt_version'] = 'v2'
        with pytest.raises(ValueError, match='restricted to Critic'):
            PromptEntry.model_validate_json(json.dumps(entry))
        request = requests[index].model_dump(mode='json')
        request['prompt_version'] = 'v2'
        with pytest.raises(ValueError, match='restricted to Critic'):
            PreparedRoleRequest.model_validate_json(json.dumps(request))
