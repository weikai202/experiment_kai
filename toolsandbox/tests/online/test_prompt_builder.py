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


def contexts(tmp_path, content="Please look up Alice"):
    snapshot = load(generation(tmp_path))
    _, _, kwargs = dependencies()
    prompts = load_prompts(ROOT)
    current = state(content)
    decision = ControllerDecision.model_validate_json(json.dumps(dict(blocking_codes=[], critic_trigger_codes=["MEDIUM_OR_HIGH_RISK"],
        evidence=[dict(code="MEDIUM_OR_HIGH_RISK", source_kind="tool_metadata", source_ref="search_contacts:private")])))
    with EmbeddingCache(tmp_path / "queries", EmbeddingIdentity(), 2) as cache:
        service = RetrievalService(snapshot, cache=cache, **kwargs)
        initial = initial_context(current, service.retrieve_policy_skills(current, canonical_to_agent=MAPPING), prompts[0])
        world = service.retrieve_world(current, action(), controller_decision=decision)
        critic = critic_context(initial, action(), decision, world)
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
