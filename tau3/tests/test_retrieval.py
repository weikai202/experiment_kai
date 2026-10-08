import pytest

from tau3_evolution import EMBEDDING_MODEL
from tau3_evolution.canonical import atomic_write_json, canonical_sha256, read_json
from tau3_evolution.generation_store import (
    GenerationState,
    GenerationStore,
    MemoryRecord,
    SkillRecord,
)
from tau3_evolution.retrieval import (
    EmbeddingCache,
    EmbeddingResponse,
    StoredVector,
    rank_vectors,
)


class Provider:
    def __init__(self, vectors=None, model=EMBEDDING_MODEL):
        self.vectors = vectors or {}
        self.model = model
        self.calls = []
        self.receipts = {}

    def embed(self, text, *, model, logical_request_id):
        self.calls.append((text, model, logical_request_id))
        vector = self.vectors.get(text, (1.0, 0.0))
        response = EmbeddingResponse(
            self.model,
            vector,
            f"attempt:{logical_request_id}",
            2,
            0,
            canonical_sha256({"text": text, "model": self.model, "vector": vector}),
        )
        self.receipts[logical_request_id] = response
        return response

    def recover(self, *, logical_request_id):
        return self.receipts.get(logical_request_id)


def test_embedding_cache_is_exact_model_content_bound_and_has_no_fallback(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.json")
    provider = Provider()
    first, events, content_sha = cache.embed(
        "record", provider=provider, scope_id="scope", purpose="index"
    )
    second, cached_events, cached_sha = cache.embed(
        "record", provider=provider, scope_id="other", purpose="query"
    )
    assert first == second == (1.0, 0.0)
    assert content_sha == cached_sha == canonical_sha256("record")
    assert len(events) == 2 and cached_events == () and len(provider.calls) == 1
    assert provider.calls[0][1] == "text-embedding-3-small"

    with pytest.raises(ValueError, match="no fallback"):
        EmbeddingCache(tmp_path / "other.json").embed(
            "record", provider=Provider(model="fallback-model"), scope_id="scope", purpose="index"
        )


def test_cosine_ranking_has_deterministic_id_tie_break_and_strict_dimension():
    rows = (
        StoredVector("skill", "z", 1, "sha256:z", EMBEDDING_MODEL, 2, (1.0, 0.0)),
        StoredVector("skill", "a", 2, "sha256:a", EMBEDDING_MODEL, 2, (1.0, 0.0)),
        StoredVector("skill", "low", 1, "sha256:l", EMBEDDING_MODEL, 2, (0.0, 1.0)),
    )
    assert [row.record_id for row in rank_vectors((1.0, 0.0), rows)] == ["a", "z", "low"]
    with pytest.raises(ValueError, match="dimension"):
        rank_vectors(
            (1.0, 0.0, 0.0),
            (StoredVector("skill", "x", 1, "sha256:x", EMBEDDING_MODEL, 2, (1.0, 0.0)),),
        )


def test_generation_retrieval_keeps_policy_world_and_skill_proof_distinct(tmp_path):
    provider = Provider(
        {
            "policy text": (1.0, 0.0),
            "world text": (0.0, 1.0),
            "skill text": (0.8, 0.2),
            "query": (1.0, 0.0),
        }
    )
    store = GenerationStore(tmp_path / "generations")
    memories = (
        MemoryRecord("policy", 1, "policy", ("airline",), ("lookup",), "policy text"),
        MemoryRecord("world", 2, "world", ("airline",), ("lookup",), "world text"),
    )
    skills = (
        SkillRecord("skill", 3, ("airline",), ("lookup",), "skill text", "effect", "sha256:dev"),
    )
    vectors = []
    for kind, identity, version, content in (
        ("memory", "policy", 1, "policy text"),
        ("memory", "world", 2, "world text"),
        ("skill", "skill", 3, "skill text"),
    ):
        vector, _ = store.embed_record(
            record_kind=kind,
            record_id=identity,
            version=version,
            content=content,
            provider=provider,
            scope_id="bootstrap",
        )
        vectors.append(vector)
    store.publish(
        GenerationState("g000", None, memories, skills, tuple(vectors), "sha256:seed-library")
    )
    result = store.retrieve(
        "g000",
        domain="airline",
        tool_names=("lookup",),
        query="query",
        provider=provider,
        scope_id="round",
    )
    assert result.context.policy_memory_records == ("policy text",)
    assert result.context.world_memory_records == ("world text",)
    assert result.context.skill_records == ("skill text",)
    assert result.context.selected_policy_memory_ids == ("policy",)
    assert result.context.selected_world_memory_ids == ("world",)
    assert result.context.selected_skill_versions == (("skill", 3),)
    assert len(result.ledger_events) == 2


def test_multi_tool_record_is_not_retrieved_with_partial_tool_visibility(tmp_path):
    provider = Provider({"skill": (1.0, 0.0), "query": (1.0, 0.0)})
    store = GenerationStore(tmp_path / "generations")
    skill = SkillRecord(
        "requires-two", 1, ("airline",), ("lookup", "refund"), "skill", "effect", "sha256:dev"
    )
    vector, _ = store.embed_record(
        record_kind="skill",
        record_id=skill.skill_id,
        version=skill.version,
        content=skill.content,
        provider=provider,
        scope_id="bootstrap",
    )
    store.publish(GenerationState("g000", None, (), (skill,), (vector,), "sha256:seed-library"))
    result = store.retrieve(
        "g000",
        domain="airline",
        tool_names=("lookup",),
        query="query",
        provider=provider,
        scope_id="round",
    )
    assert result.context.skill_records == ()
    assert result.context.selected_skill_versions == ()


def test_generation_publish_and_load_reject_duplicate_identities(tmp_path):
    provider = Provider({"one": (1.0, 0.0)})
    store = GenerationStore(tmp_path / "generations")
    skill = SkillRecord("skill", 1, ("airline",), (), "one", "effect", "sha256:dev")
    vector, _ = store.embed_record(
        record_kind="skill",
        record_id="skill",
        version=1,
        content="one",
        provider=provider,
        scope_id="bootstrap",
    )
    with pytest.raises(ValueError, match="duplicate"):
        store.publish(
            GenerationState(
                "g000", None, (), (skill, skill), (vector, vector), "sha256:seed-library"
            )
        )

    store.publish(GenerationState("g000", None, (), (skill,), (vector,), "sha256:seed-library"))
    path = tmp_path / "generations" / "g000.json"
    document = read_json(path)
    document["skills"].append(document["skills"][0])
    document["vectors"].append(document["vectors"][0])
    base = {key: value for key, value in document.items() if key != "state_sha256"}
    document["state_sha256"] = canonical_sha256(base)
    atomic_write_json(path, document)
    with pytest.raises(ValueError, match="duplicate"):
        store.load("g000")
