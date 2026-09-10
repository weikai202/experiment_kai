from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.retrieval.contracts import EmbeddingCacheKey, ResolvedEmbedding
from toolsandbox_pipeline.schemas.generation import RetrievalIndexEntry
from toolsandbox_pipeline.schemas.memory import PolicyMemory
from toolsandbox_pipeline.schemas.offline_memory import PolicyMemoryCandidate


def digest(value):
    return "sha256:" + value * 64


def record(value):
    return PolicyMemory(
        memory_id="pm_" + value * 64,
        scope=f"scope {value}", applicability=(), action_guidance=f"guidance {value}", avoid=(),
        evidence_trajectory_ids=(f"trajectory-{value}",), support_count=1, success_rate=1.0,
        confidence=1 / 3, created_version="g000", status="active",
    )


class Resolver:
    def __init__(self):
        self.calls = 0

    def resolve_one(self, text):
        self.calls += 1
        return ResolvedEmbedding(
            key=EmbeddingCacheKey(input_sha256=canonical_sha256(__import__("json").loads(text))),
            vector=(1.0, 0.0), cache_hit=False, source_attempt_id="embedding-attempt",
        )


def test_role_local_top_three_and_utf8_identity_tie_breaking():
    records = tuple(record(value) for value in "dcba")
    indexes = tuple(
        RetrievalIndexEntry(
            generation_id="g000", record_kind="policy", record_id=item.memory_id,
            document_sha256=digest("f"),
            embedding_cache_key=EmbeddingCacheKey(input_sha256=digest("f")),
            vector_dimension=2, vector=(1.0, 0.0),
        ) for item in records
    )
    resolver = Resolver()
    retriever = MemoryCandidateRetriever(
        generation_id="g000", policy_records=records, world_records=(), indexes=indexes,
        embedding_resolver=resolver,
    )
    candidate = PolicyMemoryCandidate(
        result="CANDIDATE", role="policy",
        candidate={"scope": "general scope", "applicability": [], "action_guidance": "general guidance", "avoid": []},
    )
    result = retriever.retrieve(candidate)
    assert [item.memory_id for item in result.records] == sorted(item.memory_id for item in records)[:3]
    assert resolver.calls == 1


def test_staged_same_round_record_is_visible():
    resolver = Resolver()
    retriever = MemoryCandidateRetriever(
        generation_id="g000", policy_records=(), world_records=(), indexes=(), embedding_resolver=resolver
    )
    staged = record("a")
    retriever.remember_staged(staged, semantic_vector=(1.0, 0.0))
    candidate = PolicyMemoryCandidate(
        result="CANDIDATE", role="policy",
        candidate={"scope": "scope", "applicability": [], "action_guidance": "guidance", "avoid": []},
    )
    assert retriever.retrieve(candidate).records == (staged,)
