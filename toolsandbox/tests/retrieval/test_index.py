import pytest
from toolsandbox_pipeline.retrieval.index import cosine, rank, validate_indexes
from toolsandbox_pipeline.schemas.generation import RetrievalIndexEntry
from toolsandbox_pipeline.retrieval.contracts import EmbeddingCacheKey
from tests.memory.test_generation_store import generation, load


def test_cosine_and_ties():
    assert cosine((1.0, 0.0), (0.0, 1.0)) == 0.0
    assert cosine((1.0, 0.0), (-1.0, 0.0)) == -1.0
    entries = tuple(RetrievalIndexEntry(generation_id="g000", record_kind="skill", record_id=name, record_version="v1.0",
        document_sha256="sha256:" + "a" * 64, embedding_cache_key=EmbeddingCacheKey(input_sha256="sha256:" + "a" * 64),
        vector_dimension=2, vector=(1.0, 0.0)) for name in ("z", "a", "b", "c"))
    assert [e.record_id for e, _ in rank(entries, (1.0, 0.0))] == ["a", "b", "c"]
    assert rank((), (1.0, 0.0)) == ()


def test_missing_index_entry(tmp_path):
    snapshot = load(generation(tmp_path))
    with pytest.raises(ValueError):
        validate_indexes(snapshot.model_copy(update={"indexes": snapshot.indexes[:1]}))


def test_duplicate_records_rejected_before_embedding_or_writes(tmp_path):
    from toolsandbox_pipeline.retrieval.index import build_indexes
    from tests.memory.test_records import policy
    from tests.retrieval.test_embedding_cache import dependencies
    transport, _, kwargs = dependencies()
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(ValueError):
        build_indexes(staging, generation_id="g000", policy_memory=(policy(), policy()), world_memory=(), skills=(),
                      store_entries=(), tool_inventory=("search_contacts",), cache=None, **kwargs)
    assert not transport.calls and not (staging / "retrieval_indexes").exists()
