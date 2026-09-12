import json
import sqlite3
import pytest
from toolsandbox_pipeline.providers.embedding import EmbeddingGateway
from toolsandbox_pipeline.providers.contracts import RequestContext, ProviderRole, TransportResponse
from toolsandbox_pipeline.schemas.runtime import EmbeddingConfig
from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity, EmbeddingBatchConfig
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache


class Transport:
    def __init__(self):
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        data = dict(model="text-embedding-3-small", object="list",
                    data=[dict(object="embedding", index=i, embedding=[1.0, 0.0]) for i, _ in enumerate(request["input"])],
                    usage=dict(prompt_tokens=len(request["input"]), total_tokens=len(request["input"])))
        return TransportResponse(raw_body=json.dumps(data).encode(), data=data)


def dependencies():
    transport = Transport()
    gateway = EmbeddingGateway(EmbeddingConfig(expected_dimension=2), transport=transport)
    contexts, durable = [], []
    def context_factory(index, inputs):
        n = len(contexts)
        context = RequestContext(logical_request_id=f"logical-{n}", attempt_id=f"attempt-{n}", role=ProviderRole.EMBEDDING,
                                 phase="unit", unit_reference="synthetic", input_fingerprint=canonical_sha256(list(inputs)),
                                 replayed_after_unknown_outcome=False, manifest_identity="sha256:" + "0" * 64)
        contexts.append(context)
        return context
    def record(response):
        durable.append(response.attempt)
        return True
    return transport, durable, dict(gateway=gateway, context_factory=context_factory, record_durable=record)


def test_hit_miss_order_usage_and_reopen(tmp_path):
    transport, durable, kwargs = dependencies()
    path = tmp_path / "cache.sqlite"
    with EmbeddingCache(path, EmbeddingIdentity(), 2) as cache:
        results = cache.resolve(["a", "b", "a"], **kwargs)
        assert len(transport.calls) == len(durable) == 1
        assert transport.calls[0]["input"] == ["a", "b"]
        assert results[0] == results[2]
        assert durable[0].metrics.usage.total_tokens == 2
        mixed = cache.resolve(["b", "c", "a"], **kwargs)
        assert [r.cache_hit for r in mixed] == [True, False, True]
    with EmbeddingCache(path, EmbeddingIdentity(), 2) as cache:
        assert all(r.cache_hit for r in cache.resolve(["a", "b", "c"], **kwargs))
        assert len(durable) == 2
    assert b"Please" not in path.read_bytes()


def test_byte_and_item_batch_boundaries(tmp_path):
    transport, _, kwargs = dependencies()
    with EmbeddingCache(tmp_path / "cache", EmbeddingIdentity(), 2) as cache:
        texts = [str(i).zfill(4) + "x" * 7996 for i in range(36)]
        cache.resolve(texts, **kwargs)
        assert [len(r["input"]) for r in transport.calls] == [35, 1]
        before = len(transport.calls)
        with pytest.raises(ValueError):
            cache.resolve(["x" * 280001], **kwargs)
        assert len(transport.calls) == before
    transport, _, kwargs = dependencies()
    with EmbeddingCache(tmp_path / "count", EmbeddingIdentity(), 2) as cache:
        cache.resolve([str(i) for i in range(2049)], **kwargs)
        assert [len(r["input"]) for r in transport.calls] == [2048, 1]


def test_ledger_ack_and_transaction_rollback(tmp_path):
    _, _, kwargs = dependencies()
    path = tmp_path / "cache"
    with EmbeddingCache(path, EmbeddingIdentity(), 2) as cache:
        with pytest.raises(ValueError):
            cache.resolve(["a"], **{**kwargs, "record_durable": lambda _: False})
        with sqlite3.connect(path) as db:
            assert db.execute("select count(*) from vectors").fetchone()[0] == 0
            db.execute("CREATE TRIGGER fail_second BEFORE INSERT ON vectors WHEN (SELECT count(*) FROM vectors)=1 BEGIN SELECT RAISE(ABORT, 'fixture'); END")
        with pytest.raises(sqlite3.IntegrityError):
            cache.resolve(["a", "b"], **kwargs)
        with sqlite3.connect(path) as db:
            assert db.execute("select count(*) from provenance").fetchone()[0] == 0
            assert db.execute("select count(*) from vectors").fetchone()[0] == 0


@pytest.mark.parametrize("payload", ['[0.0,0.0]', '[1.0]', '[NaN,0.0]', '[0.0,1.0]'])
def test_corrupt_vector(tmp_path, payload):
    _, _, kwargs = dependencies()
    path = tmp_path / "cache"
    with EmbeddingCache(path, EmbeddingIdentity(), 2) as cache:
        cache.resolve(["a"], **kwargs)
        with sqlite3.connect(path) as db:
            db.execute("update vectors set vector_json=?", (payload,))
        with pytest.raises(ValueError):
            cache.resolve(["a"], **kwargs)


def test_unknown_schema_unchanged(tmp_path):
    path = tmp_path / "cache"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        EmbeddingCache(path, EmbeddingIdentity(), 2)
    assert path.read_bytes() == before
