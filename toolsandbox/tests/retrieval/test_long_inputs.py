import json
import math
import sqlite3
import pytest
from toolsandbox_pipeline.retrieval.long_inputs import chunks, aggregate, STRATEGY
from toolsandbox_pipeline.providers.embedding_limits import validate_embedding_input
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
from tests.retrieval.test_embedding_cache import dependencies


@pytest.mark.parametrize('text', ['short input', '\x00'*8192, '汉字🙂'*5000, '<|endoftext|>'*3000])
def test_lossless_chunks_obey_provider_limit(text):
    parts, weights = chunks(text)
    assert ''.join(parts) == text
    assert len(parts) == len(weights)
    assert all(validate_embedding_input(p) == p for p in parts)
    assert all(w > 0 for w in weights)
    if text == 'short input':
        assert parts == (text,)


def test_weighted_aggregation_and_cancellation():
    value = aggregate(((1., 0.), (0., 1.)), (3, 1))
    assert value == pytest.approx((3/math.sqrt(10), 1/math.sqrt(10)))
    with pytest.raises(ValueError):
        aggregate(((1., 0.), (-1., 0.)), (1, 1))


def test_long_cache_preserves_native_response_provenance_and_reopens(tmp_path):
    transport, durable, kwargs = dependencies()
    path = tmp_path/'cache'
    long = '\x00'*8192
    with EmbeddingCache(path, EmbeddingIdentity(), 2) as cache:
        result = cache.resolve([long, 'short', long], **kwargs)
        assert len(transport.calls) == len(durable) == 1
        assert transport.calls[0]['input'] == ['\x00'*8191, '\x00', 'short']
        assert result[0] == result[2]
        assert result[0].vector == (1., 0.)
        assert result[0].source_attempt_id == durable[0].context.attempt_id
    with sqlite3.connect(path) as db:
        row = db.execute('select strategy,token_weights_json from derivations where input_sha256=?', (result[0].key.input_sha256,)).fetchone()
        assert row[0] == STRATEGY
        assert json.loads(row[1]) == [8191, 1]
    with EmbeddingCache(path, EmbeddingIdentity(), 2) as cache:
        assert cache.resolve([long], **kwargs)[0].cache_hit
        assert len(transport.calls) == 1


def test_reject_prior_cache_schema(tmp_path):
    path = tmp_path/'old'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=1')
    with pytest.raises(ValueError, match='unknown cache schema'):
        EmbeddingCache(path, EmbeddingIdentity(), 2)
