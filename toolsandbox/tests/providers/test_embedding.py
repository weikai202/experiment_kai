import pytest
from toolsandbox_pipeline.providers import EmbeddingGateway
from toolsandbox_pipeline.providers.contracts import ProviderRole, ProviderRequestError
from toolsandbox_pipeline.schemas.runtime import EmbeddingConfig
from tests.providers.test_request_identity import context, FakeTransport


def embedding_response(count=1):
    return dict(model="text-embedding-3-small", object="list",
                data=[dict(object="embedding", index=i, embedding=[0.1, 0.2]) for i in range(count)],
                usage=dict(prompt_tokens=4, total_tokens=4))


@pytest.mark.parametrize("inputs", ["one", ["one"], ["one", "two"]])
def test_valid_embeddings(inputs):
    transport = FakeTransport(embedding_response(len(inputs) if isinstance(inputs, list) else 1))
    gateway = EmbeddingGateway(EmbeddingConfig(), transport=transport)
    result = gateway.embed(context(ProviderRole.EMBEDDING), inputs)
    assert transport.calls == [dict(model="text-embedding-3-small", input=inputs, encoding_format="float")]
    assert gateway.dimension == 2
    assert all(v == (0.1, 0.2) for v in result.value)
    assert result.attempt.metrics.usage.input_tokens == result.attempt.metrics.usage.total_tokens == 4
    assert result.attempt.metrics.usage.output_tokens == 0


@pytest.mark.parametrize("inputs", ["", [], [""], [1], 1, None, ("one",), ["x"] * 2049,
                                     "x" * 8001, "\u6c49" * 2667, ["x" * 8000] * 36])
def test_invalid_inputs_before_dispatch(inputs):
    transport = FakeTransport()
    with pytest.raises(ProviderRequestError) as caught:
        EmbeddingGateway(EmbeddingConfig(), transport=transport).embed(context(ProviderRole.EMBEDDING), inputs)
    assert not transport.calls
    assert caught.value.attempt.status.value == "rejected_before_dispatch"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(model="other"),
    lambda r: r.update(object="embedding"),
    lambda r: r.update(data=[]),
    lambda r: r["data"][0].update(index=1),
    lambda r: r["data"][0].update(index=True),
    lambda r: r["data"][0].update(object="other"),
    lambda r: r["data"][0].pop("embedding"),
    lambda r: r["data"][0].update(embedding=[]),
    lambda r: r["data"][0].update(embedding=[float("nan")]),
    lambda r: r["data"][0].update(embedding=[float("inf")]),
    lambda r: r["data"][0].update(embedding=[float("-inf")]),
    lambda r: r["data"][0].update(embedding=["0.1"]),
    lambda r: r["data"][0].update(embedding=[True]),
    lambda r: r["usage"].update(total_tokens=5),
])
def test_invalid_response_hard_failure(mutation):
    raw = embedding_response()
    mutation(raw)
    transport = FakeTransport(raw)
    gateway = EmbeddingGateway(EmbeddingConfig(), transport=transport)
    with pytest.raises(ProviderRequestError):
        gateway.embed(context(ProviderRole.EMBEDDING), "input")
    assert len(transport.calls) == 1
    assert gateway.dimension is None


@pytest.mark.parametrize("indices", [[1, 0], [0, 0], [0, 2]])
def test_bad_batch_indices(indices):
    raw = embedding_response(2)
    for item, index in zip(raw["data"], indices):
        item["index"] = index
    with pytest.raises(ProviderRequestError):
        EmbeddingGateway(EmbeddingConfig(), transport=FakeTransport(raw)).embed(context(ProviderRole.EMBEDDING), ["a", "b"])


def test_dimension_drift_and_no_partial_commit():
    raw = embedding_response(2)
    raw["data"][1]["embedding"] = [0.1]
    gateway = EmbeddingGateway(EmbeddingConfig(), transport=FakeTransport(raw))
    with pytest.raises(ProviderRequestError):
        gateway.embed(context(ProviderRole.EMBEDDING), ["a", "b"])
    assert gateway.dimension is None
    transport = FakeTransport(embedding_response())
    gateway = EmbeddingGateway(EmbeddingConfig(), transport=transport)
    gateway.embed(context(ProviderRole.EMBEDDING), "a")
    transport.payload["data"][0]["embedding"].append(0.3)
    with pytest.raises(ProviderRequestError):
        gateway.embed(context(ProviderRole.EMBEDDING, "attempt-2"), "b")
    assert gateway.dimension == 2
    with pytest.raises(ProviderRequestError):
        EmbeddingGateway(EmbeddingConfig(expected_dimension=3), transport=FakeTransport(embedding_response())).embed(context(ProviderRole.EMBEDDING), "a")


def test_missing_usage_is_incomplete_without_estimation():
    raw = embedding_response()
    raw.pop("usage")
    result = EmbeddingGateway(EmbeddingConfig(), transport=FakeTransport(raw)).embed(context(ProviderRole.EMBEDDING), "a")
    assert not result.attempt.metrics.usage.usage_complete
    assert result.attempt.metrics.usage.total_tokens is None
    assert result.attempt.metrics.usage.output_tokens == 0


def test_byte_limits_allow_exact_boundary():
    transport = FakeTransport(embedding_response(35))
    gateway = EmbeddingGateway(EmbeddingConfig(), transport=transport)
    gateway.embed(context(ProviderRole.EMBEDDING), ["x" * 8000] * 35)
    assert len(transport.calls) == 1
