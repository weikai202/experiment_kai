import pytest
from pydantic import ValidationError
from toolsandbox_pipeline.metrics.usage import normalize_usage, UsageRecorder
from toolsandbox_pipeline.schemas.usage import TokenUsage
from toolsandbox_pipeline.providers.contracts import ProviderRole, physical_request, ProviderRequestError
from tests.providers.test_request_identity import context, FakeTransport, chat_response


def test_actual_cache_formula():
    usage = normalize_usage(dict(prompt_tokens=10, completion_tokens=3, total_tokens=13,
                                 prompt_tokens_details=dict(cached_tokens=4, cache_write_tokens=2)))
    assert usage.uncached_input_tokens == 4
    assert usage.cache_read_input_tokens == 4
    assert usage.cache_write_input_tokens == 2
    assert usage.input_tokens == 10 and usage.total_tokens == 13 and usage.usage_complete
    with pytest.raises(ValidationError):
        usage.total_tokens = 0


@pytest.mark.parametrize("raw", [
    dict(prompt_tokens=True), dict(prompt_tokens=-1), dict(prompt_tokens="1"),
    dict(prompt_tokens=3, completion_tokens=2, total_tokens=4),
    dict(prompt_tokens=1, prompt_tokens_details=dict(cached_tokens=2)),
    dict(prompt_tokens=3, prompt_tokens_details=dict(cached_tokens=-1)),
    dict(prompt_tokens_details="bad"),
    dict(prompt_tokens_details=dict(cached_tokens=1), cache_read_input_tokens=2),
])
def test_invalid_actual_usage(raw):
    with pytest.raises(ValueError):
        normalize_usage(raw)


@pytest.mark.parametrize("raw", [None, {}, {"prompt_tokens": 3}, {"total_tokens": 3},
                                     {"prompt_tokens": 3, "completion_tokens": 2}])
def test_missing_usage_not_estimated(raw):
    usage = normalize_usage(raw)
    assert not usage.usage_complete
    assert usage.total_tokens == (raw or {}).get("total_tokens")


def test_embedding_usage_consistency():
    usage = normalize_usage(dict(prompt_tokens=3, total_tokens=3), embedding=True)
    assert usage.output_tokens == usage.cache_read_input_tokens == usage.cache_write_input_tokens == 0
    assert usage.usage_complete
    with pytest.raises(ValueError):
        normalize_usage(dict(prompt_tokens=3, total_tokens=4), embedding=True)


def test_replays_count_by_attempt_not_logical_request():
    recorder = UsageRecorder(ProviderRole.POLICY)
    transport = FakeTransport(chat_response())
    for index in range(2):
        result = physical_request(context(attempt_id=f"attempt-{index}", replayed_after_unknown_outcome=index > 0),
                                  "Qwen/Qwen3-32B", prepare=lambda: (transport, {}),
                                  validate=lambda raw, data: data, recorder=recorder)
        recorder.record(result.attempt)
    assert len(recorder.attempts) == 2
    assert recorder.token_totals().total_tokens == 16
    transport.payload.pop("usage")
    physical_request(context(attempt_id="attempt-3"), "Qwen/Qwen3-32B", prepare=lambda: (transport, {}),
                     validate=lambda raw, data: data, recorder=recorder)
    assert recorder.token_totals().total_tokens is None
    assert not recorder.token_totals().usage_complete


def test_recorder_role_and_id_conflicts():
    recorder = UsageRecorder(ProviderRole.POLICY)
    transport = FakeTransport(chat_response())
    result = physical_request(context(), "Qwen/Qwen3-32B", prepare=lambda: (transport, {}),
                              validate=lambda raw, data: data, recorder=recorder)
    with pytest.raises(ValueError, match="role"):
        UsageRecorder(ProviderRole.EMBEDDING).record(result.attempt)
    with pytest.raises(ValueError, match="conflicting"):
        recorder.record(result.attempt.model_copy(update={"model": "different"}))
