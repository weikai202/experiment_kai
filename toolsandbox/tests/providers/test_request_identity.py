import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from hashlib import sha256

import pytest
from pydantic import ValidationError
from toolsandbox_pipeline.providers.contracts import (
    RequestContext, ProviderRole, TransportResponse, ProviderRequestError,
    PhysicalAttemptStatus, RejectedBeforeDispatch, physical_request,
)
from toolsandbox_pipeline.metrics.usage import UsageRecorder


def context(role=ProviderRole.POLICY, attempt_id="attempt-1", **changes):
    data = dict(logical_request_id="logical-1", attempt_id=attempt_id, role=role,
                phase="online", unit_reference="state-1", input_fingerprint="sha256:" + "a" * 64,
                replayed_after_unknown_outcome=False, manifest_identity="sha256:" + "b" * 64)
    data.update(changes)
    return RequestContext(**data)


class FakeTransport:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []
        self.closed = False

    def create(self, **request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return TransportResponse(json.dumps(self.payload).encode(), self.payload)

    def close(self):
        self.closed = True


def chat_response(content='{"action":{"type":"assistant_message","content":"ok"}}', model="Qwen/Qwen3-32B"):
    return dict(id="fake", object="chat.completion", created=0, model=model,
                choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content=content))],
                usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8))


def test_immutable_context_and_validation():
    ctx = context()
    with pytest.raises(ValidationError):
        ctx.attempt_id = "changed"
    for changes in ({"attempt_id": ""}, {"input_fingerprint": "bad"},
                    {"replayed_after_unknown_outcome": 1}, {"extra": 1}):
        with pytest.raises(ValidationError):
            context(**changes)


@pytest.mark.parametrize("error,status,calls", [
    (TimeoutError("secret-sentinel"), PhysicalAttemptStatus.UNKNOWN_OUTCOME, 1),
    (ConnectionError("secret-sentinel"), PhysicalAttemptStatus.UNKNOWN_OUTCOME, 1),
    (RuntimeError("secret-sentinel"), PhysicalAttemptStatus.UNKNOWN_OUTCOME, 1),
    (RejectedBeforeDispatch("secret-sentinel"), PhysicalAttemptStatus.REJECTED, 1),
])
def test_single_attempt_failure(error, status, calls, caplog):
    transport = FakeTransport(error=error)
    recorder = UsageRecorder(ProviderRole.POLICY)
    with pytest.raises(ProviderRequestError) as caught:
        physical_request(context(), "Qwen/Qwen3-32B", prepare=lambda: (transport, {}),
                         validate=lambda raw, data: data, recorder=recorder)
    attempt = caught.value.attempt
    assert attempt.status == status
    assert len(transport.calls) == calls
    assert attempt.metrics.latency_seconds >= 0
    assert attempt.metrics.started_at.utcoffset().total_seconds() == 0
    assert not attempt.metrics.usage.usage_complete
    assert recorder.attempts == (attempt,)
    assert "secret-sentinel" not in str(caught.value) + repr(caught.value) + caplog.text + repr(attempt)
    assert caught.value.__cause__ is None
    with pytest.raises(ValidationError):
        attempt.status = PhysicalAttemptStatus.COMPLETED
    with pytest.raises(ValidationError):
        attempt.metrics.latency_seconds = 100.0


def test_preparation_failure_does_not_dispatch():
    transport = FakeTransport()
    def prepare():
        raise ValueError("secret-sentinel")
    with pytest.raises(ProviderRequestError) as caught:
        physical_request(context(), "model", prepare=prepare, validate=lambda *a: None)
    assert caught.value.attempt.status == PhysicalAttemptStatus.REJECTED
    assert caught.value.attempt.metrics.latency_seconds == 0
    assert caught.value.raw_response_body is None
    assert transport.calls == []


def test_success_raw_hash_and_immutable_result():
    transport = FakeTransport(chat_response())
    result = physical_request(context(), "Qwen/Qwen3-32B", prepare=lambda: (transport, {}),
                              validate=lambda raw, data: "ok")
    assert result.value == "ok"
    expected_raw = json.dumps(transport.payload).encode()
    assert result.raw_response_body == expected_raw
    assert result.attempt.response_hash == "sha256:" + sha256(expected_raw).hexdigest()
    assert result.attempt.metrics.usage.total_tokens == 8
    assert expected_raw.decode() not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.attempt = None


def test_received_invalid_response_bytes_are_restricted_but_available_for_ledger():
    payload = chat_response("secret-response-sentinel")
    transport = FakeTransport(payload)
    with pytest.raises(ProviderRequestError) as caught:
        physical_request(
            context(),
            "Qwen/Qwen3-32B",
            prepare=lambda: (transport, {}),
            validate=lambda raw, data: (_ for _ in ()).throw(ValueError("invalid")),
        )
    expected_raw = json.dumps(payload).encode()
    assert caught.value.raw_response_body == expected_raw
    assert caught.value.attempt.response_hash == "sha256:" + sha256(expected_raw).hexdigest()
    assert "secret-response-sentinel" not in str(caught.value) + repr(caught.value)
