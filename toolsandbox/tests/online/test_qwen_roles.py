import json
from dataclasses import replace

import pytest
from toolsandbox_pipeline.providers.qwen import QwenGateway
from toolsandbox_pipeline.providers.contracts import RequestContext, ProviderRole, TransportResponse, ProviderRequestError
from toolsandbox_pipeline.schemas.runtime import QwenConfig
from toolsandbox_pipeline.online import qwen_roles
from toolsandbox_pipeline.online.qwen_roles import (
    CriticRunner,
    InitialPolicyRunner,
    QwenResponseDurabilityError,
    RevisionRunner,
)
from tests.online.test_prompt_builder import contexts, prepared

MANIFEST = "sha256:" + "1" * 64


def request_context(request, n=0):
    return RequestContext(logical_request_id=f"logical-{n}", attempt_id=f"attempt-{n}", role=ProviderRole(request.role),
        phase="offline", unit_reference=request.state_id, input_fingerprint=request.canonical_input_fingerprint,
        replayed_after_unknown_outcome=False, manifest_identity=MANIFEST)


class Chat:
    def __init__(self, content, finish="stop", tokens=10):
        self.content, self.finish, self.tokens = content, finish, tokens
        self.calls = []
        self.raw_bodies = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = dict(model="Qwen/Qwen3-32B", choices=[dict(finish_reason=self.finish, message=dict(content=self.content))],
                    usage=dict(prompt_tokens=20, completion_tokens=self.tokens, total_tokens=20 + self.tokens))
        raw_body = json.dumps(data).encode()
        self.raw_bodies.append(raw_body)
        return TransportResponse(raw_body=raw_body, data=data)


@pytest.mark.parametrize("index,runner_type", [(0, InitialPolicyRunner), (1, CriticRunner), (2, RevisionRunner)])
def test_single_role_call(tmp_path, index, runner_type):
    request = prepared(contexts(tmp_path)[index], index)
    content = '{"verdict":"accept","predicted_outcome":"success","predicted_effect":"A result","error_codes":[],"correction":""}' if index == 1 else '{"action":{"type":"assistant_message","content":"Clarify please"}}'
    transport = Chat(content)
    runner = runner_type(QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport), manifest_identity=MANIFEST)
    result = runner.run(request, request_context(request))
    assert len(transport.calls) == 1 and result.attempt.metrics.usage.total_tokens == 30
    assert result.output is not None
    assert "tools" not in transport.calls[0] and "top_p" not in transport.calls[0]
    assert transport.calls[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    if index == 2:
        with pytest.raises(ValueError):
            runner.run(request, request_context(request, 1))
        assert len(transport.calls) == 1


@pytest.mark.parametrize("content,finish", [("broken", "stop"), ("", "length"), ("<think>no</think>", "stop")])
def test_invalid_output_no_retry(tmp_path, content, finish):
    request = prepared(contexts(tmp_path)[0], 0)
    transport = Chat(content, finish)
    runner = InitialPolicyRunner(QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"), transport=transport), manifest_identity=MANIFEST)
    with pytest.raises(ProviderRequestError):
        runner.run(request, request_context(request))
    assert len(transport.calls) == 1


def test_exact_response_persisted_before_role_result_and_raw_bytes_hidden(
    tmp_path, monkeypatch, caplog
):
    request = prepared(contexts(tmp_path)[0], 0)
    transport = Chat(
        '{"action":{"type":"assistant_message","content":"persist-sentinel"}}'
    )
    events = []
    captured = []

    class Seam:
        def load_completed_response(self, context):
            events.append("load")
            return None

        def persist_completed_response(self, response):
            events.append("persist")
            captured.append(response)

    real_role_call_result = qwen_roles.RoleCallResult

    def record_role_call_result(**kwargs):
        events.append("result")
        return real_role_call_result(**kwargs)

    monkeypatch.setattr(qwen_roles, "RoleCallResult", record_role_call_result)
    runner = InitialPolicyRunner(
        QwenGateway(
            QwenConfig(structured_output_wire_mode="guided_json"),
            transport=transport,
        ),
        manifest_identity=MANIFEST,
        durability_seam=Seam(),
    )
    result = runner.run(request, request_context(request))

    assert events == ["load", "persist", "result"]
    response, = captured
    assert response.raw_response_body == transport.raw_bodies[0]
    assert result.attempt == response.attempt
    assert transport.raw_bodies[0].decode() not in repr(result) + result.model_dump_json() + caplog.text


def test_completed_response_reuse_preserves_source_attempt_without_dispatch(
    tmp_path,
):
    request = prepared(contexts(tmp_path)[0], 0)
    source_transport = Chat(
        '{"action":{"type":"assistant_message","content":"saved"}}'
    )
    captured = []

    class CaptureSeam:
        def load_completed_response(self, context):
            return None

        def persist_completed_response(self, response):
            captured.append(response)

    source_runner = InitialPolicyRunner(
        QwenGateway(
            QwenConfig(structured_output_wire_mode="guided_json"),
            transport=source_transport,
        ),
        manifest_identity=MANIFEST,
        durability_seam=CaptureSeam(),
    )
    source_context = request_context(request)
    source_runner.run(request, source_context)
    completed, = captured

    persist_calls = []

    class ReuseSeam:
        def load_completed_response(self, context):
            return completed

        def persist_completed_response(self, response):
            persist_calls.append(response)

    unused_transport = Chat(
        '{"action":{"type":"assistant_message","content":"new"}}'
    )
    recovery_data = source_context.model_dump()
    recovery_data.update(
        attempt_id="recovery-attempt",
        replayed_after_unknown_outcome=True,
    )
    recovery_context = RequestContext(**recovery_data)
    recovery_runner = InitialPolicyRunner(
        QwenGateway(
            QwenConfig(structured_output_wire_mode="guided_json"),
            transport=unused_transport,
        ),
        manifest_identity=MANIFEST,
        durability_seam=ReuseSeam(),
    )
    result = recovery_runner.run(request, recovery_context)

    assert unused_transport.calls == []
    assert persist_calls == []
    assert result.output.action.content == "saved"
    assert result.attempt.context.attempt_id == source_context.attempt_id
    assert result.attempt.context.attempt_id != recovery_context.attempt_id


@pytest.mark.parametrize(
    "index,runner_type",
    [(0, InitialPolicyRunner), (1, CriticRunner), (2, RevisionRunner)],
)
def test_every_online_role_persists_through_the_same_seam(
    tmp_path, index, runner_type
):
    request = prepared(contexts(tmp_path)[index], index)
    content = (
        '{"verdict":"accept","predicted_outcome":"success",'
        '"predicted_effect":"A result","error_codes":[],"correction":""}'
        if index == 1
        else '{"action":{"type":"assistant_message","content":"ok"}}'
    )
    transport = Chat(content)
    persisted = []

    class Seam:
        def load_completed_response(self, context):
            return None

        def persist_completed_response(self, response):
            persisted.append(response)

    runner = runner_type(
        QwenGateway(
            QwenConfig(structured_output_wire_mode="guided_json"),
            transport=transport,
        ),
        manifest_identity=MANIFEST,
        durability_seam=Seam(),
    )
    result = runner.run(request, request_context(request))

    response, = persisted
    assert response.attempt.context.role is ProviderRole(request.role)
    assert result.attempt == response.attempt


def test_revision_allows_same_logical_response_recovery_only(tmp_path):
    request = prepared(contexts(tmp_path)[2], 2)
    transport = Chat(
        '{"action":{"type":"assistant_message","content":"saved revision"}}'
    )

    class Seam:
        completed = None

        def load_completed_response(self, context):
            return self.completed

        def persist_completed_response(self, response):
            self.completed = response

    seam = Seam()
    runner = RevisionRunner(
        QwenGateway(
            QwenConfig(structured_output_wire_mode="guided_json"),
            transport=transport,
        ),
        manifest_identity=MANIFEST,
        durability_seam=seam,
    )
    source_context = request_context(request)
    first = runner.run(request, source_context)
    recovery_data = source_context.model_dump()
    recovery_data.update(
        attempt_id="revision-recovery-attempt",
        replayed_after_unknown_outcome=True,
    )
    recovered = runner.run(request, RequestContext(**recovery_data))

    assert len(transport.calls) == 1
    assert recovered.output == first.output
    assert recovered.attempt.context.attempt_id == source_context.attempt_id

    with pytest.raises(ValueError, match="second Revision forbidden"):
        runner.run(request, request_context(request, 99))
    assert len(transport.calls) == 1


def test_recovered_response_identity_or_raw_hash_mismatch_fails_closed(tmp_path):
    request = prepared(contexts(tmp_path)[0], 0)
    source_transport = Chat(
        '{"action":{"type":"assistant_message","content":"saved"}}'
    )
    captured = []

    class CaptureSeam:
        def load_completed_response(self, context):
            return None

        def persist_completed_response(self, response):
            captured.append(response)

    source_runner = InitialPolicyRunner(
        QwenGateway(
            QwenConfig(structured_output_wire_mode="guided_json"),
            transport=source_transport,
        ),
        manifest_identity=MANIFEST,
        durability_seam=CaptureSeam(),
    )
    source_runner.run(request, request_context(request))
    completed, = captured
    invalid = replace(completed, raw_response_body=b"different")

    class InvalidSeam:
        def load_completed_response(self, context):
            return invalid

        def persist_completed_response(self, response):
            pytest.fail("invalid recovered response must not persist")

    unused_transport = Chat(
        '{"action":{"type":"assistant_message","content":"new"}}'
    )
    runner = InitialPolicyRunner(
        QwenGateway(
            QwenConfig(structured_output_wire_mode="guided_json"),
            transport=unused_transport,
        ),
        manifest_identity=MANIFEST,
        durability_seam=InvalidSeam(),
    )
    with pytest.raises(QwenResponseDurabilityError, match="invalid completed"):
        runner.run(request, request_context(request))
    assert unused_transport.calls == []


@pytest.mark.parametrize("failure_point", ["load", "persist"])
def test_durability_failure_is_sanitized_and_never_retried(
    tmp_path, failure_point
):
    request = prepared(contexts(tmp_path)[0], 0)
    transport = Chat(
        '{"action":{"type":"assistant_message","content":"raw-sentinel"}}'
    )

    class Seam:
        def load_completed_response(self, context):
            if failure_point == "load":
                raise RuntimeError("callback-secret-sentinel")
            return None

        def persist_completed_response(self, response):
            if failure_point == "persist":
                raise RuntimeError(response.raw_response_body.decode())

    runner = InitialPolicyRunner(
        QwenGateway(
            QwenConfig(structured_output_wire_mode="guided_json"),
            transport=transport,
        ),
        manifest_identity=MANIFEST,
        durability_seam=Seam(),
    )
    with pytest.raises(QwenResponseDurabilityError) as caught:
        runner.run(request, request_context(request))

    assert len(transport.calls) == (0 if failure_point == "load" else 1)
    assert caught.value.__cause__ is None
    assert "sentinel" not in str(caught.value) + repr(caught.value)
