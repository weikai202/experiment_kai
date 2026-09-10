import json

import pytest
from openai import NOT_GIVEN
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser, GPT_4_o_2024_05_13_User
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.common.execution_context import RoleType
from toolsandbox_pipeline.providers.user_simulator import (
    InstrumentedGPT4oMiniUser,
    UserResponseDurabilityError,
    UserSimulatorGateway,
)
from toolsandbox_pipeline.providers.contracts import ProviderRole, ProviderRequestError
from toolsandbox_pipeline.schemas.runtime import UserSimulatorConfig
from tests.providers.test_request_identity import context, FakeTransport, chat_response


def test_upstream_behavior_inherited_exactly():
    assert issubclass(InstrumentedGPT4oMiniUser, OpenAIAPIUser)
    assert InstrumentedGPT4oMiniUser.respond is OpenAIAPIUser.respond
    assert InstrumentedGPT4oMiniUser.to_openai_messages is OpenAIAPIUser.to_openai_messages
    assert InstrumentedGPT4oMiniUser.filter_messages.__func__ is OpenAIAPIUser.filter_messages.__func__
    assert InstrumentedGPT4oMiniUser.role_type is RoleType.USER
    assert InstrumentedGPT4oMiniUser.model_name == "gpt-4o-mini-2024-07-18"
    assert GPT_4_o_2024_05_13_User.model_name == "gpt-4o-2024-05-13"
    assert not hasattr(InstrumentedGPT4oMiniUser.model_inference, "retry")


def test_native_message_conversion_and_tool_omission():
    transport = FakeTransport(chat_response("ok", "gpt-4o-mini-2024-07-18"))
    user = InstrumentedGPT4oMiniUser(context_provider=lambda: context(ProviderRole.USER_SIMULATOR), transport=transport)
    messages = [Message(RoleType.SYSTEM, RoleType.USER, "synthetic system"),
                Message(RoleType.USER, RoleType.AGENT, "synthetic user"),
                Message(RoleType.AGENT, RoleType.USER, "synthetic agent")]
    wire = user.to_openai_messages(messages)
    assert wire == [{"role": "system", "content": "synthetic system"},
                    {"role": "assistant", "content": "synthetic user"},
                    {"role": "user", "content": "synthetic agent"}]
    response = user.model_inference(wire, NOT_GIVEN)
    assert response.choices[0].message.content == "ok"
    request, = transport.calls
    assert set(request) == {"model", "messages", "tools"}
    assert request["model"] == "gpt-4o-mini-2024-07-18"
    assert request["tools"] is NOT_GIVEN
    attempt, = user.gateway.recorder.attempts
    assert attempt.context.role is ProviderRole.USER_SIMULATOR
    assert attempt.metrics.usage.total_tokens == 8


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4o-2024-05-13", "Qwen/Qwen3-32B", "secret-sentinel"])
def test_snapshot_mismatch_no_fallback(model):
    transport = FakeTransport(chat_response("ok", model))
    user = InstrumentedGPT4oMiniUser(context_provider=lambda: context(ProviderRole.USER_SIMULATOR), transport=transport)
    with pytest.raises(ProviderRequestError) as caught:
        user.model_inference([{"role": "user", "content": "x"}], NOT_GIVEN)
    assert len(transport.calls) == 1
    assert "secret-sentinel" not in repr(caught.value.attempt)


def test_timeout_has_no_upstream_retries():
    transport = FakeTransport(error=TimeoutError("secret-sentinel"))
    user = InstrumentedGPT4oMiniUser(context_provider=lambda: context(ProviderRole.USER_SIMULATOR), transport=transport)
    with pytest.raises(ProviderRequestError):
        user.model_inference([{"role": "user", "content": "x"}], NOT_GIVEN)
    assert len(transport.calls) == 1
    assert len(user.gateway.recorder.attempts) == 1


def test_native_user_respond_and_termination_tool(monkeypatch):
    from tool_sandbox.tools.user_tools import end_conversation
    transport = FakeTransport(chat_response(None, "gpt-4o-mini-2024-07-18"))
    transport.payload["choices"][0]["message"]["tool_calls"] = [
        {"id": "end-call", "type": "function", "function": {"name": "end_conversation", "arguments": "{}"}}]
    user = InstrumentedGPT4oMiniUser(context_provider=lambda: context(ProviderRole.USER_SIMULATOR), transport=transport)
    monkeypatch.setattr(user, "get_messages", lambda **kw: [Message(RoleType.AGENT, RoleType.USER, "done")])
    monkeypatch.setattr(user, "get_available_tools", lambda: {"end_conversation": end_conversation})
    appended = []
    monkeypatch.setattr(user, "add_messages", appended.extend)
    user.respond()
    assert len(transport.calls) == 1
    assert transport.calls[0]["tools"][0]["function"]["name"] == "end_conversation"
    assert len(appended) == 1
    assert appended[0].sender is RoleType.USER
    assert appended[0].recipient is RoleType.EXECUTION_ENVIRONMENT
    assert "end_conversation" in appended[0].content


def test_system_turn_no_client_no_context(monkeypatch):
    def unexpected():
        pytest.fail("context unexpectedly consumed")
    user = InstrumentedGPT4oMiniUser(context_provider=unexpected)
    monkeypatch.setattr(user, "get_messages", lambda **kw: [Message(RoleType.SYSTEM, RoleType.USER, "synthetic")])
    user.respond()
    assert user.gateway._transport is None


def test_durability_seam_persists_exact_response_before_message_visibility(
    monkeypatch, caplog
):
    payload = chat_response("raw-response-sentinel", "gpt-4o-mini-2024-07-18")
    transport = FakeTransport(payload)
    events = []
    captured = []

    class Seam:
        def load_completed_response(self, request_context):
            events.append("load")
            assert request_context.role is ProviderRole.USER_SIMULATOR
            return None

        def persist_completed_response(self, response):
            assert len(transport.calls) == 1
            events.append("persist")
            captured.append(response)

    user = InstrumentedGPT4oMiniUser(
        context_provider=lambda: context(ProviderRole.USER_SIMULATOR),
        transport=transport,
        durability_seam=Seam(),
    )
    monkeypatch.setattr(
        user,
        "get_messages",
        lambda **kw: [Message(RoleType.AGENT, RoleType.USER, "synthetic agent")],
    )
    monkeypatch.setattr(user, "get_available_tools", lambda: {})
    appended = []

    def add_messages(messages):
        events.append("add")
        appended.extend(messages)

    monkeypatch.setattr(user, "add_messages", add_messages)
    user.respond()

    assert events == ["load", "persist", "add"]
    response, = captured
    expected_raw = json.dumps(payload).encode()
    assert response.raw_response_body == expected_raw
    assert appended[0].content == "raw-response-sentinel"
    assert expected_raw.decode() not in repr(response) + caplog.text


def test_completed_response_reuse_does_not_dispatch_or_persist_again():
    saved_transport = FakeTransport(
        chat_response("saved", "gpt-4o-mini-2024-07-18")
    )
    saved_context = context(ProviderRole.USER_SIMULATOR, attempt_id="saved-attempt")
    completed = UserSimulatorGateway(
        UserSimulatorConfig(), transport=saved_transport
    ).chat(saved_context, [{"role": "user", "content": "x"}], NOT_GIVEN)

    events = []

    class Seam:
        def load_completed_response(self, request_context):
            events.append(("load", request_context.attempt_id))
            return completed

        def persist_completed_response(self, response):
            pytest.fail("a reused response must not be persisted twice")

    unused_transport = FakeTransport(
        chat_response("new", "gpt-4o-mini-2024-07-18")
    )
    user = InstrumentedGPT4oMiniUser(
        context_provider=lambda: context(
            ProviderRole.USER_SIMULATOR, attempt_id="unused-attempt"
        ),
        transport=unused_transport,
        durability_seam=Seam(),
    )
    value = user.model_inference([{"role": "user", "content": "x"}], NOT_GIVEN)

    assert value is completed.value
    assert events == [("load", "unused-attempt")]
    assert unused_transport.calls == []
    assert user.gateway.recorder.attempts == ()


def test_persistence_failure_propagates_before_user_message_append(monkeypatch):
    transport = FakeTransport(
        chat_response("raw-response-sentinel", "gpt-4o-mini-2024-07-18")
    )

    class Seam:
        def load_completed_response(self, request_context):
            return None

        def persist_completed_response(self, response):
            raise RuntimeError(response.raw_response_body.decode())

    user = InstrumentedGPT4oMiniUser(
        context_provider=lambda: context(ProviderRole.USER_SIMULATOR),
        transport=transport,
        durability_seam=Seam(),
    )
    monkeypatch.setattr(
        user,
        "get_messages",
        lambda **kw: [Message(RoleType.AGENT, RoleType.USER, "synthetic agent")],
    )
    monkeypatch.setattr(user, "get_available_tools", lambda: {})
    appended = []
    monkeypatch.setattr(user, "add_messages", appended.extend)

    with pytest.raises(UserResponseDurabilityError) as caught:
        user.respond()

    assert len(transport.calls) == 1
    assert len(user.gateway.recorder.attempts) == 1
    assert appended == []
    assert caught.value.__cause__ is None
    assert "raw-response-sentinel" not in str(caught.value) + repr(caught.value)


def test_completed_response_lookup_failure_prevents_dispatch():
    class Seam:
        def load_completed_response(self, request_context):
            raise RuntimeError("lookup-detail-sentinel")

        def persist_completed_response(self, response):
            pytest.fail("persistence must not run after lookup failure")

    transport = FakeTransport(
        chat_response("new", "gpt-4o-mini-2024-07-18")
    )
    user = InstrumentedGPT4oMiniUser(
        context_provider=lambda: context(ProviderRole.USER_SIMULATOR),
        transport=transport,
        durability_seam=Seam(),
    )

    with pytest.raises(UserResponseDurabilityError) as caught:
        user.model_inference([{"role": "user", "content": "x"}], NOT_GIVEN)

    assert transport.calls == []
    assert user.gateway.recorder.attempts == ()
    assert caught.value.__cause__ is None
    assert "lookup-detail-sentinel" not in str(caught.value) + repr(caught.value)


def test_reused_response_identity_mismatch_fails_without_dispatch():
    saved_transport = FakeTransport(
        chat_response("saved", "gpt-4o-mini-2024-07-18")
    )
    completed = UserSimulatorGateway(
        UserSimulatorConfig(), transport=saved_transport
    ).chat(
        context(ProviderRole.USER_SIMULATOR),
        [{"role": "user", "content": "x"}],
        NOT_GIVEN,
    )

    class Seam:
        def load_completed_response(self, request_context):
            return completed

        def persist_completed_response(self, response):
            pytest.fail("invalid reuse must not persist")

    unused_transport = FakeTransport(
        chat_response("new", "gpt-4o-mini-2024-07-18")
    )
    user = InstrumentedGPT4oMiniUser(
        context_provider=lambda: context(
            ProviderRole.USER_SIMULATOR,
            input_fingerprint="sha256:" + "c" * 64,
        ),
        transport=unused_transport,
        durability_seam=Seam(),
    )

    with pytest.raises(UserResponseDurabilityError, match="invalid completed"):
        user.model_inference([{"role": "user", "content": "x"}], NOT_GIVEN)
    assert unused_transport.calls == []
