"""Instrument the generic upstream User without copying its prompt/turn logic.

Model selection intentionally differs from GPT_4_o_2024_05_13_User. Both SDK
retries and the generic user's tenacity-decorated inference are bypassed.
"""
from hashlib import sha256
from typing import Protocol

from openai import NOT_GIVEN
from openai.types.chat import ChatCompletion
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

from toolsandbox_pipeline.schemas.runtime import UserSimulatorConfig
from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    PhysicalAttemptStatus,
    ProviderRole,
    RequestContext,
    physical_request,
)
from toolsandbox_pipeline.metrics.usage import UsageRecorder


class UserResponseDurabilitySeam(Protocol):
    """Caller-owned durable-response boundary used by episode recovery.

    Implementations may query and write a Task 011 request ledger. Provider code
    deliberately knows nothing about the ledger or its storage format.
    """

    def load_completed_response(
        self, context: RequestContext
    ) -> GatewayResponse | None: ...

    def persist_completed_response(self, response: GatewayResponse) -> None: ...


class UserResponseDurabilityError(RuntimeError):
    """Sanitized failure from the injected durability boundary."""


class UserSimulatorGateway:
    def __init__(self, config: UserSimulatorConfig, *, transport=None, recorder=None):
        self.config = config
        self._transport = transport
        self.recorder = recorder if recorder is not None else UsageRecorder(ProviderRole.USER_SIMULATOR)

    def chat(self, context, messages, tools=NOT_GIVEN):
        def prepare():
            if context.role is not ProviderRole.USER_SIMULATOR:
                raise ValueError("user simulator role required")
            if not isinstance(messages, list) or not messages:
                raise ValueError("user messages required")
            for message in messages:
                if type(message) is not dict or set(message) != {"role", "content"}:
                    raise ValueError("invalid upstream user message")
                if message["role"] not in ("system", "user", "assistant") or type(message["content"]) is not str:
                    raise ValueError("invalid upstream user message")
            if self._transport is None:
                from toolsandbox_pipeline.providers.openai_clients import create_transport
                self._transport = create_transport(self.config)
            # Preserve the exact upstream request, including NOT_GIVEN tools.
            return self._transport, dict(model=self.config.model, messages=messages, tools=tools)

        def validate(raw, data):
            if raw.get("model") is not None and raw["model"] != self.config.model:
                raise ValueError("user snapshot mismatch")
            choices = raw.get("choices")
            if type(choices) is not list or len(choices) != 1:
                raise ValueError("one user choice required")
            message = choices[0]["message"]
            if message.get("tool_calls") is None and type(message.get("content")) is not str:
                raise ValueError("missing user content")
            if isinstance(data, ChatCompletion):
                return data
            return ChatCompletion.model_validate(raw)

        return physical_request(context, self.config.model, prepare=prepare,
                                validate=validate, recorder=self.recorder)


class InstrumentedGPT4oMiniUser(OpenAIAPIUser):
    model_name = "gpt-4o-mini-2024-07-18"

    def __init__(self, *, context_provider, config=None, transport=None,
                 recorder=None, durability_seam=None):
        # Do not call upstream __init__: it eagerly resolves environment secrets.
        self._context_provider = context_provider
        self._durability_seam: UserResponseDurabilitySeam | None = durability_seam
        self.gateway = UserSimulatorGateway(config or UserSimulatorConfig(),
                                            transport=transport, recorder=recorder)

    def model_inference(self, openai_messages, openai_tools):
        context = self._context_provider()
        if not isinstance(context, RequestContext):
            raise TypeError("validated RequestContext required")
        if context.role is not ProviderRole.USER_SIMULATOR:
            raise ValueError("user simulator role required")

        if self._durability_seam is not None:
            try:
                completed = self._durability_seam.load_completed_response(context)
            except Exception:
                raise UserResponseDurabilityError(
                    "completed user response lookup failed"
                ) from None
            if completed is not None:
                return self._validate_reusable_response(context, completed).value

        response = self.gateway.chat(context, openai_messages, openai_tools)
        if self._durability_seam is not None:
            try:
                self._durability_seam.persist_completed_response(response)
            except Exception:
                raise UserResponseDurabilityError(
                    "completed user response persistence failed"
                ) from None
        # OpenAIAPIUser.respond() parses this value and calls add_messages only
        # after model_inference returns, so persistence necessarily happens first.
        return response.value

    def _validate_reusable_response(
        self, context: RequestContext, response: GatewayResponse
    ) -> GatewayResponse:
        if not isinstance(response, GatewayResponse):
            raise UserResponseDurabilityError("invalid completed user response")
        attempt = response.attempt
        saved_context = attempt.context
        if (
            attempt.status is not PhysicalAttemptStatus.COMPLETED
            or attempt.model != self.gateway.config.model
            or attempt.returned_model not in (None, self.gateway.config.model)
            or saved_context.role is not ProviderRole.USER_SIMULATOR
            or context.role is not ProviderRole.USER_SIMULATOR
        ):
            raise UserResponseDurabilityError("invalid completed user response")
        current_identity = (
            context.logical_request_id,
            context.role,
            context.phase,
            context.unit_reference,
            context.input_fingerprint,
            context.manifest_identity,
        )
        saved_identity = (
            saved_context.logical_request_id,
            saved_context.role,
            saved_context.phase,
            saved_context.unit_reference,
            saved_context.input_fingerprint,
            saved_context.manifest_identity,
        )
        raw_body = response.raw_response_body
        if (
            current_identity != saved_identity
            or type(raw_body) is not bytes
            or attempt.response_hash
            != "sha256:" + sha256(raw_body).hexdigest()
            or not isinstance(response.value, ChatCompletion)
        ):
            raise UserResponseDurabilityError("invalid completed user response")
        return response
