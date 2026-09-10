"""Frozen Qwen generation with explicit structured-output wire modes."""
import json
from pydantic import Field
from typing import Literal
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.runtime import QwenConfig, RoleTokenLimitConfig
from toolsandbox_pipeline.providers.contracts import (
    OutputTruncated,
    ProviderRole,
    QWEN_GENERATION_ROLES,
    ValidatedProviderResponse,
    physical_request,
)


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class QwenGateway:
    def __init__(self, config: QwenConfig, *, transport=None, recorder=None):
        self.config = config
        self._transport = transport
        self.recorder = recorder

    def generate(self, context, messages, output_model, *, max_tokens, token_limit_config=None):
        """Use an explicit versioned selection for calibrated values.

        Omitting it selects the documented bootstrap config for offline probes.
        Formal calibration/evidence verification belongs to the caller.
        """
        def prepare():
            if context.role not in QWEN_GENERATION_ROLES:
                raise ValueError("Qwen generation role required")
            if type(max_tokens) is not int or max_tokens <= 0:
                raise ValueError("positive calibrated token limit required")
            selected_limit = token_limit_config
            if selected_limit is None:
                selected_limit = RoleTokenLimitConfig(version="section15-bootstrap-v1", role=context.role.value,
                                                     stage="bootstrap", max_tokens=max_tokens)
            if not isinstance(selected_limit, RoleTokenLimitConfig):
                raise ValueError("validated token-limit config required")
            if selected_limit.role != context.role.value or selected_limit.max_tokens != max_tokens:
                raise ValueError("selected token-limit config mismatch")
            if self.config.output_limit is not None and max_tokens > self.config.output_limit:
                raise ValueError("deployment output limit exceeded")
            if not isinstance(messages, list) or not messages:
                raise ValueError("non-empty messages required")
            wire_messages = [ChatMessage.model_validate(m.model_dump() if isinstance(m, ChatMessage) else m).model_dump() for m in messages]
            if not isinstance(output_model, type) or not issubclass(output_model, StrictModel):
                raise ValueError("strict output model required")
            if not output_model.model_config.get("strict") or output_model.model_config.get("extra") != "forbid":
                raise ValueError("strict output model required")
            from toolsandbox_pipeline.schemas.action import ActionEnvelope
            from toolsandbox_pipeline.schemas.critic import CriticOutput
            expected_model = {ProviderRole.VANILLA: ActionEnvelope,
                              ProviderRole.POLICY: ActionEnvelope, ProviderRole.REVISION: ActionEnvelope,
                              ProviderRole.CRITIC: CriticOutput}.get(context.role)
            if expected_model is not None and output_model is not expected_model:
                raise ValueError("role output schema mismatch")
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
            schema = output_model.model_json_schema()
            if self.config.structured_output_wire_mode == "guided_json":
                extra["guided_json"] = schema
            else:
                extra["structured_outputs"] = {"json": schema}
            request = dict(model=self.config.model, messages=wire_messages,
                           temperature=0.0, seed=0, max_tokens=max_tokens, extra_body=extra)
            if self._transport is None:
                from toolsandbox_pipeline.providers.openai_clients import create_transport
                self._transport = create_transport(self.config)
            return self._transport, request

        def validate(raw, data):
            if raw.get("model") != (self.config.served_model_id or self.config.model):
                raise ValueError("model mismatch")
            choices = raw.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("one choice required")
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            if finish_reason not in ("stop", "length"):
                raise ValueError("unsupported or missing finish reason")
            if any(raw.get(key) not in (None, "") for key in ("reasoning", "reasoning_content")):
                raise ValueError("reasoning output")
            message = choice["message"]
            if any(message.get(key) not in (None, "") for key in ("reasoning", "reasoning_content")):
                raise ValueError("reasoning output")
            if finish_reason == "length":
                raise OutputTruncated
            content = message.get("content")
            if type(content) is not str or not content.strip() or "<think" in content.lower() or "</think" in content.lower():
                raise ValueError("invalid non-thinking content")
            if message.get("tool_calls") or message.get("function_call"):
                raise ValueError("native tool calling forbidden")
            parsed = json.loads(content, object_pairs_hook=_unique_object,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            return ValidatedProviderResponse(
                value=output_model.model_validate(parsed), finish_reason=finish_reason
            )

        return physical_request(context, self.config.served_model_id or self.config.model,
                                prepare=prepare, validate=validate, recorder=self.recorder)
