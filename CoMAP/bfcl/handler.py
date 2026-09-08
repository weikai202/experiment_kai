from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


def as_dict(message: Any) -> dict:
    return (
        message if isinstance(message, dict) else message.model_dump(exclude_none=True)
    )


def action_from_message(message: Any) -> dict:
    message = as_dict(message)
    calls = message.get("tool_calls") or []
    return {
        "content": message.get("content") or "",
        "tool_calls": [
            {
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
            for call in calls
        ],
    }


def validate_action(action: Any, tools: list[dict]) -> dict:
    if not isinstance(action, dict) or not isinstance(action.get("content"), str):
        raise ValueError("Final action must have string content")
    calls = action.get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError("Final action must have a tool_calls list")
    names = {tool["function"]["name"] for tool in tools}
    for call in calls:
        if (
            not isinstance(call, dict)
            or call.get("name") not in names
            or not isinstance(call.get("arguments"), dict)
        ):
            raise ValueError("Invalid tool call in reflection")
    if not calls and not action["content"].strip():
        raise ValueError("Empty final action")
    return {"content": action["content"], "tool_calls": calls}


def message_from_action(action: dict) -> ChatCompletionMessage:
    calls = [
        {
            "id": "call_" + uuid4().hex,
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
            },
        }
        for call in action["tool_calls"]
    ]
    return ChatCompletionMessage(
        role="assistant", content=action["content"] or None, tool_calls=calls or None
    )


@dataclass
class CoMAPResponse:
    response: Any
    trace: dict
    input_tokens: int
    output_tokens: int


class CoMAPHandler(OpenAICompletionsHandler):
    """Draft -> predicted consequence -> gated reflection, with BFCL execution."""

    def __init__(
        self, model_name, temperature, registry_name, is_fc_model=True, **kwargs
    ):
        if not is_fc_model:
            raise ValueError("CoMAP BFCL currently requires function-calling mode")
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.wm_model = os.environ.get("COMAP_WM_MODEL")
        if not self.wm_model:
            raise ValueError(
                "Set COMAP_WM_MODEL to a served world-model checkpoint name"
            )
        self.wm_client = OpenAI(
            api_key=os.getenv("COMAP_WM_API_KEY") or self.client.api_key,
            base_url=os.getenv("COMAP_WM_BASE_URL") or str(self.client.base_url),
            timeout=float(os.getenv("COMAP_TIMEOUT", "120")),
            max_retries=2,
        )
        self.client = self.client.with_options(
            timeout=float(os.getenv("COMAP_TIMEOUT", "120")), max_retries=2
        )
        self.revise_threshold = float(os.getenv("COMAP_REVISE_THRESHOLD", "0.5"))
        self.confidence_threshold = float(
            os.getenv("COMAP_CONFIDENCE_THRESHOLD", "0.6")
        )
        for value in (self.revise_threshold, self.confidence_threshold):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("CoMAP thresholds must be between 0 and 1")
        self.max_tokens = int(os.getenv("COMAP_MAX_TOKENS", "1024"))
        self.wm_max_tokens = int(os.getenv("COMAP_WM_MAX_TOKENS", "512"))
        self.policy_extra_body = json.loads(os.getenv("COMAP_POLICY_EXTRA_BODY", "{}"))
        self.wm_extra_body = json.loads(os.getenv("COMAP_WM_EXTRA_BODY", "{}"))
        if not isinstance(self.policy_extra_body, dict) or not isinstance(
            self.wm_extra_body, dict
        ):
            raise ValueError("CoMAP extra body settings must be JSON objects")

    def _query_FC(self, inference_data: dict):
        started = time.monotonic()
        history = deepcopy([as_dict(message) for message in inference_data["message"]])
        tools = inference_data["tools"]
        kwargs = dict(
            model=self.model_name,
            messages=history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.policy_extra_body,
        )
        if tools:
            kwargs["tools"] = tools
        draft = self.client.chat.completions.create(**kwargs)
        responses = [draft]
        trace = {"flow_mode": "draft_student_wm_reflect", "used_revised_action": False}
        final = draft
        try:
            draft_action = action_from_message(draft.choices[0].message)
        except (ValueError, KeyError, TypeError) as exc:
            trace.update(flow_mode="draft_only_fallback", fallback_reason=str(exc))
            draft_action = None

        # Match upstream's fallback for replies without an executable draft action.
        if draft_action is not None and draft_action["tool_calls"]:
            payload = {"history": history, "tools": tools, "draft_action": draft_action}
            prediction = self.wm_client.chat.completions.create(
                model=self.wm_model,
                temperature=0,
                max_tokens=self.wm_max_tokens,
                extra_body=self.wm_extra_body,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Predict the next tool observations and state changes for a candidate "
                            "BFCL action using only the visible conversation and tool definitions. "
                            "Identify likely errors, missing information and uncertainty. "
                            "Do not execute tools. Do not assume access to hidden environment "
                            "state or reference answers. This is a hypothetical prediction."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
            )
            responses.append(prediction)
            predicted_state = prediction.choices[0].message.content or "Unknown."
            instruction = (
                "Review the draft action against the hypothetical world-model prediction. "
                "Revise only if it is invalid, unhelpful, harmful, redundant or clearly "
                "worse than an available action. Otherwise keep it. The prediction is "
                "not a real tool result. Return only a JSON object with keys reflection "
                "(string), decision (KEEP or REVISE), revise_probability (number 0 to 1), "
                "and final_action. final_action has content (string) and tool_calls "
                '(list of {"name": string, "arguments": object}). '
                "An empty tool_calls list is allowed for a response requesting clarification.\n"
                + json.dumps(
                    {
                        "draft_action": draft_action,
                        "predicted_next_state": predicted_state,
                        "available_tools": tools,
                    },
                    ensure_ascii=False,
                )
            )
            reflection = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=history + [{"role": "user", "content": instruction}],
                response_format={"type": "json_object"},
                extra_body=self.policy_extra_body,
            )
            responses.append(reflection)
            trace.update(
                draft_action=draft_action,
                pred_next_state_student=predicted_state,
                reflect_response=reflection.choices[0].message.content,
            )
            try:
                review = json.loads(reflection.choices[0].message.content or "")
                revised = validate_action(review["final_action"], tools)
                probability = review["revise_probability"]
                if (
                    type(probability) not in (float, int)
                    or not math.isfinite(probability)
                    or not 0 <= probability <= 1
                ):
                    raise ValueError("Invalid revise_probability")
                decision = review.get("decision")
                if decision not in ("KEEP", "REVISE"):
                    raise ValueError("Invalid reflection decision")
                # Upstream uses a textual confidence proxy, not token probabilities.
                confidence = 1.0
                use_revision = (
                    decision == "REVISE"
                    and probability > self.revise_threshold
                    and confidence > self.confidence_threshold
                    and revised != draft_action
                )
                trace.update(
                    revise_probability=probability,
                    reflection_confidence=confidence,
                    used_revised_action=use_revision,
                )
                if use_revision:
                    final = draft.model_copy(deep=True)
                    final.choices[0].message = message_from_action(revised)
            except (ValueError, KeyError, TypeError) as exc:
                trace["fallback_reason"] = str(exc)
        else:
            trace["flow_mode"] = "draft_only_fallback"

        trace.update(
            policy_model=self.model_name,
            world_model=self.wm_model,
            revise_threshold=self.revise_threshold,
            confidence_threshold=self.confidence_threshold,
            policy_extra_body=self.policy_extra_body,
            wm_extra_body=self.wm_extra_body,
            final_message=as_dict(final.choices[0].message),
        )
        inference_data["inference_input_log"] = {"message": history, "tools": tools}
        return (
            CoMAPResponse(
                final,
                trace,
                sum(
                    response.usage.prompt_tokens
                    for response in responses
                    if response.usage
                ),
                sum(
                    response.usage.completion_tokens
                    for response in responses
                    if response.usage
                ),
            ),
            time.monotonic() - started,
        )

    def _parse_query_response_FC(self, api_response: CoMAPResponse) -> dict:
        result = super()._parse_query_response_FC(api_response.response)
        result.update(
            input_token=api_response.input_tokens,
            output_token=api_response.output_tokens,
            reasoning_content=json.dumps(api_response.trace, ensure_ascii=False),
        )
        return result
