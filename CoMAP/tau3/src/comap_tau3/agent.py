"""Draft -> student world model -> gated reflection, without environment peeking."""

import copy
import json
import math
from dataclasses import asdict

import jsonschema
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall

from .prompts import policy_prompt, reflect_prompt, world_prompt


def visible_message(message):
    """Allowlist: exclude evaluator state, user private tools, timing and model metadata."""
    if message.role == "tool" and message.requestor != "assistant":
        raise ValueError("User-private tool result cannot enter the agent")
    if message.role == "user" and getattr(message, "tool_calls", None):
        raise ValueError("User-private tool calls cannot enter the agent")
    result = {"role": message.role, "content": message.content}
    if getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ]
    if message.role == "tool":
        result.update(id=message.id, error=message.error)
    return result


def parse_json(text):
    # Permit code fences, but reject prose or multiple JSON objects.
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()

    def bad_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    value = json.loads(text, parse_constant=bad_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def validate_action(action, schemas):
    if not isinstance(action, dict):
        raise ValueError("Action must be an object")
    if action.get("type") == "message":
        if (
            set(action) != {"type", "content"}
            or not isinstance(action["content"], str)
            or not action["content"].strip()
        ):
            raise ValueError("Invalid message action")
    elif action.get("type") == "tool_calls":
        if (
            set(action) != {"type", "calls"}
            or not isinstance(action["calls"], list)
            or not action["calls"]
        ):
            raise ValueError("Invalid tool action")
        by_name = {
            schema["function"]["name"]: schema["function"]["parameters"] for schema in schemas
        }
        for call in action["calls"]:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise ValueError("Invalid tool call fields")
            if (
                not isinstance(call["name"], str)
                or call["name"] not in by_name
                or not isinstance(call["arguments"], dict)
            ):
                raise ValueError("Unknown tool or invalid arguments")
            schema = copy.deepcopy(by_name[call["name"]])
            schema["additionalProperties"] = False
            jsonschema.validate(call["arguments"], schema)
    else:
        raise ValueError("Unknown action type")
    return action


class CoMAPAgent(HalfDuplexAgent[list]):
    """Only receives tools/schema, policy and agent-visible messages."""

    def __init__(self, tools, domain_policy, policy_backend, world_backend, *, options=None):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.policy_backend = policy_backend
        self.world_backend = world_backend
        self.options = options or {}
        self.schemas = [tool.openai_schema for tool in tools]
        self.traces = []
        self.calls = []

    def set_seed(self, seed):
        for backend in (self.policy_backend, self.world_backend):
            if hasattr(backend, "set_seed"):
                backend.set_seed(seed)

    def get_init_state(self, message_history=None):
        self.traces = []
        self.calls = []
        return [visible_message(message) for message in (message_history or [])]

    def _call(self, backend, messages, phase, temperature):
        completion = backend.generate(
            messages,
            max_tokens=self.options.get(phase + "_max_tokens", 512),
            temperature=temperature,
            phase=phase,
        )
        self.calls.append({"phase": phase, "model": backend.model_id, **asdict(completion)})
        return completion.text

    def generate_next_message(self, message, state):
        incoming = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        observed = [visible_message(item) for item in incoming]
        if self.traces:
            self.traces[-1]["next_observation"] = copy.deepcopy(observed)
        state = copy.deepcopy(state) + observed
        history = copy.deepcopy(state)
        base = policy_prompt(self.domain_policy, self.schemas, history)
        temperature = self.options.get("temperature", 0.7)
        raw_draft = self._call(self.policy_backend, base, "draft", temperature)
        try:
            draft = validate_action(parse_json(raw_draft), self.schemas)
        except (ValueError, jsonschema.ValidationError):
            retry = base + [
                {
                    "role": "user",
                    "content": "Previous output was malformed. Return one valid action JSON object.\n"
                    "Previous output:\n" + raw_draft,
                }
            ]
            raw_draft = self._call(self.policy_backend, retry, "draft_retry", temperature)
            draft = validate_action(parse_json(raw_draft), self.schemas)
        prediction = self._call(
            self.world_backend,
            world_prompt(self.domain_policy, self.schemas, history, draft),
            "wm",
            0.0,
        )
        reflection_messages = reflect_prompt(base, draft, prediction)
        raw_reflection = self._call(
            self.policy_backend, reflection_messages, "reflect", temperature
        )
        candidate, probability, confidence, used, error = draft, 0.0, 0.0, False, None
        try:
            reflection = parse_json(raw_reflection)
            candidate = validate_action(reflection["action"], self.schemas)
            probability = reflection["revise_probability"]
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Invalid revision probability")
            if reflection.get("decision") not in {"KEEP", "REVISE"}:
                raise ValueError("Missing explicit KEEP/REVISE decision")
            # Upstream textual confidence proxy, not a calibrated probability.
            confidence = 1.0
            used = (
                reflection["decision"] == "REVISE"
                and candidate != draft
                and probability > self.options.get("revise_probability_threshold", 0.5)
                and confidence > self.options.get("reflection_confidence_threshold", 0.6)
            )
        except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as exc:
            error = type(exc).__name__
        final = candidate if used else draft
        step = len(self.traces)
        if final["type"] == "message":
            response = AssistantMessage(role="assistant", content=final["content"])
        else:
            response = AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id=f"comap_{step}_{index}", **call)
                    for index, call in enumerate(final["calls"])
                ],
            )
        self.traces.append(
            {
                "step_id": step,
                "policy": self.domain_policy,
                "tools": self.schemas,
                "history": history,
                "draft": draft,
                "raw_draft": raw_draft,
                "prediction": prediction,
                "raw_reflection": raw_reflection,
                "candidate": candidate,
                "revise_probability": probability,
                "reflection_confidence": confidence,
                "used_revised_action": used,
                "reflection_parse_error": error,
                "action": final,
                "next_observation": None,
                "policy_prompt": base,
                "reflection_prompt": reflection_messages,
            }
        )
        state.append(visible_message(response))
        return response, state
