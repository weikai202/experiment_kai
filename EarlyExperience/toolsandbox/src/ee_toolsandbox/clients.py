"""Explicit endpoint/model configuration and usage accounting."""

import json
import os

from openai import OpenAI
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

from .protocol import canonical, parse_action, policy_messages, validate_action

REFLECTION_PROMPT = """Write a self-reflection explaining why the action the agent takes here is suitable.
Analyze the current situation and goal; compare the possible actions and their consequences;
justify the action using the supplied observed outcomes; identify relevant clues and constraints.
Stay strictly within the supplied information. Avoid meta-commentary about being an AI.
Use natural step-by-step reasoning and focus on logical decision making.
This text will be generated BEFORE acting at evaluation: reason anticipatorily, never claim to
have observed a future result, never call an action expert/correct, never reference numbered options.
The recorded outcomes are evidence for writing, not facts already available to the acting policy.
Return only the reflection prose, no action JSON, headings or reflection tags. Target 200-400 words,
with a soft maximum of 500 words."""


class ModelClient:
    def __init__(self, config, usage):
        self.model = config["model"]
        self.config, self.usage = config, usage
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=os.environ[config["api_key_env"]],
            timeout=config.get("timeout", 120),
            max_retries=0,
        )

    def complete(self, messages, temperature=0):
        request = dict(model=self.model, messages=messages, temperature=temperature)
        if "seed" in self.config:
            request["seed"] = self.config["seed"]
        if "enable_thinking" in self.config:
            request["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.config["enable_thinking"]
                }
            }
        attempt = {"model": self.model, "usage": None, "status": "dispatched"}
        self.usage.append(attempt)
        try:
            response = self.client.chat.completions.create(**request)
            attempt["usage"] = response.usage.model_dump() if response.usage else None
            choice = response.choices[0]
            content = choice.message.content
            if choice.finish_reason != "stop" or not content:
                raise RuntimeError(
                    "Incomplete model response; no training record emitted"
                )
            if self.config.get("enable_thinking") is False:
                fields = choice.message.model_dump()
                if (
                    fields.get("reasoning_content")
                    or fields.get("reasoning")
                    or "<think>" in content
                    or "</think>" in content
                ):
                    raise RuntimeError("Server did not honor non-thinking mode")
            attempt["status"] = "completed"
            return content
        except BaseException as error:
            attempt.update(status="failed", error_type=type(error).__name__)
            raise

    def action(self, state):
        return parse_action(self.complete(policy_messages(state)))

    def alternatives(self, state, expert, k):
        prompt = (
            "Propose "
            + str(k + 2)
            + " distinct alternative actions at this same state. "
            "Return a JSON array of action objects using the action schema below. "
            "Do not include the demonstrated action: " + canonical(expert)
        )
        # The expert label is only used offline for distinct alternatives, never in policy SFT inputs.
        response = self.complete(
            policy_messages(state) + [{"role": "user", "content": prompt}], 0.7
        )
        values = json.loads(response)
        if not isinstance(values, list):
            raise ValueError("Proposer must return an array")
        available = {t["function"]["name"] for t in state["tools"]}
        result, seen = [], {canonical(expert)}
        for value in values:
            action = validate_action(value)
            if any(c["name"] not in available for c in action.get("tool_calls", [])):
                raise ValueError(
                    "Proposer returned unavailable tool; batch needs inspection"
                )
            key = canonical(action)
            if key not in seen:
                seen.add(key)
                result.append(action)
        if len(result) < k:
            raise ValueError(
                f"Only {len(result)} distinct alternatives for requested K={k}; no silent fallback"
            )
        return result[:k]

    def reflect(self, record, alternative):
        payload = {
            "situation": record["state"],
            "action_taken": record["expert_action"],
            "observed_outcome": record["expert_observation"],
            "other_action_and_outcome": alternative,
        }
        text = self.complete(
            [
                {"role": "system", "content": REFLECTION_PROMPT},
                {"role": "user", "content": canonical(payload)},
            ],
            0.7,
        ).strip()
        if "<reflection>" in text or "</reflection>" in text:
            raise ValueError("Reflection includes protocol delimiters")
        return text


class UserSimulator(OpenAIAPIUser):
    def __init__(self, config, usage):
        self.model_name = config["model"]
        self.usage = usage
        self.openai_client = OpenAI(
            base_url=config["base_url"],
            api_key=os.environ[config["api_key_env"]],
            timeout=config.get("timeout", 120),
            max_retries=0,
        )

    def model_inference(self, openai_messages, openai_tools):
        attempt = {
            "model": self.model_name,
            "role": "user",
            "usage": None,
            "status": "dispatched",
        }
        self.usage.append(attempt)
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name, messages=openai_messages, tools=openai_tools
            )
            attempt["usage"] = response.usage.model_dump() if response.usage else None
            if response.choices[0].finish_reason not in ("stop", "tool_calls"):
                raise RuntimeError("Incomplete user response")
            attempt["status"] = "completed"
            return response
        except BaseException as error:
            attempt.update(status="failed", error_type=type(error).__name__)
            raise


class ReplayPolicy:
    """User-supplied expert action sequence; assert state hash when supplied."""

    def __init__(self, entries):
        self.entries, self.index = entries, 0

    def action(self, state):
        from .protocol import digest

        entry = self.entries[self.index]
        if "state_hash" in entry and entry["state_hash"] != digest(state):
            raise ValueError("Expert replay state differs from recorded state")
        self.index += 1
        return validate_action(entry["action"])
