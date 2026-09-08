"""Qwen3 non-thinking through vLLM's OpenAI-compatible HTTP interface."""

import os
import time

from openai import OpenAI

from .backends import Completion


class VLLMBackend:
    def __init__(self, config):
        self.model_id = config["model"]
        self.seed = 0
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            timeout=config.get("timeout_seconds", 300),
            max_retries=0,
        )

    def set_seed(self, seed):
        self.seed = seed

    def generate(self, messages, *, max_tokens, temperature, phase):
        started = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            seed=self.seed,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        message = response.choices[0].message
        content = message.content or ""
        if (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or "<think>" in content
            or "</think>" in content
        ):
            raise ValueError("vLLM returned thinking output despite enable_thinking=false")
        if response.model != self.model_id:
            raise ValueError(f"Unexpected served model: {response.model}")
        usage = response.usage
        return Completion(
            content,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            None,
            time.monotonic() - started,
        )
