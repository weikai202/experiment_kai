import hashlib
import json
import os
from pathlib import Path


def transport_messages(messages):
    """Qwen's native tool-role serialization, without invalid OpenAI tool names.

    The BFCL prompting handler names tools using Python call strings. Text-mode
    chat APIs reject those names; Qwen treats tool responses as wrapped user text.
    Merge consecutive responses exactly as its native template does.
    """
    result = []
    previous_tool = False
    for message in messages:
        if message["role"] == "tool":
            wrapped = "<tool_response>\n" + message["content"] + "\n</tool_response>"
            if previous_tool:
                result[-1]["content"] += "\n" + wrapped
            else:
                result.append({"role": "user", "content": wrapped})
            previous_tool = True
        else:
            result.append({"role": message["role"], "content": message["content"]})
            previous_tool = False
    return result


class Client:
    def __init__(self, config, cache):
        from openai import OpenAI
        if config.get("enable_thinking") is not False:
            raise ValueError("This experiment requires enable_thinking=false")
        self.config = config
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.api = OpenAI(base_url=config["base_url"],
                          api_key=os.environ.get(config.get("api_key_env", "EE_API_KEY"), "EMPTY"),
                          timeout=config.get("timeout", 600), max_retries=2)

    def complete(self, messages, *, temperature=None):
        request = {"model": self.config["model"], "messages": transport_messages(messages),
                   "temperature": self.config.get("temperature", 0.7) if temperature is None else temperature,
                   "top_p": self.config.get("top_p", 0.8),
                   "max_tokens": self.config.get("max_tokens", 8192),
                   "seed": self.config.get("seed", 42),
                   "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        key = hashlib.sha256(json.dumps({"request": request, "endpoint": self.config["base_url"],
                                         "revision": self.config.get("revision", "unspecified")},
                                        sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["text"]
        response = self.api.chat.completions.create(**request)
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise RuntimeError(f"Incomplete generation ({choice.finish_reason}); increase context/output limit")
        if getattr(choice.message, "reasoning_content", None):
            raise RuntimeError("Server returned thinking content despite non-thinking configuration")
        text = choice.message.content
        if not text or "<think>" in text or "</think>" in text:
            raise RuntimeError("Expected non-thinking text response")
        from .io import write_json
        write_json(path, {"text": text, "model": response.model,
                          "usage": response.usage.model_dump() if response.usage else None})
        return text

    def ask(self, instruction):
        return self.complete([{"role": "user", "content": instruction}])
