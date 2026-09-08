from types import SimpleNamespace as NS

import pytest

from comap_tau3.vllm_backend import VLLMBackend


def test_non_thinking_request_and_usage(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return NS(
            model="Qwen/Qwen3-32B",
            choices=[NS(message=NS(content="{}"))],
            usage=NS(prompt_tokens=9, completion_tokens=2),
        )

    backend = VLLMBackend({"model": "Qwen/Qwen3-32B", "base_url": "http://localhost:8000/v1"})
    monkeypatch.setattr(backend.client.chat.completions, "create", create)
    backend.set_seed(0)
    result = backend.generate(
        [{"role": "user", "content": "hello"}], max_tokens=32, temperature=0, phase="draft"
    )
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert captured["seed"] == 0 and captured["temperature"] == 0
    assert "top_p" not in captured
    assert (result.input_tokens, result.output_tokens) == (9, 2)


def test_thinking_output_rejected(monkeypatch):
    backend = VLLMBackend({"model": "Qwen/Qwen3-32B", "base_url": "http://localhost:8000/v1"})
    monkeypatch.setattr(
        backend.client.chat.completions,
        "create",
        lambda **kw: NS(
            model="Qwen/Qwen3-32B",
            choices=[NS(message=NS(content="<think>reason</think>{}"))],
            usage=None,
        ),
    )
    with pytest.raises(ValueError, match="thinking"):
        backend.generate([], max_tokens=32, temperature=0, phase="draft")
