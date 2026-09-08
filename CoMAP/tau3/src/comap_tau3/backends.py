"""Explicit API or local HF generation; API mode does not imply model training."""

import time
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    elapsed_seconds: float = 0.0


class APIBackend:
    def __init__(self, config):
        self.config = config
        self.model_id = config["model"]
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def generate(self, messages, *, max_tokens, temperature, phase):
        from tau2.data_model.message import SystemMessage, UserMessage
        from tau2.utils.llm_utils import generate

        classes = {"system": SystemMessage, "user": UserMessage}
        args = dict(self.config.get("args", {}))
        args.update(max_tokens=max_tokens, temperature=temperature)
        if self.seed is not None:
            args["seed"] = self.seed
        started = time.monotonic()
        result = generate(
            model=self.model_id,
            messages=[classes[m["role"]](**m) for m in messages],
            call_name="comap_" + phase,
            **args,
        )
        usage = result.usage or {}
        return Completion(
            result.content or "",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            result.cost,
            time.monotonic() - started,
        )


def load_hf(path, device="cuda", dtype="bfloat16", revision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=getattr(torch, dtype), revision=revision
    )
    return model.to(device), tokenizer


def prompt_ids(tokenizer, messages):
    # Same template for training and inference. No silent context truncation.
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


class HFBackend:
    def __init__(self, config):
        self.config = config
        self.model_id = config["model"]
        self.model, self.tokenizer = load_hf(
            self.model_id, config.get("device", "cuda"), config.get("dtype", "bfloat16")
        )
        self.model.eval()

    def set_seed(self, seed):
        from transformers import set_seed

        set_seed(seed)

    def generate(self, messages, *, max_tokens, temperature, phase):
        import torch

        ids = prompt_ids(self.tokenizer, messages)
        limit = self.config.get("max_context", self.model.config.max_position_embeddings)
        if len(ids) + max_tokens > limit:
            raise ValueError(f"{phase}: context overflow {len(ids)}+{max_tokens}>{limit}")
        inputs = torch.tensor([ids], device=self.model.device)
        args = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            args.update(temperature=temperature, top_p=self.config.get("top_p", 0.9))
        started = time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), **args)
        generated = outputs[0, len(ids) :]
        return Completion(
            self.tokenizer.decode(generated, skip_special_tokens=True),
            len(ids),
            len(generated),
            None,
            time.monotonic() - started,
        )


def make_backend(config):
    if not config.get("model") or config["model"].startswith("REQUIRED"):
        raise ValueError("Set an explicit model/checkpoint in the configuration")
    if config["backend"] == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(config)
    if config["backend"] == "api":
        return APIBackend(config)
    if config["backend"] == "hf":
        return HFBackend(config)
    raise ValueError("backend must be vllm, api or hf")
