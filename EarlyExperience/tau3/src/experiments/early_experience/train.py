"""Full-parameter SFT: IL, IWM warm-up then IL, or expert+reflection SR."""

import argparse
from copy import deepcopy
from pathlib import Path

from .pipeline import read_jsonl


def hf_messages(messages):
    """HF tool templates expect dict arguments; API messages carry JSON strings."""
    import json

    result = deepcopy(messages)
    for msg in result:
        for call in msg.get("tool_calls", []):
            args = call["function"]["arguments"]
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
    return result


def encode_example(tokenizer, row, max_length):
    """Mask everything except the final completion; never truncate action tokens."""
    messages = hf_messages(row["messages"])
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT example must end with an assistant target")
    kwargs = {"tools": row["tools"]} if row.get("tools") else {}
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, **kwargs
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **kwargs
    )
    if not full.startswith(prompt):
        raise ValueError(
            "Chat template changes the assistant prefix; select a compatible template"
        )
    # Tokenize together: refuse boundary merges instead of supervising prompt tokens.
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
        raise ValueError("Invalid prompt/completion token boundary")
    if len(ids) > max_length:
        raise ValueError(
            f"Example has {len(ids)} tokens > {max_length}; increase max length (no silent truncation)"
        )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix) + ids[len(prefix) :],
    }


def stages(method, directory, epochs, iwm_epochs):
    """Return the exact baseline curriculum, each initialized independently."""
    directory = Path(directory)
    expert = read_jsonl(directory / "expert_sft.jsonl")
    if method == "il":
        return [("final", expert, epochs)]
    if method == "iwm":
        return [
            ("iwm_warmup", read_jsonl(directory / "iwm_sft.jsonl"), iwm_epochs),
            ("final", expert, epochs),
        ]
    if method == "sr":
        return [
            ("final", expert + read_jsonl(directory / "reflection_sft.jsonl"), epochs)
        ]
    raise ValueError(f"Unknown method: {method}")


def main():
    """Train with transformers Trainer; resume IWM weights into imitation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["il", "iwm", "sr"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--iwm-epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if Path(args.output).exists():
        raise ValueError("Use a new output directory for each baseline/run")
    if (
        min(
            args.epochs,
            args.iwm_epochs,
            args.batch_size,
            args.gradient_accumulation,
            args.max_length,
            args.learning_rate,
        )
        <= 0
    ):
        raise ValueError("Training parameters must be positive")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        raise ValueError("Model must supply a tool-compatible chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    model.config.use_cache = False

    def collate(examples):
        width = max(len(x["input_ids"]) for x in examples)
        return {
            key: torch.tensor(
                [x[key] + [pad] * (width - len(x[key])) for x in examples]
            )
            for key, pad in [
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]
        }

    for name, rows, epochs in stages(
        args.method, args.data, args.epochs, args.iwm_epochs
    ):
        if not rows:
            raise ValueError(f"Empty training stage: {name}")
        dataset = [encode_example(tokenizer, row, args.max_length) for row in rows]
        output = str(Path(args.output) / name)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=output,
                num_train_epochs=epochs,
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                bf16=args.bf16,
                seed=args.seed,
                data_seed=args.seed,
                save_strategy="epoch",
                logging_steps=1,
                report_to=[],
            ),
            train_dataset=dataset,
            data_collator=collate,
        )
        trainer.train()
        trainer.save_model(output)
        tokenizer.save_pretrained(output)
        # Same model weights; fresh optimizer for the next stage.
        del trainer
    import json

    Path(args.output, "training_config.json").write_text(
        json.dumps(vars(args), indent=2)
    )


if __name__ == "__main__":
    main()
