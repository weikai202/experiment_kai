"""Last-completion-only SFT. Run in the separate training environment."""

import argparse
import hashlib
import json
from pathlib import Path

from .pipeline import read_jsonl, write_json


def encode_example(tokenizer, messages, max_length):
    if messages[-1]["role"] != "assistant":
        raise ValueError("Last message must be the supervised assistant completion")
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Chat template does not preserve assistant prefix; choose a compatible template"
        )
    if len(input_ids) > max_length:
        raise ValueError(
            f"Example has {len(input_ids)} tokens > max_length={max_length}; no silent truncation"
        )
    if len(input_ids) == len(prompt_ids):
        raise ValueError("Empty completion")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["il", "iwm", "sr"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--iwm-epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
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

    set_seed(args.seed)
    data, output = Path(args.data), Path(args.output)
    provenance = json.loads((data / "provenance.json").read_text())
    if not provenance["complete"] or provenance["source"]["split"] != "train":
        raise ValueError("Training requires complete train-only SFT provenance")
    if provenance.get("format", "ee-json-action-v1") != "ee-json-action-v1":
        raise ValueError("Only standalone EarlyExperience SFT is supported")
    for category, expected_hash in provenance.get("files", {}).items():
        if (
            hashlib.sha256((data / (category + "_sft.jsonl")).read_bytes()).hexdigest()
            != expected_hash
        ):
            raise ValueError("SFT data changed after preparation")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    model.config.use_cache = False
    stages = (
        [("iwm", ["iwm"], args.iwm_epochs), ("policy", ["expert"], args.epochs)]
        if args.method == "iwm"
        else [
            (
                "policy",
                ["expert", "reflection"] if args.method == "sr" else ["expert"],
                args.epochs,
            )
        ]
    )
    paths = sorted(
        {
            data / (category + "_sft.jsonl")
            for _, categories, _ in stages
            for category in categories
        }
    )
    write_json(
        output / "training_run.json",
        {
            "arguments": vars(args),
            "data_provenance": provenance,
            "files": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
            "complete": False,
        },
    )

    def collate(features):
        width = max(len(f["input_ids"]) for f in features)
        return {
            key: torch.tensor(
                [f[key] + [padding] * (width - len(f[key])) for f in features]
            )
            for key, padding in [
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]
        }

    # Preflight ALL stages before any optimizer step, including context lengths and missing SR.
    encoded = {}
    for path in paths:
        encoded[path.stem.removesuffix("_sft")] = [
            encode_example(tokenizer, r["messages"], args.max_length)
            for r in read_jsonl(path)
        ]
        if not encoded[path.stem.removesuffix("_sft")]:
            raise ValueError("Empty SFT category: " + str(path))
    for name, categories, epochs in stages:
        records = [record for category in categories for record in encoded[category]]
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(output / name),
                num_train_epochs=epochs,
                learning_rate=args.lr,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                seed=args.seed,
                data_seed=args.seed,
                bf16=args.bf16,
                report_to=[],
                save_strategy="no",
                logging_steps=1,
                remove_unused_columns=False,
                gradient_checkpointing=True,
            ),
            train_dataset=records,
            data_collator=collate,
        )
        trainer.train()
        trainer.save_model(str(output / name))
        tokenizer.save_pretrained(output / name)
        # Same in-memory parameters flow from IWM into imitation; optimizer resets at the stage boundary.
        model = trainer.model
        del trainer
    run = json.loads((output / "training_run.json").read_text())
    run["complete"] = True
    run["format"] = provenance.get("format", "ee-json-action-v1")
    run["enable_thinking"] = False
    write_json(output / "training_run.json", run)


if __name__ == "__main__":
    main()
