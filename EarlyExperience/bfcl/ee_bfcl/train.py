"""Final-assistant-only SFT; IWM stage 2 inherits stage 1 model weights."""
import argparse
import json
from pathlib import Path

from .client import transport_messages
from .io import digest, read_jsonl, write_json


def encode_example(tokenizer, messages, max_length):
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT example must end in its supervised assistant target")
    # Build the exact non-thinking inference prefix, including Qwen3's empty
    # <think></think> block. Do not let a full-chat template move the boundary.
    prefix = tokenizer.apply_chat_template(transport_messages(messages[:-1]), tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False, return_dict=False)
    completion = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")
    target = completion + [tokenizer.eos_token_id]
    if len(prefix) + len(target) > max_length:
        raise ValueError(f"Example needs {len(prefix) + len(target)} tokens, exceeding max_length={max_length}; "
                         "increase max_length or explicitly design a separate truncation experiment")
    return {"input_ids": prefix + target, "attention_mask": [1] * (len(prefix) + len(target)),
            "labels": [-100] * len(prefix) + target}


def plan(prepared, collected, output, model="Qwen/Qwen3-32B", allow_partial=False):
    prepared, collected, output = Path(prepared).resolve(), Path(collected).resolve(), Path(output).resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    report = json.loads((collected / "report.json").read_text())
    metadata = json.loads((collected / "collection.json").read_text())
    if metadata["manifest_sha256"] != digest(prepared / "manifest.json"):
        raise ValueError("Collection came from a different split")
    if report["partial"] and not allow_partial:
        raise ValueError("Partial smoke collection cannot be used as full baseline training data")
    expert = prepared / "expert_sft_text.jsonl"
    iwm = collected / "iwm_sft_text.jsonl"
    sr = collected / "reflection_sft_text.jsonl"
    for p in (expert, iwm, sr):
        if not read_jsonl(p):
            raise ValueError(f"Empty training corpus: {p}")
    stages = [
        {"name": "il", "model": model, "data": [str(expert)], "output": str(output / "il")},
        {"name": "iwm_stage1", "model": model, "data": [str(iwm)], "output": str(output / "iwm_stage1")},
        {"name": "iwm_il", "model": str(output / "iwm_stage1"), "data": [str(expert)], "output": str(output / "iwm_il")},
        {"name": "sr_il", "model": model, "data": [str(expert), str(sr)], "output": str(output / "sr_il")},
    ]
    result = {"stages": stages, "train_ids": manifest["train_ids"], "heldout_ids": manifest["heldout_ids"],
              "enable_thinking": False, "sr_mixing": "concatenate expert and SR once each, shuffle each epoch",
              "files_sha256": {str(p): digest(p) for p in (expert, iwm, sr)},
              "partial": report["partial"], "notes": "Default 1 epoch per stage; configurable, not a claim of original hyperparameter reproduction."}
    write_json(output / "training_plan.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--tokenizer", help="Optional tokenizer path, e.g. the original base model")
    parser.add_argument("--validate-only", action="store_true", help="Tokenize all data and validate masking, without loading model weights")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    adapter_file = Path(args.model) / "adapter_config.json"
    adapter_base = json.loads(adapter_file.read_text())["base_model_name_or_path"] if adapter_file.exists() else None
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or adapter_base or args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows = [r for p in args.data for r in read_jsonl(p)]
    if not rows:
        raise ValueError("No training examples")
    encoded = []
    for i, row in enumerate(rows):
        try:
            encoded.append(encode_example(tokenizer, row["messages"], args.max_length))
        except ValueError as error:
            raise ValueError(f"Training row {i}: {error}") from error
    stats = {"examples": len(encoded), "tokens": sum(len(r["input_ids"]) for r in encoded),
             "supervised_tokens": sum(sum(x != -100 for x in r["labels"]) for r in encoded),
             "max_tokens": max(len(r["input_ids"]) for r in encoded), "arguments": vars(args),
             "data_sha256": {p: digest(p) for p in args.data}, "enable_thinking": False}
    print(json.dumps(stats, indent=2))
    if args.validate_only:
        return
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Training output exists; use a new path or explicitly resume a checkpoint")
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(adapter_base or args.model, torch_dtype=torch.bfloat16)
    if adapter_base:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model, is_trainable=True)
    elif args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank,
                                               lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                               target_modules="all-linear"))
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    def collate(batch):
        longest = max(len(row["input_ids"]) for row in batch)
        pads = {"input_ids": tokenizer.pad_token_id, "attention_mask": 0, "labels": -100}
        return {key: torch.tensor([row[key] + [pad] * (longest - len(row[key])) for row in batch])
                for key, pad in pads.items()}

    training_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        logging_steps=1, save_strategy="epoch", save_total_limit=2,
        report_to="none", seed=args.seed, data_seed=args.seed, remove_unused_columns=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded, data_collator=collate)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)
        stats["world_size"] = training_args.world_size
        stats["effective_batch_size"] = args.batch_size * args.gradient_accumulation * training_args.world_size
        write_json(output / "training_metadata.json", stats)


if __name__ == "__main__":
    main()
