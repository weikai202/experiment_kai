"""Merge a trained LoRA adapter for use by a standard vLLM model server."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Merge destination must not already exist")
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, torch_dtype=torch.bfloat16,
                                                   device_map="cpu")
    model = model.merge_and_unload()
    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)


if __name__ == "__main__":
    main()
