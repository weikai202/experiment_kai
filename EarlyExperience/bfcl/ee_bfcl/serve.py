"""Serve the pinned Hugging Face Qwen3-32B checkpoint with vLLM."""
import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"


def command(args, template):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL, "--revision", REVISION, "--tokenizer-revision", REVISION,
            "--served-model-name", MODEL, "--dtype", "bfloat16",
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--max-model-len", str(args.max_model_len), "--max-num-seqs", "4",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--host", args.host, "--port", str(args.port),
            "--chat-template", str(template), "--generation-config", "vllm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=Path("outputs/vllm"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tensor_parallel_size < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("Invalid parallel size or GPU memory utilization")
    if not 1 <= args.max_model_len <= 32768:
        parser.error("This baseline uses native context <=32768; longer contexts require an explicit YaRN experiment")
    template = args.output.resolve() / "qwen3_non_thinking.jinja"
    cmd = command(args, template)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if importlib.util.find_spec("vllm") is None:
        parser.error("vLLM is not installed in this Python environment. Install it in a dedicated GPU serving environment.")
    import torch
    if torch.cuda.device_count() < args.tensor_parallel_size:
        parser.error("Not enough visible GPUs for tensor parallelism; use an allocated GPU node")
    for i in range(args.tensor_parallel_size):
        free, total = torch.cuda.mem_get_info(i)
        if free < total * args.gpu_memory_utilization:
            parser.error(f"GPU {i} has only {free / 2**30:.1f} GiB free; requested "
                         f"{total * args.gpu_memory_utilization / 2**30:.1f} GiB. Use free allocated GPUs.")
    from huggingface_hub import hf_hub_download
    config = json.loads(Path(hf_hub_download(MODEL, "tokenizer_config.json", revision=REVISION)).read_text())
    # The original template and empty-thinking prefix stay identical to training.
    # Force the hard switch even when an external client omits or overrides it.
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{% set enable_thinking = false %}\n" + config["chat_template"])
    from .io import write_json
    write_json(args.output / "server.json", {"model": MODEL, "revision": REVISION,
               "enable_thinking": False, "command": cmd,
               "base_url": f"http://{args.host}:{args.port}/v1"})
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
