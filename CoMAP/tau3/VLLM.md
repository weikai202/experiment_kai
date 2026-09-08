# Qwen3-32B non-thinking with vLLM

The active configuration is `configs/pipeline_aligned.yaml`:

- Source: `Qwen/Qwen3-32B` on Hugging Face.
- Initial weight/tokenizer revision: `9216db5781bf21249d130ec9da846c4624c16137`.
- vLLM OpenAI-compatible endpoint: `http://127.0.0.1:8000/v1`.
- Both roles initially use the same pretrained model; their prompts differ.
- Each request explicitly sends `chat_template_kwargs.enable_thinking=false`,
  temperature 0 and seed 0 during evaluation. `top_p` is omitted.
- The client rejects returned thinking content and a mismatched served model ID.
- No SDK automatic retries are enabled; reported token usage is retained.

[Qwen's official vLLM guide](https://qwen.readthedocs.io/en/stable/deployment/vllm.html)
describes the non-thinking request parameter.

## Start on allocated GPUs

Use a separate vLLM environment compatible with the host CUDA/driver. Its version
must be recorded and fixed for both methods. The baseline CPU test environment
does not install or run vLLM. The script uses the flags documented in
[the vLLM serve reference](https://docs.vllm.ai/en/v0.16.0/cli/serve/).

```bash
cd experiment_kai/CoMAP/tau3
CUDA_VISIBLE_DEVICES=0,1,2,3 TENSOR_PARALLEL_SIZE=4 \
  bash scripts/serve_qwen3.sh
```

This downloads the pinned Hugging Face weights when not cached. The script uses
BF16, a 16384-token context limit, and vLLM generation defaults rather than
unrecorded model-repository sampling defaults. Change GPU selection to the actual
allocation. These are launch defaults, not a measurement of available capacity.

Then run a development rollout with the baseline environment:

```bash
uv run comap-tau3 run \
  --config configs/pipeline_aligned.yaml \
  --manifest configs/splits.official178.json \
  --domain airline --split train --round 0 --limit 1 \
  --output outputs/vllm-airline-smoke
```

The user simulator still needs `OPENAI_API_KEY`. Local vLLM optionally uses the
separate `VLLM_API_KEY` environment variable. No keys go in saved configuration.

## Training and subsequent rounds

vLLM handles inference; gradients still require HF/PyTorch training. Each role's
`checkpoint` and `revision` fields identify training weights independently of
its served `model` name and `base_url`.

`train --kind policy` and `train --kind world_model` can read this vLLM config.
For example, after a complete round-0 rollout:

```bash
uv run --extra train comap-tau3 train \
  --config configs/pipeline_aligned.yaml --manifest configs/splits.official178.json \
  --domain airline --round 0 --kind world_model \
  --transitions outputs/round0/transitions.jsonl --output outputs/wm-round0
```

Train the policy analogously using `--kind policy` and a separate output directory.
After updates, serve the **new student checkpoints** under distinct endpoint/model
names and update the next round config's `checkpoint`, `model`, `base_url` and
clear `revision` for local saved checkpoints. A launch example:

```bash
SERVED_MODEL_NAME=comap-policy-r1 VLLM_PORT=8000 \
  bash scripts/serve_qwen3.sh /absolute/path/to/policy-round0/student
```

Deploy the world-model checkpoint independently, e.g. under port 8001 and served
name `comap-wm-r1`. Initial shared serving must not continue after weights diverge.

**Current limitation:** `evolve` automatically manages only the original HF
inference workflow. vLLM endpoint lifecycle/reloading is explicit, using staged
`run`/`train` commands; automatic three-round vLLM deployment is not implemented.
The current full-parameter single-device reference trainer is tested on tiny
models, not on 32B. Four occupied L40S GPUs do not establish that 32B full-parameter
training will fit; a distributed/offloaded training implementation or larger
training allocation is still needed for a practical full-size reproduction.

No vLLM server or 32B training job was started by this configuration change.
