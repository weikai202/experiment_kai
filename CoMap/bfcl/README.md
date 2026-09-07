# CoMAP on BFCL with Qwen3-32B

BFCL v4 inference baseline using **Qwen/Qwen3-32B for both policy model (PM)
and world model (WM)**. The two roles share a model server and use different
prompts. The loop is draft action -> predicted consequence -> gated reflection.
Only the selected action is executed by the official BFCL executor.

This is an inference adaptation, not a full CoMAP training reproduction.
It does not implement world-model self-distillation or policy SDPO updates.
Both roles initially use the unmodified backbone, not separately trained models.
No real-model benchmark scores are included. Tests use scripted model responses
with the official BFCL parser, tool executor and AST scorer.

## Sources

- [CoMAP](https://github.com/loyiv/CoMAP), revision
  `a5f6e2d2a0aff571f9a5441bf82a796a724ea94d`.
- [Official BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard),
  revision `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (v4).
- [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) and
  [Qwen vLLM documentation](https://qwen.readthedocs.io/en/latest/deployment/vllm.html).

See `NOTICE` for adaptation details and `LICENSE` for the upstream Apache 2.0 license.

## Setup

All commands below run from the **experiment_kai repository root**. Requires
Python 3.10+; the evaluator was tested on Python 3.10.21. BFCL is installed as
an external dependency; this repository does not duplicate its dataset or source.

```bash
git clone https://github.com/ShishirPatil/gorilla.git ../gorilla
git -C ../gorilla checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
uv venv --python 3.10 CoMap/bfcl/.venv
uv pip install --python CoMap/bfcl/.venv/bin/python \
  -e ../gorilla/berkeley-function-call-leaderboard \
  -r CoMap/bfcl/requirements.txt --torch-backend cpu
```

The CPU evaluator environment does not serve the model. Use a separate CUDA
environment with vLLM (the prepared cluster launch uses an existing vLLM 0.12.0
environment). `soundfile` covers an undeclared Qwen-agent import in BFCL.

## Serve The Backbone

On a GPU node with the CUDA vLLM environment activated:

```bash
bash CoMap/bfcl/serve_qwen3_32b.sh
```

Defaults: one 80GB GPU, BF16, 16384-token context, one concurrent sequence,
Hermes tool-call parser and localhost port 8000. Set `COMAP_TP_SIZE=2` for two
suitable GPUs, `COMAP_MODEL_PATH` for an existing local snapshot, or `VLLM_BIN`
for a specific vLLM executable. Missing weights are downloaded on first launch.

## Smoke Run And Scoring

In a second shell on the same GPU node:

```bash
source CoMap/bfcl/qwen3_32b.env.sh
CoMap/bfcl/.venv/bin/python -m CoMap.bfcl.smoke \
  --result-dir "$PWD/CoMap/bfcl/results/smoke-001"
CoMap/bfcl/.venv/bin/python -m CoMap.bfcl.run evaluate \
  --model "$COMAP_REGISTRY_NAME" --test-category simple_python,multi_turn_base \
  --result-dir "$PWD/CoMap/bfcl/results/smoke-001" \
  --score-dir "$PWD/CoMap/bfcl/scores/smoke-001" --partial-eval
```

This runs the first real entry of each category. `--count N` selects the first
N entries per category. Use a fresh result directory each time. Such small
samples validate execution, not statistically meaningful benchmark accuracy.

For a full category:

```bash
CoMap/bfcl/.venv/bin/python -m CoMap.bfcl.run generate \
  --model "$COMAP_REGISTRY_NAME" --test-category multi_turn_base \
  --temperature 0 --num-threads 1 --include-input-log \
  --result-dir "$PWD/CoMap/bfcl/results/full-001"
CoMap/bfcl/.venv/bin/python -m CoMap.bfcl.run evaluate \
  --model "$COMAP_REGISTRY_NAME" --test-category multi_turn_base \
  --result-dir "$PWD/CoMap/bfcl/results/full-001" \
  --score-dir "$PWD/CoMap/bfcl/scores/full-001"
```

Use the wrapper for both generation and evaluation to register the custom model.
BFCL's own `.env` has upstream precedence during CLI execution; check it if
endpoint settings do not take effect. Other BFCL categories may need additional
upstream resources and have not all been validated by this adaptation.

## Slurm

The provided script targets the WFU `gpu_small` partition with one A100-80GB,
8 CPUs, 128GB host RAM and a 1-hour limit. Adjust directives for another cluster.
It starts the model, runs smoke inference and official partial scoring, then
terminates its model server. Startup has a 20-minute timeout.

```bash
export VLLM_BIN=/path/to/cuda-environment/bin/vllm
sbatch CoMap/bfcl/qwen3_smoke.sbatch
```

Submit from the repository root. `COMAP_REPO_ROOT` and `COMAP_EVAL_PYTHON`
override repository and evaluator paths. Logs and results are local artifacts;
do not commit model weights, API keys or run outputs.

## Method And Configuration

The supplied profile disables thinking in all three calls and uses deterministic
temperature-0 decoding. This is a non-thinking experiment profile, not the Qwen
model card's recommended sampling recipe. PM and WM can later use separate
endpoints/checkpoints through `COMAP_WM_MODEL` and `COMAP_WM_BASE_URL`.

Only visible history, tools and the draft are supplied to WM. Predictions are
hypothetical and never become real tool observations or reveal hidden state.
Revision requires an explicit REVISE decision, probability > 0.5, textual
confidence > 0.6 and a changed action. This explicit decision gate is an addition
to upstream. Malformed reflections keep the draft; transport failures propagate.
Text-only drafts skip WM/reflection, matching the no-action fallback upstream.

`reasoning_content` stores the draft, prediction, reflection, selected action
and model options. Usage aggregates all calls; latency covers the entire step.
Combined tokens do not imply dollar costs when PM and WM prices differ.

Environment overrides include `COMAP_POLICY_MODEL`, `COMAP_WM_MODEL`,
`COMAP_REGISTRY_NAME`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `COMAP_WM_BASE_URL`,
`COMAP_WM_API_KEY`, `COMAP_REVISE_THRESHOLD`, `COMAP_CONFIDENCE_THRESHOLD`,
`COMAP_MAX_TOKENS`, `COMAP_WM_MAX_TOKENS`, `COMAP_TIMEOUT`,
`COMAP_POLICY_EXTRA_BODY` and `COMAP_WM_EXTRA_BODY` (JSON objects).
Use distinct registry names for different experiment configurations.

## Tests

```bash
CoMap/bfcl/.venv/bin/python -m pytest CoMap/bfcl/test_handler.py -q
```

Tests cover revision gates, malformed JSON, token accounting, parallel tool
calls, text replies, Qwen request options, real BFCL multi-turn execution and
official AST scoring. They require no GPU or API credentials.
