# Early Experience on BFCL - Qwen3-32B non-thinking

An independent implementation of Early Experience baselines using the Gorilla/BFCL v4 simulators and evaluation checkers.

| Baseline | Initialization | Training |
|---|---|---|
| IL | Qwen3-32B | Expert SFT |
| IWM -> IL | Qwen3-32B | Predict observed action outcomes, then continue expert SFT from the resulting checkpoint |
| SR + IL | Qwen3-32B | Mix expert examples with reflections grounded in actual environment feedback |

All model requests explicitly set `chat_template_kwargs.enable_thinking=false`. SR uses visible `<reflection>...</reflection>` training targets, separately from Qwen's built-in thinking mode. The executor parses only the Python function calls following the reflection.

References: [EarlyExperience](https://github.com/OSU-NLP-Group/EarlyExperience), [BFCL reproduction notes](https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/envs/bfcl_v4/README.md), and [the paper](https://arxiv.org/abs/2510.08558). This implementation covers **BFCL v4 multi-turn** tasks, not the entire leaderboard's single-turn, web-search, or memory categories.

## Validation status

The following results were obtained on the original development machine:

- Public Opus expert trajectories and official scores were downloaded and SHA256-verified.
- The 162 successful Base cases were split into 121 training and 41 held-out cases with seed 42.
- Preparation produced 1,196 expert training examples and 398 held-out examples. The three OOD categories never enter training data.
- All tool observations from the 162 successful cases were replayed with zero mismatches.
- The five-state collection integration test uses synthetic generator text and real simulator execution. These fixtures are not Qwen3-generated experiment data.
- No real Qwen3 model requests, 32B training runs, or model accuracy measurements have been completed. The existing development BFCL environment has CPU-only PyTorch; GPU training requires a separate environment.

Generated data, caches, tokenizer files, model weights, and validation reports are not included in Git. Tests requiring locally downloaded artifacts explicitly skip when those artifacts are absent.

## Installation

The paths below illustrate the original development setup. On another machine, replace the project path, `BFCL_PY`, and Gorilla installation path.

```bash
cd /path/to/experiment_kai/EarlyExperience/bfcl
BFCL_PY=/home/weik/CoMAP/.venv-bfcl/bin/python
"$BFCL_PY" -m ee_bfcl --help
"$BFCL_PY" -m pytest -q
```

Create a separate training environment to avoid changing CoMAP dependencies. Python 3.10 is recommended. Install a PyTorch build compatible with your cluster's CUDA configuration.

```bash
python3.10 -m venv .venv-train
.venv-train/bin/pip install -e /home/weik/gorilla/berkeley-function-call-leaderboard
.venv-train/bin/pip install -e '.[train,test]'
```

The validated BFCL commit is `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. Check out that commit before installing Gorilla on a new machine. It differs from the December 2025 version requested by the reference repository. This project adapts the method to Qwen3; it does not claim to reproduce the reference Qwen2.5 scores. Runtime checks verify the data, simulator, and evaluator fingerprints recorded during preparation.

## 1. Prepare expert data

The original development machine already has `data/prepared`. For a fresh clone, download the public expert inputs and prepare the data below. Use a new output directory when repeating preparation.

```bash
"$BFCL_PY" -m ee_bfcl.download --output data/source
"$BFCL_PY" -m ee_bfcl prepare \
  --results data/source/opus_base_result.jsonl \
  --scores data/source/opus_base_score.jsonl \
  --output data/prepared --seed 42
```

`manifest.json` records the exact case split, input hashes, BFCL commit, and source hashes. The official score file contains an aggregate header and failed cases only. The parser validates completeness before deriving the successful set; missing scores are not assumed to indicate success.

`expert_records.jsonl` preserves case IDs, turns, actions, and observations. `expert_sft_text.jsonl` uses the standard `{"messages": [...]}` schema. Held-out examples are stored separately and are not used by the training commands. Each user turn receives a `[]` stop example, and the expert's natural-language closing response is normalized to `[]` consistently across all three baselines. Parallel expert calls remain one action.

## 2. Configure Qwen3 and collect IWM/SR data

Edit `base_url` and `model` in `configs/qwen3_32b.json`. Set `revision` to identify the actual served checkpoint. Authentication uses the `EE_API_KEY` environment variable; the configuration does not store credentials. The server must support vLLM's `chat_template_kwargs` parameter.

A basic vLLM example follows. Adjust GPU count and context length for your hardware, or use the pinned Hugging Face launcher documented below.

```bash
vllm serve /path/to/Qwen3-32B \
  --served-model-name Qwen/Qwen3-32B \
  --tensor-parallel-size 4 --max-model-len 32768 --port 8000
```

Preview collection without making model requests:

```bash
"$BFCL_PY" -m ee_bfcl collect --prepared data/prepared \
  --output outputs/smoke --config configs/qwen3_32b.json --limit 5
```

Run a real five-state smoke test: 65 logical requests before retries, with actual token usage recorded in the API cache.

```bash
"$BFCL_PY" -m ee_bfcl collect --prepared data/prepared \
  --output outputs/smoke --config configs/qwen3_32b.json --limit 5 --execute
```

Inspect candidate actions, actual outputs, summaries, and reflections in `outputs/smoke/states/*.json`, then collect the full dataset:

```bash
"$BFCL_PY" -m ee_bfcl collect --prepared data/prepared \
  --output outputs/full --config configs/qwen3_32b.json --filter-leaks --execute
```

At each training action state, sample K=10 distinct names from public, documented simulator methods, excluding names in the expert action. One model request fills arguments for all candidates. Each candidate executes on an independent `deepcopy`. Execution errors are retained; next states come from actual execution. Descriptive summaries are generated afterward, and K=3 candidates are used for SR comparisons.

IWM uses a separate world-model system prompt and targets starting with `Observation:`. IL and SR use the action format. SR targets append the original expert action deterministically. By default, reflection vocabulary leakage is audited only. `--filter-leaks` enables the reference repository's vocabulary filter and records discarded counts in `reflection_audit.jsonl` and `report.json`. It does not remove IWM error observations.

Per-state outputs and API responses support cached resumption. Use a new output directory if the model, split, filtering options, or limit changes. Update `revision` when replacing a checkpoint under the same served model name. API timeouts and truncated generations raise errors while preserving progress; truncated text is not accepted as training data.

## 3. Train the baselines

Generate a training plan with input hashes and stage dependencies. Partial smoke collections are rejected for full experiment training by default.

```bash
"$BFCL_PY" -m ee_bfcl plan-training --prepared data/prepared \
  --collected outputs/full --output checkpoints
```

Run the following four stages in the training environment. Full-parameter 32B training generally requires sharding; use your cluster's Accelerate/FSDP/DeepSpeed launch configuration, replacing `python -m` with the corresponding `accelerate launch -m` invocation. The defaults are batch size 1 per device and gradient accumulation 32. Effective batch size also includes the number of processes; explicitly keep it consistent across comparisons.

```bash
# IL
python -m ee_bfcl.train --model Qwen/Qwen3-32B \
  --data data/prepared/expert_sft_text.jsonl --output checkpoints/il

# IWM stage 1
python -m ee_bfcl.train --model Qwen/Qwen3-32B \
  --data outputs/full/iwm_sft_text.jsonl --output checkpoints/iwm_stage1

# IWM stage 2: inherit stage 1 weights
python -m ee_bfcl.train --model checkpoints/iwm_stage1 \
  --data data/prepared/expert_sft_text.jsonl --output checkpoints/iwm_il

# SR + IL: initialize from the same base model
python -m ee_bfcl.train --model Qwen/Qwen3-32B \
  --data data/prepared/expert_sft_text.jsonl outputs/full/reflection_sft_text.jsonl \
  --output checkpoints/sr_il
```

Defaults are one epoch per stage, learning rate 1e-5, a cosine schedule, and warmup ratio 0.03. These are configurable starting points, not a claim of matching the paper's hyperparameters. Only the final assistant target contributes to the loss; history and padding are masked. The native non-thinking prefix is masked as part of the prompt. SR reflection text and action tokens are both supervised. Overlong examples raise an error instead of being silently truncated or dropped.

Validate real token lengths without loading model weights:

```bash
python -m ee_bfcl.train --model Qwen/Qwen3-32B \
  --data data/prepared/expert_sft_text.jsonl --output checkpoints/il --validate-only
```

For LoRA training, add `--lora --lora-rank 32` to the three commands that start from the base model. IWM stage 2 automatically detects and continues the stage 1 adapter. Use the same training approach across baselines. LoRA is an additional experimental choice and should be reported explicitly. To create a merged checkpoint:

```bash
python -m ee_bfcl.merge_adapter --adapter checkpoints/iwm_il --output checkpoints/iwm_il_merged
```

Merging a 32B adapter requires sufficient CPU RAM. Resume interrupted training with `--resume-from-checkpoint /path/to/checkpoint-N`.

## 4. Evaluate on BFCL

Serve each final checkpoint separately. Copy the JSON configuration and set that baseline's `model`, `base_url`, and unique `revision`. Training and evaluation use BFCL's Python text-call protocol, not Qwen's native XML/JSON function-calling protocol.

```bash
"$BFCL_PY" -m ee_bfcl evaluate --prepared data/prepared \
  --output outputs/eval_il/base --config configs/il_eval.json \
  --category multi_turn_base --execute

for category in multi_turn_long_context multi_turn_miss_func multi_turn_miss_param; do
  "$BFCL_PY" -m ee_bfcl evaluate --prepared data/prepared \
    --output "outputs/eval_il/$category" --config configs/il_eval.json \
    --category "$category" --execute
done
```

Create `configs/il_eval.json` by copying and editing `configs/qwen3_32b.json`; use separate configurations and output directories for IWM and SR. Omit `--execute` to preview case counts and request limits. Use `--limit 1` and a separate output directory for a service smoke test.

Base evaluation includes only the 41 held-out cases. Each of the other three categories uses its complete OOD set. Missing functions are introduced into the prompt only at the designated turn. Evaluation calls the official `multi_turn_checker` and `multi_turn_irrelevance_checker`, comparing actual states and responses rather than exact call strings. Model and reference actions use the same AST literal-only executor instead of unrestricted Python evaluation.

`results.jsonl` stores raw outputs, tool feedback, decoded calls, and official scoring details for each case. `metrics.json` stores category accuracy. Report categories separately; these results are not the complete BFCL leaderboard overall score. Long-context OOD tasks may require a larger serving context. If the server rejects an overlong request, adjust the configuration and rerun in a new output directory rather than truncating the task and claiming complete evaluation.

## Source layout

- `prepare.py`: expert harvesting, case-level splitting, and observation replay checks.
- `environment.py`: simulator copies, action parsing, and official scorer integration.
- `collect.py`: candidate actions, outcome summaries, IWM/SR data, and leakage auditing.
- `client.py`: non-thinking requests, caching, and equivalent Qwen tool-response formatting.
- `train.py`: IL, two-stage IWM, mixed SR training, and final-assistant loss masking.
- `evaluate.py`: multi-turn interaction, delayed function prompts, scoring, and resumption.
- `tests/test_baseline.py`: simulator integration, fixture-based collection, tokenizer, and isolation checks.

For fair comparisons, share the manifest's case split, tool prompts, stopping rules, inference budget, and scoring interface across your method and these baselines. Do not train on Base cases and then report accuracy over the entire Base dataset.

## Hugging Face to vLLM launcher

The launcher pins `Qwen/Qwen3-32B` to Hugging Face revision `9216db5781bf21249d130ec9da846c4624c16137`. Create a dedicated serving environment on an allocated, available GPU node. Install vLLM with a compatible GPU driver/CUDA combination. The referenced Qwen deployment guide specifies vLLM >=0.8.5; keep its PyTorch dependencies separate from BFCL and training environments.

```bash
# Run from the project root in the Python environment containing vLLM.
python -m ee_bfcl.serve --dry-run
python -m ee_bfcl.serve
```

The launcher downloads weights using the Hugging Face model ID. Defaults are BF16, tensor parallelism 4, context length 32768, and port 8000. It builds a forced non-thinking template from the same pinned revision. `configs/qwen3_32b.json` connects to `http://localhost:8000/v1`; requests also explicitly disable thinking. Native function-call parsing remains disabled to preserve the BFCL text protocol.

Adjust GPU count with `--tensor-parallel-size`. Insufficient free GPU memory raises an error before weight download. When serving on another node, forward port 8000 over SSH or explicitly configure the server host and client base URL.

Once the service is ready, run a real five-state smoke test:

```bash
/home/weik/CoMAP/.venv-bfcl/bin/python -m ee_bfcl collect \
  --prepared data/prepared --output outputs/qwen3_hf_smoke \
  --config configs/qwen3_32b.json --limit 5 --execute
```

The serving setup follows the [Qwen vLLM deployment guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md). During the recorded development check, all four L40S GPUs were occupied, so the service was not started and 32B weights were not downloaded. Local paths, prepared data, and test reports describe that development environment and are not bundled repository artifacts.
