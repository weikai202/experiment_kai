# Early Experience on tau3-bench

This module implements the IL, IWM -> IL, and SR baselines from
[EarlyExperience](https://github.com/OSU-NLP-Group/EarlyExperience) for
**training followed by evaluation on unseen tasks**.
Pinned tau3 source revision: `17e07b1da2bbc0cadfddeea36412686e0604127b`.

## Methods and scope

- **IL**: expert state -> expert action.
- **IWM**: execute K distinct candidate actions at expert states and train next-observation
  prediction on their actual outcomes. Then perform IL using **the same model weights**.
  IWM and IL have separate system prompts and target formats.
- **SR**: the base model reads actual outcomes of expert and alternative actions and
  generates one reflection per alternative. Train on reflection-plus-expert-action
  targets mixed with expert examples.
- Supports `retail`, `airline`, `telecom`, and `mock` for offline tests, in text
  half-duplex mode only. **`banking_knowledge`, voice/full-duplex, and solo/GT agents
  are not supported.** Knowledge retrieval environments use external resources;
  `deepcopy` cannot be assumed to isolate their state.
- At the time of implementation, upstream EE had not released its tau-bench adapter.
  This is a tau3 adaptation based on its method descriptions, not an author-released
  tau3 result or a guarantee of reproducing the paper's reported scores.

Implementation choices: K=3 by default. One request proposes multiple candidates,
which are deduplicated; this is batched candidate proposal rather than K independent
policy samples. Expert next observations come from demonstration trajectories, with
additional replay verification for tool outcomes. Alternative conversational actions
call a user simulator with the same configuration, so user responses remain stochastic.
All tool-error responses are retained. Demonstrations are not automatically filtered
by reward; experimenters should verify the quality of imported trajectories first.
If a stronger model generates alternatives or reflections, report the setting as
teacher-assisted EE. The paper-style self setting uses the same base model that will
be trained. Reflection prompts request 200-400 words, with a soft limit of 500 words.

## Installation

Run from the upstream repository root. The Python package remains `tau2`, following
the upstream tau3 naming convention.

```bash
uv sync --extra dev --extra voice --extra knowledge
# Building PyAudio requires a system PortAudio development package,
# such as portaudio-devel or portaudio19-dev.
# Training dependencies; choose the PyTorch CUDA build for your training machine:
uv pip install 'transformers==4.57.1' 'accelerate==1.11.0' torch
.venv/bin/python -m experiments.early_experience --help
```

This upstream revision imports some voice/knowledge modules even through the text
entry point, so `uv sync` alone is insufficient. Use `.venv/bin/python` to avoid a
later `uv run` synchronization removing additional training dependencies.
Configure API keys through environment variables, not configuration files or
`--generator-args`. This module does not launch a model server automatically; use
LiteLLM-supported model names and `api_base` to connect to your endpoints.

## 1. Fix the task split

```bash
.venv/bin/python -m experiments.early_experience split \
  --domain retail --train-split train --eval-split test --output retail_split.json
```

Alternatively, provide `{"domain":"retail","train_ids":["..."],"eval_ids":["..."]}`.
Both ID lists must be nonempty, contain no duplicates, and have no overlap. After
sampling or training on `train`, the full `base` set containing those training tasks
cannot serve as a leakage-free test set. Keep splits, base models, seeds, user
simulators, and trial counts fixed across methods.

## 2. Collect or import expert demonstrations

Create `teacher.json`. The model names below are placeholders; replace them with
the models used in your experiment.

```json
{
  "domain": "retail",
  "agent": "llm_agent",
  "llm_agent": "openai/teacher",
  "llm_args_agent": {"api_base": "http://localhost:8000/v1", "temperature": 0},
  "llm_user": "openai/user-model",
  "llm_args_user": {"api_base": "http://localhost:8001/v1", "temperature": 0},
  "num_trials": 1,
  "max_steps": 100,
  "seed": 42
}
```

```bash
.venv/bin/python -m experiments.early_experience collect \
  --config teacher.json --split retail_split.json --output teacher_results.json
```

This command calls real models and the native evaluator and can incur charges.
Existing native `Results` JSON can be used directly in the next step, provided it
contains only training tasks and uses ordinary `llm_agent` / `ee_agent` and
`user_simulator` implementations. GT agents exposing task `evaluation_criteria`
as visible input are rejected.
Initial scripted history and the default greeting are context only. Every subsequent
assistant decision requires an actual next observation. Missing outcomes or
non-replayable trajectories raise errors instead of being silently ignored.

## 3. Sample real branches, generate reflections, and export

Start with a five-state smoke run. Inspect the actual actions, outcomes, and
reflection text in `transitions.jsonl`.

```bash
.venv/bin/python -m experiments.early_experience generate \
  --results teacher_results.json --split retail_split.json --output ee_retail \
  --generator-model openai/base-policy \
  --generator-args '{"api_base":"http://localhost:8002/v1","temperature":0.7}' \
  --k 3 --max-states 5
.venv/bin/python -m experiments.early_experience export --data ee_retail
```

Remove `--max-states` and rerun to continue. Each completed state is flushed to disk,
and completed states are skipped on restart. Resume is rejected if the configuration
or source-file hash differs. An interrupted state must be regenerated. If a file
contains a partial line, JSON parsing raises an error; back up the file before
repairing its incomplete final line.

Each state requires approximately one candidate-generation call and K reflection
calls, plus user-simulation calls for alternative conversational actions.
`--max-states` limits newly added states in the current invocation. Estimate full-run
costs from smoke-run usage. Fewer than K distinct candidates, truncated generation,
or an excessive user-tool loop causes an explicit error.

Outputs are `expert_sft.jsonl`, `iwm_sft.jsonl`, and `reflection_sft.jsonl`, with
`transitions.jsonl` and `manifest.json` for auditing. SFT preserves native multi-turn
messages and structured `tool_calls`; tool schemas are stored in the top-level
`tools` field. IWM predicts the next agent-visible observation, not the hidden DB.

## 4. Train the three baselines

Start every method from **the same original base checkpoint**. The IWM command
runs both stages internally.

```bash
for method in il iwm sr; do
  .venv/bin/python -m experiments.early_experience.train \
    --method "$method" --model /path/to/base-model --data ee_retail \
    --output "checkpoints/$method" --epochs 1 --iwm-epochs 1 \
    --learning-rate 2e-5 --batch-size 1 --gradient-accumulation 32 \
    --max-length 16384 --bf16 --seed 42
done
```

This is full-parameter SFT. The model must provide a tool-compatible Hugging Face
chat template. Loss is computed only on each example's final assistant completion;
history, user messages, and tool outputs are masked. Overlong examples raise errors
to avoid truncating action targets. IWM saves its first stage to `iwm_warmup`, then
continues the same model weights with a fresh optimizer. Deployable final models
are saved to each run's `final` directory.
Default epochs and learning rate are configurable starting points, not paper-recommended
tau3 hyperparameters. SR uses one IL example plus K SR examples per state by default;
IWM adds a warm-up stage. Report their additional training volume in the experiment.

## 5. Deploy and evaluate with native tau3

Deploy `checkpoints/<method>/final` to an OpenAI-compatible service supporting the
model's tool-call parser. In `eval.json`, using the same structure as `teacher.json`,
point `llm_agent` and `api_base` to that service. Fix `llm_user`, user parameters,
and the seed, and set `num_trials`, for example to 4.

```bash
.venv/bin/python -m experiments.early_experience evaluate \
  --config eval.json --split retail_split.json \
  --manifest ee_retail/manifest.json --output il_results.json
```

For IWM/SR, change the endpoint and output file while keeping other conditions fixed.
The command enforces test IDs and validates the data manifest. It uses the native
runner/evaluator to produce native Results and metrics. Tau-bench's `pass^k` measures
consistency across k successful trials, unlike the at-least-one-success `pass@k` metric.
This module does not redefine rewards or success criteria.

All three methods use the same `ee_agent` prompt. SR `<think>...</think>` text is
removed before tool execution or delivery to the user. Subsequent history also
omits private reflections, matching the visible history used during data generation.
Unclosed tags raise an error. The inference server must preserve generated tags or
parse reasoning according to the model's conventions, and correctly parse tool calls.

## Offline validation

```bash
.venv/bin/python -m pytest src/experiments/early_experience/tests -q
.venv/bin/ruff check src/experiments/early_experience
.venv/bin/ruff format --check src/experiments/early_experience
```

Tests use real mock-environment tools and cover isolation, within-batch call order,
error retention, private-information visibility, reflection/action formatting,
training masks, task splits, and all three curricula. Tests do not call paid APIs.

References: [method definitions](https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/skill/METHOD.md),
[implementation notes](https://github.com/OSU-NLP-Group/EarlyExperience/blob/main/skill/method_recap.md),
and [paper](https://arxiv.org/abs/2510.08558).

### Recorded implementation validation

- Python 3.12.14; 14 data/environment/native-evaluator tests passed.
- Real retail, airline, and telecom lookup tools were executed and verified in isolated copies.
- Three randomly initialized tiny-model CPU training tests passed, covering IL,
  two-stage IWM weight updates, and mixed SR training.
- Ruff lint/format passed. The native retail train/test split contains 74/40 tasks with no overlap.
- No real generator-model data collection, full-scale base-model training, or official
  benchmark scoring has been run.
