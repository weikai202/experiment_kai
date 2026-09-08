# CoMAP on τ³ core text

**Active model setup: Qwen3-32B non-thinking via vLLM. See [VLLM.md](VLLM.md)
for the pinned Hugging Face revision, serving commands and staged training workflow.
The automatic `evolve` commands below apply to `configs/pipeline_aligned.hf.yaml`;
vLLM server lifecycle is not automatically managed.**

This is a **CoMAP adaptation**, not a claim of reproducing the authors' ALFWorld
numbers. It implements draft → learned world-model prediction → gated reflection,
real on-policy transition collection, world-model CE + WMSD, and policy
feedback-conditioned self-distillation with auxiliary supervision. The native τ³
orchestrator, tools, user simulator and evaluator remain unchanged.

Pinned sources:

- τ³ v1.0.0: `17e07b1da2bbc0cadfddeea36412686e0604127b` in
  [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench/tree/17e07b1da2bbc0cadfddeea36412686e0604127b).
- CoMAP: `a5f6e2d2a0aff571f9a5441bf82a796a724ea94d` in
  [loyiv/CoMAP](https://github.com/loyiv/CoMAP/tree/a5f6e2d2a0aff571f9a5441bf82a796a724ea94d).

## Experimental partition (current confirmed protocol)

All **178 official training tasks** are used for offline updates. No development
holdout is reserved. Final evaluation uses the 100 official held-out tasks.

| Domain | Official train | Baseline round 0 / 1 / 2 | Dev | Official test |
|---|---:|---:|---:|---:|
| Airline | 30 | 10 / 10 / 10 | 0 | 20 |
| Retail | 74 | 25 / 25 / 24 | 0 | 40 |
| Telecom | 74 | 25 / 25 / 24 | 0 | 40 |
| Total | 178 | 60 / 60 / 58 | 0 | 100 |

`configs/splits.official178.json` uses the exact official train/test IDs. The
baseline partitions the official training list into three contiguous, balanced,
non-overlapping shards. **This round schedule is an implementation choice, not
an observed pipeline round manifest.** Aggregate train/test membership matches
the confirmed protocol; round-level exposure parity remains to be verified if
the pipeline reuses tasks or follows another schedule.

The older `configs/splits.candidate.seed42.json` and `heldout_dev` protocol remain
for reproducibility only and must not be used for this comparison. They implement
a different, superseded 144-train/34-dev experiment.

Mock is only for sanity checks. Banking Knowledge and voice are excluded.
Reference-action statistics supplied in the experiment description are not
model inputs and were not independently recomputed from held-out task content.

## Confirmed model alignment

Use `configs/pipeline_aligned.yaml` with `configs/splits.official178.json`.
Policy and world model independently start from Qwen/Qwen3-32B; evaluation uses
temperature 0, seed 0, and HF non-thinking chat formatting. CoMAP updates its
weights while the original pipeline can evolve external artifacts: this is a
method difference, not a reason to freeze CoMAP's weights.

The user simulator is `openai/gpt-4o-mini-2024-07-18`. Its temperature, top_p and
seed are omitted. `PipelineUserSimulator` retains native tau prompts/tool behavior
but explicitly overrides the seed hook so the orchestrator cannot inject a
provider seed. Remote simulator outputs are not claimed to be deterministic.

Before execution, resolve both Qwen model entries to the same **pinned local
checkpoint/tokenizer revision**. The model names alone do not lock a revision.
Step/error limits, output/context budgets and trial count are still explicitly
marked baseline defaults. The HF JSON protocol remains different from the
pipeline's vLLM structured-output interface; strict request-level parity is not
claimed. The policy training temperature stays 0.7 for on-policy sampling.

## Install

Expected directory layout:

```text
<workspace>/experiment_kai/CoMAP/tau3/      # this package
<workspace>/tau2-bench/     # clean checkout at the pinned commit
```

From the workspace directory, prepare the separate benchmark checkout:

```bash
git clone https://github.com/sierra-research/tau2-bench.git tau2-bench
git -C tau2-bench checkout --detach 17e07b1da2bbc0cadfddeea36412686e0604127b
```

The benchmark lives alongside `experiment_kai`, outside this repository. Run
all baseline commands from `experiment_kai/CoMAP/tau3`; configuration `tau_root`
and the editable dependency both resolve to `../../../tau2-bench`.

The relative `tool.uv.sources.tau2.path` in `pyproject.toml` refers to the second
checkout and installs it editable. The runner checks both the imported package
location and clean Git HEAD before execution. Keep τ³ read-only.

```bash
cd experiment_kai/CoMAP/tau3
uv sync --extra dev --extra train
uv run comap-tau3 validate-splits \
  --tau-root ../../../tau2-bench \
  --manifest configs/splits.official178.json
uv run --extra train --extra dev pytest -q
```

The pinned τ³ release eagerly imports voice and knowledge modules even for text
runs. Its optional dependency groups are therefore declared for import
compatibility; they do not enable those experiments. Python 3.13 also needs
`audioop-lts`. PyAudio needs PortAudio development headers and a linkable library
installed in the build environment. Host-specific `.native/` files are not included.

For CPU-only validation, use the CPU PyTorch wheel rather than CUDA packages:

```bash
uv sync --extra dev
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install 'transformers>=4.51,<5'
uv run --no-sync pytest -q
```

The tiny-model tests construct random weights locally, run actual optimizer
steps, and reload student and teacher checkpoints. They do not download a model
or call an API. Mock integration uses a scripted model/user with real native
Mock tools and rewards. These are implementation checks, not benchmark scores.

## Configuration and execution

Copy `configs/pipeline_aligned.yaml` to `configs/local.yaml`. Verify:

- `policy.model`: initial local policy checkpoint;
- `world_model.model`: initial world-model checkpoint (or explicitly chosen base
  checkpoint if studying cold-start evolution; no hidden warmup is performed);
- `user_simulator.model` and decoding arguments: exactly the pipeline's simulator;
- `manifest_sha256`: the hash printed by `validate-splits`, for the
  shared official split;
- model devices, context lengths, steps/trials and training hyperparameters to
  match the agreed comparison. Example values are not inferred pipeline settings.

Use provider environment variables for API credentials. Do not place secrets in
configuration files: configurations are saved with run artifacts.

First run one training task as a development smoke test:

```bash
uv run --extra train comap-tau3 run \
  --config configs/local.yaml --manifest configs/splits.official178.json \
  --domain airline --split train --round 0 --limit 1 \
  --output outputs/airline-smoke
```

Run three offline-evolution rounds for a domain:

```bash
uv run --extra train comap-tau3 evolve \
  --config configs/local.yaml --manifest configs/splits.official178.json \
  --domain airline --output outputs/airline-evolution
```

Repeat for Retail and Telecom. This implements **separate domain-specific model
chains**; pooled cross-domain training is not silently assumed. Each round uses
only its assigned shard, then updates WM and policy. Teacher EMA checkpoints
carry across rounds. No automatic dev tuning or test evaluation occurs inside
`evolve`.

Evaluate the resulting checkpoints using the unchanged official evaluator:

```bash
uv run --extra train comap-tau3 run \
  --config outputs/airline-evolution/final_config.json \
  --manifest configs/splits.official178.json \
  --domain airline --split test --output outputs/airline-test
```

Use the corresponding final configuration for each other domain. Test selection
is explicitly `test`, never the default `base` split. Task limiting is forbidden
for final test runs. The manifest hash must match. CLI output directories cannot
be overwritten; an interrupted run leaves partial artifacts and a failure marker
and must not be reported as a complete benchmark.

Individual `train --kind world_model|policy --transitions ... --round ...` stages
are also available for debugging. API backends (`backend: api`, LiteLLM `model`
and optional `args`) support inference only; `evolve` rejects API-only models.
API inference is not equivalent to a trained CoMAP baseline.

## Source mapping and adaptation details

| Mechanism | Public CoMAP source | τ³ implementation |
|---|---|---|
| Draft/WM/reflection | `comap/integrations/eto/eval_agent/agents/coevolving_local_agent.py` | `src/comap_tau3/agent.py` |
| Next-state targets | `comap/world_model/datasets.py` | `prompts.py`, `training.py` |
| WMSD loss | `comap/world_model/wmsd.py` | target CE + forward KL on aligned target positions |
| Policy SDPO | `comap/integrations/sdpo/verl/trainer/ppo/core_algos.py` | on-policy sampled responses, full-logit KL/JSD |
| Auxiliary supervision | `comap/bridge/build_sdpo_buffer.py`, patched `dp_actor.py` | reflection CE, executed-action anchors, successful revision draft distillation |
| Round loop | `comap/orchestrator/round_manager.py` | `cli.py evolve` |

Changes that must be disclosed in a paper/report:

1. ALFWorld action strings become schema-validated JSON customer messages or
   native τ³ tool-call batches. History is serialized consistently for local HF
   and API backends. This changes the prompt/wire format relative to τ³'s vanilla
   native function-calling agent, so compare with that difference documented.
2. WM predicts the next **agent-visible** observation: tool returns or a customer
   reply, not hidden database state. Unobserved terminal transitions are skipped.
   Private user tools and evaluator/reference fields are excluded from prompts.
3. Upstream's revision probability threshold (0.5) and textual confidence proxy
   (threshold 0.6) are retained. Explicit `KEEP` is honored; malformed reflection
   falls back to the validated draft. This tightens the minimal upstream parser,
   which does not check the content of its `Decision:` field. A malformed draft
   gets one format retry, then the run fails rather than inventing an action.
4. The public repo does not include `world_model.train_wm`, although its launch
   script references it. This package supplies a full-parameter, single-device HF
   reference trainer. It uses actual parameter EMA (rate 0.01), while the public
   orchestrator's `ema` branch only materializes a snapshot and metadata. This is
   not the authors' distributed verl/LoRA configuration; large models need enough
   memory for student, frozen teacher, gradients and Adam state.
5. Policy SDPO samples a fresh response under the current student, compares its
   token distributions to a feedback-conditioned frozen EMA teacher on the same
   continuation positions, and adds CE supervision. Default alpha=0.5 follows
   the final upstream config override. Offline feedback uses real observed next
   observations and episode success; no held-out labels or expected actions are
   used. Auxiliary weights are reflect 1.0, anchor 0.5, successful revision-to-draft
   0.25. These categories are adapted to JSON responses, not byte-identical to the
   ALFWorld buffer builder. No unverifiable token confidence or reward oracle is
   introduced.
6. No implicit prompt truncation: context overflow fails explicitly. The example
   token limits are enlarged for tool JSON and must be fixed before comparison.

## Artifacts and current limits

Each rollout stores the exact manifest/config, native `*.simulation.json`, and
`*.trace.json` with drafts, predictions, reflections, final actions, actual next
observations, model calls and reported usage. Only train runs export
`transitions.jsonl`; training revalidates task IDs, round, split and manifest hash.
Each update saves reloadable `student/`, `teacher/` and numeric training logs.

A successful Mock or tiny-model test establishes implementation behavior only.
No core benchmark performance is established until the shared split, model
checkpoints/simulator and experiment settings are provided and actual runs finish.
Full-parameter distributed training, automatic recovery/resumption and exact
reproduction of the unpublished upstream training entry point are not claimed.
