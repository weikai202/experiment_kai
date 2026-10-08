# Standalone tau3 Evolution Pipeline

This project ports the ToolSandbox Policy-Controller-Critic evolution method to the actual tau3
Airline, Retail, and Telecom text task structure. It is standalone: there are no imports from the
ToolSandbox, CoMAP, or EarlyExperience sibling directories.

Implemented contracts include:

- the exact seed-0 `144 train / 34 Dev / 100 official test` task-level protocol;
- an explicit, non-comparable `official178` compatibility protocol;
- hashed manifests, sealed test membership, and audited train/Dev access;
- a native half-duplex Agent adapter that excludes evaluator and private user state;
- deterministic, relevance-filtered paired Dev Mini-Bench selection (maximum 20 tasks);
- a checked-in round workflow composing native episodes, Policy-Controller-Critic-Revision,
  generation-scoped Memory/Skill retrieval, offline proposals, paired Dev acceptance, and publication;
- exactly three generations, ledger-bound atomic checkpoints, and no-redispatch recovery;
- a pinned public-Agent-schema compiler and content-bound G000 Skill library;
- a mandatory real-train-smoke output-limit receipt with deterministic headroom;
- direct per-round latency, all-token, and effective-Qwen-output-token accounting;
- exact failure lineage and linked Skill-use repair analysis; and
- a per-unit crash-resumable Vanilla/G000/G003 executor with trusted traces, runner-owned latency,
  and an immutable report.

See [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for the data and evaluation definitions.

## Offline setup

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check .
```

Regenerate manifests only from the verified pinned tau checkout:

```bash
uv run python scripts/build_manifests.py \
  --tau-root /path/to/pinned/tau2-bench \
  --output-root configs
```

No live model, simulator, or official evaluation is performed by the offline suite. A reviewed live composition must supply the pinned tau runtime, model transports, native episode

Compile G000 through the explicit two-environment schema-only seam. First run the exporter in the
pinned tau Python 3.12 environment, then compile under this project's Python 3.10 environment:

```bash
python3.12 scripts/export_tau_schemas.py \
  --tau-root /path/to/pinned/tau2-bench --output /tmp/tau-public-schemas.json
uv run python scripts/build_seed_library.py \
  --schema-export /tmp/tau-public-schemas.json --output seed_library.json
```
executor, offline proposal/review implementations, Dev branch executor, and durable artifact location.
The checked-in `PipelineRoundEngine` owns their order, access boundaries, acceptance rule, accounting,
and generation publication; injected components perform benchmark I/O or model inference only. Secrets are process-injected and never stored.
