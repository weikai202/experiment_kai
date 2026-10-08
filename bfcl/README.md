# BFCL Evolution Pipeline

This is a standalone, pipeline-aligned adaptation of the ToolSandbox
Policy-Controller-Critic evolution method to BFCL v4 multi-turn. It is an
experimental protocol and **not** the official BFCL leaderboard protocol.

The package keeps BFCL's fixed dataset user turns, stateful multi-turn simulators, delayed-function
semantics, missing-parameter cases, long-context cases, and official evaluator
interfaces. Generated actions are decoded with a literal-only AST parser;
generated Python is never evaluated.

## Frozen protocol

- Benchmark source revision: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- Pipeline model: `Qwen/Qwen3-32B`
- Decoding: temperature 0, seed 0, and
  `chat_template_kwargs={"enable_thinking": false}`
- Retrieval embeddings: OpenAI `text-embedding-3-small` only, with no fallback
- Scope: the 200 indices shared by Base, Miss Func, Miss Param, and Long Context
- Split after one shuffle: first 40 sealed test families, next 40 dev
  families, and remaining 120 train families
- Evolution: three train rounds of 40 families / 160 cases
- Dev Mini-Bench: at most five relevant complete families / 20 cases per Skill
- Final systems: Vanilla, G000, and fixed G003, in that order
- Execution: one process because the official evaluator seam is scoped and
  serialized

The four same-index records become a family only after their tool path,
involved classes, and excluded-function provenance agree. An ID suffix alone is
not proof of family membership. A family and all its turns and variants remain
in one split and, for train, in one round.

## Access boundary

Train is used for evolution. Dev can only accept or reject an already-created
Skill candidate through paired A/B evaluation. Test is inaccessible through
ordinary loaders and requires a frozen plan hash, the sealed manifest hash, and
a durable one-use authority marker. Updated always means G003; a best observed
checkpoint is diagnostic only.

No datasets, model responses, credentials, run outputs, or checkpoint artifacts
are committed. The code does not make model calls by itself; a reviewed external
runtime must inject Qwen roles and the pinned BFCL installation.

Build a content-hashed split manifest from a pinned BFCL checkout with:

```bash
python -m bfcl_pipeline.manifest_cli \
  --data-dir /path/to/bfcl_eval/data \
  --output /secure/setup/split_manifest.json \
  --seed-library-output /secure/setup/g000_seed_library.json
```

This setup command records IDs and hashes and deterministically compiles G000
Skills from public BFCL function schemas only; it does not execute any case.
A formal run also requires a hash-verified max-token calibration receipt from a
real train-only smoke. There is no arbitrary formal `max_tokens` default.

## Offline validation

```bash
python3.10 -m venv .venv
uv sync --frozen --extra test
uv run pytest -q
```

Tests use synthetic BFCL-shaped records and fake roles. Optional integration
checks may use the locally installed official BFCL package, but default tests do
not execute a test split or contact a model endpoint. Synthetic calibration
fixtures test validation only and are not evidence of a real smoke.

The official adapter runtime is deliberately a separate heavy extra:

```bash
uv sync --frozen --extra test --extra bfcl
```

See [pipeline.md](pipeline.md) for the complete experiment contract and
[docs/FAILURE_MODE_ATTRIBUTION.md](docs/FAILURE_MODE_ATTRIBUTION.md) for the
observational repair analysis.
