# ToolSandbox Evolution Pipeline

This is a standalone Python 3.10 implementation of a reproducible
Policy–Controller–Critic pipeline for Apple ToolSandbox. It pins the upstream
benchmark source, isolates all compatibility code in an adapter, evolves
Policy/World memories and Skills over exactly three train shards, and evaluates
only Vanilla, Generation 0, and the fixed final Generation 3 system.

The repository contains implementation and offline validation code. Generated
datasets, model responses, run artifacts, credentials, and final benchmark
results are deliberately excluded.

## Environment

- Python `>=3.10,<3.11`
- `uv`
- pinned Apple ToolSandbox commit
  `165848b9a78cead7ca7fe7c89c688b58e6501219`

```bash
uv sync --frozen
uv run python -c "import toolsandbox_pipeline; import tool_sandbox"
uv run pytest -q
```

Do not activate or reuse a parent environment. The project-local `.venv` is
created from `uv.lock` and is not published.

## Fixed Model Boundary

- Pipeline model: `Qwen/Qwen3-32B`, non-thinking
  (`enable_thinking=false`)
- Retrieval embeddings: OpenAI `text-embedding-3-small` only, with no fallback
- ToolSandbox User Simulator: `gpt-4o-mini-2024-07-18`

The two OpenAI roles share one process-injected `OPENAI_API_KEY`, but use
separate clients, request ledgers, retry state, and usage accounting. Qwen uses
`QWEN_BASE_URL` and `QWEN_API_KEY`. Never place any credential in this
repository or an Agent transcript; see `docs/SECRET_INJECTION.md`.

## Data and Experiment Protocol

- Train is the only split used for iterative updates and calibration.
- Dev is available only to the deterministic Skill Mini-Bench for accepting or
  rejecting an already-created candidate.
- Test remains sealed until a frozen, approved one-time final plan runs the
  three systems in the order Vanilla, Generation 0, Updated.
- Updated is always `g003`; diagnostic best-observed checkpoints are retained
  for inspection but cannot select the final system.

Each training round records its own direct wall-clock latency, all physical
provider tokens, and non-monetary `qwen_effective_output_tokens` cost. This
cost includes only Qwen output tokens causally linked to a committed substantive
pipeline effect; rejected, NONE, SKIP, and no-op outputs contribute zero.

## Failure-Mode and Stability Analysis

Training produces sanitized failure-mode lineage records. Final reporting uses
exact host-derived signatures to identify directly related test variants and
counts a repair only when Generation 0 fails, Updated succeeds, one lineage
matches, and the Updated committed action chain proves use of the linked evolved
Skill. The report also measures stability across all eight ToolSandbox variants
in every scenario family. This is mechanism-linked observational attribution,
not a causal claim.

See `docs/FAILURE_MODE_ATTRIBUTION.md` for the frozen definitions.

## Commands

The train coordinator exposes a fixed allowlist:

```bash
uv run python -m toolsandbox_pipeline.orchestration.cli validate-config --help
uv run python -m toolsandbox_pipeline.orchestration.cli query-checkpoints --help
```

The final reporting layer exposes a separate fixed allowlist:

```bash
uv run python -m toolsandbox_pipeline.reporting.cli validate-plan --help
uv run python -m toolsandbox_pipeline.reporting.cli verify-report --help
```

Offline validation/build/query commands need no credentials. Model-backed smoke,
training, and one-time final evaluation run only through the reviewed
coordinator-owned launcher and immutable manifests. The current instance launcher and supervisor templates are included in
`scripts/runtime/`; see `docs/RUNTIME_DEPLOYMENT.md` for their fixed paths,
external prerequisites, and validation status. Generic CLI operations still fail
closed without an injected reviewed operation map.
A fake-client test is not a real-model result, and a train smoke is not a final
dataset result.

## Specifications

`pipeline.md` is authoritative. `AGENTS.md` defines ownership and access
boundaries, and `tasks/001_*.md` through `tasks/018_*.md` contain the reviewed
implementation contracts and acceptance checks.

## Live runtime snapshot

This snapshot includes the full live bootstrap, online/offline calibration,
three-round training composition, native Dev validation, Frankfurter currency
backend, host-assigned call IDs with raw-output audit, and lock-free progress
monitoring. Full end-to-end completion is not yet verified: the latest live campaign completed 29 calibration episodes, then stopped on
a Policy output truncation before offline calibration or formal training. See
[deployment instructions](docs/RUNTIME_DEPLOYMENT.md).
