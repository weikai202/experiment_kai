# Policy Memory reflection smoke

This is a development-only, one-round smoke on the existing fixed two-case train
selection. It exercises real Qwen, embeddings, native User/EE, native evaluation,
memory candidate generation/review, semantic deduplication, memory mutation, G001
index construction and online retrieval. It does not fine-tune weights, rewrite
Skills, update World memory, run dev/test, or constitute formal three-round training.

The operator's owner-only launcher exposes the fixed command:

```bash
/root/.local/bin/toolsandbox-with-secrets reflection-smoke
```

The command maps to `uv run --frozen python -m
toolsandbox_pipeline.orchestration.reflection_smoke` and accepts no extra arguments.
Runtime configuration and pinned local model are under `/root/toolsandbox-runtime`;
no credentials are stored in the repository. `configs/run/reflection_smoke_v1.json`
records the scope and fixed scenario IDs. Online limits use the length-smoke config;
offline candidate/review limits remain the provisional 512/256 tokens. A length or
schema failure stops the smoke and is recorded; limits are not adapted in-flight.

Flow:

1. Create a new private run and preflight the three providers.
2. Build G000 and execute the two train scenarios as fresh eligible round-0 inputs.
3. Seal one Policy projection per completed, evaluated trajectory. Projections
   contain only the original online-visible state/semantic retrieval/actions/tool
   outcomes and native scalar score/success label; no hidden evaluator definitions.
4. Use the existing MemoryUpdateOrchestrator to generate and review candidates.
   `NONE` and `SKIP` remain valid outcomes; no memory is hand-written to force improvement.
5. Apply approved Policy memory overlays and build a separate G001 directory.
6. Execute the same two train scenarios with G001 and save native results. Online
   inputs use the normal retrieval interface and do not consume the reflection buffer.

The retest is an in-sample diagnostic. The remote User simulator is not frozen, so
before/after wording and trajectories can differ; scores do not identify a causal
effect or establish generalization. No formal generation pointer is changed.

Artifacts are under `/root/toolsandbox-runtime/runs/reflection-smoke-*`, located by
`/root/toolsandbox-runtime/latest-reflection-run.json`:

- `manifest.json`, `preflight.json`, `dataset-access.json`;
- `g000-episode-*.json`, `g001-episode-*.json`;
- `reflection-inputs.json`, `reflection-result.json`, `g001-ready.json`;
- `phase-metrics.json`: training, reflection/build, direct round-0 total, retest;
- `provider-attempts.json`, `summary.json`, and the durable checkpoint ledger.

The direct round-0 row includes online training through G001 construction. Stage
rows are its breakdown, not additional rounds to sum. Preflights and G000 build
are setup. The effective-output cost is nonmonetary and excludes NONE/SKIP effects.
Failures save sanitized classes/stack locations and durable attempt metadata.
There is no automatic resume or automatic fresh-run retry in this smoke entry.

## Retrieval-length compatibility fix

The first fresh run stopped before reflection because a complete state query was
8,332 bytes, above the original 8,000-byte guard. It was only 2,625 cl100k_base
tokens. Retrieval and the embedding gateway now share an 8,191-token check with
a pinned local vocabulary (tiktoken 0.7.0). The batch resource ceiling remains
280,000 bytes. No input is shortened and there is no embedding fallback.

The model's documented input limit is 8,192 tokens:
https://developers.openai.com/api/docs/guides/embeddings

The failed run remains archived, and its incomplete trajectory is not consumed
by reflection. The next launch creates a fresh run under the updated lock/hash.

## JSON whitespace calibration (2026-09-12)

Run `reflection-smoke-20260912T073435-63e96c` stopped during G000 Critic, before evaluation/reflection. An exact-input local replay verified the original request fingerprint and reproduced a 768-token truncation: after `verdict=accept` and `predicted_outcome=success`, decoding repeated whitespace rather than completing the JSON.

The local vLLM launch now sets `structured_outputs_config.disable_any_whitespace=true` with xgrammar. This constrains JSON formatting, including inter-field whitespace; string contents retain ordinary spaces. Model weights, role prompts, temperature, seed and output token ceilings are unchanged. The complete launch configuration is versioned in `configs/run/qwen_reflection_smoke.json` and recorded in each run manifest. A new run is required because decoding configuration changed.

Reflection smoke role runners now use `offline` execution mode with development token limits. The earlier `calibration` runner converted truncation exceptions into calibration observations, which the episode backend rejected without persisting the failure. Offline execution preserves the provider failure, raw response and token usage in the ledger and does not retry silently. A real runner-to-ledger regression verifies this behavior. Failed episodes remain excluded from the reflection buffer.

## Long retrieval inputs (2026-09-12)

Run `reflection-smoke-20260912T074542-1fcfc5` reached a genuine 9,559-token retrieval query (26,646 UTF-8 bytes) after repeated tool results. This is distinct from the earlier overly strict byte guard. It stopped before evaluation/reflection.

`retrieval/long_inputs.py` now defines `utf8-token-chunks-weighted-l2-v1`: keep complete text; split at cl100k token boundaries, moving boundaries to preserve valid UTF-8; recheck each chunk against 8,191 tokens; send chunks in one native embedding batch; average vectors weighted by each chunk's actual token count and L2-normalize. Short inputs retain their original embedding vector. Agent/Critic visible input text is unchanged. Long query ranking may change; no performance equivalence is claimed.

The native provider response and usage are persisted before aggregation. Cache schema 2 stores the aggregation strategy, chunk hashes and token weights alongside each derived vector. Old cache schemas are refused; fresh reflection runs create fresh caches. Query hashes still cover complete original input text, and the run manifest records algorithm version and source hashes. Total text is bounded at 280,000 bytes; custom smaller batch limits fail before dispatch when one input cannot fit.

The real failed query now splits into 8,191 + 1,368 tokens and reconstructs exactly. Unicode, aggregation, provenance, cache reopen and relevant pipeline tests: 160 passed. Full reflection smoke must be restarted from a secret-bearing terminal.

## Execution identity repair (user-authorized, 2026-09-12)

Model-local call IDs can repeat across turns. Native dispatch now uses deterministic execution IDs derived from the online decision identity and original action; original Policy/Critic decisions and their hashes remain unchanged. The same durable decision reconstructs the same execution IDs, while distinct turns receive different IDs. Parallel calls still require distinct model-local IDs within the action.

The smoke outcome builder joins each binding to its committed transaction using the action ordinal, then selects exactly its result_message_indices and execution call IDs. It never joins a model-local call ID against all historical results. This prevents repeated IDs from multiplying completed calls. Model repetition remains a separate behavior to evaluate with smoke.

Unattended launch is available through the external owner-only credential launcher and supervisor program toolsandbox-reflection-smoke; setup/operations are documented at /root/toolsandbox-runtime/unattended-operation.md.

## Packed Policy reflection v2 (user-approved)

The development smoke now selects `input_representation=packed-v2` for Policy candidate requests. Existing callers default to v1. The original complete `PolicyTrajectoryProjection` remains in the sealed buffer and is used unchanged by visibility and sensitive-literal checks. Only the candidate's model input is encoded.

The codec stores state deltas and a per-trajectory pool of exact repeated JSON subtrees (threshold32 characters). Retrieval occurrences retain order. Reserved `$ref`/`$literal` keys are escaped. Before dispatch, decoding must reproduce the original canonical JSON exactly. The v2 envelope records its version and original projection hash; the system prompt adds explicit reconstruction instructions before the unchanged reflection criteria. Prepared requests record prompt_version=v2 and independent prompt/input hashes.

The runner checks actual local Qwen token count plus the reserved output ceiling against the unchanged context limit before each offline dispatch. Codec, role and orchestrator source hashes are included in the run manifest. No summarization, truncation, cross-trajectory pool, fabricated memory or wider context setting is introduced. GPU probes returning NONE establish validity, not improved reflection quality.

Integrated tests:135 passed, including lossless codec cases and a real MemoryRoleRunner boundary test confirming that v2 is dispatched and the original projection is unchanged.
