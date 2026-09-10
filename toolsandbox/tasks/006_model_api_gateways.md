# Task 006: Model/API Gateways, Request Usage, and Explicit Preflights

Status: `approved`

## Objective

Implement strict provider boundaries for:

- frozen `Qwen/Qwen3-32B` chat requests through one OpenAI-compatible vLLM endpoint;
- OpenAI `text-embedding-3-small` requests with no alternate model or retrieval fallback;
- the project-selected `gpt-4o-mini-2024-07-18` User Simulator with separate instrumentation;
- normalized physical-attempt latency and actual token usage;
- explicit, individually selectable Qwen, embedding, and User Simulator preflight commands.

Provider gateways perform one physical request per invocation and return validated data plus immutable attempt metrics. They do not retry, build prompts, perform retrieval, route Controller decisions, mutate pipeline state, write checkpoints, choose datasets, or aggregate experiment results.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3, 6, 8, 11-15, 20-24;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. `tasks/002_core_contracts.md`;
6. `tasks/004_toolsandbox_adapter.md`;
7. this task file;
8. the pinned upstream generic User Simulator behavior, its concrete GPT-4o class for comparison, and only the provider SDK interfaces required by this task.

Use these primary documentation contracts without substituting model names:

- <https://developers.openai.com/api/docs/models/text-embedding-3-small>
- <https://developers.openai.com/api/reference/resources/embeddings/methods/create>
- <https://developers.openai.com/api/docs/models/gpt-4o-mini>
- <https://qwen.readthedocs.io/en/stable/deployment/vllm.html>
- <https://docs.vllm.ai/en/v0.17.0/features/structured_outputs/>

## Access

```text
Access class: real_data
Setup network: none
Data splits: none
Secrets: real-data-runner only
Allowed external operations: explicit Qwen, embedding, and User Simulator preflights only
```

The assigned development Agent runs all implementation tests with fake transports and no secrets. After offline tests pass, it submits preflight requests to the coordinator. Only the dedicated real-data runner injects credentials and contacts the selected endpoint. This task does not authorize train, dev, test, RapidAPI, generic OpenAI models, or arbitrary URLs.

## Preconditions

- Tasks 001-005 are complete and accepted.
- The project environment directly contains `openai==1.17.0`.
- `QWEN_BASE_URL` identifies an operator-approved OpenAI-compatible vLLM endpoint when the Qwen preflight is requested.
- The formal vLLM version is not yet selected. Code must support the two explicit wire modes below without selecting one automatically.
- If `gpt-4o-mini-2024-07-18` is unavailable to the configured OpenAI project, report the preflight as blocked. Do not change the model.

## Coordinator-Approved User Durability Amendment

The user has authorized this Task-owned interface/test amendment required by
Task 014. Implement an injected optional durable callback or equivalent
prepare/complete seam in `InstrumentedGPT4oMiniUser` that:

- receives the exact successful `GatewayResponse`, including raw bytes and
  physical-attempt metrics;
- allows caller-owned Task 011 logic to persist that response before upstream
  `OpenAIAPIUser.respond()` appends/returns the resulting User message;
- supports reuse of a previously completed response without a second OpenAI
  dispatch;
- preserves upstream prompt construction, message conversion, tool-call parsing,
  stop behavior, and omitted decoding fields;
- never exposes raw response bytes through repr, logs, native messages, prompts,
  reports, or metrics;
- remains optional for isolated Task 006 unit tests and does not import
  checkpoint/ledger modules into provider code.

This amendment is authorized only within the existing Owned Files below. Task 006
still does not own persistence or retry policy.

## Owned Files

The assigned Agent may create or edit only:

```text
src/toolsandbox_pipeline/schemas/runtime.py
src/toolsandbox_pipeline/schemas/usage.py
src/toolsandbox_pipeline/providers/__init__.py
src/toolsandbox_pipeline/providers/contracts.py
src/toolsandbox_pipeline/providers/openai_clients.py
src/toolsandbox_pipeline/providers/qwen.py
src/toolsandbox_pipeline/providers/embedding.py
src/toolsandbox_pipeline/providers/user_simulator.py
src/toolsandbox_pipeline/providers/preflight.py
src/toolsandbox_pipeline/metrics/__init__.py
src/toolsandbox_pipeline/metrics/usage.py
tests/providers/test_runtime_config.py
tests/providers/test_request_identity.py
tests/providers/test_qwen.py
tests/providers/test_embedding.py
tests/providers/test_user_simulator.py
tests/providers/test_preflight.py
tests/metrics/test_usage.py
```

Do not edit prompts, Action/State/Controller schemas, Adapter files, dependency files, upstream source, retrieval, checkpointing, CLI, dataset code, aggregate metric schemas, or generated run artifacts.

## Runtime Configuration Contracts

`schemas/runtime.py` defines strict configuration models. They contain endpoint identities and environment-variable names, never resolved secret values.

### Qwen

The fixed request contract is:

```yaml
provider: vllm_openai_compatible
model: Qwen/Qwen3-32B
temperature: 0.0
seed: 0
top_p: omitted
enable_thinking: false
structured_output_wire_mode: guided_json | structured_outputs_json
base_url_env: QWEN_BASE_URL
api_key_env: QWEN_API_KEY
```

Requirements:

- `model`, temperature, seed, omitted `top_p`, and disabled thinking cannot be overridden per call.
- Maximum output tokens are supplied by a validated versioned role token-limit
  config. Bootstrap values and calibrated formal values must follow Section 15.
- The configuration records expected vLLM version, container digest, served-model ID, structured-output backend, server generation-config policy, context limit, and output limit.
- Unresolved deployment identity is valid only for offline construction/tests and invalid for external preflight or formal run.
- The code never loads Qwen weights, vLLM, Transformers, CUDA, or model files into the project process. The model server is a separate frozen deployment.

### Embeddings

The fixed contract is:

```yaml
provider: openai
model: text-embedding-3-small
base_url: https://api.openai.com/v1
api_key_env: OPENAI_API_KEY
encoding_format: float
dimensions: omitted
```

A non-default base URL is allowed only as an explicitly manifest-recorded identity and must still return the exact model contract. There is no configurable alternate embedding model.

### User Simulator

The fixed contract is:

```yaml
provider: openai
model: gpt-4o-mini-2024-07-18
base_url: https://api.openai.com/v1
api_key_env: OPENAI_API_KEY
temperature: omitted
top_p: omitted
seed: omitted
```

Use the dated snapshot rather than the moving `gpt-4o-mini` alias. This selection intentionally differs from the pinned upstream concrete `GPT_4_o_2024_05_13_User`; unavailability is a blocking preflight failure and does not authorize an alias or model change.

## Client and Secret Isolation

- Construct provider clients only inside an explicitly invoked gateway/preflight, never at module import.
- Disable SDK automatic retries with `max_retries=0`; each gateway invocation represents exactly one physical attempt.
- Set an explicit finite timeout from validated configuration.
- `text-embedding-3-small` and the User Simulator read the same `OPENAI_API_KEY` but use separate client instances, roles, request IDs, recorders, rate-limit state, and usage totals.
- Qwen reads only `QWEN_BASE_URL` and `QWEN_API_KEY`; it must not inherit `OPENAI_BASE_URL` or use `OPENAI_API_KEY`.
- A deployment-local `QWEN_API_KEY="EMPTY"` is allowed only when explicitly selected for a no-auth local endpoint.
- Missing credentials report only the environment-variable name. Never expose values, lengths, hashes, prefixes, suffixes, headers, or exception strings containing headers.
- Tests inject fake transports/clients directly and must not read the real environment.

## Request Identity and Physical Attempt Boundary

`providers/contracts.py` defines at least:

```python
ProviderRole
RequestContext
PhysicalAttemptStatus
PhysicalAttemptResult
GatewayResponse
ProviderRequestError
ChatTransport
EmbeddingTransport
```

`RequestContext` receives a prepared `logical_request_id`, unique `attempt_id`, role, phase, state/offline-unit reference, canonical input fingerprint, replay-after-unknown-outcome flag, and manifest identity. The later checkpoint/request-ledger task owns derivation, persistence, and retry authorization. This task validates the identifiers but does not invent a second ledger.

Every provider gateway:

1. receives a `RequestContext` before dispatch;
2. records UTC and monotonic start immediately before the one physical client call;
3. dispatches exactly once;
4. records monotonic completion and raw response hash or sanitized exception class;
5. normalizes returned usage without estimation;
6. returns or raises with one complete immutable `PhysicalAttemptResult`.

A successful `GatewayResponse` also carries the exact received raw response bytes
as a frozen `repr=False` field. `ProviderRequestError.raw_response_body` contains
those bytes only when a response was actually received; preparation and transport
failures expose `None`. Raw bytes never appear in exception text, repr, preflight,
ordinary logs, or metrics. The later request ledger persists them under the
restricted checkpoint boundary before applying a validated value. Gateways still
perform no filesystem write and import no checkpoint module.

Timeout, connection loss, or an exception after dispatch uses `unknown_outcome` unless the transport proves rejection before dispatch. The gateway never retries. Later orchestration may create a new `attempt_id` for the same logical request only under Section 21 recovery rules.

## Qwen Gateway

The Qwen gateway accepts validated chat messages, one Task 002 Pydantic output
model/JSON Schema, the fixed provider role, and that role's selected value from a
validated versioned Section 15 token-limit config.

Build the request with exactly:

```text
model = Qwen/Qwen3-32B
temperature = 0.0
seed = 0
max_tokens = role-specific value from the selected token-limit config
chat_template_kwargs = {enable_thinking: false}
```

Do not include `top_p`. Do not use native tool calling for Policy/Critic/Revision; current augmented tool schemas are serialized by the later prompt task.

For `guided_json`, send only the configured legacy `guided_json` JSON Schema field. For `structured_outputs_json`, send only `structured_outputs: {json: <schema>}`. Both travel as vLLM-specific extra request body fields. Sending both, auto-probing one after another, or falling back after a rejection is forbidden.

Requirements:

- Confirm the returned model identity matches the configured served-model contract.
- Require exactly one choice, one explicit `finish_reason`, and non-empty string
  content when the response is not truncated.
- Preserve the returned `finish_reason` in `PhysicalAttemptResult`. Treat
  `finish_reason: length` as a typed `output_truncated` failure while retaining
  response hash and actual usage for calibration/accounting.
- A reasoning field with non-empty content is a hard error.
- Preflight uses a fixed schema/prompt and rejects raw thinking tags or an ignored non-thinking switch.
- Parse JSON once, then locally validate it with the exact Task 002 strict Pydantic model used to generate the schema.
- Unparseable or schema-invalid output is a hard provider-output failure; do not
  repair JSON or retry. Only an explicit external calibration controller may issue
  a new logical request with a different candidate `max_tokens` after a typed
  length truncation.
- Return validated wire data, exact repr-hidden raw response bytes, raw response
  hash, and normalized actual usage. Raw content persistence belongs to the later
  request ledger.

The project intentionally keeps temperature zero despite Qwen's different general recommendation; the gateway must not silently change it.

## Embedding Gateway

The embedding gateway accepts one non-empty string or an ordered non-empty list of non-empty strings and sends:

```text
model = text-embedding-3-small
encoding_format = float
dimensions = omitted
```

Requirements:

- Reject empty strings, empty batches, batches larger than the official API input-count limit, and non-string inputs before dispatch.
- Do not truncate or locally retokenize over-limit input; provider rejection is a hard failure.
- Require returned model identity `text-embedding-3-small`, object type, item count, unique contiguous indices, and original order.
- Require every vector to be non-empty, finite, and the same dimension. Record the first valid dimension and require the configured run/generation dimension to match thereafter.
- Normalize usage from provider `prompt_tokens` and `total_tokens`, require consistency, and record zero output tokens.
- Do not provide BM25, local vectors, another embedding model, stale-cache substitution, or zero vectors after failure.
- This gateway does not cache or rank vectors; the later retrieval task owns those functions.

## Instrumented User Simulator

Implement a thin project-owned `InstrumentedGPT4oMiniUser` subclass/wrapper that reuses the pinned upstream `OpenAIAPIUser` behavior while fixing `model_name = "gpt-4o-mini-2024-07-18"`. The upstream concrete `GPT_4_o_2024_05_13_User` is comparison evidence for the original configuration; do not instantiate it or inherit its old model identity for project runs.

Requirements:

- Preserve upstream prompt/few-shot construction, message conversion, tool schema, stop condition, and omission of temperature/top-p/seed.
- Replace only client construction/physical-call instrumentation so the client uses explicit base URL, `max_retries=0`, timeout, RequestContext, and usage recorder.
- Keep `role: user_simulator`; never count it as Policy/Critic/Revision.
- Do not expose pipeline memories, skills, Critic output, Controller sidecars, or evaluator state.
- Require the response model to be the exact configured snapshot when the API returns model identity.
- Do not fall back to `gpt-4o-mini`, GPT-4o, another dated snapshot, a ChatGPT alias, Qwen, or a fake client during an authorized real run.
- Fake clients are mandatory for unit tests and forbidden as evidence of real snapshot availability.
- Invoke the approved optional durability seam after the one successful
  `GatewayResponse` exists and before the upstream User response becomes visible
  to the episode. Propagate seam failure without appending a User message.

If preserving upstream behavior requires copying or materially rewriting upstream prompt logic, stop and request an interface decision rather than forking it silently.

## Usage Normalization

`schemas/usage.py` and `metrics/usage.py` define strict immutable records for:

```python
TokenUsage
PhysicalAttemptMetrics
```

Use:

```text
input_tokens = uncached_input_tokens + cache_read_input_tokens + cache_write_input_tokens
total_tokens = input_tokens + output_tokens
```

Requirements:

- Qwen/User Simulator map provider prompt/completion usage to input/output fields and retain cache details when actually returned.
- Embeddings map `prompt_tokens` to uncached input, set output/cache counts to zero, and require provider `total_tokens` consistency.
- Count usage by physical `attempt_id`, including replay attempts.
- Missing actual usage produces null affected counts and `usage_complete: false`.
- Never estimate missing usage or label tokenizer estimates as actual.
- Record per-attempt monotonic latency in seconds from the gateway boundary.
- Do not compute aggregate `total_tokens` or substantive-effect Qwen-output
  `total_cost` in this task; later ledger/metrics tasks consume these attempt
  records and committed effect links.
- Do not aggregate `total_running_time_seconds` by summing attempts; later run orchestration measures end-to-end wall time directly.

## Explicit Preflight Module

`python -m toolsandbox_pipeline.providers.preflight` exposes exactly one selected mode per invocation:

```text
qwen
embedding
user-simulator
```

No default mode and no `all` mode are allowed.

Each preflight:

- validates fully resolved non-secret configuration before retrieving a credential;
- performs one minimal fixed request;
- emits sanitized JSON containing mode, pass/fail/blocked status, expected/returned model identity, configured wire mode where relevant, vector dimension where relevant, latency, actual input/output/total tokens, `usage_complete`, and response hash;
- never emits prompt/response content, environment values, URLs containing credentials, headers, stack traces with request data, or raw exception bodies;
- exits non-zero on missing credentials, unavailable exact model, schema mismatch, reasoning output, invalid vectors, missing usage, or endpoint failure.

Preflight results are setup evidence, not experiment metrics, and must not be included in training/evaluation `total_running_time_seconds` or `total_tokens`.

## Required Offline Tests

With fake transports only, test at least:

1. strict runtime configuration and rejection of model/decoding overrides;
2. separate OpenAI and Qwen credential/base-URL routing without reading real variables;
3. exactly one physical transport call and disabled implicit retries;
4. immutable RequestContext/attempt metrics for success, pre-dispatch rejection, timeout, and unknown outcome;
5. both Qwen structured-output wire modes and proof that neither sends the other's field;
6. omitted `top_p`, fixed temperature/seed, selected config `max_tokens`, and
   exact non-thinking request body;
7. Qwen model mismatch, multiple/empty choices, missing/unknown finish reason,
   typed length truncation with preserved usage, reasoning content, malformed JSON,
   and local schema failure;
8. valid single/batched embeddings, order, dimension, finite values, model identity, and usage normalization;
9. empty/oversized/invalid embedding input, bad indices, missing vectors, dimension drift, NaN/infinity, model mismatch, and missing usage;
10. User Simulator exact mini snapshot, inherited upstream User-role behavior, omitted decoding fields, separate role/client/accounting, and no snapshot fallback;
11. actual token formulas, cache fields, replay-attempt counting inputs, and incomplete-usage behavior;
12. preflight mode isolation, sanitized output, non-zero failure exits, and absence of an `all` mode;
13. imports perform no client construction, environment read, network access, or filesystem write;
14. secret-like test values never appear in logs, exceptions, reprs, or preflight output.
15. User durability seam ordering before User-message visibility, exact raw-byte
    handoff, completed-response reuse without dispatch, callback failure
    propagation, and no raw-byte repr/log leakage.

## Required External Validation

After every offline test passes, submit three separate runner requests:

```text
Allowlisted mode: preflight-qwen
Access used: QWEN_BASE_URL, QWEN_API_KEY
```

```text
Allowlisted mode: preflight-embedding
Access used: OPENAI_API_KEY
```

```text
Allowlisted mode: preflight-user
Access used: OPENAI_API_KEY
```

The same OpenAI key is used for the last two requests, but they run separately and produce separate usage records. A failure of one must not trigger either other preflight.

If the model server or key is not yet available, code may be reported complete but Task 006 remains externally unvalidated. Record the missing preflight by name without exposing credential details.

## Acceptance Commands

Development Agent runs:

```bash
uv sync --frozen
uv run pytest -q tests/providers tests/metrics/test_usage.py
uv run python -c "from toolsandbox_pipeline.providers import QwenGateway, EmbeddingGateway; from toolsandbox_pipeline.providers.user_simulator import InstrumentedGPT4oMiniUser"
git diff --check
git status --short
```

The coordinator-owned runner separately executes the three explicit preflights through the project-external launcher after their CLI modes are installed. Do not place credential-prefixed commands in this task file.

## Acceptance Criteria

- Offline provider and metrics tests pass with zero external calls.
- Each gateway makes exactly one physical request and records actual latency/usage without hidden retry.
- Qwen request fields and selected structured-output mode exactly match configuration.
- Embeddings strictly use `text-embedding-3-small` with no fallback.
- User Simulator strictly uses `gpt-4o-mini-2024-07-18`, preserves upstream generic User-role behavior, and records the model-selection deviation.
- The approved optional User durability seam is implemented without provider-owned
  persistence and is verified before Task 014 begins.
- Secret/client/role/accounting isolation is demonstrated.
- All available explicit runner preflights pass; unavailable infrastructure is reported as external validation outstanding.
- No unowned or upstream file is modified.

## Completion Report Additions

Include:

```text
Qwen wire modes tested offline:
Qwen preflight: pass | fail | not run
Embedding preflight: pass | fail | not run
User Simulator preflight: pass | fail | not run
User durability seam tests:
Returned model identities:
Selected token-limit config path and SHA-256:
Qwen finish reasons observed:
Embedding dimension:
Physical attempts by role:
Preflight latency and actual tokens:
Usage complete:
External validation blockers:
Dependency/interface change requests:
```

Preflight timing/tokens are reported separately as setup evidence. This task runs no dataset experiment, so do not label preflight totals as round/run experimental metrics.
