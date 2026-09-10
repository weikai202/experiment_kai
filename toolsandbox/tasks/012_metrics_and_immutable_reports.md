# Task 012: Metrics and Immutable Accounting Artifacts

Status: `approved`

## Objective

Implement the pure accounting and artifact layer required by Pipeline Sections
22-23. This task owns:

- strict schemas for physical-request, scenario-task, round, and run metrics;
- direct monotonic wall-time measurement for scenario, round, and run scopes;
- aggregation of actual provider usage across every dispatched physical attempt;
- an explicit non-monetary `total_cost` proxy counting only Qwen output tokens
  linked to a committed substantive pipeline effect;
- separate logical-call, physical-dispatch, retry, failure, and incomplete-usage
  counts;
- deterministic, append-only logical records materialized as immutable JSONL/JSON;
- fail-closed completeness propagation and sanitized accounting diagnostics.

It does not call a model or tool, open a dataset, run an episode, invoke the native
evaluator, decide retry/recovery policy, estimate missing usage, calculate money,
or orchestrate training/evaluation. Later orchestration supplies strict accounting
inputs from the Task 011 ledger and calls this task's pure APIs.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 6, 17, and 20-25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `tasks/002_core_contracts.md`;
5. `tasks/006_model_api_gateways.md`;
6. `tasks/009_dataset_manifest_and_split_access.md`;
7. `tasks/010_rapidapi_fixtures_and_network_boundary.md`;
8. `tasks/011_checkpoints_request_ledger_and_recovery.md`;
9. this task file.

Do not inspect real scenarios, prompts, provider responses, dataset rows, evaluator
targets, fixture bodies, trajectories, or generated experiment artifacts.

## Access

```text
Access class: offline
Setup network: none
Data splits: none
Secrets: none
External operations: none
```

All tests use synthetic usage, identities, timestamps, and temporary
directories. The development Agent must not use an OpenAI/Qwen/RapidAPI key, query
provider billing APIs, make a model/tool request, or read a real run directory.

## Preconditions

- Tasks 001-011 are complete and accepted.
- Task 006 immutable `TokenUsage`, `PhysicalAttemptMetrics`,
  `RequestContext`, and `PhysicalAttemptResult` contracts are stable.
- Task 011 stable `logical_request_id`/`attempt_id`, physical attempt states,
  and checkpoint accumulator slots are stable.
- Qwen response application is durably identifiable from the Task 011 logical
  response application marker.

Task 012 consumes Task 006 usage objects and Task 011-compatible identities. It
must not edit provider or checkpointing files. If those interfaces are missing or
incompatible, stop and request a coordinator-owned amendment.

## Owned Files

The assigned Agent may create or edit only:

```text
src/toolsandbox_pipeline/schemas/accounting.py
src/toolsandbox_pipeline/metrics/__init__.py
src/toolsandbox_pipeline/metrics/timing.py
src/toolsandbox_pipeline/metrics/aggregation.py
src/toolsandbox_pipeline/metrics/artifacts.py
tests/metrics/test_timing.py
tests/metrics/test_aggregation.py
tests/metrics/test_artifacts.py
```

The existing Task 006-owned files:

```text
src/toolsandbox_pipeline/schemas/usage.py
src/toolsandbox_pipeline/metrics/usage.py
tests/metrics/test_usage.py
```

are read-only inputs. Do not edit online/offline orchestration, providers,
checkpointing, Adapter, evaluator, retrieval, dataset/fixture code, prompts,
dependency files, CLI entry points, upstream source, or generated `runs/` and
`artifacts/` content.

## Accounting Boundary

Accounting uses two different units and never conflates them:

1. A logical request is identified by one Task 011 `logical_request_id`. It may
   be prepared once, applied at most once, and have multiple physical attempts.
2. A physical dispatch is an attempt that durably reached Task 011 `in_flight`
   before the gateway was invoked. Every such `attempt_id` counts separately,
   including failures, timeouts, invalid responses, and recovery attempts.

An attempt left `allocated` and later marked `abandoned_before_dispatch` is not
a physical dispatch, has no model usage or effective-output cost, and cannot make usage
incomplete. It remains visible in diagnostic lifecycle counts. A
`rejected_before_dispatch` attempt is treated the same unless evidence proves a
provider dispatch occurred; ambiguous evidence fails validation.

A dispatched `unknown_outcome` attempt is a real physical attempt. Its missing
provider usage makes every containing token aggregate incomplete even if a later
recovery attempt succeeds. It does not enter effective-output cost because it has
no durably applied response.

## Strict Input Projection

`schemas/accounting.py` defines frozen Pydantic v2 models with
`extra="forbid"` and no coercion:

```text
PhysicalAttemptAccountingInput
LogicalRequestAccountingInput
ToolAttemptAccountingInput
ScopeTimingInput
TaskAccountingInput
RoundAccountingInput
RunAccountingInput
RequestMetricRecord
TaskMetricRecord
RoundMetricRecord
RunMetricRecord
AccountingTotals
AccountingBreakdown
```

`PhysicalAttemptAccountingInput` contains only:

```text
run_id
round_index | null
task_id | null
scenario_family_id | null
scenario_id | null
system_variant | null
logical_request_id
attempt_id
attempt_ordinal
role
phase
provider
model
endpoint_kind
dispatched
replayed_after_unknown_outcome
status
started_at_utc | null
completed_at_utc | null
latency_seconds | null
usage
response_sha256 | null
unknown_outcome_kind: timeout | connection | other | null
```

Task 011 persists transport timeouts as `unknown_outcome`. The projection keeps
that authoritative status and adds only the sanitized subtype above. A timeout is
therefore included in `unknown_outcome_attempt_count` and also in the diagnostic
`timeout_attempt_count`; it is not remapped into a contradictory terminal state.

`LogicalRequestAccountingInput` contains the corresponding non-content run,
round/task, logical-request, role/phase/model, terminal logical status, and
`source_attempt_id | null` needed to count `prepared`, `response_completed`,
`applied`, and `terminal_failure` without inferring them from physical attempts.
`source_attempt_id` is required exactly for `applied` and must identify a completed
attempt under that logical request.

`ToolAttemptAccountingInput` contains only run/round/task/scenario identity,
logical call ID, physical tool-attempt ID, profile, execution mode/effect class,
fixture hit/miss where applicable, terminal status, UTC timestamps, and real
monotonic latency. It contains no tool arguments, results, response bodies, or
canonical-name mapping. Tool latency/counts remain separate from model tokens and
the Qwen effective-output cost proxy.

Allowed `system_variant` values are `vanilla`, `generation_0`, `updated`,
or null for training/offline work. Allowed roles include `policy`, `critic`,
`revision`, `skill_updater`, `memory_updater`, `memory_reviewer`,
`embedding`, and `user_simulator`. Unknown roles/phases are rejected rather
than grouped under `other`.

No input or metric record contains prompts, outputs, raw response bytes, exception
text, tool arguments/results, evaluator content, endpoint URLs, headers, or
environment values. Provider/model/config identities are non-secret exact strings.

The later orchestration task maps validated Task 011 rows and Task 010 attempt
records into these inputs. Task 012 does not import `sqlite3`, open the ledger
database, infer status from filesystem state, or define a second source of
request truth.

## Actual Token Aggregation

Use only Task 006 `TokenUsage` returned by the provider/server. Do not tokenize a
prompt, count characters, infer output length, query billing, or synthesize usage.

For one included set of dispatched attempts:

```text
input_tokens
uncached_input_tokens
cache_read_input_tokens
cache_write_input_tokens
output_tokens
total_tokens
usage_complete
```

follow these rules:

1. Validate each complete attempt using Task 006 identities:
   `input = uncached + cache_read + cache_write` and
   `total = input + output`.
2. Include every unique dispatched `attempt_id` exactly once.
3. Identical duplicate records are idempotent; a conflicting duplicate is a hard
   error.
4. If all included attempts have complete actual usage, sum every field with
   unbounded Python integers and set `usage_complete: true`.
5. If any included dispatched attempt lacks any required count, set
   `usage_complete: false` and set every headline aggregate token field,
   including `total_tokens`, to null.
6. An empty set has all six token totals equal to zero and
   `usage_complete: true`.
7. Embeddings retain provider-reported input/total counts with output and cache
   counts equal to zero under the Task 006 contract.
8. User-simulator tokens remain in role `user_simulator`; they are included in
   total experiment tokens but excluded from Policy/Critic/Revision logical-call
   counts and from `total_cost`.

Known counts from an incomplete aggregate may appear only in an optional nested
`diagnostics.observed_partial_usage` object labeled
`experimental_result_eligible: false`. They must never populate headline
fields, tables, comparisons, or acceptance claims.

Breakdowns are emitted by exact:

```text
provider
model
role
phase
system_variant
provider + model
role + phase
```

The overall aggregate is computed directly from attempts, not by adding breakdown
rows. This avoids double counting overlapping dimensions.

## Call and Outcome Counts

Each scope records:

```text
prepared_logical_request_count
dispatched_logical_request_count
applied_logical_response_count
physical_dispatch_count
recovery_dispatch_count
abandoned_before_dispatch_count
completed_attempt_count
failed_attempt_count
timeout_attempt_count
unknown_outcome_attempt_count
incomplete_usage_attempt_count
```

Logical counts deduplicate by `logical_request_id`; physical counts deduplicate by
`attempt_id`. `physical_dispatch_count` includes every dispatched status.
`recovery_dispatch_count` counts dispatched attempts whose immutable replay flag
is true. Outcome categories must be mutually exclusive and sum to the applicable
attempt population.

Role-specific logical call counts for Policy/Critic/Revision count unique
dispatched logical requests. They do not count preparation-only rows, extra
physical retries, embeddings, user simulator, or offline updater/reviewer calls.
Those calls remain visible in their own breakdowns and overall physical totals.

## Monotonic Timing

`metrics/timing.py` supplies explicit scope timers for:

```text
scenario_task
round_online
round_offline
round_total
evaluation_run
training_run
```

Use injected `time.monotonic_ns` and `time.time_ns` callables in tests. A timer
records UTC timestamps for audit display, but elapsed time is calculated only from
monotonic nanoseconds:

```text
total_running_time_seconds =
    Decimal(end_monotonic_ns - start_monotonic_ns) / Decimal(1_000_000_000)
```

Rules:

- Start a scenario timer immediately before its first state dispatch.
- Stop it only after native evaluator output and the durable final scenario
  checkpoint.
- Start a training-round total timer immediately before its first online scenario
  dispatch.
- Stop it only after online evaluation, every offline unit, generation
  publication, and the durable final round checkpoint.
- Start an evaluation-run timer immediately before its first scenario dispatch.
- Stop it only after the last native evaluator result and durable final run
  checkpoint.
- Preflight, dependency installation, operator approval time before dispatch, and
  post-finalization report rendering are outside the experimental boundary.
- Model/tool waits, retries, recovery dispatches, checkpointing, orchestration
  overhead, and same-boot crash/restart downtime inside an open scope are included.
- Never compute task, round, or run wall time by summing request latency, task
  times, or online/offline segments. Those spans may overlap.

Persist the Linux boot identity with each open scope. A resumed scope may subtract
stored monotonic readings only on the same boot. If the boot identity changed,
`timing_complete: false` and `total_running_time_seconds: null`; do not fall
back to UTC subtraction or silently omit crash downtime. Such a run cannot satisfy
formal headline timing acceptance.

Negative elapsed values, stop-before-start, double start/stop with conflicting
values, mismatched scope identity, or non-integral nanosecond inputs fail.
Identical repeated close operations are idempotent.

Per-request `latency_seconds` is the Task 006 gateway boundary measurement.
External-tool attempt latency from Task 010 is a separate non-model breakdown.
Neither substitutes for direct task/round/run timing.

## Effective Qwen Output Cost

`total_cost` is a non-monetary experiment metric. Its unit is the number of
actual Qwen output tokens that passed strict validation and are linked by Task 011
to a committed substantive pipeline effect:

```text
total_cost: int | null
cost_unit: "qwen_effective_output_tokens"
cost_complete: bool
```

It never represents USD, API price, GPU time, electricity, amortization, or any
other currency/resource conversion. Do not create a pricing manifest, retrieve
provider prices, or emit currency fields.

One Qwen logical response contributes exactly when all conditions hold:

1. the provider/model identity is the run-pinned Qwen3-32B endpoint;
2. Task 011 records the logical response as `applied`;
3. the applied marker identifies the single completed physical attempt whose
   validated output was used;
4. a committed `QwenEffectiveEffect` record lists that application as causally
   required for its committed online action or durable offline semantic mutation;
5. that attempt reports an actual non-null `output_tokens` count.

For an online action, count the applied Initial Policy even if Revision later
supersedes it, plus only the Critic/Revision outputs actually consumed by that
committed decision chain. For an offline update, count candidate, reviewer, and
validation-chain Qwen outputs only when they causally produce a persistent semantic
mutation. A logical response that merely completed, was audited, or recorded a
no-change decision does not contribute.

Never include:

- Qwen input, cache-read, or cache-write tokens;
- a Qwen response that failed parsing/schema/reasoning/truncation validation;
- a timeout, transport failure, unknown outcome, or attempt abandoned/rejected
  before dispatch;
- an unused duplicate/recovery attempt whose output was not the applied response;
- a prepared or `response_completed` logical request not yet applied;
- an applied `NONE`, `SKIP`, duplicate/no-op, or failure-mode no-change
  decision;
- a rejected Skill candidate and its Mini-Bench decision chain when no accepted
  Skill mutation commits;
- any applied response lacking a committed substantive-effect link;
- `text-embedding-3-small` tokens;
- GPT-4o mini User Simulator tokens;
- tool calls, evaluator work, or host computation.

Deduplicate cost by the applications listed across committed effect records and
their single source `attempt_id`, not by every physical attempt. If every linked
Qwen response has actual output usage, sum those output counts and set
`cost_complete: true`. If any linked response lacks actual output usage, set
`total_cost: null` and `cost_complete: false`; do not estimate it or sum only
the known outputs. An empty eligible set has `total_cost: 0` and
`cost_complete: true`.

`total_tokens` remains a separate physical-resource metric. It includes actual
input and output tokens from every dispatched model attempt, including retries,
unknown outcomes when usage exists, embeddings, and the User Simulator. Therefore
`total_cost` must never be substituted for or expected to equal
`total_tokens`.

## Immutable Metric Records

`RequestMetricRecord` emits one record per dispatched `attempt_id`, plus
optional lifecycle-only records for attempts abandoned before dispatch. It
contains identities, status, replay flag, latency, actual usage, whether it is the
source of an applied Qwen response, its effective-output contribution, and a
sanitized response hash only.

`TaskMetricRecord` covers one completed/terminal scenario execution and contains
direct task wall time, attempt-derived totals and breakdowns, model call counts,
tool latency/count summaries supplied by later orchestration, evaluator-result
hash, final-context hash, and completion status. It does not reproduce the native
evaluator body.

`RoundMetricRecord` contains direct online, offline, publication, and total
time; the total is measured independently. It contains request/task aggregates,
generation input and published-generation IDs, scenario counts, and completion
status.

There is exactly one immutable `RoundMetricRecord` for each formal round index
0, 1, and 2. Each record independently carries start/end UTC evidence, direct
`total_running_time_seconds`, `total_tokens`/`usage_complete`, and
`total_cost`/`cost_unit`/`cost_complete`, including a terminal partial row
when a round fails. Run-level summaries cannot replace, average, or omit these
three per-round headline records.

`RunMetricRecord` contains one training or one named evaluation run, direct
headline time/tokens, request/task/round aggregates, manifest hashes, profile,
system variant, completion status, and all completeness flags.

Every record has:

```text
schema_version
record_id
recorded_at_utc
run_id
record_kind
content_sha256
```

`record_id` and `content_sha256` are derived from canonical JSON with those
self-identifying fields omitted according to a documented domain-separated
algorithm. IDs cannot depend on file position, current working directory, Python
hash order, locale, or float rendering.

## Artifact Materialization

The output names are exactly:

```text
<run-root>/metrics/request_metrics.jsonl
<run-root>/metrics/task_metrics.jsonl
<run-root>/metrics/round_metrics.jsonl
<run-root>/metrics/run_metrics.json
<run-root>/metrics/manifest.json
```

Task 012 implements an explicit writer rooted at an already authorized absolute
run root. It rejects symlinks, traversal, wrong run identity, and unexpected
existing files. Importing or constructing accounting objects writes nothing.

The Task 011 ledger remains authoritative during execution. Metric files are
deterministic materialized views:

1. validate the complete supplied projection and its ledger high-water mark;
2. canonicalize records and sort request rows by attempt identity, tasks by
   manifest scenario order, and rounds by round index;
3. verify every existing JSONL record is an exact prefix with matching IDs and
   hashes;
4. write the expanded view to a same-directory temporary file with mode `0600`;
5. fsync the file, atomically replace the materialized view, and fsync its parent;
6. write `run_metrics.json` only when the run scope is closed;
7. write `manifest.json` with file hashes, record counts, ledger high-water
   marks, schema versions, `cost_unit`, and pinned Qwen model identity.

“Immutable” means a published record can never be edited, removed, reordered, or
reused with different content. The containing JSONL file may be atomically
rematerialized only to append an exact validated suffix. Identical replay is a
no-op. A non-prefix file, duplicate conflicting ID, changed prior record, missing
source attempt, decreasing high-water mark, or hash disagreement fails closed.

A crash before rename leaves the old valid view authoritative. A crash after
rename is recovered by verifying it against the ledger projection; no attempt or
applied Qwen response is double counted. Artifacts never become a second mutable
checkpoint store.

## Security and Diagnostic Rules

- Never serialize credentials, URLs with credentials, headers, prompts, responses,
  raw blobs, tool inputs/results, evaluator bodies, environment dumps, or traceback
  locals.
- Exceptions contain error type, record/attempt ID, and sanitized field path only.
- `repr`, logs, test output, and artifact content are scanned with secret/raw-body
  sentinels.
- Unknown fields, NaN/infinity, negative counts/times, naive datetimes, non-UTC
  datetimes, unknown cost units, and invalid hashes fail.
- Diagnostic partial totals are always nested, explicitly incomplete, and excluded
  from headline serialization helpers.

## Required Offline Tests

Use synthetic inputs and temporary directories. Cover at least:

1. strict/frozen schema behavior, unknown-field rejection, UTC validation, fixed
   cost-unit validation, and no numeric/string coercion;
2. complete actual usage aggregation across Qwen, embedding, User Simulator,
   offline roles, retry, and recovery attempts;
3. Qwen Policy/Critic/Revision outputs in a committed online-action effect and
   Qwen memory/Skill chains in a committed semantic-mutation effect contributing
   their actual output tokens exactly once;
4. revised actions proving both the applied Initial Policy and applied Revision
   outputs count, without counting Qwen input/cache tokens;
5. invalid/truncated/unapplied/unknown Qwen attempts, applied NONE/SKIP/no-op
   decisions, rejected Skill candidate/Mini-Bench chains, embeddings, User
   Simulator, tools, and evaluator work excluded from `total_cost`;
6. one substantive-effect-linked Qwen output with missing output usage producing
   `total_cost: null`/`cost_complete: false`, without changing the independent
   `usage_complete` rule;
7. unused recovery attempts excluded from cost while all dispatched attempts
   remain in `total_tokens`;
8. one unknown dispatched attempt poisoning headline token totals even when a
   later attempt succeeds;
9. abandoned/rejected-before-dispatch attempts not poisoning usage, while lifecycle
   counts remain correct;
10. attempt-ID deduplication, identical replay idempotence, conflicting duplicate
   rejection, logical versus physical call counts, and role exclusions;
11. exactly one direct immutable headline record for each round 0/1/2, including
    failed-round timing/cost completeness, and no replacement by an average;
12. empty-scope zero totals/cost and incomplete diagnostic partial totals never
   entering headline fields;
13. exact provider/model/role/phase/system-variant breakdowns and proof that the
    overall total is not obtained by double-counting overlapping breakdowns;
14. direct monotonic scenario/round/run timing, overlapping task spans, model/tool
    wait inclusion, non-additivity of child spans, and idempotent close;
15. same-boot recovery including downtime, boot-identity change producing
    incomplete timing, negative/mismatched/double-close denial, and no UTC elapsed
    fallback;
16. deterministic record IDs/order/canonical JSON, file modes, fsync/atomic replace
    paths, exact-prefix append, identical no-op, and conflicting-prefix denial;
17. injected crashes before and after file rename with recovery from authoritative
    projection and no duplicate records;
18. run metrics unavailable until scope close, manifest file hashes/counts/high
    water marks, cost unit, and Qwen model identity propagation;
19. secret, raw response, prompt, tool/evaluator-content, endpoint, and exception
    sentinels absent from repr/log/artifact output;
20. imports/construction perform no filesystem write, ledger open, environment
    read, provider/tool/evaluator call, billing query, or network access.

## Acceptance Commands

The development Agent runs:

```bash
uv sync --frozen
uv run pytest -q tests/metrics
uv run python -c "from toolsandbox_pipeline.metrics import MetricsAggregator, ScopeTimer, MetricsArtifactWriter"
git diff --check
git status --short
```

No acceptance command may access a credential, provider, model server, dataset,
scenario, fixture body, native tool/evaluator, real ledger, or existing run.

## Acceptance Criteria

- Headline time is direct monotonic elapsed time at each required boundary.
- Headline tokens include every dispatched physical model attempt exactly once.
- Logical calls and physical dispatches are separately and correctly counted.
- Missing actual usage propagates null and false completeness without estimation
  or known-only headline totals.
- `total_cost` counts only actual output tokens from Qwen responses linked to a
  committed substantive effect, deduplicated by application/source attempt and
  labeled with its fixed non-monetary unit.
- Applied NONE/SKIP/no-op and rejected Skill candidate chains contribute zero to
  `total_cost` while remaining present in `total_tokens`.
- User Simulator and embedding usage remain separately attributable and contribute
  to `total_tokens`, but never to `total_cost`.
- Request/task/round/run records are strict, deterministic, sanitized, recoverable,
  and immutable at the record level.
- Artifact materialization reconciles only against supplied authoritative ledger
  projections and never opens or mutates Task 011 storage.
- Offline tests pass with zero external access and no unowned files are modified.

## Completion Report Additions

Include:

```text
Pinned Qwen model identity:
Cost unit: qwen_effective_output_tokens
Substantive Qwen effect links counted:
Qwen outputs excluded:
Cost completeness cases:
Usage completeness cases:
Logical requests / physical dispatches / recovery dispatches:
Timing scopes and boot-change cases:
Round 0/1/2 direct latency and total_cost records:
Artifact files and schema versions:
Ledger high-water reconciliation cases:
Crash points tested:
Sensitive-output scan:
External calls: 0
Total tokens: 0
Usage complete: true
Dependency/interface change requests:
```

These are development-test facts. Unit-test duration and synthetic token fixtures
must not be reported as experimental `total_running_time_seconds`,
`total_tokens`, or `total_cost`.
