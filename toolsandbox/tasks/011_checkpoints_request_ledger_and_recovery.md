# Task 011: Checkpoints, Request Ledger, and Crash Recovery

Status: `approved`

## Objective

Implement the durable transaction layer used by later online, offline, and
evaluation orchestration. This task owns:

- content-addressed restricted blobs and atomic checkpoint snapshots;
- stable logical LLM request identities and unique physical-attempt identities;
- prepare/in-flight/completed/applied LLM state transitions;
- durable raw-response and validated-output persistence before application;
- pre-action and committed ToolSandbox ExecutionContext persistence;
- safe replay classification for local, fixture-backed, and live tool actions;
- idempotent offline-unit and generation-publication commit markers;
- deterministic recovery plans and injected crash-point tests.

It provides primitives only. It does not build prompts, call a provider directly,
run an episode/tool/evaluator, select a dataset, aggregate
metrics, update memories/skills, publish a generation, or decide whether an
experiment should continue after a terminal failure.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3-6, 15-17, and 20-25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. `tasks/002_core_contracts.md`;
6. `tasks/004_toolsandbox_adapter.md`;
7. `tasks/005_tool_metadata_controller.md`;
8. `tasks/006_model_api_gateways.md`;
9. `tasks/009_dataset_manifest_and_split_access.md`;
10. `tasks/010_rapidapi_fixtures_and_network_boundary.md`;
11. this task file;
12. only the pinned upstream `ExecutionContext.to_dict/from_dict` and native
    ExecutionEnvironment behavior required for context recovery.

Do not inspect real scenario definitions, prompts, databases, evaluator targets,
dev/test IDs, fixture bodies, or prior experiment trajectories.

## Access

```text
Access class: offline
Setup network: none
Data splits: none
Secrets: none
External operations: none
```

All tests use synthetic contexts, responses, actions, ledger rows, and subprocess
crash injection under temporary directories. No model, OpenAI/RapidAPI credential,
dataset manifest, real scenario, tool, evaluator, or network access is permitted.

## Preconditions

- Tasks 001-006 and 009-010 are complete and accepted.
- Task 002 canonical JSON and strict action/controller/critic contracts are stable.
- Task 006 exposes immutable `RequestContext`, `PhysicalAttemptResult`, actual
  usage, response hash, and the raw-response interface amendment below.
- Task 009 exposes canonical non-console ExecutionContext hashing and environment
  identity.
- Task 010 exposes external-read mode/effect evidence sufficient to distinguish
  fixture replay from `official_live` unknown outcomes.
- Only Python 3.10 standard-library `sqlite3`, `hashlib`, `json`, `base64`,
  filesystem APIs, and already locked dependencies may be used.

## Satisfied Task 006 Interface Amendment

Section 21 requires the raw provider response to be durably stored before its
validated value is applied. The coordinator has amended Task 006-owned code and
tests so that:

```python
GatewayResponse.raw_response_body: bytes  # frozen, repr=False
ProviderRequestError.raw_response_body: bytes | None  # repr-hidden
```

Requirements:

- a successful gateway response carries the exact `TransportResponse.raw_body`;
- an error after a response was received may carry the exact bytes for restricted
  failure audit/calibration, while preparation/transport failures carry `None`;
- raw bytes never appear in `repr`, exception text, preflight output, ordinary
  logs, or metrics;
- the existing response hash must equal SHA-256 of those exact bytes;
- provider gateways still perform one physical dispatch and do not write files or
  import checkpoint code.

This remains Task 006 file ownership. The Task 011 Agent must verify the interface
is present before implementation; if it is absent or incompatible, stop and
request it rather than wrapping transports, fabricating raw responses, or editing
provider files.

## Owned Files

The assigned Agent may create or edit only:

```text
configs/reproducibility/checkpointing_v1.json
src/toolsandbox_pipeline/schemas/checkpoint.py
src/toolsandbox_pipeline/checkpointing/__init__.py
src/toolsandbox_pipeline/checkpointing/blob_store.py
src/toolsandbox_pipeline/checkpointing/execution_context_codec.py
src/toolsandbox_pipeline/checkpointing/store.py
src/toolsandbox_pipeline/checkpointing/identities.py
src/toolsandbox_pipeline/checkpointing/llm_ledger.py
src/toolsandbox_pipeline/checkpointing/tool_ledger.py
src/toolsandbox_pipeline/checkpointing/unit_ledger.py
src/toolsandbox_pipeline/checkpointing/recovery.py
tests/checkpointing/test_blob_store.py
tests/checkpointing/test_execution_context_codec.py
tests/checkpointing/test_store.py
tests/checkpointing/test_identities.py
tests/checkpointing/test_llm_ledger.py
tests/checkpointing/test_tool_ledger.py
tests/checkpointing/test_unit_ledger.py
tests/checkpointing/test_recovery.py
tests/checkpointing/test_crash_injection.py
```

Do not edit providers, metrics, online/offline code, Adapter, Controller, dataset or
fixture code, dependencies, prompts, generation stores, CLI entry points, upstream
source, or generated run artifacts.

## Versioned Durability Configuration

`configs/reproducibility/checkpointing_v1.json` is a strict non-secret contract:

```yaml
schema_version: 1
checkpoint_protocol_version: checkpoint-v1
hash_algorithm: sha256
database: sqlite3
journal_mode: DELETE
synchronous: FULL
foreign_keys: true
busy_timeout_ms: 0
single_writer: true
file_mode: "0600"
directory_mode: "0700"
max_blob_bytes: 268435456
max_checkpoint_bytes: 1073741824
```

Unknown fields, coercion, different SQLite pragmas, disabled fsync/foreign keys,
multi-writer mode, weaker permissions, relative size strings, or unbounded blobs
fail validation. Formal manifests pin the exact config path and SHA-256.

The store requires an explicit absolute run root, rejects symlinks/path traversal,
and never invents a default under the current working directory. It creates only
its assigned new run directory or opens an existing directory whose run/config/
environment identities match exactly.

## Restricted Content-Addressed Blobs

`blob_store.py` writes immutable values under:

```text
<run-root>/checkpointing/blobs/sha256/<first-two-hex>/<remaining-hex>
```

Blob identity is SHA-256 of exact bytes. Writes use a same-directory temporary
file, mode `0600`, file fsync, atomic non-overwriting rename/link semantics, and
parent-directory fsync. A collision path with different bytes is a hard error.
Identical existing bytes are verified and reused.

Every blob reference contains:

```text
sha256
byte_count
media_type
schema_name
schema_version
content_visibility
```

Allowed media types are canonical JSON, UTF-8 JSONL, raw provider JSON bytes, and
the explicit ExecutionContext checkpoint envelope. Unknown media/schema,
oversized content, noncanonical JSON where canonical form is required, symlinks,
wrong owner/mode, short reads, hash/size mismatch, or unexpected mutable files fail
before deserialization.

Blobs are never deleted or garbage-collected during an active run. A crash may
leave an unreferenced fully written blob; recovery ignores it. It must never leave
a committed reference to a missing blob.

Restricted blobs may contain model responses, visible trajectories, native
contexts, and post-episode evaluator records. They are not prompt inputs by virtue
of being stored. Secret values, headers, environment dumps, or credentials are
forbidden in every blob.

## ExecutionContext Checkpoint Codec

ToolSandbox's stateful `InteractiveConsole` must survive recovery without
re-executing already committed tool effects. `execution_context_codec.py` therefore
uses the pinned upstream:

```python
ExecutionContext.to_dict(serialize_console=True)
ExecutionContext.from_dict(...)
```

The envelope contains:

```text
codec_version
upstream_commit
python_patch_version
environment_identity
non_console_context_sha256
serialized_context_with_base64_console
serialized_context_sha256
```

Convert only the upstream console bytes to canonical base64; canonicalize all
remaining dict/table/enum content with the Task 009 rules. The non-console hash
must equal Task 009's canonical context hash before and after round-trip. The full
checkpoint hash is local recovery identity and is not required to be identical
across independently executed runs.

Deserializing console bytes can execute pickle/dill reconstruction logic. It is
allowed only for coordinator-owned local blobs under the validated run root after
file owner/mode, exact hash, run/environment/upstream/Python identity, size, codec,
and schema checks all pass. Never deserialize user input, downloaded content,
provider output, fixtures, a different run's blob, or an unresolved environment.

Codec calls are explicit and side-effect free until invoked. Decode returns a new
deeply isolated `ExecutionContext`; it never installs it as current context.

## SQLite Transaction Store

`store.py` owns one SQLite database:

```text
<run-root>/checkpointing/ledger.sqlite3
```

On every open, validate application ID, schema version, run ID, profile,
environment identity, dataset/config/prompt/generation/fixture hashes, permissions,
and required pragmas. Schema migration and best-effort repair are forbidden in
this task. An unknown/newer/partial schema stops recovery.

All mutations use `BEGIN IMMEDIATE`, one writer, explicit foreign keys, and a
commit before returning. The database stores identifiers, statuses, canonical
fingerprints, blob references, attempt metadata, event ordinals, and timestamps;
large/raw content lives only in verified blobs.

The authoritative tables cover:

```text
run_identity
checkpoint_events
checkpoint_components
logical_llm_requests
physical_llm_attempts
llm_response_applications
tool_action_transactions
tool_execution_attempts
offline_unit_transactions
generation_publication_transactions
```

Each row has strict allowed state transitions. UPDATE is permitted only for the
next defined state with immutable identity fields unchanged. DELETE, replace,
upsert-overwrite, status rewind, ID reuse with changed inputs, and cascade deletion
are forbidden.

## Stable Identities

`identities.py` uses Task 002 canonical JSON bytes and lowercase SHA-256.

Logical LLM request identity is:

```text
logical_request_id = "llm-" + SHA256(canonical JSON of
  {
    "protocol": "llm-request-v1",
    "run_id": run_id,
    "role": role,
    "phase": phase,
    "unit_reference": state_id_or_offline_unit_id,
    "input_fingerprint": canonical_input_fingerprint,
    "model": exact_model,
    "decoding_configuration_sha256": decoding_configuration_sha256,
    "output_schema_sha256": output_schema_sha256_or_null
  }
).hexdigest()
```

The same inputs in one run produce the same ID. Different role, phase, state/unit,
prompt input, model, decoding/token limit, structured-output mode, or schema must
produce a different ID. Calibration attempts with a different `max_tokens` are
different logical requests, not retries.

Each physical attempt ordinal is allocated durably under its logical request. Its
ID is:

```text
attempt_id = "attempt-" + <logical hex> + "-" + zero-padded attempt ordinal
```

The ordinal is never reused after crash, failure, or unknown outcome. IDs are
validated against Task 006 `RequestContext` before dispatch.

Tool-action transaction identity is derived from run/scenario, pre-action context
hash, canonical final `ActionEnvelope` fingerprint, ordered call IDs, and action
ordinal. Offline-unit/publication identities similarly include run/round,
generation input, unit kind/key, canonical input fingerprint, and expected output
artifact identities.

## LLM Request State Machine

Logical request statuses are:

```text
prepared → response_completed → applied
prepared → terminal_failure
```

Physical attempt statuses are:

```text
allocated → abandoned_before_dispatch
          → in_flight → completed
                      → rejected_before_dispatch
                      → failed
                      → unknown_outcome
```

The exact workflow is:

1. validate/derive the logical ID and atomically insert `prepared` plus the complete
   immutable request identity before a provider call;
2. write the required pre-request checkpoint event;
3. allocate a never-reused attempt ID as `allocated`, then atomically mark it
   `in_flight` immediately before calling Task 006;
4. call one gateway once with the matching `RequestContext`;
5. atomically store its full `PhysicalAttemptResult`, raw-response blob when one
   exists, response hash, actual usage, and terminal attempt status;
6. for a valid response, store the exact strict validated-output canonical JSON
   blob and atomically move the logical request to `response_completed` before any
   downstream application;
7. later orchestration computes the deterministic post-application state without
   external side effects, then atomically stores that state/checkpoint and the
   `applied` marker before installing the stored state as current. The application
   row stores a foreign-keyed `source_attempt_id` identifying the one completed
   physical attempt whose validated output was applied; it must belong to the same
   logical request and can never change.

The raw-body hash, attempt response hash, and validated-output relationship are
verified. Do not parse, repair, or regenerate provider output in the ledger.

### Substantive-effect cost links

An `applied` logical response is necessary but not sufficient for
`total_cost`. The ledger also stores append-only `QwenEffectiveEffect` records
created atomically with a committed externally meaningful pipeline effect:

```text
effect_id
effect_kind: committed_online_action | policy_memory_mutation |
             world_memory_mutation | failure_mode_mutation |
             accepted_skill_mutation
effect_artifact_id
effect_artifact_sha256
ordered_application_ids
committed_checkpoint_id
```

Each listed application identifies its immutable source attempt. One application
may belong to at most one effective-effect record. Online records contain only
Policy/Critic/Revision outputs actually consumed by the final action committed to
the episode. Offline records are created only for a real durable semantic
mutation; they may include the candidate/reviewer/validation chain causally
required for that mutation.

Never create an effective-effect record for `NONE`, `SKIP`, a duplicate/no-op,
an audit/checkpoint/metrics write, or a rejected Skill candidate/Mini-Bench unit
that leaves no accepted Skill mutation. Those logical responses remain applied
and fully auditable, and every physical attempt remains available to
`total_tokens`, but their effective-output contribution is zero. Effect records
are immutable and idempotent; recovery may reuse an identical record but cannot
attach new applications or change eligibility after commitment.

Recovery rules:

- `applied`: load the committed post-application checkpoint; never dispatch or
  apply again;
- `response_completed`: return the stored validated output and raw-response
  evidence without dispatch, then complete application once;
- `prepared` with no in-flight attempt: a new first attempt is allowed;
- an attempt left `allocated` is durably marked `abandoned_before_dispatch`; a new
  attempt may be authorized with `replayed_after_unknown_outcome=false` because
  dispatch was never permitted before the committed `in_flight` transition;
- an attempt left `in_flight` after process death becomes `unknown_outcome`; a new
  physical attempt for the same logical ID may be authorized with
  `replayed_after_unknown_outcome=true`;
- `rejected_before_dispatch`: orchestration may authorize a new attempt only after
  the unchanged request is still valid; it is never labeled replay-after-unknown;
- timeout/connection/unknown result: retain incomplete usage and all attempt rows;
- provider-output/schema/reasoning/truncation failures are terminal for that
  logical request, except that token calibration may create a different logical
  request with a larger pinned limit;
- a completed response with missing actual usage remains reusable, while all later
  token aggregates must be marked incomplete; substantive-effect Qwen-output cost
  is incomplete only if a response linked by a committed effect record lacks
  actual output usage.

No ledger method loops, sleeps, backs off, or retries a provider.

## Tool Action State Machine

A logical tool-action transaction covers one final routed `ActionEnvelope`. A
`parallel_batch` is one logical transaction around the native all-permutations
execution; Task 010 still records every external physical read produced by
permutations.

Logical transaction statuses are:

```text
prepared → committed
         → failed_committed
         → reconciliation_required
```

Every native ExecutionEnvironment invocation has a never-reused physical tool
attempt ID and status:

```text
allocated → abandoned_before_execution
          → in_flight → completed
                      → unknown_outcome
```

Before native execution, persist:

- transaction/call IDs and action fingerprint;
- ordered native messages/call IDs;
- Controller blocking-check result and effect classes;
- pre-action complete ExecutionContext blob and non-console hash;
- profile plus pinned fixture/backend manifest identities;
- a pre-action checkpoint event.

Allocate an attempt and commit `in_flight` immediately before native execution.
Immediately after native execution, persist the complete post-action context,
non-console hash, visible native result/exception identity, external-read attempt
references, physical attempt completion, and logical committed status before
exposing the new context to later online state construction.

Recovery rules:

- committed/failed-committed: restore the stored post-action context and never
  execute again;
- an allocated attempt abandoned before execution is marked accordingly; allocate
  a new attempt from the stored pre-action context without claiming an unknown
  outcome;
- prepared with no in-flight attempt: execute once from the stored pre-action
  context;
- abandoned in-flight containing only `sandbox_read`/`sandbox_write`: mark that
  physical attempt `unknown_outcome`, reset to the stored pre-action context, and
  allow one new physical execution attempt;
- abandoned in-flight with `external_read` under `strict_replay`: allow replay only
  through the identical pinned fixture store using a new physical execution
  attempt;
- abandoned in-flight with any `official_live` external read: mark
  `reconciliation_required`, checkpoint, and stop; never retry automatically;
- changed call ID/action fingerprint/pre-context/profile/fixture identity is a hard
  error.

Every physical tool attempt remains recorded. The ledger does not claim that a live
external call did or did not happen when its outcome is unknown.

## Offline Unit and Generation Publication Markers

`unit_ledger.py` provides generic idempotency primitives for later tasks:

```text
prepared → outputs_staged → committed
prepared/outputs_staged → terminal_failure
```

An offline unit records its current-round input buffer hash, generation input,
unit type/key, request IDs, staged-output blob hashes, and final committed artifact
hashes. Re-entry with identical committed inputs returns the stored outputs; any
identity change under the same unit ID fails.

A generation publication transaction records expected staging manifest/content
hashes, mini-bench decision hash where applicable, destination generation ID, and
pre/post publication checkpoints. This task stores markers only. Later generation
code owns validation, filesystem staging, atomic rename, and reader activation.

## Checkpoint Snapshots

Every checkpoint has a strictly increasing event ordinal and phase-specific event
kind. The snapshot contains direct values or verified blob references for at least:

```text
run/profile/phase/round/shard/family/scenario/state identity
completed scenario IDs
serialized ExecutionContext
pinned Policy/World/Skill generation IDs
retrieval results
Controller/Critic/Revision state
current trajectory buffer
staged memory and Skill updates
Skill statistics
failure-mode buffers
config/prompt/schema/dataset/fixture/environment hashes
Python and component random states
metrics accumulators
LLM/tool/offline/publication ledger high-water marks
```

Fields irrelevant to the current phase are explicit `null` or empty values, not
silently omitted. Later tasks supply already validated component payloads with
schema name/version; Task 011 stores and verifies them without weakening their
contracts or exposing a generic unvalidated pickle field.

Required event kinds include after Agent-visible message; before/after each LLM
request and tool action; timeout/hard failure; after evaluator result; after every
offline unit; and before/after generation publication. Later orchestrators are
responsible for calling these primitives at the required boundaries.

The authoritative checkpoint row and component references commit in SQLite first.
An optional canonical snapshot export and `latest` pointer are then written
atomically. On disagreement, recovery trusts the verified database and repairs only
the derived pointer/export by creating a new file; it never changes authoritative
history.

## Recovery Planning

`recovery.py` is pure decision logic over a validated durable snapshot. It returns
one strict plan such as:

```text
reuse_completed_llm_response
dispatch_first_llm_attempt
dispatch_recovery_llm_attempt
apply_stored_llm_response
restore_committed_tool_context
replay_local_tool_from_pre_context
replay_fixture_tool_from_pre_context
resume_offline_unit
finalize_generation_publication
reconciliation_required
terminal_failure
run_complete
```

It never performs the action, contacts a service, changes a status, or guesses from
filesystem timestamps. Ambiguous/missing/corrupt/conflicting state returns a hard
error rather than the most convenient plan.

Resume requires exact run/profile/environment/dataset/config/prompt/schema/model/
generation/fixture hashes. A caller cannot resume under updated code, prompt,
token-limit config, fixture generation, split manifest, world clock, or model
deployment and call it the same run.

## Required Tests

Use temporary directories and synthetic strict records. Cover at least:

1. exact durability config, SQLite pragmas, permissions, absolute-root and
   symlink/path rejection;
2. blob SHA-256 golden vectors, atomic publication, parent fsync path, identical
   reuse, collision/missing/extra/wrong-mode/oversize rejection;
3. ExecutionContext full-console round trip plus stable Task 009 non-console hash;
4. refusal to deserialize wrong-run/environment/upstream/Python/hash/schema blobs;
5. logical request and attempt ID golden vectors and sensitivity to every identity
   field, including token limits and schema;
6. every legal LLM state transition and rejection of skip/rewind/overwrite/reused
   identity transitions;
7. raw response persisted and hash-verified before response reuse/application;
8. completed response reuse with zero second physical dispatch;
9. unknown attempt recovery with same logical ID, new attempt ID, and correct
   replay flag;
10. preservation of every physical attempt and incomplete actual usage;
11. at-most-once response application with committed post-state restoration;
12. pre/post tool context, call/action identity, batch transaction, and visible
    result/exception persistence;
13. local and fixture-backed safe replay versus `official_live`
    reconciliation-required behavior;
14. idempotent offline-unit and publication markers with changed-input rejection;
15. complete checkpoint component/event/high-water-mark validation;
16. recovery-plan truth table for every supported durable state;
17. corruption, torn derived pointer/export, orphan blob, stale lock, and partial
    authoritative-state failures;
18. immutable substantive-effect links, source-application uniqueness, online
    committed-action chains, offline mutation chains, and zero cost eligibility
    for NONE/SKIP/no-op/rejected candidates;
19. subprocess termination at every boundary before/after LLM allocation,
    in-flight mark, response persistence, application commit, tool pre/in-flight/
    post commit, offline unit, checkpoint export, and publication marker;
20. crash tests prove at-most-once logical application/tool/offline/publication
    commitment while retaining all possible physical attempts;
21. repr/log/exception/output scanning for synthetic response, context, evaluator,
    secret, header, and environment sentinels;
22. imports/construction perform no directory creation, database open, context
    decode, environment read, provider/tool/evaluator call, or network access.

## Acceptance Commands

The development Agent runs:

```bash
uv sync --frozen
uv run pytest -q tests/checkpointing
uv run python -c "from toolsandbox_pipeline.checkpointing import CheckpointStore, RecoveryPlanner"
git diff --check
git status --short
```

No acceptance command may use a real provider response, credential, dataset,
scenario, fixture body, native tool execution, evaluator, or network service.

## Acceptance Criteria

- Durable identities and state transitions match Sections 21-22 exactly.
- A valid raw response and strict output are durable before one-time application.
- Completed responses are reused without another physical dispatch; unknown
  outcomes retain every attempt and permit only the specified recovery attempt.
- Native tool effects restore from committed contexts and never commit twice.
- Local/fixture replay is explicit; live unknown outcomes stop for reconciliation.
- Offline units and generation publication have idempotent commit markers without
  implementing their domain behavior.
- Checkpoints contain all required state through strict schema-bound references.
- Corruption or configuration drift fails closed; no best-effort repair changes
  authoritative history.
- Restricted content and secrets cannot leak through ordinary diagnostics.
- Offline and crash-injection tests pass with zero external access.
- No unowned or upstream file is modified.

## Completion Report Additions

Include:

```text
Checkpoint-config path and SHA-256:
Task 006 raw-response interface present:
SQLite schema/application version and pragmas:
Blob/context codec cases:
LLM state/recovery cases:
Physical attempts retained:
Tool state/recovery cases:
Offline/publication idempotency cases:
Crash points tested:
At-most-once assertions:
Corruption/drift denial cases:
Sensitive-output scan:
External calls: 0
Total tokens: 0
Usage complete: true
Dependency/interface change requests:
```

Unit-test wall time is development evidence only. It is not experimental
`total_running_time_seconds`.
