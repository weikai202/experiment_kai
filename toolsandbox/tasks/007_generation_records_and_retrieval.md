# Task 007: Generation Records, Embedding Cache, and Deterministic Retrieval

Status: `approved`

## Objective

Implement the immutable online-read boundary for generation-scoped Policy memory,
World memory, and Skill records, plus strict `text-embedding-3-small` caching and
deterministic vector retrieval.

This task owns record validation, complete-generation loading, retrieval-text
serialization, embedding-cache transactions, generation index build/load, Skill
prefiltering, and immutable retrieval results. It does not generate or rewrite
records, update statistics, select datasets, build prompts, call Qwen, route
actions, publish generations, write checkpoints, or aggregate experiments.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3, 7-10, 14, 17-22, and 25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. `tasks/002_core_contracts.md`;
6. `tasks/003_compact_state.md`;
7. `tasks/004_toolsandbox_adapter.md`;
8. `tasks/005_tool_metadata_controller.md`;
9. `tasks/006_model_api_gateways.md`;
10. <https://developers.openai.com/api/reference/resources/embeddings/methods/create>;
11. this task file.

## Access

```text
Access class: real_data
Setup network: none
Implementation-test data splits: none
Deferred external-validation data splits: train only
Secrets: real-data-runner only
Allowed external operations: assigned retrieval train smoke only
```

The assigned development Agent uses literal records, fake embedding transports,
and temporary directories only. It receives no credential and does not load an
upstream scenario.

After the dataset manifest and Generation-0 seed artifacts exist, the coordinator
may run the explicitly assigned retrieval smoke on manifest-listed train scenarios.
Dev and test scenarios are forbidden. This deferred check is required before
retrieval is considered externally validated, but its unavailable prerequisites do
not block offline implementation.

## Preconditions

- Tasks 001-006 are complete and accepted.
- Task 006 exposes the strict Embedding Gateway and physical-attempt records.
- Task 002 canonical JSON helpers and Task 003 `CompactVerifiedState` are stable.
- Task 005 exposes `RetrievedSkillControllerView` and predicate semantics.
- Use only locked dependencies and Python 3.10 standard-library `sqlite3`,
  `json`, and `math`. Do not add NumPy, FAISS, a vector database, BM25, or a
  tokenizer dependency.

## Owned Files

The assigned Agent may create or edit only:

```text
src/toolsandbox_pipeline/schemas/memory.py
src/toolsandbox_pipeline/schemas/skill.py
src/toolsandbox_pipeline/schemas/generation.py
src/toolsandbox_pipeline/memory/__init__.py
src/toolsandbox_pipeline/memory/store.py
src/toolsandbox_pipeline/skills/__init__.py
src/toolsandbox_pipeline/skills/store.py
src/toolsandbox_pipeline/skills/views.py
src/toolsandbox_pipeline/retrieval/__init__.py
src/toolsandbox_pipeline/retrieval/contracts.py
src/toolsandbox_pipeline/retrieval/queries.py
src/toolsandbox_pipeline/retrieval/embedding_cache.py
src/toolsandbox_pipeline/retrieval/index.py
src/toolsandbox_pipeline/retrieval/service.py
tests/memory/test_records.py
tests/memory/test_generation_store.py
tests/skills/test_records.py
tests/skills/test_views.py
tests/retrieval/test_queries.py
tests/retrieval/test_embedding_cache.py
tests/retrieval/test_index.py
tests/retrieval/test_service.py
```

Do not edit existing shared schemas, provider gateways, Controller/Adapter code,
dependency files, prompts, seed artifacts, dataset manifests, checkpoint code,
offline updaters, CLI code, or generated artifacts.

## Strict Record Contracts

All public records are frozen strict Pydantic v2 models. Unknown fields, scalar
coercion, non-finite numbers, duplicate list items, invalid conditional fields, and
caller-owned mutation are rejected. JSONL loaders preserve no unvalidated dicts.

### Skill state predicates

`schemas/skill.py` defines a `SkillStatePredicate` matching Section 9:

```text
path
op
value, only when required
```

- `path` is an RFC 6901 pointer into `CompactVerifiedState`.
- `exists/not_exists` omit `value`.
- `eq/neq/in/contains` require `value`.
- `in` requires an array value.
- Comparisons use strict JSON type and value equality.

This wire record has no Controller `code`. The view adapter creates a stable,
opaque code from `skill_id`, predicate group, and zero-based index when converting
it to Task 005's `MetadataPredicate`; it does not change predicate semantics.

### Policy memory

`PolicyMemory` contains exactly:

```text
memory_id
scope
applicability
action_guidance
avoid
evidence_trajectory_ids
support_count
success_rate
confidence
created_version
status
```

- `memory_id` matches `pm_<64 lowercase hex characters>`.
- Natural-language fields contain 1-512 characters.
- Lists contain at most five unique items unless they are evidence IDs.
- Evidence IDs are unique and stored in UTF-8 byte order.
- Counts are non-negative; rates are finite and in `[0, 1]`.
- `success_rate * support_count` is integral within a documented fixed tolerance.
- `confidence` equals `support_count / (support_count + 2)` within that tolerance.
- `created_version` is a generation ID; `status` is `active | deprecated`.

### World memory

`WorldMemory` contains exactly:

```text
memory_id
action_pattern
state_conditions
schema_conditions
likely_error_codes
outcome_calibration
correction_principle
evidence_trajectory_ids
support_count
empirical_failure_rate
confidence
created_version
status
```

It applies the same ID, text, uniqueness, ordering, count, rate, confidence, and
status rules as Policy memory, using the `wm_<64 lowercase hex characters>`
prefix. `likely_error_codes` uses the fixed Critic error-code enum. World memory
cannot contain a complete replacement action.

### Skill records

`SkillRecord` and its nested records exactly represent Section 9:

```text
skill_id
name
description
applicability
required_inputs
expected_outputs
tool_dependencies
success_criteria
failure_mode_buffer
cost_profile
risk_profile
instruction
online_statistics
validation
version
status
```

Requirements:

- IDs and natural-language fields are non-empty; collections are immutable and
  contain no duplicates.
- `tool_dependencies` contains canonical execution-facing tool names in sorted
  UTF-8 byte order.
- `version` matches `v1.<non-negative integer>`; status is
  `active | deprecated`.
- Counters and rates are internally consistent:
  `evaluated_uses = successes + failures` and
  `success_rate = successes / evaluated_uses`, with zero when unused.
- A failure-mode buffer contains at most five unique stable mode IDs, non-negative
  support, and host-owned non-negative observation sequence numbers.
- Validation counts/sums are finite, non-negative, and internally possible.
- Host-owned statistics, evidence, validation, failure buffers, status, and version
  are never accepted as model-produced semantic content by this task.

To preserve ToolSandbox name scrambling, `name`, `description`,
`best_used_when`, `expected_outputs`, `success_criteria`, failure-mode text,
and `instruction` must not contain any canonical tool identifier from the pinned
public inventory as an exact token. Tool identity may occur only in the structured
`tool_dependencies` field. A violation stops loading; text is never rewritten.

## Complete Generation Contract

`schemas/generation.py` defines:

```python
GenerationFileEntry
RetrievalIndexEntry
GenerationManifest
GenerationSnapshot
```

Generation IDs match `g<three decimal digits>`. A complete generation directory is:

```text
gNNN/
  manifest.json
  policy_memory.jsonl
  world_memory.jsonl
  skills.jsonl
  retrieval_indexes/
    policy.jsonl
    world.jsonl
    skill.jsonl
    manifest.json
```

The manifest pins at least:

- schema version, generation ID, optional parent generation ID, and
  `publication_status: complete`;
- pinned upstream commit and Controller tool-inventory hash;
- exact embedding provider, base-URL identity without credentials, model,
  `encoding_format: float`, omitted dimensions setting, and observed vector
  dimension;
- record counts and SHA-256 hashes for all three JSONL stores and all four index
  files.

The loader:

1. accepts an explicit generation directory, never “latest” or a mutable staging
   path;
2. rejects symlinks, path traversal, unexpected files, incomplete publication,
   wrong generation/parent identity, and hash/count mismatch;
3. requires memory records sorted by `memory_id` and Skill records sorted by
   `(skill_id, numeric version)`;
4. rejects duplicate IDs or duplicate Skill versions;
5. requires exactly one active latest version per `skill_id`; older versions are
   deprecated;
6. verifies that every `tool_dependencies` value exists in the pinned canonical
   public-tool inventory;
7. returns one immutable `GenerationSnapshot` and closes all files.

This task reads completed generations and may build index files inside an explicit
staging directory. Atomic generation publication and “last complete generation”
selection belong to the later transaction task.

## Retrieval Text and Query Serialization

Serialization is versioned and deterministic. It uses Task 002 canonical JSON
bytes, decoded as UTF-8, with no prose template, timestamps, environment data, or
locale-dependent formatting.

`RetrievalStateProjectionV1` is constructed only from one validated
`CompactVerifiedState` and contains exactly:

```text
query_version = retrieval-state-v1
visible_messages = sender, recipient, and verbatim content only
current_observation = verbatim content only
verified_facts = fact pointer, value, agent-facing tool name, and result pointer
completed_tool_calls = agent-facing tool name, arguments, result, structured flag
failed_actions = agent-facing tool name and visible error
pending_dependencies = agent-facing tool name, prerequisite code, and description
available_tool_names = ordered agent-facing names only
conversation_status
```

It excludes episode, scenario, family, state, message, call, and metadata-record
IDs; agent-turn counters; source references; full tool JSON Schemas; canonical
names; and every Controller/offline field. Nothing is summarized or inferred.

`RetrievalActionProjectionV1` preserves the validated action's type and semantic
content: agent-facing tool name and arguments, ordered batch calls, or assistant
content. It excludes `call_id` and `selected_skill_id`.

- Policy/Skill query input is canonical JSON of
  `RetrievalStateProjectionV1`.
- World query input is canonical JSON of exactly
  `{"state": <RetrievalStateProjectionV1>, "action": <RetrievalActionProjectionV1>}`.
- Controller provenance sidecars, canonical name mappings, hidden databases,
  evaluator data, Critic output, and user-simulator state are rejected by type and
  never serialized.

Retrieval documents contain semantic fields only:

- Policy memory: `scope`, `applicability`, `action_guidance`, and `avoid`;
- World memory: `action_pattern`, state/schema conditions, likely error codes,
  outcome calibration, and correction principle;
- Skill: tool-name-free semantic fields, structured predicates, cost/risk profiles,
  and instruction.

IDs, evidence IDs, statistics, confidence, generation, version, status, validation,
failure buffers, and canonical `tool_dependencies` are excluded from embedding
documents. Each exact UTF-8 input has a canonical `sha256:` content hash.

Every query or document input must be non-empty and at most 8,191 cl100k_base tokens, using the pinned offline vocabulary.
Over-limit input fails before provider dispatch; it is never truncated, summarized,
split, or replaced by a hash-only embedding input.

## Embedding Cache

`EmbeddingCache` is a project-owned SQLite store. It is opened only from an
explicit path and never at import. Its schema version and migrations are strict;
unknown or newer schemas fail without modification.

An item key contains exactly:

```text
provider
base_url_identity
model
encoding_format
dimensions_setting
input_sha256
```

For this project the provider/model/encoding are fixed to
`openai/text-embedding-3-small/float`, and `dimensions_setting` records the
omitted setting explicitly. The cache never stores credentials, raw URLs containing
credentials, headers, or raw input text.

Each vector row stores finite non-empty float values, dimension, vector-payload
hash, sanitized response hash, and a source `attempt_id`. A separate
cache-provenance row stores that source attempt ID and its immutable usage snapshot
once for the whole batch; vectors only reference it. This cache provenance is not
the authoritative request ledger, does not contain raw response content or request
status transitions, and is never used directly for experiment aggregation. The
later request-ledger task remains the sole owner of authoritative attempt
persistence.

Resolution rules:

1. validate and hash every ordered input;
2. resolve exact hits and verify every stored hash/vector/dimension;
3. deterministically deduplicate misses by first occurrence;
4. send the remaining ordered misses in deterministic manifest-pinned batches of
   at most 2,048 items and 280,000 aggregate UTF-8 bytes through Task 006's
   Embedding Gateway, using a caller-supplied `RequestContext` per physical batch;
5. after the caller has durably recorded Task 006's result in the authoritative
   ledger, commit the cache-provenance row and all returned item rows in one SQLite
   transaction;
6. return vectors in the caller's original order.

A cache hit creates no physical attempt and contributes zero new tokens to the
current run. Historical usage retained with a cached response is provenance only
and must not be re-added to current metrics. Cache corruption, identity collision,
dimension drift, partial provider output, or failed transaction is a hard error;
there is no stale, zero-vector, alternate-model, or lexical fallback.

## Generation Retrieval Indexes

Build one immutable index each for active Policy memory, active World memory, and
the latest active Skill records. Every eligible record appears exactly once with:

```text
generation_id
record_kind
record_id
record_version, for skills only
document_sha256
embedding_cache_key
vector_dimension
vector
```

Index files are sorted by UTF-8 `record_id`, with numeric Skill version as the
secondary key. Their manifest binds record-store hashes, embedding identity,
dimension, query/document serialization versions, counts, and file hashes.
Loading rejects missing/extra/stale records or an index built from another
generation/configuration.

Cosine similarity uses deterministic Python float operations over finite,
non-zero vectors of the exact pinned dimension. Rank by descending score, then
UTF-8 `record_id`, then numeric Skill version. Return the first three or all
available records when fewer than three remain. Do not use a score threshold,
random tie breaking, approximate nearest-neighbor search, BM25, or another fallback.

The BM25 ordering used for presenting all tool schemas is a separate upstream
concern and must not enter these memory/Skill indexes.

## Skill Prefilter and Views

Before scoring Skills:

1. retain only the snapshot's latest active version;
2. require every canonical `tool_dependencies` entry to be available in the
   Controller-only current tool mapping;
3. require every `required_state` and `required_inputs` predicate to hold;
4. reject any Skill whose `forbidden_state` predicate holds;
5. do not interpret `best_used_when` or natural-language instruction as a rule.

Prefiltering may receive `CompactVerifiedState` and an explicit canonical-to-agent
mapping, but no hidden ToolSandbox state. It returns:

- `RetrievedSkillControllerView` using canonical dependencies and deterministic
  opaque predicate codes, with no prose;
- `RetrievedSkillPolicyView` using only currently mapped agent-facing dependency
  names and allowed semantic fields, with no canonical names, statistics,
  validation, failure buffer, or Controller sidecar.

An unmappable dependency makes the Skill ineligible. It is never partially mapped.

## Retrieval Service Contract

The service exposes immutable results with record ID, version when applicable,
generation ID, rank, finite cosine score, document/query hashes, and cache-hit or
physical-attempt references.

For each Agent state:

1. require one explicitly pinned `GenerationSnapshot`;
2. compute/resolve one Policy/Skill query embedding;
3. independently return top-three Policy memories and top-three prefiltered Skills;
4. only after the Controller triggers the Critic, compute/resolve one
   action-conditioned query and return top-three World memories;
5. pass the original immutable Policy/Skill retrieval bundle to Revision.

Revision has no method that can trigger retrieval. Repeated reads of the same
snapshot, inputs, indexes, and cache state return identical IDs, ranks, and scores.
This task does not persist those results; the later checkpoint task owns that.

## Failure and Ownership Boundaries

- Embedding failure or invalid output propagates as a hard typed failure. The
  orchestrator later writes the checkpoint and stops the run.
- This task never catches a failure to return empty synthetic results.
- Empty valid corpora return empty hit lists without fallback.
- No record is added, merged, deprecated, rewritten, statistically updated, or
  published here.
- No dev/test artifact, evaluator result, Critic prediction, or hidden database
  content can enter records, queries, cache rows, or indexes.
- Importing modules performs no filesystem, environment, network, provider-client,
  or SQLite access.

## Required Offline Tests

Using literal records, temporary directories, and fake embedding transports, test:

1. every valid record/nested record and all strict invalid forms;
2. memory ID prefixes, text/list limits, evidence ordering, count/rate/confidence
   invariants, and non-finite rejection;
3. Skill version/statistics/failure-buffer/validation invariants;
4. canonical tool-name rejection from Skill prose and structured dependency
   acceptance;
5. exact generation layout, hash/count/order/tool-inventory verification, immutable
   snapshot loading, symlink/path/unexpected-file/incomplete-generation rejection;
6. byte-exact state/action projection, Policy/Skill query, and World query fixtures,
   including proof that IDs, full schemas, selected Skill IDs, private sidecars, and
   evaluator content cannot enter them;
7. exact semantic document projections proving all host/statistical fields are
   excluded;
8. SQLite creation at an explicit path, schema-version rejection, exact composite
   keys, and absence of secret/raw-input columns;
9. hit, miss, repeated input, mixed hit/miss, deterministic item/aggregate-byte
   batching, original order restoration, and one usage record per physical batch;
10. no current-run token count on cache hit and one count per real fake-transport
    attempt on miss;
11. rollback on partial writes and rejection of corrupt hashes, NaN/infinity,
    zero/empty vectors, collisions, and dimension drift;
12. index completeness, generation/config binding, deterministic serialization,
    cosine fixtures, top-three truncation, fewer-than-three behavior, and exact tie
    breaking;
13. Skill latest-active selection, tool/predicate prefilters, all-or-nothing name
    mapping, Policy-view non-leakage, and Controller-view compatibility;
14. one shared Policy/Skill query, Critic-only World query, and Revision reuse with
    proof that Revision cannot call retrieval;
15. provider/cache/index errors propagate without BM25, stale, empty-result,
    alternate-model, or zero-vector fallback;
16. imports perform no I/O, environment read, client construction, or SQLite open.
17. exact acceptance/rejection at the 8,191-token per-input and 280,000-byte
    per-batch boundaries, with no truncation or provider call after rejection.

## Deferred Real-Data Validation

After the scenario-family manifest, Generation-0 seed files/index staging, and Task
006 embedding preflight are available, submit one coordinator runner request:

```text
Allowlisted mode: train-smoke
Purpose: retrieval
Split: train
Scenario IDs: first three eligible train scenarios in manifest order
Models: text-embedding-3-small only for this task boundary
```

The integrated runner builds the three initial compact states from real pinned
ToolSandbox scenarios, retrieves twice against the same generation, and verifies:

- exact returned embedding model and one stable vector dimension;
- identical query/document hashes, IDs, ranks, and scores on repeat;
- the second pass uses valid cache hits and adds no physical embedding usage;
- no canonical tool name appears in a scrambled Policy Skill view;
- latency and actual embedding tokens are separately reported.

This is a train-only engineering check, not a benchmark result or retrieval-quality
claim. Dev/test access and any threshold tuning are forbidden. Later train
calibration and the Dev Skill Mini-Bench own behavior adjustment.

## Acceptance Commands

The development Agent runs:

```bash
uv sync --frozen
uv run pytest -q tests/memory tests/skills tests/retrieval
uv run python -c "from toolsandbox_pipeline.retrieval import RetrievalService; from toolsandbox_pipeline.memory.store import load_generation"
git diff --check
git status --short
```

No development-Agent acceptance command may access a credential, dataset, network,
or external model.

## Acceptance Criteria

- Strict records and complete-generation verification fail closed.
- Embedding inputs contain only permitted visible/semantic data.
- Cache and indexes are bound to the exact embedding and generation identities.
- Physical batch usage is counted once; cache hits add no current-run usage.
- Policy, Skill, and World top-three results are deterministic.
- Skill prefiltering and Policy/Controller views preserve scrambling boundaries.
- No fallback, record mutation, generation publication, or checkpoint behavior is
  implemented.
- All offline tests pass; deferred real-data validation is passed or explicitly
  reported outstanding by prerequisite.
- No unowned file is modified.

## Completion Report Additions

Include:

```text
Record and manifest schemas added:
Retrieval serialization versions:
Embedding cache schema version:
Index identity and vector dimension:
Deterministic ranking fixture:
Physical embedding attempts in tests:
Cache-hit accounting verified:
Canonical-name leakage tests:
Deferred train retrieval smoke: pass | fail | not run
External validation blockers:
Dependency/interface change requests:
```

The train smoke reports its wall time and actual token usage as setup evidence. It
does not count as a training round or Vanilla/Generation-0/Updated evaluation run.
