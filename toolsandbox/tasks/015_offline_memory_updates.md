# Task 015: Offline Policy and World Memory Updates

Status: `approved`

## Objective

Implement deterministic, checkpointed offline updates for Policy memory and World
memory using only eligible trusted trajectories from the current training round.
For each eligible trajectory, a frozen Qwen role may return at most one candidate
or `NONE`; a second frozen Qwen reviewer returns `ADD`, `MERGE`, or `SKIP`.
Host code alone assigns identities, evidence, statistics, confidence, generation,
and status.

This task stages memory changes for the next generation. It does not update Skills,
run a Dev Mini-Bench, publish a generation, read dev/test data, execute scenarios,
or choose a checkpoint.

## Required Reading

Read, in order:

1. `pipeline.md`, Sections 3, 5, 8, 15-18, and 21-25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. `tasks/002_core_contracts.md`;
6. `tasks/006_model_api_gateways.md`;
7. `tasks/007_generation_records_and_retrieval.md`;
8. `tasks/008_online_prompts_and_qwen_roles.md`;
9. `tasks/009_dataset_manifest_and_split_access.md`;
10. `tasks/011_checkpoints_request_ledger_and_recovery.md`;
11. `tasks/012_metrics_and_immutable_reports.md`;
12. `tasks/014_episode_execution_evaluation_and_trajectories.md`;
13. this task file.

Do not inspect dev/test scenarios, hidden evaluator definitions, previous-round raw
trajectories, real provider responses, or final test results.

## Access

```text
Access class: offline
Setup network: none
Data splits: synthetic train only
Secrets: none
External operations: none
```

Unit tests use synthetic current-round train trajectories, fake Qwen/embedding
executors, literal generations, and temporary stores. Real model/API/data access
is deferred to the coordinator-owned train runner after offline acceptance.

## Preconditions

- Tasks 001-014 are implemented and accepted.
- The caller supplies one validated current-generation snapshot, one sealed
  current-round train trajectory buffer, and expected next generation ID.
- Every trajectory is Task 014 `completed_evaluated`, manifest-bound, and
  explicitly train/current-round eligible.
- Task 011 offline-unit markers and durable model request ledger are available.
- The formal runner supplies calibrated offline role token limits produced only
  from train calibration inputs.

## Owned Files

The assigned Agent may create or edit only:

```text
prompts/offline/memory_candidate_v1.txt
prompts/offline/memory_review_v1.txt
prompts/offline/memory_manifest.json
configs/offline_memory_token_limits.provisional.json
src/toolsandbox_pipeline/schemas/offline_memory.py
src/toolsandbox_pipeline/offline/__init__.py
src/toolsandbox_pipeline/offline/memory_projection.py
src/toolsandbox_pipeline/offline/memory_prompts.py
src/toolsandbox_pipeline/offline/memory_roles.py
src/toolsandbox_pipeline/offline/memory_retrieval.py
src/toolsandbox_pipeline/offline/memory_updates.py
src/toolsandbox_pipeline/offline/memory_orchestrator.py
tests/offline_memory/test_projection.py
tests/offline_memory/test_prompts.py
tests/offline_memory/test_roles.py
tests/offline_memory/test_retrieval.py
tests/offline_memory/test_updates.py
tests/offline_memory/test_orchestrator.py
tests/offline_memory/test_recovery.py
tests/offline_memory/test_leakage.py
```

Do not edit existing memory/generation/retrieval stores, provider gateways,
checkpointing, online/episode code, metrics, dataset code, Skill files, CLI,
dependencies, upstream ToolSandbox, or generated formal artifacts.

## Strict Schemas

`schemas/offline_memory.py` defines frozen strict discriminated unions:

```text
MemoryUpdateIdentity
PolicyTrajectoryProjection
WorldTrajectoryProjection
PolicyMemoryCandidate
WorldMemoryCandidate
MemoryCandidateNone
MemoryCandidateDecision
MemoryReviewAdd
MemoryReviewMerge
MemoryReviewSkip
MemoryReviewDecision
StagedMemoryMutation
MemoryUpdateUnitResult
MemoryRoundResult
```

Model-controlled fields are limited to the Section 18 semantic content:

```text
Policy candidate:
  scope
  applicability[0..5]
  action_guidance
  avoid[0..5]

World candidate:
  action_pattern
  state_conditions[0..5]
  schema_conditions[0..5]
  likely_error_codes[0..5]
  outcome_calibration
  correction_principle
```

Natural-language fields contain 1-512 characters. Reviewer `reason` contains
1-240 characters. Lists are unique. Unknown/conditional fields and coercion are
forbidden.

Models cannot output memory IDs, target IDs except the reviewer's supplied
`target_memory_id`, evidence IDs, trajectory IDs, support counts, rates,
confidence, generation, timestamps, status, model identity, or hashes.

## Sealed Input Buffer and Eligibility

The orchestrator accepts a sealed canonical manifest of current-round trajectory
references. Validate:

- run/round/shard/generation/dataset/config identities match;
- every entry belongs to the current train shard and appears once;
- entries follow Task 009 manifest order;
- every trajectory is complete, evaluated, and hash-valid;
- no dev/test/calibration/mini-bench/final-evaluation entry appears;
- no entry from an earlier/later round appears;
- the buffer hash matches the checkpoint and offline-unit identity.

Policy eligibility includes current-round evaluated train trajectories. Select at
most the 50 highest manifest positions (the most recent), then process that
selected set in ascending manifest order.

World eligibility additionally requires that Critic was actually called and its
draft action has a conservative attributable label. A World failure label is
allowed only when the draft action has a directly verifiable real tool exception,
fixture miss, selected related milestone below perfect similarity, or minefield
hit. If host code cannot attribute trusted evidence to the draft action, the World
candidate decision must be `NONE`.

Critic predictions, Revision text, confidence, and model claims are never trusted
labels.

## Offline Prompt Projection

`memory_projection.py` constructs role-specific canonical JSON from one
trajectory at a time.

Policy projection may include:

- Agent-visible compact states and augmented schemas;
- retrieved Policy/Skill semantic views used online;
- proposed/final actions and deterministic Controller codes;
- committed visible tool results/exceptions;
- terminal native score and `fully_successful`;
- generalized host-derived success/failure attribution.

World projection may include the same permitted information plus the actual
Critic call/output, Revision occurrence, and only the trusted attributable outcome
for that draft action.

Neither projection may include:

- raw provider bodies, vectors, secrets, endpoint data, or headers;
- hidden starting/final databases;
- milestone/minefield definitions, target DataFrames, similarity code, or mapping
  details beyond the minimal trusted generalized label;
- canonical tool names when the trajectory variant hid/scrambled them from the
  online model;
- another trajectory, previous-round raw data, dev/test content, or Skill updater
  state;
- concrete temporary entities or answers copied as proposed reusable memory.

The model sees one trajectory projection only. Batch prompting and concatenating
multiple trajectories are forbidden.

Host code builds a deterministic sensitive-literal set only from concrete values
already present in that trajectory's permitted offline projection: Agent-visible
message/state fields, action arguments, visible tool results, and call IDs. It may
not open or derive literals from hidden databases, evaluator definitions/targets,
canonical-name mappings, another trajectory, or dev/test data. Candidate fields
matching a protected literal or containing an exact nontrivial projected value
are rejected. Common public tool/schema vocabulary is allowlisted. The guard is a
conservative rejection layer, not an attempt to infer semantic leakage.

## Versioned Prompts and Token Limits

The two prompt files are exact Section 18 contracts with untrusted-data framing:

- Candidate: return at most one reusable role-appropriate candidate or `NONE`;
  never preserve concrete answers/entities, hidden evaluator content, unsupported
  hypotheses, or Critic predictions as facts.
- Reviewer: return `ADD` only for a reusable nonduplicate, `MERGE` only for a
  supplied semantic equivalent, otherwise `SKIP`; never rewrite existing memory.

`memory_manifest.json` pins exact bytes, role, version, and output schema.
`offline_memory_token_limits.provisional.json` contains bootstrap ceilings:

```text
memory_candidate: 512
memory_review: 256
```

They are calibration ceilings, not formal values. Reuse Task 008's train-only
calibration algorithm: actual completion usage, zero length finishes, strict
validity, 1.25 headroom rounded to 64, complete rerun, then immutable promotion.
No dev/test trajectory may calibrate limits.

## Durable Offline Qwen Roles

`memory_roles.py` implements exactly one Qwen gateway call per prepared request
for provider roles `memory_candidate` and `memory_review`.

Every request pins:

```text
model = Qwen/Qwen3-32B
temperature = 0.0
seed = 0
top_p omitted
enable_thinking = false
selected structured-output mode
calibrated role max_tokens
```

Task 011 records prepare/in-flight/raw response/actual usage/validated output and
application. No SDK/task retry, repair, fallback, alternate model, or dynamic
token limit is allowed.

A valid `NONE`, `ADD`, `MERGE`, or `SKIP` response remains an applied and
auditable Qwen output whose application row links the exact Task 011 source
attempt. Application alone does not make it cost-eligible. Only a candidate and
reviewer chain ending in a committed `ADD` or non-no-op `MERGE` semantic
mutation receives a Task 011 substantive-effect link and contributes to Task 012
`total_cost`. `NONE`, `SKIP`, duplicate/no-op outcomes, and a candidate later
skipped by the reviewer contribute zero to `total_cost`; all physical attempts
remain in `total_tokens`.

## Candidate Similarity and Reviewer Inputs

For a valid candidate:

1. serialize only its semantic candidate content using the matching Task 007
   retrieval text contract;
2. resolve exactly one `text-embedding-3-small` embedding through the durable
   cache/request path;
3. rank only active Policy or active World memories in the current generation;
4. select at most top three using Task 007 deterministic cosine/tie rules;
5. supply the candidate plus those complete permitted semantic records to the
   matching reviewer.

Do not search the other memory role, staged same-round candidates, Skills,
deprecated records, dev/test artifacts, or external stores. Embedding failure
checkpoints and stops; it never implies no duplicate.

`MERGE.target_memory_id` must be exactly one supplied top-three active memory.
`ADD` is allowed with an empty current store. `SKIP` makes no change.

## Host-Owned Update Rules

### ADD

Canonicalize candidate semantic content and assign:

```text
policy: memory_id = "pm_" + sha256(content)
world:  memory_id = "wm_" + sha256(content)
evidence_trajectory_ids = [current_trajectory_id]
support_count = 1
policy success_rate = current fully_successful label
world empirical_failure_rate = current attributable binary failure label
confidence = 1 / 3
created_version = next_generation_id
status = active
```

Reject an ID collision with different content. If identical content already exists
in current/staged records, the reviewer should have returned `MERGE`; fail closed
rather than silently convert the decision.

### MERGE

- target exactly one supplied active memory of the matching role;
- reject duplicate trajectory evidence;
- preserve every semantic field, ID, created version, and status;
- append evidence ID then sort IDs by UTF-8 order;
- increment support once;
- update the binary rate as
  `(old_rate * old_support + label) / new_support`;
- recompute `confidence = new_support / (new_support + 2)`;
- stage the updated record for the next generation.

### SKIP/NONE

Persist the applied decision and unit completion marker, but stage no memory
mutation.

Models never directly mutate stores. This task creates an immutable staged mutation
set only; a later publication task combines it with unchanged records.

## Ordering, Same-Round Semantics, and Recovery

Process selected trajectories in ascending manifest order. For each trajectory,
process Policy candidate/review before World candidate/review. Each
`(round, trajectory, memory_role)` is one Task 011 offline unit.

Reviewer duplicate search always uses the current generation snapshot plus already
staged same-round mutations in deterministic unit order, so repeated candidates in
one round merge rather than create duplicates. The staged view is private to this
offline round and is not visible to online episodes.

After every candidate decision, embedding, review decision, and staged mutation,
write a checkpoint. Commit the offline unit only after its exact staged output hash
is durable. Recovery reuses applied model outputs/embeddings/unit results and
continues at the next missing step. It never reruns a committed unit or changes
processing order.

Terminal invalid output, collision, duplicate evidence, changed buffer/generation,
missing trusted label, or hash drift stops the round without publishing a partial
generation.

## Output Contract

`MemoryRoundResult` contains:

```text
run/round/current_generation/next_generation identities
sealed_input_buffer_sha256
selected Policy trajectory IDs
selected World trajectory IDs
ordered offline unit IDs
candidate/reviewer logical and source-attempt IDs
ADD/MERGE/SKIP/NONE counts by role
staged Policy records
staged World records
unchanged source-generation hashes
last checkpoint/high-water marks
accounting projections
result_sha256
```

It contains references/hashes rather than raw trajectories or responses. It is
not a complete generation and cannot be loaded by online retrieval.

## Required Offline Tests

Cover at least:

1. all strict candidate/reviewer/input/output schemas and conditional fields;
2. prompt/manifest exact bytes, hash/path/newline checks, and injection framing;
3. sealed buffer run/round/shard/generation/hash/order validation and rejection of
   dev/test/previous-round/missing-evaluator entries;
4. Policy last-50 selection and ascending deterministic processing;
5. World Critic-called and conservative attributable-label eligibility;
6. no Critic prediction/model claim used as trusted evidence;
7. role projections include only allowed fields and one trajectory;
8. hidden database/evaluator definition/mapping/raw response/vector/canonical-name
   leakage denial;
9. sensitive-literal candidate rejection with public vocabulary allowance;
10. exact Qwen request configuration, calibrated limit enforcement, and one
    physical call;
11. candidate/reviewer application linkage, ADD/MERGE substantive-effect cost
    linkage, and zero cost for NONE/SKIP/duplicate/no-op chains;
12. invalid/truncated/missing-usage/unknown outputs retained without repair/retry;
13. candidate embedding, active matching-role top-three, deterministic ties, empty
    store, and embedding failure stop;
14. reviewer target membership and role/generation/status validation;
15. ADD ID golden vectors, collision/duplicate denial, host-owned fields, and
    binary initial rates;
16. MERGE semantic immutability, evidence order/uniqueness, exact rate/confidence
    updates, and seed integral-count validation;
17. staged same-round duplicate visibility without online publication;
18. fixed Policy-then-World and manifest ordering;
19. checkpoint/offline-unit crash injection at candidate, embedding, review,
    mutation, and commit boundaries;
20. resume without repeated physical calls, duplicate evidence, reordered units,
    or partial publication;
21. result identities/counts/hashes/accounting projections;
22. imports/construction perform no file, environment, provider, dataset,
    evaluator, or network access.

## Deferred Real-Train Validation

After offline acceptance and Task 017 runner integration, the coordinator may run
an allowlisted train-only memory smoke:

```text
Allowlisted mode: train-smoke
Purpose: offline-memory-update
Split: train only
Secrets: QWEN_BASE_URL, QWEN_API_KEY, OPENAI_API_KEY
```

Use a small manifest-pinned set of completed train trajectories. Report only
sanitized IDs/hashes, decision counts, latency, actual tokens, `total_cost`, and
completeness. This smoke cannot read dev/test or publish a formal generation.

## Acceptance Commands

```bash
uv sync --frozen
uv run pytest -q tests/offline_memory tests/memory tests/retrieval
uv run python -c "from toolsandbox_pipeline.offline.memory_orchestrator import MemoryUpdateOrchestrator"
git diff --check
git status --short
```

## Acceptance Criteria

- Only sealed current-round evaluated train trajectories are eligible.
- Policy/World projections obey their distinct trust and visibility rules.
- Each model controls semantic candidate/review content only.
- Host code alone owns identity, evidence, statistics, confidence, and staging.
- Reviewer search is exact, role-local, generation-pinned, and deterministic.
- ADD/MERGE/SKIP/NONE semantics and same-round staged visibility are exact.
- Every external attempt and applied Qwen output is durably/accountably linked,
  while only committed ADD/MERGE semantic mutations receive cost-effect links.
- Recovery never repeats a committed unit or exposes partial staged data online.
- Offline tests pass with zero external access and no unowned files are modified.

## Completion Report Additions

```text
Buffer eligibility/last-50 cases:
Policy/World projection cases:
World attribution cases:
Candidate/reviewer schema cases:
ADD/MERGE/SKIP/NONE cases:
Memory ID/rate/confidence cases:
Same-round staged-view cases:
Offline unit crash/recovery cases:
Applied Qwen output links:
Substantive ADD/MERGE cost-effect links:
NONE/SKIP/no-op zero-cost cases:
Leakage/sensitive-output scan:
External calls: 0
Total tokens: 0
Usage complete: true
Deferred train smoke: not run
Dependency/interface change requests:
```

Unit-test wall time is development evidence only, not experimental
`total_running_time_seconds`.
