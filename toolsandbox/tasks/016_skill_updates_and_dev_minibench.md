# Task 016: Skill Updates and Dev Mini-Bench

Status: `approved`

## Objective

Implement current-round Skill statistics, failure-mode buffering, deterministic
rewrite triggering, one frozen-Qwen rewrite candidate per triggered Skill, and the
paired Dev Mini-Bench that alone accepts or rejects that candidate.

This task may use dev scenarios only through an explicitly selected, deterministic
A/B evaluation boundary. Dev content/results never enter a model prompt, memory
update, failure buffer, candidate generation, aggregate Skill statistics, or
checkpoint selection. This task stages accepted/rejected Skill state; a later task
publishes the complete next generation.

## Required Reading

Read, in order:

1. `pipeline.md`, Sections 3, 5, 8-9, 15-17, and 19-25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. `tasks/005_tool_metadata_controller.md`;
6. `tasks/006_model_api_gateways.md`;
7. `tasks/007_generation_records_and_retrieval.md`;
8. `tasks/008_online_prompts_and_qwen_roles.md`;
9. `tasks/009_dataset_manifest_and_split_access.md`;
10. `tasks/011_checkpoints_request_ledger_and_recovery.md`;
11. `tasks/012_metrics_and_immutable_reports.md`;
12. `tasks/014_episode_execution_evaluation_and_trajectories.md`;
13. `tasks/015_offline_memory_updates.md`;
14. this task file.

Do not inspect test data or use any dev/test artifact except through the explicitly
defined Mini-Bench selector/runner interface.

## Access

```text
Development access class: offline
Development data: synthetic only
Development secrets: none
External operations: none

Deferred Mini-Bench access class: real_data
Deferred split: selected dev scenarios only
Deferred secrets: QWEN_BASE_URL, QWEN_API_KEY, OPENAI_API_KEY
```

Unit tests use synthetic train attributions, Skills, dev metadata, A/B result
summaries, fake model executors, and temporary stores. The development Agent must
not run a real Mini-Bench.

## Preconditions

- Tasks 001-015 are implemented and accepted.
- The current generation contains one valid active version per Skill.
- Task 014 supplies sealed current-round train trajectories, trusted evaluator
  records, and conservative executed Skill-use attributions.
- Task 009 supplies a validated dev manifest and nominal no-distraction necessary
  canonical-tool metadata through an offline-selector-only view.
- Task 014 EpisodeRunner can execute a caller-supplied scenario/generation branch.
- The caller supplies expected next generation and staging identities.

## Owned Files

The assigned Agent may create or edit only:

```text
prompts/offline/failure_mode_update_v1.txt
prompts/offline/skill_candidate_v1.txt
prompts/offline/skill_manifest.json
configs/offline_skill_token_limits.provisional.json
src/toolsandbox_pipeline/schemas/offline_skill.py
src/toolsandbox_pipeline/offline/skill_projection.py
src/toolsandbox_pipeline/offline/skill_prompts.py
src/toolsandbox_pipeline/offline/skill_roles.py
src/toolsandbox_pipeline/offline/skill_statistics.py
src/toolsandbox_pipeline/offline/failure_modes.py
src/toolsandbox_pipeline/offline/failure_lineage.py
src/toolsandbox_pipeline/offline/skill_rewrite.py
src/toolsandbox_pipeline/offline/dev_selector.py
src/toolsandbox_pipeline/offline/dev_minibench.py
src/toolsandbox_pipeline/offline/skill_orchestrator.py
tests/offline_skill/test_projection.py
tests/offline_skill/test_prompts.py
tests/offline_skill/test_roles.py
tests/offline_skill/test_statistics.py
tests/offline_skill/test_failure_modes.py
tests/offline_skill/test_failure_lineage.py
tests/offline_skill/test_rewrite.py
tests/offline_skill/test_dev_selector.py
tests/offline_skill/test_dev_minibench.py
tests/offline_skill/test_orchestrator.py
tests/offline_skill/test_recovery.py
tests/offline_skill/test_leakage.py
```

Do not edit existing Skill/generation/retrieval stores, memory updates, online/
episode/provider/checkpoint/metrics/dataset code, dependency files, CLI, upstream
ToolSandbox, or generated formal artifacts.

## Strict Schemas

`schemas/offline_skill.py` defines:

```text
SkillRoundIdentity
SkillFailureEvidence
FailureModeAdd
FailureModeMerge
FailureModeSkip
FailureModeDecision
SkillContentCandidate
PreparedSkillRewrite
DevScenarioSelection
DevBranchResult
DevMiniBenchResult
StagedSkillMutation
SkillUpdateUnitResult
SkillRoundResult
```

All models are frozen, strict, `extra="forbid"`, and finite.

Failure-mode model-controlled output is exactly:

```json
{"decision":"ADD","task_condition":"...","failure_mode":"..."}
{"decision":"MERGE","mode_id":"..."}
{"decision":"SKIP","reason":"..."}
```

Only MERGE contains `mode_id`, which must be one supplied current mode. ADD text
and SKIP reason contain 1-240 characters.

Rewrite output is exactly:

```json
{"candidate": <SkillContent>}
```

`SkillContent` contains the existing semantic fields:

```text
skill_id
name
description
applicability
required_inputs
expected_outputs
tool_dependencies
success_criteria
cost_profile
risk_profile
instruction
```

It excludes failure buffer, statistics, validation, version, status, candidate ID,
evidence, generation, hashes, and acceptance decision. `skill_id` must exactly
match the triggered Skill.

## Train-Only Attribution and Statistics

Validate the sealed current-round train buffer exactly as Task 015 does. Process
Task 014 `SkillUseAttribution` records in scenario manifest order, then Skill ID
UTF-8 order.

For each unique Skill/episode attribution:

- increment `evaluated_uses` once;
- increment success when `fully_successful`, otherwise failure;
- recompute `success_rate = successes / evaluated_uses`;
- never count blocked, unexecuted, rolled-back, missing-evaluator, or duplicate
  uses;
- never treat one episode's multiple calls as multiple evaluated uses.

Statistics are staged for the next generation and remain invisible to current
online episodes.

Only actual train failures produce `SkillFailureEvidence`. Allowed evidence is a
generalized projection of real visible tool exceptions, fixture misses,
non-perfect selected milestones, minefield hit, or visible terminal state,
together with the trusted task failure label. Critic predictions and hidden
definitions are excluded.

For later analysis, host code also derives an immutable, sanitized
`FailureModeLineageRecord`. Its match signature contains only the owning
`skill_id`, evidence kind, sorted public canonical tool dependencies, and a
sanitized exception/controller outcome class. It links the signature to the
committed `mode_id`, source train evidence hash, producing round, and optionally
the accepted evolved Skill version/effect ID. It contains no scenario text,
entity, raw exception/tool content, hidden evaluator definition, or model prose.
The signature and lineage ID are canonical hashes; identical replay is
idempotent and conflicts fail closed.

## Failure-Mode Updates

For every attributed train failure, call the frozen Qwen failure-mode updater once
with exactly one generalized failure projection and that Skill's currently staged
buffer.

Host rules:

### ADD

- assign `mode_id = "fm_" + sha256(skill_id + canonical semantic content)`;
- initialize `support_count = 1`;
- allocate one monotonically increasing round-global `last_observed_seq`;
- reject collision/equivalent existing content rather than silently merge.

### MERGE

- target one supplied mode under the same Skill;
- preserve `task_condition`, `failure_mode`, and ID exactly;
- increment support once;
- update `last_observed_seq` to the next host-owned global sequence.

### SKIP

- stage no buffer mutation but durably apply/record the decision.

After mutation, retain at most five modes by descending support count, then
descending `last_observed_seq`, then UTF-8 `mode_id`. Persist the retained buffer
in stable `mode_id` order for canonical serialization. Eviction never changes
the retained records.

Each valid ADD/MERGE/SKIP remains an applied and auditable Qwen output. Only an
ADD or MERGE for which the final post-cap canonical buffer hash differs from the
pre-update hash receives a Task 011 substantive-effect link and contributes to
Task 012 `total_cost`. SKIP/duplicate/no-op/added-then-evicted outcomes contribute
zero; all physical attempts remain in `total_tokens`.

## Rewrite Trigger

After all current-round statistics/failure-mode updates for one Skill, trigger a
rewrite iff:

```text
evaluated_uses >= 10
failures / evaluated_uses > 0.25
evaluated_uses > last_update_attempt_at_use_count
```

Use exact integer comparison `failures * 4 > evaluated_uses`; do not rely on
rounded floats. Trigger decisions are host-only and deterministic.

Process triggered Skills in UTF-8 `skill_id` order. Before calling the candidate
model, stage `last_update_attempt_at_use_count = evaluated_uses` so rejection or
crash recovery cannot repeat the attempt without a new evaluated use.

## Rewrite Projection and Qwen Role

The candidate request receives only:

- one current active Skill;
- its staged train-only statistics and at most five failure modes;
- relevant public canonical tool schemas for its declared dependencies;
- relevant generalized current-round train trajectories attributed to that Skill.

It cannot receive Policy/World memory, another Skill, dev/test data/results,
hidden evaluator definitions/databases, raw responses, vectors, or concrete
scenario answers/entities.

The versioned prompt is the Section 19 rewrite contract. The model must preserve
`skill_id`, cannot add/split/merge Skills, and cannot change another Skill.

`skill_roles.py` performs one Qwen call with:

```text
provider role: skill_candidate
model: Qwen/Qwen3-32B
temperature: 0.0
seed: 0
top_p: omitted
enable_thinking: false
structured-output mode: pinned
max_tokens: calibrated skill_candidate value
```

The provisional bootstrap ceiling is 2048. Use the same train-only calibration
protocol as Task 008/015. Task 011 owns attempts/application. A valid candidate is
an applied and auditable Qwen output, but it and its paired Mini-Bench Qwen
decision chains contribute to `total_cost` only if the candidate is accepted and
the new Skill version commits. A rejected/incomplete/no-relevant-dev candidate
unit contributes zero to `total_cost`, while all of its physical attempts remain
in `total_tokens`.

Host validation requires:

- unchanged `skill_id`;
- all declared dependencies exist and remain sorted;
- no canonical tool identifier appears in prohibited prose fields;
- predicates/fields satisfy Task 007 Skill schema;
- no concrete scenario entity/evaluator content is copied;
- candidate semantic content differs from current version;
- no unsupported dependency or scope expansion.

Invalid candidate stops the Skill update unit; it is not repaired.

## Deterministic Dev Selection

`dev_selector.py` is the only component allowed to inspect the restricted
selector view of dev metadata. It never exposes that view to a model.

For the triggered Skill:

1. obtain the canonical necessary-tool set of each nominal no-distraction dev
   family from Task 009 restricted metadata;
2. remove `end_conversation`;
3. retain a family iff the set intersects the Skill's canonical
   `tool_dependencies`;
4. include all eligible expanded dev scenario variants from retained families;
5. sort by
   `(SHA256(data_seed + NUL + skill_id + NUL + scenario_id), UTF-8 scenario_id)`;
6. take at most 20.

Selection stores only IDs/hashes/counts in the candidate unit. An empty selection
deterministically rejects the candidate with reason `no_relevant_dev_scenarios`;
it never accepts without evidence or broadens tool matching.

## Paired Dev Mini-Bench

For every selected scenario, run two fresh Task 014 episodes:

- branch A: current active Skill version;
- branch B: candidate Skill content.

Pin identically across branches:

```text
starting context and evaluator definition
scenario order and augmentation
User Simulator and its request input
world clock and fixture store
Qwen/embedding models and decoding
all prompts/schemas/token limits
Policy and World memories
all other Skills
previously accepted staged Skill updates
runtime/environment/config hashes
```

The evaluated Skill version/content is the only intentional difference. Branches
use distinct run/episode/request IDs and fresh deep-copied contexts; results cannot
be reused across branches. Scenario-level seeds derive from scenario ID and branch
only where required by a pinned component; deterministic Qwen settings remain
fixed.

When multiple Skills trigger, later A/B comparisons include every earlier accepted
staged Skill update in both branches. They exclude rejected candidates and the
current Skill candidate from branch A.

Dev trajectories/evaluator results are stored in a dedicated restricted
`dev_minibench` namespace. They cannot enter:

- candidate/failure-mode/memory prompts;
- train statistics or failure buffers;
- retrieval corpora;
- subsequent candidate generation;
- final test selection;
- persisted aggregate statistics other than the candidate's fixed validation
  summary.

## Acceptance Rule

Aggregate each branch over the exact selected scenario IDs:

```text
full_success_count
similarity_sum
minefield_hit_count
```

Use native finite values without rounding. Accept exactly when:

1. candidate `full_success_count` is greater; or
2. full-success counts tie, candidate `similarity_sum` is greater, and candidate
   minefield-hit count does not increase.

Reject exact ties, lower full success, lower/equal tied similarity, increased
minefields under the tie path, incomplete branches, and improvements only in turn
count/latency/tokens.

## Accepted and Rejected State

Accepted candidate:

- new version is next patch `v1.n`;
- old active version becomes deprecated;
- new version becomes active;
- semantic content is candidate content;
- failure buffer is empty;
- online statistics are all zero;
- `last_update_attempt_at_use_count = 0`;
- validation stores the exact seven Section 19 fields/counts/sums;
- version history remains ordered and exactly one latest version is active.

Rejected candidate:

- consumes no version number;
- active semantic content/version/status remains unchanged;
- staged train statistics/failure buffer remain;
- `last_update_attempt_at_use_count` remains updated to current evaluated uses;
- validation result/reason is stored in the offline unit audit, not attached as a
  new Skill version.

No automatic Skill creation, deletion, split, merge, or cross-Skill edit exists.

## Durability, Recovery, and Metrics

Each failure decision and each rewrite/Mini-Bench is a Task 011 offline unit.
Checkpoint before/after every Qwen request, candidate stage, dev selection, branch
episode, aggregate, decision, and staged mutation.

Recovery:

- reuses applied updater/candidate outputs;
- reuses completed branch episode/evaluator results;
- resumes the exact next scenario/branch;
- never reruns a committed unit or accepts from a partial branch;
- preserves sorted Skill processing and earlier accepted staged updates.

All Qwen/embedding/User attempts in train update or Mini-Bench contribute to
`total_tokens` for their enclosing operation. Only Qwen chains linked to a
committed failure-mode or accepted Skill semantic mutation contribute to
`total_cost`; SKIP/no-op/rejected candidate units, User, and embedding outputs do
not. Dev Mini-Bench
latency/tokens/cost are reported separately from training-round headline metrics
unless the final orchestration specification explicitly includes selection work
inside the training-round boundary. Pipeline Section 22 includes offline update
and generation publication in round time; therefore the Mini-Bench selection work
for that round is included in its total round wall time and physical-token total,
while still labeled `phase: dev_minibench`.

## Required Offline Tests

Cover at least:

1. strict failure/candidate/selection/branch/result schemas and conditionals;
2. train buffer/attribution identity, ordering, duplicate, and eligibility checks;
3. one evaluated use per Skill/episode and exact staged success/failure statistics;
4. actual-failure evidence allowlist and Critic/hidden-definition exclusion;
5. ADD/MERGE/SKIP schemas, one-call routing, IDs, support/sequence updates;
6. capacity-five retention ordering and stable serialization;
7. exact integer rewrite threshold boundaries and last-attempt gating;
8. sorted triggered-Skill processing and crash-safe attempt marking;
9. rewrite projection single-Skill/tool-schema/train-only visibility;
10. prompt/config hashes, bootstrap/calibrated limits, exact Qwen fields, and no
    retry/repair/fallback;
11. candidate Skill identity/dependency/predicate/prose/entity validation;
12. deterministic dev family intersection, end-conversation exclusion, hash sort,
    at-most-20 limit, empty selection, and no test IDs;
13. paired A/B equality of every pinned field except evaluated Skill;
14. fresh branch contexts/IDs and no cross-branch result reuse;
15. later Skill branches include earlier accepted updates equally;
16. strict dev isolation from all model prompts/statistics/buffers/retrieval;
17. every acceptance-rule branch, exact tie rejection, minefield constraint, and
    no latency/turn/token tie-break;
18. accepted version increment/deprecation/reset/validation and rejected
    no-version/attempt-count semantics;
19. checkpoint/crash recovery across failure updates, rewrite, every dev
    scenario/branch, aggregation, and decision;
20. `total_tokens` versus substantive-effect `total_cost` phase accounting,
    including zero cost for SKIP/no-op/rejected candidate units;
21. secret/raw-response/dev-evaluator/hidden-database/concrete-entity sentinel
    scans;
22. deterministic failure-signature/lineage construction, accepted-Skill linkage,
    idempotence, ambiguity retention, and sensitive-field denial;
23. imports/construction perform no filesystem, environment, dataset, role,
    evaluator, provider, or network access.

## Deferred Dev Mini-Bench Validation

After all offline tests and Task 017 integration, the coordinator may run:

```text
Allowlisted mode: dev-mini-bench
Split: deterministic selected dev IDs only
Secrets: QWEN_BASE_URL, QWEN_API_KEY, OPENAI_API_KEY
Publication: forbidden
```

The result contains selected ID/hash, branch config equality hash, native aggregate
scores, decision, latency/tokens/`total_cost`, completeness, and artifact hashes.
It cannot return raw dev trajectories to a coding Agent.

## Acceptance Commands

```bash
uv sync --frozen
uv run pytest -q tests/offline_skill tests/skills tests/episode
uv run python -c "from toolsandbox_pipeline.offline.skill_orchestrator import SkillUpdateOrchestrator"
git diff --check
git status --short
```

## Acceptance Criteria

- Train statistics/failure modes use only executed, evaluated current-round uses.
- Rewrite triggers and same-round ordering are exact and deterministic.
- Candidate generation sees no dev/test or other forbidden state.
- Mini-Bench selection is relevant, dev-only, deterministic, and at most 20.
- A/B branches differ only in the evaluated Skill version.
- Acceptance uses only full success, similarity, and minefield counts as specified.
- Accepted/rejected version/statistics semantics are exact.
- Dev artifacts cannot influence future generation except fixed validation fields.
- Recovery cannot repeat calls/branches/decisions or partially publish a Skill.
- Offline tests pass with zero external access and no unowned files are modified.

## Completion Report Additions

```text
Skill statistics cases:
Failure ADD/MERGE/SKIP/capacity cases:
Rewrite trigger boundaries:
Candidate validation cases:
Dev selector golden IDs:
A/B equality checks:
Acceptance/rejection cases:
Version/reset cases:
Crash/recovery cases:
Dev leakage scan:
External calls: 0
Total tokens: 0
Usage complete: true
Deferred Dev Mini-Bench: not run
Dependency/interface change requests:
```

Unit-test wall time is development evidence only, not experimental
`total_running_time_seconds`.
