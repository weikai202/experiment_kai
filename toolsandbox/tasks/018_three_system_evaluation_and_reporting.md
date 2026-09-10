# Task 018: Three-System Evaluation and Reporting

Status: `approved`

## Objective

Implement and validate the final evaluation/reporting layer for exactly:

1. Vanilla frozen Qwen3-32B;
2. the complete Generation-0 pipeline using G000;
3. the Updated pipeline using G003 after all three training rounds.

All three run once on the same frozen test manifest with identical environment
inputs. This task computes native-score summaries, family-cluster statistics,
latency/token/substantive-effect Qwen-output cost metrics, reproducibility checks,
and
immutable sanitized reports.

This task does not update memories/Skills, select a different checkpoint, tune a
prompt/token limit, rerun a completed system, or authorize the formal experiment
without explicit user approval.

## Required Reading

Read, in order:

1. `pipeline.md` in full;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/FAILURE_MODE_ATTRIBUTION.md`;
5. `docs/SECRET_INJECTION.md`;
6. Tasks 001-017 in numeric order;
7. this task file.

Do not inspect test scenario content or results before the frozen plan is committed
and the explicit final-test capability is granted.

## Access

```text
Development access class: offline
Development data: synthetic only
Development secrets: none

Deferred Vanilla calibration: real_data, deterministic train-only IDs
Deferred final evaluation: coordinator-owned final-test
Deferred final split: test exactly once per system
Deferred secrets: profile-selected Qwen/OpenAI/live-tool credentials
```

The development Agent writes code/tests with synthetic records and fake Episode
runners. Only the project-external launcher may open real train/test data or
credentials.

## Preconditions

- Tasks 001-017 are implemented, reviewed, and accepted.
- All offline and integration tests pass.
- Required preflights and train-only token calibrations are complete.
- G000 and G003 are complete, immutable, validated, and hash-pinned.
- The formal three-round training run completed without selecting a checkpoint
  from dev/test outcomes.
- One final evaluation profile is selected and feasible.
- The user explicitly approves the frozen final-test plan after reviewing smoke
  results.

With the currently selected remote `gpt-4o-mini-2024-07-18` User Simulator, the
formal profile is `official_live`. That profile also preserves upstream live
RapidAPI behavior and therefore requires the corresponding authorized live-tool
access. It cannot silently combine the remote User with `strict_replay` fixtures.
If live-tool access is unavailable, final evaluation remains blocked until the
user changes the formal profile or supplies its prerequisites.

## Owned Files

The assigned Agent may create or edit only:

```text
prompts/vanilla_agent_v1.txt
prompts/vanilla_manifest.json
configs/vanilla_token_limit.provisional.json
configs/run/final_evaluation_plan.schema.json
src/toolsandbox_pipeline/schemas/reporting.py
src/toolsandbox_pipeline/reporting/__init__.py
src/toolsandbox_pipeline/reporting/vanilla_responder.py
src/toolsandbox_pipeline/reporting/evaluation_plan.py
src/toolsandbox_pipeline/reporting/system_runner.py
src/toolsandbox_pipeline/reporting/aggregates.py
src/toolsandbox_pipeline/reporting/cluster_statistics.py
src/toolsandbox_pipeline/reporting/failure_mode_analysis.py
src/toolsandbox_pipeline/reporting/reproducibility.py
src/toolsandbox_pipeline/reporting/artifacts.py
src/toolsandbox_pipeline/reporting/cli.py
tests/reporting/test_vanilla_responder.py
tests/reporting/test_evaluation_plan.py
tests/reporting/test_system_runner.py
tests/reporting/test_aggregates.py
tests/reporting/test_cluster_statistics.py
tests/reporting/test_failure_mode_analysis.py
tests/reporting/test_reproducibility.py
tests/reporting/test_artifacts.py
tests/reporting/test_cli.py
tests/reporting/test_test_once_guard.py
tests/reporting/test_leakage.py
```

Do not edit earlier components, formal generations/training artifacts, upstream
ToolSandbox, dependencies, dataset/fixture manifests, secret launcher, or real
result files.

Use `python -m toolsandbox_pipeline.reporting.cli`; no packaging edit is needed.

## Vanilla System

Vanilla replaces only the Agent role with a direct frozen Qwen3-32B responder. It
uses:

- the current Agent-visible native messages;
- the current augmented agent-facing tool schemas;
- one versioned Vanilla system prompt;
- the same strict `ActionEnvelope`;
- the same Task 004 action-to-native-message conversion;
- the same Qwen endpoint/decoding/wire mode and Task 011 durability.

It does not use State Builder summaries, Policy/World memory, Skills, retrieval,
Controller, Critic, Revision, or offline artifacts. It may not inspect canonical
tool mappings except through the trusted Adapter conversion after producing an
agent-facing action.

The prompt requires one next action only, current agent-facing names, grounded
arguments from visible messages/results, no predicted result/plan/reasoning, and
strict JSON.

`vanilla_token_limit.provisional.json` begins with the Policy bootstrap ceiling
256 but is not formal. Before freezing the final plan, calibrate Vanilla using the
same train-only actual-usage/length/schema-validity algorithm as Task 008. No test
input may affect it.

A Vanilla valid Qwen response contributes its actual output tokens to
`total_cost` only when its action is committed and the Task 011 substantive-effect
link exists. Every physical attempt contributes to `total_tokens` under Task 012.

## Frozen Final Evaluation Plan

`FinalEvaluationPlan` is canonical, strict, non-secret, and committed before test
access. It contains:

```text
plan/protocol/schema version and plan SHA-256
user approval reference and frozen timestamp
training run/G000/G003 identities and hashes
checkpoint observation registry hash and non-selecting diagnostic pointers
ordered three system IDs
test dataset manifest/hash and exact ordered test IDs
family/category/variant membership hashes
starting-context/evaluator/tool-schema hashes per test ID
ToolSandbox/source/dependency/image/environment hashes
profile/world clock/fixture-or-live-tool configuration
Qwen/embedding/User identities and decoding
all prompt/schema/calibrated-token-limit hashes
per-system allowed components and generation ID
process count/order/seed policy
checkpoint/metrics/reporting schema identities
cluster statistics algorithm/seed/replicate count
absolute output root
```

The system order is fixed:

```text
vanilla
generation_0
updated
```

Generation-0 pins G000. Updated pins G003. No plan field accepts G001/G002 or a
score-selected generation.

Plan validation occurs before test materialization. Any provisional limit,
generation drift, split mismatch, missing family, duplicate scenario, component
difference not explicitly system-specific, secret/raw URL, relative/symlink path,
or prior test result fails.

## One-Time Test Guard

The output root contains a durable plan-bound test ledger with states:

```text
authorized_not_started
running
completed
terminal_failure
reconciliation_required
```

For each system/scenario there is exactly one logical episode identity. Starting a
system is an irreversible durable event. Recovery resumes that same run/episode
from Task 011 checkpoints; it is not a new evaluation and does not consume a
second test attempt.

After a system reaches `completed`, any command that would execute one of its test
scenarios again is rejected, even with a new output path/run ID. A different
prompt/config/generation/profile cannot be called a resume.

A terminal provider/schema/config failure is reported, not tuned and rerun. An
`official_live` unknown external-read outcome requires reconciliation and cannot
be automatically retried. Test results cannot modify the remaining systems'
frozen plan.

## Fair System Execution

For every test scenario, all three systems share exact:

```text
scenario ID/order/family/category/variant
deep-copied starting ExecutionContext
native evaluator definition and max messages
agent-facing tool augmentation/schema/order
world clock
fixture/live external-tool configuration
User Simulator model/prompt/few-shot/tools/stop behavior
Qwen checkpoint/server/decoding/wire mode
embedding model where a system uses retrieval
container/dependencies/process policy
```

Intentional differences are only:

- Vanilla: direct responder and no generation/retrieval/controller/critic;
- Generation-0: complete pipeline pinned to G000;
- Updated: complete pipeline pinned to G003.

Generation-0 and Updated use identical pipeline prompt/token-limit/config hashes.
All three use fresh isolated contexts and distinct request/episode IDs. No model
response, embedding, User message, tool result, or context is shared across
systems. Shared immutable model/config files are allowed.

Execute scenarios in test manifest order. If scenario-level parallelism is
enabled, derive seeds from scenario ID and serialize results back into manifest
order; default remains one process.

## Native Results and Aggregates

Consume only complete Task 014 trusted evaluator records. For each system report:

```text
scenario_count and family_count
mean native similarity
mean milestone_similarity
mean minefield_similarity
fully_successful count/rate
macro and micro metrics for every native ScenarioCategories value
mean and median effective turn count
Critic trigger/accept/revise/uncertain rates
Revision count/rate and post-revision episode score
fixture hit/miss and external-tool exception rates
failure/timeout/reconciliation counts
total_running_time_seconds
total_tokens and usage_complete
total_cost, cost_unit, and cost_complete
round-0/1/2 direct latency, total cost, and completeness
per-request/task/phase/role/model/provider breakdowns
```

Mean native `similarity` is primary. Do not replace it with binary success,
milestone-only score, turn efficiency, cost, or a custom composite.

Macro category metrics average the per-category means over all categories with
test members. Micro metrics pool scenarios. A scenario with multiple native
categories contributes once to each applicable category but exactly once to the
overall metric. Preserve the complete membership denominator table.

Median uses the deterministic sorted middle value/mean of two middle values.
All finite score sums use a documented exact order and `math.fsum`; display
rounding never changes stored numeric results.

## Failure-Mode-Linked Repair Analysis

After all three systems are durably complete, consume only Task 016 sanitized
failure-mode lineage records and Task 014 trusted, content-free scenario records.
Match with the exact host-derived tuple of Skill ID, evidence kind, sorted public
canonical tool dependencies, and sanitized exception/controller outcome class.
No LLM, embedding, fuzzy/text similarity, raw exception text, tool arguments or
results, hidden evaluator definition, or manual post-hoc classification is
allowed. Preserve zero, one, and multiple matches; multiple matches are reported
as ambiguous and excluded from the repaired count.

Define a primary repaired case only when the same test scenario has
`generation_0.fully_successful == false`, `updated.fully_successful == true`, one
unambiguous lineage match, and the Updated committed action chain records use of
the linked evolved Skill version. Report:

```text
failure_mode_related_case_count
failure_mode_repaired_case_count
failure_mode_repair_rate
failure_mode_unmatched_case_count
failure_mode_ambiguous_case_count
updated_minus_generation_0_similarity_points_overall
updated_minus_generation_0_similarity_points_related_subset
secondary updated_minus_vanilla paired values
all_variants_success_family_count/rate by system
mean family_minimum_similarity by system
mean within_family_similarity_range by system
related_family_count and fully_repaired_related_family_count/rate
```

Emit one content-free attribution row per system/scenario with scenario identity,
matched lineage/mode/Skill-version IDs, baseline and Updated outcome/score fields,
evolved-Skill-use proof hash, classification, and reason code. Call this a
mechanism-linked observational attribution, not causal proof. Missing or
incomplete evidence cannot be counted as repaired.

ToolSandbox stability is evaluated at the `scenario_family_id` level across its
exact eight variants. A stable-success family requires all eight variants to be
fully successful. Compute each family's minimum native similarity and
`max(similarity) - min(similarity)` range, then report deterministic means across
the 25 families. A related family is fully repaired only when every one of its
Generation-0 failing variants that has one unambiguous failure-mode match is a
primary repaired case. Preserve variant denominators and use families, never
individual variants, as the uncertainty sampling unit.

## Family-Cluster Statistics

The 8 variants of one `scenario_family_id` are not independent observations.
Uncertainty and pairwise tests resample/compare families.

Implement a fixed paired cluster bootstrap:

```text
seed: 0
replicates: 10000
sampling unit: the 25 test family IDs
sample size per replicate: 25 families with replacement
within-family unit: all test variants, retaining multiplicity
statistic: difference in mean native similarity
pairs: updated-vanilla, updated-generation_0, generation_0-vanilla
interval: two-sided percentile 95%
```

Sort family IDs by UTF-8 bytes before sampling and use Python 3.10
`random.Random(0)`. Store the generated family-index stream hash. Percentile
indices use the documented nearest-rank rule and are tested with golden vectors.

This is a descriptive paired uncertainty analysis, not permission to select a
system. Report raw paired family differences and the interval; do not claim
statistical significance solely because an interval excludes zero unless the
report language is separately reviewed.

If the validated test family count is not exactly 25 or variants/family membership
do not match the manifest, stop rather than adapt sample size.

## Direct Timing and Token/Cost Semantics

Each system has its own direct evaluation-run timer from immediately before its
first test scenario dispatch through its last evaluator record and durable final
run checkpoint. Do not sum scenario/request latency.

`total_tokens` includes every actual physical Qwen, embedding, and GPT-4o mini
User attempt inside that system boundary, including explicit recovery attempts.
One missing dispatched usage makes it null/`usage_complete: false`.

`total_cost` is the non-monetary
`qwen_effective_output_tokens` metric and requires Task 011 committed
substantive-effect links:

- Vanilla: valid direct Qwen output whose action is committed;
- Generation-0/Updated: only Policy/Critic/Revision outputs actually consumed by
  each committed final-action chain;
- no embeddings, User, input/cache, invalid/unapplied/unknown attempts.

Evaluation performs no offline updates, so no memory/Skill updater output can
appear in final-evaluation cost. Missing output usage for a Qwen response linked
to a committed substantive effect makes `total_cost`
null/`cost_complete: false`.

The final report also reproduces the three immutable training-round headline rows
from Task 017 without recomputation: each round's direct
`total_running_time_seconds`, `total_tokens`/`usage_complete`, and
`total_cost`/`cost_complete`. It includes the G000-G003 query-only checkpoint
registry and diagnostic best-observed pointers, labeled as different-shard,
non-comparable, and forbidden for formal Updated selection.

## Reproducibility Validation

`reproducibility.py` evaluates every Section 24 item with:

```text
pass
fail
not_applicable_with_reason
not_run
```

Evidence is a hash/reference to immutable test/build/smoke artifacts, never a
free-form unsupported assertion. Required applicable failure or `not_run` forces
the conclusion `partially_reproducible`.

For `strict_replay`, two pre-final smoke runs must have identical final-context
hashes, native scores, retrieval IDs, and request-output hashes. For
`official_live`, trajectory equality is not claimed; report
`configuration_traceable_or_statistically_reproducible` and disclose the fixed
GPT-4o mini deviation from upstream GPT-4o.

Windows traceback/response-order behavior is never accepted as Linux formal
evidence.

## Immutable Report Artifacts

Write atomically under the plan output root:

```text
final_evaluation/
  plan.json
  plan.sha256
  systems/vanilla/metrics/
  systems/generation_0/metrics/
  systems/updated/metrics/
  scenario_results.jsonl
  family_results.jsonl
  category_results.jsonl
  failure_mode_case_attribution.jsonl
  failure_mode_repair_summary.json
  pairwise_cluster_bootstrap.json
  reproducibility_checklist.json
  main_table.json
  report.md
  manifest.json
```

Scenario records are ordered by system order then test manifest order. Family and
category records use UTF-8 key order. Every file is schema-versioned, canonical
where applicable, hashed, mode `0600`, and listed in `manifest.json`.

`report.md` includes:

- primary and secondary native metrics;
- category/family-aware results and bootstrap intervals;
- direct latency, actual tokens, effective Qwen output cost and completeness;
- separate round-0/1/2 training latency/cost rows and the query-only checkpoint
  observation registry;
- profile and GPT-4o mini upstream-deviation disclosure;
- all manifest/config/generation hashes;
- timeout/failure/incomplete/reconciliation counts;
- failure-mode-related/repaired case counts and rates, overall and related-subset
  Generation-0-to-Updated score points, ambiguous/unmatched counts, and the
  observational-not-causal qualification;
- family-level all-variant success, minimum similarity, within-family range, and
  related-family complete-repair counts for ToolSandbox stability;
- reproducibility conclusion and every failed/not-run item;
- a clear statement that Qwen weights were never trained or modified;
- no claim beyond the completed evidence.

Reports never contain prompts, responses, raw trajectories, evaluator definitions/
mappings, hidden databases, tool arguments/results, API keys, endpoints, headers,
stack traces, or individual sensitive entities. Restricted audit artifacts remain
outside the report tree.

## Commands

`python -m toolsandbox_pipeline.reporting.cli` exposes:

```text
validate-plan
calibrate-vanilla
dry-run-report
final-test
resume-final-test
verify-report
```

- `validate-plan`, `dry-run-report`, and `verify-report` are offline.
- `calibrate-vanilla` accepts fixed train IDs only and cannot open test.
- `final-test` requires explicit plan approval and one-time capability.
- `resume-final-test` continues only the identical running ledger.
- no per-system rerun command, arbitrary ID override, `all`, shell passthrough,
  or metric-based selection exists.
- outputs are sanitized JSON status/metric/artifact references.

The CLI never retrieves secrets directly. The external launcher provides only the
credentials required by the selected frozen profile.

## Required Offline Tests

Cover at least:

1. Vanilla visibility/prompt/schema/action conversion and absence of every full
   pipeline component;
2. exact Vanilla Qwen config/durability/application/cost linkage and train-only
   calibrated limit enforcement;
3. final plan schema/canonical hash/system order/G000/G003/test membership;
4. plan rejection for provisional limits, G001/G002 selection, component drift,
   secret/raw URL, prior result, or mismatched system environment;
5. one-time test ledger transitions, completed rerun denial, identical resume, and
   changed-config resume denial;
6. official-live unknown external reconciliation and no automatic retry;
7. all shared-input equality and only three allowed system differences;
8. fresh contexts/request IDs and no response/result sharing;
9. manifest-order execution and deterministic parallel reordering;
10. native metric/count/rate/turn/category macro/micro denominator golden cases;
11. primary similarity preservation and no custom score substitution;
12. actual request/task/phase/role/model/provider breakdown reconciliation;
13. direct run timing and incomplete usage/cost propagation;
14. Vanilla versus pipeline committed-action effect eligibility and exclusion of
    applied-but-unlinked/no-op outputs;
15. family grouping, paired differences, fixed bootstrap index stream/hash,
    percentile golden vectors, and exact 25-family requirement;
16. every reproducibility checklist status/evidence/conclusion rule;
17. strict-replay identical smoke evidence and official-live disclosure behavior;
18. deterministic artifact ordering, canonical hashes, permissions, atomic
    publication, complete manifest, and verify-report;
19. crash injection throughout each system/scenario/evaluator/metrics/report step
    with no completed-system rerun;
20. report content/completeness/profile/deviation/model-weight statements;
21. test-result non-influence on plan, later systems, generations, prompts, limits,
    or selection;
22. secret/raw response/trajectory/evaluator/hidden database/tool entity/endpoint
    sentinel scans;
23. CLI access/argument allowlist and no test opening from offline/train commands;
24. imports/construction perform no filesystem, environment, dataset, provider,
    role, evaluator, or network access.

## Staged Validation Sequence

After code review:

1. run the full offline suite;
2. validate all manifests/generations/configs;
3. run separate Qwen/embedding/User/tool preflights;
4. complete train-only Vanilla calibration;
5. run train-only integration smoke;
6. if using `strict_replay`, run the two required deterministic smokes;
7. freeze and show the final plan/hash to the user;
8. obtain explicit final-test approval;
9. run Vanilla, Generation-0, and Updated once in fixed order;
10. materialize/verify the report without any tuning or rerun.

The task file authorizes none of steps 3-10 by itself. Each external stage uses the
coordinator/launcher boundary and the final test requires new explicit approval.

## Acceptance Commands

```bash
uv sync --frozen
uv run pytest -q
uv run python -m toolsandbox_pipeline.reporting.cli validate-plan --help
uv run python -m toolsandbox_pipeline.reporting.cli verify-report --help
git diff --check
git status --short
```

## Acceptance Criteria

- Vanilla, Generation-0, and G003 Updated are the only systems.
- Test plan is immutable before one-time test access.
- All systems use identical test/environment inputs except declared components.
- Completed test systems cannot be rerun or used to tune remaining work.
- Native similarity remains primary and statistics cluster by family.
- Timing, actual tokens, and substantive-effect Qwen-output cost follow Task 012
  exactly.
- Reproducibility claims match profile and evidence.
- Final reports are deterministic, immutable, sanitized, and independently
  verifiable.
- Offline tests pass with zero external access and no unowned files are modified.

## Completion Report Additions

```text
Final plan path/SHA-256/approval:
Profile and upstream User deviation:
G000/G003 hashes:
Checkpoint observation registry/hash and non-selecting pointers:
Vanilla calibration evidence:
One-time test ledger states:
Systems completed:
Scenario/family/category counts:
Primary similarity by system:
Family-cluster intervals:
total_running_time_seconds by system:
total_tokens / usage_complete by system:
total_cost / cost_unit / cost_complete by system:
Round 0/1/2 training latency and total_cost rows:
Failure/timeout/reconciliation counts:
Reproducibility conclusion/checklist hash:
Report manifest/hash verification:
Sensitive-output scan:
External stages actually executed:
Dependency/interface change requests:
```

Do not report a system result that did not complete. Do not describe smoke,
calibration, or synthetic tests as final dataset evidence.
