# Task 014: Episode Execution, Native Evaluation, and Trusted Trajectories

Status: `approved`

## Objective

Implement one recoverable ToolSandbox scenario execution from an explicitly
supplied scenario/start context and role set. Preserve the audited native turn
semantics while adding:

- durable User, Agent, and ExecutionEnvironment boundaries;
- Task 011 pre/post tool transactions and context recovery;
- fixed message-limit and conversation-termination behavior;
- exactly one post-episode native evaluator call;
- immutable restricted trajectory/evaluator records;
- conservative Skill-use attribution inputs;
- task-level timing/usage projections for Task 012.

This task does not choose dataset splits/order, perform offline memory/Skill
updates, run a Dev Mini-Bench, publish a generation, aggregate rounds, compare
systems, expose public experiment results, or own a CLI.

## Required Reading

Read, in order:

1. `pipeline.md`, Sections 3-7, 15-17, and 20-25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. `tasks/003_compact_state.md`;
6. `tasks/004_toolsandbox_adapter.md`;
7. `tasks/005_tool_metadata_controller.md`;
8. `tasks/006_model_api_gateways.md`;
9. `tasks/009_dataset_manifest_and_split_access.md`;
10. `tasks/010_rapidapi_fixtures_and_network_boundary.md`;
11. `tasks/011_checkpoints_request_ledger_and_recovery.md`;
12. `tasks/012_metrics_and_immutable_reports.md`;
13. `tasks/013_online_turn_orchestration.md`;
14. this task file;
15. only the pinned upstream `Scenario.play`, `Evaluation.evaluate`,
    `ExecutionEnvironment.respond`, role dispatch, and current-context APIs
    necessary to reproduce their audited behavior.

The Agent may inspect evaluator return types and invocation signatures, but not
real scenario definitions, milestone/minefield contents, target DataFrames,
similarity implementation details, dev/test IDs, or actual trajectories.

## Access

```text
Access class: offline
Setup network: none
Data splits: none
Secrets: none
External operations: none
```

Tests use synthetic Scenarios, ExecutionContexts, roles, tool effects, evaluator
results, and temporary run roots. Real model/user/tool/network calls are forbidden.

## Preconditions

- Tasks 001-013 are implemented and accepted.
- Task 013 supplies an episode-scoped `DurableTurnResponder`.
- Task 011 can serialize full ExecutionContext state and transact a complete
  logical parallel batch around one native ExecutionEnvironment invocation.
- Task 010 classifies every tool effect and records fixture/live external attempts.
- Task 012 supplies task timer and accounting input schemas.

## Required Task 006 User-Simulator Seam

The current instrumented User preserves upstream `respond()` behavior but must
also allow Task 011 to persist a successful raw response before the resulting User
message is appended. The user has authorized the Task 006-owned code/test
amendment. Before assigning Task 014, the coordinator must verify that Task 006
implemented the injected optional durable callback or equivalent prepare/complete
seam that:

- receives the exact `GatewayResponse` including raw bytes and attempt metrics;
- records it through caller-owned Task 011 logic before `model_inference` returns
  the ChatCompletion to upstream `OpenAIAPIUser.respond()`;
- supports completed-response reuse without a second OpenAI dispatch;
- preserves the exact upstream prompt/message/tool-call parsing and response
  behavior;
- exposes no raw bytes to repr/log/native messages;
- remains optional for isolated Task 006 unit tests and introduces no checkpoint
  import into provider modules.

Task 014 must verify this seam exists. It must not copy upstream User prompt logic,
wrap SDK internals, edit provider files, or accept post-message best-effort
recording.

## Owned Files

The assigned Agent may create or edit only:

```text
src/toolsandbox_pipeline/schemas/trajectory.py
src/toolsandbox_pipeline/toolsandbox_adapter/native_loop.py
src/toolsandbox_pipeline/toolsandbox_adapter/transactional_roles.py
src/toolsandbox_pipeline/toolsandbox_adapter/native_evaluator.py
src/toolsandbox_pipeline/toolsandbox_adapter/trajectory_store.py
src/toolsandbox_pipeline/toolsandbox_adapter/episode_runner.py
tests/episode/conftest.py
tests/episode/test_native_loop.py
tests/episode/test_transactional_environment.py
tests/episode/test_transactional_user.py
tests/episode/test_native_evaluator.py
tests/episode/test_trajectory_store.py
tests/episode/test_episode_runner.py
tests/episode/test_episode_recovery.py
tests/episode/test_episode_visibility.py
```

Do not edit upstream ToolSandbox, existing Adapter/online/provider/checkpoint/
metrics/reproducibility files, dataset manifests, prompts, dependency files, CLI,
offline code, or generated formal-run artifacts.

## Strict Episode Contracts

`schemas/trajectory.py` defines frozen strict types:

```text
EpisodeIdentity
EpisodeExecutionStatus
EpisodeResumeInput
NativeMessageRecord
ToolActionRecord
OnlineTurnRecord
TrustedEvaluatorRecord
SkillUseAttribution
TrustedTrajectory
EpisodeResult
```

`EpisodeIdentity` contains:

```text
run_id
profile
phase
round_index | null
shard_id | null
family_id
scenario_id
episode_id
manifest_position
system_variant
generation_id
starting_context_sha256
evaluation_definition_sha256
agent_tool_schema_sha256
dataset_manifest_sha256
runtime/prompt/token-limit/fixture/environment hashes
max_messages
```

Every identity is validated against caller-supplied Task 009 manifest metadata
before installing a context or invoking a role. The task does not enumerate or
select manifest entries itself.

`EpisodeResult` returns:

```text
identity
status
ending_context_reference
ending_context_sha256
trusted_trajectory_reference
evaluator_record_reference | null
task_accounting_input
last_checkpoint_ordinal
sanitized_failure_class | null
```

Large contexts/messages/trajectory bodies are Task 011 restricted blob references,
not embedded into public metrics or exception strings.

## Audited Native Loop

`native_loop.py` implements the minimal explicit loop needed for checkpoint
recovery. It must match pinned upstream `Scenario.play` semantics:

1. Fresh execution deep-copies `scenario.starting_context` and installs it with
   upstream current-context storage.
2. Process each initial SYSTEM-to-EXECUTION_ENVIRONMENT message in original index
   order and assert no message was appended.
3. Re-read the SANDBOX database after every role response.
4. While the last native `conversation_active` value is true and the last message
   index is below `max_messages + initial_max_index`, dispatch exactly the role in
   the last message's `recipient`.
5. Preserve native handling of a User `end_conversation` call and the exact
   maximum-message stopping rule.
6. Return the current upstream ExecutionContext without formatting or writing
   upstream output files.

Resume installs only a Task 011-validated committed context. It never deep-copies
the original start again, reruns processed system setup, or replays a committed
role/tool result.

Do not call upstream `Scenario.play_and_evaluate`; it writes mutable pretty-print
and conversation files outside the project transaction/artifact contract. Do not
use `tqdm`, wall-clock delays, multiprocessing, or a second message loop.

Compatibility tests run the project loop and upstream `Scenario.play` on
equivalent isolated synthetic scenarios and compare complete non-console context
hashes, message ordering, termination index, and role invocation counts.

## Transactional ExecutionEnvironment

`TransactionalExecutionEnvironment` delegates one unchanged logical message set
to one native `ExecutionEnvironment.respond()` call. Before delegation:

1. identify the complete current message set addressed to the environment;
2. bind it to the already-committed final Action/call IDs from Task 013, or to a
   User-only conversation-control action;
3. resolve current tool effect classes through Task 005/010 host metadata;
4. prepare the Task 011 tool transaction with the complete pre-action context;
5. checkpoint before the tool action;
6. allocate and mark the physical tool attempt `in_flight`.

After normal return, capture and durably commit the entire resulting context,
visible response/exception identities, Task 010 external-attempt references, and
post-action checkpoint before permitting the next loop iteration.

The native environment remains responsible for:

- parsing/validation of execution code;
- actual local tool calls;
- all-ordering permutation checks for parallel batches;
- context mutation and rollback semantics;
- response order and `tool_trace` association.

This task does not execute callable objects directly, alter native response text,
skip permutations, or expose `tool_trace` online. A batch is one logical Task 011
transaction; Task 010 separately records every physical external read caused by
native permutation execution.

Recovery follows Task 011 exactly: reuse committed contexts; replay local or
fixture-backed unknown actions only from stored pre-context; stop for
`official_live` external-read reconciliation; reject changed action/call/context/
profile/fixture identities.

## Transactional User Role

`TransactionalUserRole` delegates to exactly one supplied User implementation:

- `official_live`: Task 006 `InstrumentedGPT4oMiniUser`;
- `strict_replay`: a separately configured frozen local deterministic User role;
- offline tests: a synthetic deterministic User.

It does not choose or fall back between them. The profile/runtime manifest pins
the class/configuration identity.

For a remote User request, use the required Task 006 durable seam and Task 011
logical/physical request ledger before any User message is appended. User
`logical_request_id` is derived from the exact filtered upstream input messages,
tools/omission contract, fixed snapshot, prompt/few-shot/tool-schema hashes,
current episode/message identity, and model config.

A completed response is reused without dispatch and passed back through unchanged
upstream parsing. An unknown outcome may receive a new physical attempt only under
Task 011 recovery. Missing/invalid usage is retained and poisons
`total_tokens` completeness. User output never contributes to `total_cost`.

After native User messages are appended, persist the complete current context and
post-User checkpoint before continuing. User tool calls addressed to
ExecutionEnvironment are executed only by the transactional environment wrapper.

## Agent Role Boundary

The Agent role is exactly Task 004 `PipelineAgent` with Task 013 responder:

- the responder commits the final action before return;
- PipelineAgent validates/converts/appends native Agent messages once;
- the episode runner stores the post-Agent context/checkpoint before dispatching
  the next recipient;
- no Agent path can call `end_conversation` directly;
- no evaluator data is present before episode completion.

If the responder or Adapter fails, append no partial Agent messages, checkpoint
the terminal failure, and stop the episode without evaluator invocation unless the
protocol explicitly defines a normally terminated context.

## Native Evaluation

After the loop ends normally, freeze the ending context reference and call exactly:

```python
scenario.evaluation.evaluate(
    execution_context=ending_context,
    max_turn_count=scenario.max_messages,
)
```

Do not call it on a partially committed context, before conversation/message-limit
termination, or more than once. Do not rewrite or precompute milestone/minefield
matching, guardrails, column similarity, effective turns, or score.

`TrustedEvaluatorRecord` strictly preserves:

```text
milestone_similarity
minefield_similarity
similarity
turn_count
milestone_mapping
minefield_mapping
fully_successful
evaluation_definition_sha256
ending_context_sha256
```

`fully_successful` is host-derived as `similarity == 1.0`. Validate the native
rule that nonzero minefield similarity forces total similarity to zero; otherwise
do not reinterpret results.

Evaluator content is written to a restricted blob after episode completion and
referenced by hash. It never re-enters online state, retrieval, model prompts, User
messages, or a later test-selection decision.

## Trusted Trajectory

`TrustedTrajectory` combines immutable references to:

- complete native message history and committed context hashes;
- Task 013 state/retrieval/Policy/Controller/Critic/Revision/final-action records;
- logical request and every physical-attempt reference;
- tool actions, committed results/exceptions, effects, fixtures, and context hashes;
- generation/profile/config/prompt/schema/manifest identities;
- one terminal evaluator record;
- explicit eligibility for train-only offline consumption.

Raw provider bodies stay in their Task 011 blobs. Hidden evaluator bodies may be
present only in the restricted evaluator section. A sanitized replay view omits
hidden/canonical/controller-only/evaluator content and is never used as a prompt
without the later offline task's explicit projection.

Trajectory identity is derived from episode identity, ordered event hashes,
ending-context hash, and evaluator hash. Records are canonical, immutable, mode
`0600`, content hashed, and atomically published through Task 011 blob/commit
primitives. A repeated identical finalization is a no-op; conflicts fail.

## Conservative Skill Attribution

For each Skill ID:

1. require at least one actually executed committed call whose final routed action
   explicitly bound that Skill;
2. require one trusted evaluator record;
3. emit at most one `SkillUseAttribution` per Skill per episode;
4. label success only when `fully_successful`;
5. omit blocked, unexecuted, rolled-back, or missing-evaluator calls.

Attribution records the Skill ID/version/generation, executed call IDs, canonical
tool IDs in restricted offline provenance, and evaluator reference. It cannot use
Critic prediction as a label or create call-level correctness.

## Failure and Resume Semantics

Typed statuses are:

```text
running
completed_evaluated
terminal_failure_before_evaluation
reconciliation_required
```

Every role boundary and every new Agent-visible User/environment message receives
a durable checkpoint. On a hard error, persist only sanitized class plus latest
valid references. Do not evaluate a failed partial episode or fabricate a score.

Resume validates every identity/hash, selects a Task 011 recovery plan, restores a
new isolated context, and resumes at one exact role boundary. It cannot skip an
unfinished recipient, repeat a committed tool effect, regenerate an applied model
response, or change generation/profile/User/fixture configuration.

## Metrics Boundary

Start Task 012 scenario timing immediately before installing/dispatching the first
fresh state; stop after evaluator record and durable final episode checkpoint.
Same-boot resume includes downtime. Emit strict accounting projections for:

- all model logical requests and physical attempts;
- applied Qwen source attempts;
- tool attempts and real latency;
- direct task elapsed time;
- Policy/Critic/Revision/User call counts;
- terminal/evaluation status.

This task does not write round/run metric artifacts. Preflight/setup time is
excluded.

## Required Offline Tests

Cover at least:

1. strict episode/trajectory schemas, identity/hash sensitivity, and no coercion;
2. fresh context deep-copy isolation and exact current-context installation;
3. initial system-to-environment processing without appended messages;
4. native-recipient loop, database re-read, exact message cap, and User-only
   termination;
5. compatibility parity with upstream `Scenario.play` on synthetic assistant,
   function-call, parallel-batch, error, termination, and max-message cases;
6. no `play_and_evaluate`, tqdm, output-directory, multiprocessing, or hidden
   fallback path;
7. transactional single/parallel/User-control action prepare/in-flight/commit
   ordering and complete context snapshots;
8. proof native ExecutionEnvironment performs parsing, calls, permutations,
   rollback, response order, and tool-trace handling unchanged;
9. local/fixture recovery replay, committed reuse, allocated abandonment, and
   official-live reconciliation stop;
10. Task 010 external physical-attempt linkage for permutation execution;
11. transactional local/remote synthetic User success, completed reuse, unknown
    recovery, invalid/missing usage, and post-message context commit;
12. Task 006 durable callback ordering before User message append and raw-data
    non-disclosure;
13. Agent final-action commit, native append, and post-Agent checkpoint order;
14. exactly one evaluator call only after normal completion with exact native
    arguments;
15. evaluator rule validation, `fully_successful`, restricted storage, and
    online inaccessibility before/after finalization;
16. no evaluator call/record for partial terminal failure or reconciliation;
17. trajectory completeness, canonical identity, atomic identical publication,
    conflict denial, and restricted permissions;
18. Skill attribution executed/evaluated/once-per-skill rules and no Critic-label
    substitution;
19. crash injection before/after each role, tool, message, evaluator, trajectory,
    and final-checkpoint boundary;
20. resume at the exact recipient with no duplicate message/effect/request/score;
21. task timing and accounting projections including substantive-effect
    Qwen-output cost;
22. secret/raw-response/hidden database/evaluator/prompt/canonical-name sentinel
    scans;
23. imports/construction perform no context install, file creation, environment
    read, model/tool/evaluator/network call.

## Acceptance Commands

```bash
uv sync --frozen
uv run pytest -q tests/episode tests/online tests/checkpointing
uv run python -c "from toolsandbox_pipeline.toolsandbox_adapter.episode_runner import EpisodeRunner"
git diff --check
git status --short
```

## Acceptance Criteria

- Fresh and resumed execution preserve audited native Scenario semantics.
- All native role/tool effects are committed before later consumption.
- User model responses are durably recorded before native message application.
- Tool recovery never duplicates committed effects or retries unknown live reads.
- Native evaluator runs exactly once, after normal episode completion only.
- Evaluator/trusted trajectory data never crosses the online visibility boundary.
- Skill attribution is conservative and task-level.
- Task metrics cover the exact episode boundary and requested token/cost semantics.
- Offline tests pass with zero external access and no unowned files are modified.

## Completion Report Additions

```text
Native-loop parity cases:
Role invocation/message-limit cases:
Tool transaction/recovery cases:
User durable-seam cases:
Evaluator invocation count/arguments:
Trajectory/attribution cases:
Episode crash points:
Resume no-duplicate assertions:
Task timing/accounting projections:
Visibility/sensitive-output scan:
External calls: 0
Total tokens: 0
Usage complete: true
Task 006 interface amendment present:
Dependency/interface change requests:
```

Unit-test wall time is development evidence only, not experimental
`total_running_time_seconds`.
