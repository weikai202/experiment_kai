# Task 013: Durable Online Turn Orchestration

Status: `approved`

## Objective

Implement the project-owned `PipelineResponder` for exactly one real Agent turn.
It composes the already-reviewed State Builder, pinned-generation retrieval,
Initial Policy, deterministic Controller, optional Critic, at most one Revision,
Task 011 request durability, and final Action delivery to the thin Task 004
`PipelineAgent`.

This task owns online routing and durable model-output application. It does not
play an entire scenario, invoke User or ExecutionEnvironment roles, execute tools,
call the native evaluator, select a dataset split, perform offline updates,
publish generations, aggregate run metrics, or expose a CLI.

## Approved Review Decisions

- The fixed routing algorithm below is approved.
- The deterministic Controller remains authoritative; Critic cannot override a
  blocking code.
- A completely clean initial action skips Critic.
- One state permits at most one Revision.
- If the revised action still has a blocking code, return the versioned
  clarification message described below; do not terminate the episode solely for
  that condition.

## Required Reading

Read, in order:

1. `pipeline.md`, Sections 3-4, 7-15, and 20-22;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `tasks/002_core_contracts.md`;
5. `tasks/003_compact_state.md`;
6. `tasks/004_toolsandbox_adapter.md`;
7. `tasks/005_tool_metadata_controller.md`;
8. `tasks/006_model_api_gateways.md`;
9. `tasks/007_generation_records_and_retrieval.md`;
10. `tasks/008_online_prompts_and_qwen_roles.md`;
11. `tasks/010_rapidapi_fixtures_and_network_boundary.md`;
12. `tasks/011_checkpoints_request_ledger_and_recovery.md`;
13. `tasks/012_metrics_and_immutable_reports.md`;
14. this task file.

Do not inspect scenario definitions, evaluator internals, hidden databases,
milestones, minefields, real prompts/responses, dev/test IDs, or trajectories.

## Access

```text
Access class: offline
Setup network: none
Data splits: none
Secrets: none
External operations: none
```

All tests use literal Adapter turns, synthetic generations, fake embedding/role
executors, temporary checkpoint stores, and deterministic injected failures. No
real model, OpenAI key, Qwen endpoint, dataset, tool, evaluator, or network access
is allowed.

## Preconditions

- Tasks 001-012 are implemented and accepted.
- Task 011 exposes stable logical/attempt operations and recovery plans without
  provider retry loops.
- Task 011 application rows expose the immutable completed
  `source_attempt_id`.
- Task 012 schemas can consume logical/physical projections but are not called to
  finalize task/round/run metrics here.
- The caller supplies exactly one immutable generation, runtime identities, and
  already-validated host-only metadata for the whole episode.

## Owned Files

The assigned Agent may create or edit only:

```text
src/toolsandbox_pipeline/schemas/online_turn.py
src/toolsandbox_pipeline/online/turn_context.py
src/toolsandbox_pipeline/online/durable_roles.py
src/toolsandbox_pipeline/online/routing.py
src/toolsandbox_pipeline/online/turn_responder.py
tests/online/test_turn_context.py
tests/online/test_durable_roles.py
tests/online/test_routing.py
tests/online/test_turn_responder.py
tests/online/test_turn_recovery.py
tests/online/test_turn_visibility.py
```

Do not edit the existing State Builder, Controller, prompt builders, Qwen role
runners, retrieval implementation, provider gateways, checkpoint implementation,
metrics implementation, Task 004 Adapter, upstream ToolSandbox, configs, prompts,
dependencies, or generated artifacts.

If a required existing API is absent, stop with one exact interface request. Do
not copy or fork another task's implementation.

## Public Contracts

`schemas/online_turn.py` defines frozen, strict models:

```text
OnlineTurnIdentity
OnlineTurnInput
OnlineTurnStage
OnlineTurnDecision
OnlineTurnAuditRecord
OnlineTurnFailure
```

`OnlineTurnIdentity` contains:

```text
run_id
profile
phase
round_index | null
shard_id | null
family_id
scenario_id
episode_id
agent_turn_index
expected_generation_id
dataset_manifest_sha256
runtime_config_sha256
prompt_manifest_sha256
token_limit_config_sha256
fixture_manifest_sha256 | null
environment_identity
```

`OnlineTurnInput` contains the Task 004 `AdapterTurn`, identity, committed
Agent-visible tool outcomes, pending-dependency inputs, prior visible failed-action
history, and `structured_constraint_tension`. It contains no evaluator value,
hidden database, raw context, raw response, secret, or canonical mapping in its
prompt-facing branch.

`OnlineTurnDecision` contains exactly:

```text
identity
state_id
generation_id
final_action
initial_policy_logical_request_id
critic_logical_request_id | null
revision_logical_request_id | null
initial_controller_decision
post_revision_controller_decision | null
critic_verdict | null
revision_count
retrieval_references
audit_record_sha256
```

It implements/backs Task 004 `PipelineResponder.respond(turn)` and returns only a
locally validated `ActionEnvelope` to `PipelineAgent`. The richer decision is
retained in restricted checkpoint/audit state, never attached to a native Message.

## Explicit Dependency Injection

`TurnContext` is constructed with:

- immutable `OnlineTurnIdentity`;
- one pinned `GenerationSnapshot` and matching `RetrievalService`;
- `StateBuilder`, `Controller`, and current controller metadata;
- Task 008 Policy/Critic/Revision prepared-request builders;
- a `DurableRoleExecutor` backed by Task 011;
- one checkpoint-event sink;
- current committed-tool/pending/action-history projections;
- a deterministic safe-failure action factory;
- optional lifecycle hooks used only by tests.

Construction validates identities but performs no file open, environment read,
embedding/model call, checkpoint write, or upstream context access. Dependencies
are supplied explicitly; module globals/service locators/default clients are
forbidden.

One `DurableTurnResponder` instance is episode-scoped, not process-global. It
rejects a different run/scenario/generation/profile or a repeated/stale
`agent_turn_index`.

## State Construction Boundary

For each `AdapterTurn`:

1. Verify the Agent-visible view and host-only `ControllerToolContext` remain
   separate.
2. Verify tool mapping hash, available agent-facing names, and episode/runtime
   identities.
3. Build one Task 003 `StateBuildInput` from only visible messages, committed
   visible outcomes, pending dependencies, and augmented schemas.
4. Call the existing `StateBuilder` once.
5. Verify `state_id`, turn index, scenario/family identity, mapping hash, and
   generation pin.
6. Durably checkpoint the new Agent-visible state before retrieval or model work.

Never read SETTING/CONTACT/MESSAGING/REMINDER databases, `conversation_active`,
`tool_trace`, evaluator definitions, or raw ExecutionContext from this layer.
Canonical tool names exist only in host controller inputs/sidecars and restricted
audit records.

Repeated invocation for the exact same state uses the same logical requests and
recovery state. Reusing a turn index with a different state or mapping is a hard
identity failure.

## Durable Role Execution

`DurableRoleExecutor` is the only path from this task to a Task 008 role runner.
For each prepared Policy, Critic, or Revision request it:

1. validates role/state/generation/prompt/schema/token-limit identities;
2. asks Task 011 to prepare/reuse the stable logical request;
3. writes the required pre-request checkpoint;
4. follows the Task 011 recovery plan;
5. reuses a completed response without dispatch, or allocates/marks exactly one
   physical attempt immediately before one Task 008 runner call;
6. stores exact raw response bytes/hash, actual usage, validated output, and
   terminal status through Task 011;
7. returns a strict output plus logical/source-attempt identities;
8. marks it `applied` only in the same durable transition that stores the
   orchestrator state proving how that output informed routing.

It does not loop, sleep, retry, change token limits, catch a provider error as an
empty output, or dispatch after terminal invalid/truncated output. Only a Task 011
`dispatch_recovery_llm_attempt` plan can authorize a new attempt after an unknown
outcome.

An embedding requested by the existing RetrievalService must use the same durable
ledger/recovery discipline through its injected `record_durable` callback.
Embedding output is applied to the retrieval bundle, but it never contributes to
the substantive-effect Qwen-output `total_cost`.

## Fixed Routing Algorithm

The responder follows exactly:

1. Retrieve Policy memories and eligible Skills once for the new state.
2. Prepare and durably execute Initial Policy.
3. Build one host-only `ControllerInput` and call the existing Controller.
4. If both Controller code lists are empty, select the original action.
5. Otherwise retrieve World memory once and durably execute Critic.
6. If blocking codes are empty and Critic verdict is `accept`, select the
   original action.
7. If any blocking code exists or Critic verdict is `revise` or `uncertain`,
   prepare and durably execute Revision exactly once using the original state,
   original Policy/Skill bundle, original action, prompt-safe Controller evidence,
   and Critic output.
8. Call the same deterministic Controller on the revised action.
9. After Revision, inspect only blocking codes. Never retrieve again, call Critic
   again, or perform a second Revision.
10. If revised blocking codes are empty, select the revised action.
11. Otherwise select one deterministic assistant-message safe failure.

The Critic can never override a blocking code. A Controller trigger alone always
causes one Critic call. A Critic `accept` never causes Revision when no blocking
code exists. An initial completely clean action never calls World retrieval,
Critic, or Revision.

The safe-failure content is a versioned constant stating only that no safe action
can be taken with available information and asking for clarification. It must not
list hidden/controller evidence, canonical names, schema details, or predicted
outcomes. It is used only after one valid Revision remains blocked. Provider,
schema, checkpoint, retrieval, or identity failures are terminal typed failures
and do not become a friendly assistant message.

## Retrieval Invariants

- Policy/Skill retrieval occurs exactly once per unique state.
- World retrieval occurs only after Controller codes require Critic.
- Revision receives the original immutable Policy/Skill bundle; it cannot invoke
  retrieval.
- All bundles match state and generation exactly.
- Empty valid stores produce empty bundles without fallback.
- Embedding/cache failures checkpoint and stop; they do not return empty hits.
- Retrieved prose remains heuristic and cannot enter verified facts or ground an
  argument.
- Policy/Critic prompt projections never contain vectors, source paths, raw
  Controller refs, canonical names hidden by scrambling, or generation files.

## Controller and Revision Invariants

- Controller input uses current augmented schemas and current mapping only.
- Argument grounding uses the existing Controller; the responder never repairs
  arguments or fills missing values.
- Parallel independence is decided before Adapter conversion/native execution.
- Original and revised actions receive independent full Controller checks.
- Post-Revision trigger codes are recorded for audit but ignored for further
  model routing; blocking codes still prevent execution.
- `revision_count` is exactly zero or one per `state_id`.
- A safe-failure action binds no Skill ID and calls no tool.

## Checkpoint and Audit Stages

`OnlineTurnStage` is a closed enum with required durable events:

```text
state_built
policy_retrieval_completed
initial_policy_completed
initial_policy_applied
initial_controller_completed
world_retrieval_completed
critic_completed
critic_applied
revision_completed
revision_applied
post_revision_controller_completed
final_action_committed
terminal_failure
```

Only stages reached by the fixed route are written. Each event stores identities,
strict content hashes/blob references, retrieval IDs/scores, logical/source attempt
IDs, decisions, and final-action fingerprint. Raw response bytes remain only in
Task 011 restricted blobs.

`final_action_committed` is durable before `respond()` returns the action to
Task 004. Recovery reuses it verbatim and never reruns models/Controller. Re-entry
with different inputs fails.

`OnlineTurnAuditRecord` may contain full Controller source refs and canonical
tool attribution in a restricted artifact. Its prompt-safe projection hashes those
refs. Neither form contains credentials/raw responses, and only the safe
projection can enter Critic/Revision prompts.

## Failure Semantics

On any hard failure:

1. persist the latest valid component state and sanitized failure class;
2. write a `terminal_failure` checkpoint event;
3. append no Agent message and execute no tool;
4. return/raise one typed `OnlineTurnFailure` for the episode runner.

No exception text from SDK/provider/raw response is copied into ordinary logs.
Failure after a physical dispatch retains its attempt/usage. Failure before
dispatch cannot invent an attempt. A process crash is handled by Task 011 recovery,
not by broad exception retry.

## Metrics Boundary

This task records logical application/source-attempt links and stage timing inputs.
It does not aggregate or publish Task 012 files.

- Every physical attempt remains eligible for `total_tokens`.
- Only Qwen outputs in the final committed online decision chain are eligible for
  `total_cost`; a response application/audit row by itself is insufficient.
- Policy/Critic/Revision logical and physical counts stay distinct.
- Reused completed responses are not new physical attempts.
- Retrieval embedding attempts retain role `embedding`.
- Technical unit-test time/tokens are not experiment metrics.

## Required Offline Tests

Use strict synthetic inputs and fake injected components. Cover at least:

1. construction/import side-effect freedom and complete identity validation;
2. Agent-view/controller-sidecar separation and hidden/canonical sentinel denial;
3. one State Builder call, stable state identity, stale/reused-turn rejection, and
   state checkpoint before retrieval;
4. clean Controller path: Policy only, no World/Critic/Revision, original action;
5. trigger-only path with Critic accept and original action;
6. trigger-only path with Critic revise/uncertain and one Revision;
7. blocking path always reaching Revision even when Critic says accept;
8. revised action full blocking recheck, ignored post-revision trigger codes, and
   no second Critic/Revision;
9. revised blocking failure producing only the fixed safe clarification;
10. Policy/Skill retrieval once, conditional World retrieval once, and original
    bundles reused by Revision;
11. empty retrieval stores, retrieval failure, state/generation mismatch, and
    forbidden fallback denial;
12. deterministic Controller input construction with augmented schemas, mapping,
    metadata, Skill views, history, profile, and tension flag;
13. durable role success ordering: prepare/checkpoint/in-flight/response/application;
14. completed response recovery without dispatch and unknown-outcome recovery with
    a distinct attempt ID;
15. allocated-before-dispatch recovery, terminal invalid output, truncated output,
    missing usage, and checkpoint failure without hidden retry;
16. application rows bind exactly one completed source attempt and cannot change;
17. final action durable before return, exact recovery reuse, and conflicting
    re-entry rejection;
18. process-crash injection at every listed stage with at-most-once logical
    application and at-most-one Revision;
19. safe failure and hard failures append no unauthorized/tool messages;
20. metrics projections distinguish all physical attempts, applied Qwen outputs,
    and committed online-action effect links;
21. raw response, secret, endpoint, hidden evaluator/database, canonical-name,
    prompt, and Controller-ref sentinel scans;
22. no test performs a real embedding/model/tool/evaluator/network call.

## Acceptance Commands

```bash
uv sync --frozen
uv run pytest -q tests/online tests/checkpointing tests/retrieval
uv run python -c "from toolsandbox_pipeline.online.turn_responder import DurableTurnResponder"
git diff --check
git status --short
```

## Acceptance Criteria

- The fixed routing table is implemented exactly with zero or one Revision.
- The Controller remains deterministic and cannot be bypassed by Critic.
- Prompt-visible inputs never cross the canonical/hidden/evaluator boundary.
- Every role/retrieval request follows Task 011 durability and recovery.
- A final action is committed before the Adapter can append it.
- Re-entry and crash recovery do not repeat completed work or double-apply output.
- Only Task 004 converts/appends native messages; this task executes no tool.
- Physical usage and substantive online-action effect links support Task 012
  exactly.
- Offline tests pass with zero external access and no unowned files are modified.

## Completion Report Additions

```text
Online route cases:
Clean / Critic accept / Revision / safe-failure cases:
Policy-Skill retrieval counts:
World retrieval counts:
Revision max per state:
Durable role transitions:
Recovery/crash stages:
Final-action commit ordering:
Applied Qwen source-attempt links:
Visibility/sensitive-output scan:
External calls: 0
Total tokens: 0
Usage complete: true
Dependency/interface change requests:
```

Unit-test wall time is development evidence only, not experimental
`total_running_time_seconds`.
