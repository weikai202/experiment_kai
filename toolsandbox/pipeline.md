# ToolSandbox Policy–Critic Pipeline Specification

## 1. Purpose and Audit Baseline

This document is the implementation contract for adapting the Policy–Controller–Critic architecture, external memories, and Skill Library defined in `bfcl-online/pipeline.md` to Apple ToolSandbox. Its goals are to:

- use frozen Qwen3-32B models for the Policy, World Model/Critic, and Skill Updater;
- perform no model-weight training or modification;
- preserve ToolSandbox's four-role conversation, state databases, tool augmentations, milestone DAG, minefields, and native scoring;
- update external artifacts offline using only trusted results from the current training round;
- evaluate Vanilla, Generation-0, and Updated systems with the same data, user simulator, tool environment, and native evaluator;
- support auditable reproduction, checkpoint recovery, latency/token accounting,
  and a non-monetary substantive-effect Qwen-output cost metric.

This specification was audited against:

```yaml
toolsandbox_repository: https://github.com/apple/ToolSandbox
toolsandbox_commit: 165848b9a78cead7ca7fe7c89c688b58e6501219
audited_at: 2026-09-03
python: "3.10"
pydantic: "2.7.4"
```

Every run must pin `toolsandbox_commit`, the source-tree SHA-256, dependency-lock SHA-256, container-image digest, scenario-manifest SHA-256, and this specification's version. Changing any of them defines a new experimental protocol.

## 2. Feasibility Conclusion

Conclusion: **the pipeline is feasible under explicit conditions, but the BFCL adapter cannot be reused directly, and the unfrozen upstream default CLI is not strictly reproducible.**

Empirical and static checks on the audited commit found that:

- the project installs in a Python 3.10 environment;
- 1,032 expanded scenarios can be instantiated from 129 nominal scenario families, each with 8 variants;
- 34 registered tools can be discovered, including 5 from `rapid_api_search_tools`;
- all 4 tests in `tests/common/evaluation_test.py` pass;
- on Windows with Python 3.10.2, `execution_environment_test.py` has 4 passes and 2 failures: one is a Python traceback caret-format difference, and the other is a response-order assertion difference for an invalid parallel call; in the latter test, the sandbox state still remains unchanged.

Therefore:

1. the native milestone/minefield evaluator, ExecutionContext snapshots, and tool registry can support this pipeline;
2. formal runs must use a pinned Linux container and must not treat Windows traceback text or failed-response ordering as semantic evidence;
3. upstream dynamic time, remote user models, RapidAPI, and default multiprocessing must be constrained by the reproducibility layer in this specification;
4. the claim "reproducible experiment implemented" is allowed only after Section 24 passes. The current audit establishes structural feasibility, not a completed end-to-end Qwen3-32B reproduction.

## 3. Non-Negotiable Boundaries

1. All Qwen3-32B weights remain frozen.
2. Policy, Critic, Revision, and all offline-update calls share one vLLM OpenAI-compatible endpoint. Roles are separated only by prompts, inputs, and external memories.
3. Every Qwen call uses:

   ```yaml
   temperature: 0.0
   seed: 0
   enable_thinking: false
   # Do not override top_p.
   ```

   The OpenAI-compatible request passes the hard switch as `chat_template_kwargs: {"enable_thinking": false}`. Structured output uses one manifest-pinned `structured_output_wire_mode`: `guided_json` for a compatible legacy vLLM deployment or `structured_outputs_json` for a compatible current deployment. A run never probes, switches, or falls back between modes. The manifest pins the exact vLLM version, container digest, structured-output backend/mode, server launch configuration, and generation-config policy. Formal servers disable unrecorded model-repository sampling overrides and must not enable a reasoning parser for this non-thinking pipeline. Preflight fails on ignored parameters, reasoning content, `<think>` output, a mismatched served model, or an unsupported configured wire mode.

4. Only real user messages visible to the Agent, real execution-environment responses, committed ToolSandbox tool results, and native evaluator results produced after episode completion are facts.
5. Hidden databases, scenario-construction code, milestones, minefields, target DataFrames, similarity functions, and evaluator mappings are unavailable during online decision-making.
6. Critic predictions, Critic memory, Policy memory, skills, and learned notes are always heuristic. They cannot enter verified facts or independently ground a tool argument.
7. Each episode pins one immutable Policy-memory, World-memory, and Skill-Library generation. The generation cannot change inside the episode.
8. A real conversation state permits at most one Policy Revision. Revision loops are forbidden.
9. Dependent tool calls execute across successive real ExecutionContext states. A parallel batch contains only mutually independent calls.
10. Raw training trajectories from a round are archived after offline updates but cannot be read by later rounds. Only aggregate statistics may persist across rounds.
11. Dev trajectories and evaluator results may be used only by the deterministic Skill Dev Mini-Bench in Section 19 to accept or reject an already-generated skill candidate. They cannot generate or rewrite candidates, enter model prompts, update memories or statistics, alter retrieval indexes, or influence test-time checkpoint selection. Test trajectories and evaluator results cannot influence any update or selection decision.
12. Vanilla, Generation-0, and Updated test evaluations run exactly once. Test results cannot be used to tune settings or reselect a checkpoint.
13. The pipeline cannot silently bypass ToolSandbox's native scenarios, role visibility, tool implementations, tool augmentations, or evaluator semantics.
14. `end_conversation` remains visible only to the User role. The Agent pipeline cannot terminate a conversation directly.

## 4. ToolSandbox Integration Boundary

The native ToolSandbox roles remain unchanged:

```text
SYSTEM
USER                    # Human or pinned user simulator
AGENT                   # Replaced by this pipeline implementation
EXECUTION_ENVIRONMENT   # Upstream implementation
```

The new `PipelineAgent(BaseRole)` replaces only `RoleType.AGENT`. The scenario loop, User, ExecutionEnvironment, ExecutionContext, and `Evaluation.evaluate()` retain their native semantics.

One Agent response follows this fixed path:

```text
Agent-visible messages + current agent-facing tools
→ deterministic State Builder
→ retrieval
→ Initial Policy
→ rule-based Controller
→ optional Critic
→ at most one Revision
→ ToolSandboxAdapter writes Message objects
→ native ExecutionEnvironment or User handles them
→ new real messages/snapshots
```

Policy actions map to native messages as follows:

| Policy action | ToolSandbox message |
| --- | --- |
| `function_call` | One `AGENT → EXECUTION_ENVIRONMENT` message |
| `parallel_batch` | Consecutive `AGENT → EXECUTION_ENVIRONMENT` messages in one Agent turn |
| `assistant_message` | One `AGENT → USER` message |

The model always uses the scenario's current **agent-facing tool name**. The Adapter converts it with `ExecutionContext.get_execution_facing_tool_name()` before execution. Canonical tool names are available only to the Controller, statistics, and offline attribution. Tool-name scrambling mappings must never be shown to the Policy or Critic.

## 5. Data Units, Variants, and Leakage-Free Splits

ToolSandbox has no official train/dev/test split. The audited commit first creates 129 nominal scenarios, then generates these variants for each nominal scenario:

```text
no distraction
3 distraction tools
10 distraction tools
all tools
3 distraction + tool description scrambled
3 distraction + argument type scrambled
3 distraction + argument description scrambled
3 distraction + tool name scrambled
```

All variants of one nominal scenario belong to one indivisible `scenario_family_id`. The family registry must be built from the unaugmented results of the four `named_*_scenarios()` functions before invoking upstream augmentation logic. Inferring families by stripping string suffixes is forbidden.

The split algorithm is fixed:

```yaml
data_seed: 0
family_count_at_audited_commit: 129
test_fraction: 0.20
dev_fraction: 0.20
num_update_rounds: 3
```

1. Sort family IDs by UTF-8 byte order.
2. Shuffle exactly once with Python 3.10 `random.Random(data_seed).shuffle`.
3. Let `test_count = floor(0.20 * N)`; assign the first `test_count` families to test.
4. Let `dev_count = floor(0.20 * N)`; assign the next `dev_count` families to dev.
5. Assign all remaining families to train.
6. Expand all variants only after assigning families.

The audited commit must produce:

```yaml
train_families: 79
dev_families: 25
test_families: 25
train_scenarios: 632
dev_scenarios: 200
test_scenarios: 200
train_round_sizes: [216, 208, 208] # 27/26/26 families × 8 variants
```

Split the shuffled training families into three contiguous shards. All eight variants of a family must remain in the same shard. The manifest records ordered family IDs, expanded scenario IDs, categories, starting ExecutionContext hashes, evaluation-definition hashes, agent-facing tool-schema hashes, and tool order.

Any count, mapping, or hash mismatch stops the run before the first model call.

## 6. Reproducibility Profiles

Formal results use one of two profiles and must never combine their artifacts or metrics.

### `strict_replay`

- Pin the Linux image digest, Python patch version, `UTC` timezone, and locale.
- Fix ToolSandbox world time to the manifest's `world_epoch`.
- Route scenario construction and tool calls to controlled implementations of `datetime.now()`, `.timestamp()`, and current-year lookup.
- Use an unfrozen monotonic clock for telemetry.
- Allow RapidAPI tools to read only a hash-pinned fixture store. A fixture miss returns deterministic `EXTERNAL_FIXTURE_MISS`; it never falls back to live network access.
- Use a frozen local user-simulator checkpoint with deterministic decoding, a pinned prompt, and a pinned tokenizer.
- Default to `processes: 1`. If scenario-level parallelism is enabled, derive randomness from scenario ID and sort by scenario ID before aggregation.

### `official_live`

- Preserve upstream live RapidAPI calls and use the project-selected `gpt-4o-mini-2024-07-18` User Simulator through OpenAI Chat Completions. This fixed mini snapshot is an intentional benchmark-configuration deviation from the audited upstream `gpt-4o-2024-05-13` default and must be disclosed in every formal report. The User Simulator and `text-embedding-3-small` explicitly share one `OPENAI_API_KEY`, but retain separate clients or client roles, request records, and usage accounting.
- Still pin the commit, dependencies, scenario manifest, Policy decoding, and data split.
- Describe results only as configuration-traceable or statistically reproducible, never as trajectory-level strictly reproducible.

The paper's main table must identify the profile. The two profiles use separate memory generations and retrieval indexes.

## 7. Compact Verified State

The State Builder is a deterministic event reducer. It never calls Qwen, performs natural-language inference, or reads information invisible to the Agent.

```json
{
  "episode_id": "...",
  "scenario_id": "...",
  "scenario_family_id": "...",
  "state_id": "...",
  "agent_turn_index": 0,
  "visible_messages": [
    {"message_id": "m1", "sender": "USER", "recipient": "AGENT", "content": "..."}
  ],
  "current_observation": {"message_id": "m1", "content": "..."},
  "verified_facts": {},
  "completed_tool_calls": [],
  "failed_actions": [],
  "pending_dependencies": [],
  "available_tools": [],
  "conversation_status": "agent_turn"
}
```

The reducer follows these fixed rules:

1. Consume only SYSTEM, USER, AGENT, and EXECUTION_ENVIRONMENT messages made visible to the Agent by `BaseRole.filter_messages()`.
2. Preserve verbatim content and stable IDs in `visible_messages`; do not summarize. User-simulator few-shot messages enter only when upstream visibility exposes them to the Agent.
3. Tool-result facts must come from execution-environment messages actually received by the Agent. The Adapter may apply `ast.literal_eval` to visible `content` and strictly validate it against a registered public return contract. If validation fails, preserve only the original string.
4. Do not use internal `tool_trace` to supplement information the Agent did not receive. `tool_trace` is available only to committed logs, the evaluator, and offline auditing.
5. Every Agent-visible `verified_facts` leaf records its source message ID, call ID, result JSON Pointer, and agent-facing tool name. A later value for the same deterministic fact slot replaces the active view while the immutable event log retains all versions. Canonical tool names are stored only in the Controller/offline provenance sidecar described below.
6. Online `failed_actions` contains only real `tool_call_exception` values or fixture misses visible to the Agent. Trusted evaluator failures are added only to post-episode offline records and never to a later online state.
7. Derive `pending_dependencies` only from unsatisfied machine-readable tool metadata.
8. Populate `available_tools` from the current context's `PipelineAgent.get_available_tools()`. Retain agent-facing names and augmented schemas; expose the canonical mapping only through a Controller-only reference.
9. Compute `state_id` only over the complete Agent-visible state. Remove the `state_id` field, serialize the remaining JSON-compatible payload with Python 3.10 `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`, encode it as UTF-8, and set `state_id` to `"sha256:" + SHA256(bytes).hexdigest()`. Verification removes `state_id`, repeats the same procedure, and compares the stored value. The field being hashed must never contain `state_id` itself, and Controller-only sidecars are never part of the hash.
10. Never put milestones, minefields, target databases, hidden world state, `conversation_active`, or evaluator mappings into online state.

For each Agent-visible state, host code may construct a separate `ControllerProvenanceSidecar` keyed by `state_id` and Agent-visible fact pointer. It contains the canonical tool name and tool-name-mapping manifest hash needed for deterministic Controller checks, statistics, and offline attribution. The Policy, Critic, Revision, retrieval query, embedding input, and user simulator must never receive or serialize this sidecar. The same split is used even when tool names are not scrambled so variants do not take different visibility paths.

## 8. Tool Schemas, Retrieval, and Scrambling

The Policy and Controller receive every tool schema exposed to the Agent in the current scenario; schemas are ordered by BM25 relevance but never truncated. Generate schemas with upstream `convert_to_openai_tools()` under the current augmentation context so that description removal, argument-type removal, and agent-facing name scrambling remain intact.

Policy memory, World memory, and skills use:

```yaml
embedding_provider: openai
embedding_model: text-embedding-3-small
embedding_top_k: 3
embedding_fallback: none
embedding_failure_policy: checkpoint_and_stop
```

For each Agent state:

1. Pin the current generation.
2. Retain only the latest active version of each skill.
3. Apply deterministic prefilters using canonical tool availability, required inputs, and dependencies.
4. Compute one query embedding from the versioned Agent-visible semantic projection
   of the compact state. The projection keeps visible message content, the current
   observation, verified values, completed/failed action semantics, pending
   dependencies, and agent-facing available-tool names. It excludes all
   episode/scenario/state/message/call IDs, provenance-only fields, full tool JSON
   Schemas, canonical names, and Controller sidecars.
5. Independently retrieve the top three Policy memories and top three skills.
6. When the Controller triggers the Critic, compute an action-conditioned query and retrieve the top three World memories.
7. Reuse the Initial Policy retrieval for Revision without retrieving again.

Cache embeddings by provider, exact model name, and query hash. Every embedding
input uses canonical JSON and must contain at most 8,000 UTF-8 bytes; a batch
contains at most 2,048 inputs and 280,000 aggregate UTF-8 bytes. These conservative
pre-dispatch byte limits provide headroom beneath the provider token limits without
local tokenization. Over-limit input fails before dispatch and is never truncated.
Every embedding request must use `text-embedding-3-small`; no alternate embedding
model or lexical retrieval fallback is permitted. If the embedding API fails or
returns an invalid response, write a checkpoint and stop the run. Every retrieval
corpus must come from the currently pinned generation.

Skill `tool_dependencies` use canonical execution-facing tool names. When presenting a skill to the Policy, the Adapter maps only currently visible dependencies to agent-facing names. A skill with an unmappable dependency is not retrievable in that state.

## 9. Skill Records and Controller Metadata

Generation-0 skills may use only names, docstrings, public parameter/return schemas, and manual design principles for tools visible in the train split. They cannot use scenario user prompts, hidden initial databases, milestones, minefields, answers, tool traces, or evaluator labels.

`seed_skills.jsonl` must be generated by a reviewed deterministic project script
from the pinned public ToolSandbox tool-schema inventory; it is not handwritten or
model-generated. The script emits one initial active Skill per actionable public
tool schema, excludes `end_conversation`, and uses only schema information that is
visible under every allowed augmentation variant. Canonical tool identity appears
only in `tool_dependencies`; generated prose must pass the Section 9 canonical-name
and visibility guards. The script records the source schema hash, generator hash,
generation algorithm version, output hash, and an attestation that no scenario,
trajectory, evaluator, dev, or test artifact was opened. Optional manual Policy or
World seeds remain separately attested; they cannot silently alter the generated
Skill Library.

```json
{
  "skill_id": "...",
  "name": "...",
  "description": "...",
  "applicability": {
    "required_state": [],
    "forbidden_state": [],
    "best_used_when": []
  },
  "required_inputs": [],
  "expected_outputs": [],
  "tool_dependencies": [],
  "success_criteria": [],
  "failure_mode_buffer": [],
  "cost_profile": {
    "expected_tool_calls": 1,
    "latency": "low | medium | high",
    "token_cost": "low | medium | high"
  },
  "risk_profile": {
    "risk_if_skipped": "low | medium | high",
    "risk_if_wrong": "low | medium | high"
  },
  "instruction": "...",
  "online_statistics": {
    "evaluated_uses": 0,
    "successes": 0,
    "failures": 0,
    "success_rate": 0.0,
    "last_update_attempt_at_use_count": 0
  },
  "validation": null,
  "version": "v1.0",
  "status": "active | deprecated"
}
```

All state predicates use RFC 6901 JSON Pointers:

```json
{"path": "/verified_facts/key/value", "op": "exists | not_exists | eq | neq | in | contains", "value": "..."}
```

Comparisons use strict JSON type and value equality. `best_used_when` and natural-language instructions guide only the Policy; the Controller cannot interpret them as rules.

`exists/not_exists` omit `value`; `eq/neq/in/contains` require it. For `in`, `value` is an array containing the selected state value. For `contains`, the selected state value is an array containing `value`. Every `required_state/required_inputs` predicate must hold, and no `forbidden_state` predicate may hold. Evaluate `success_criteria` only against later real observations; never treat it as a predicted fact.

`controller_tool_metadata.jsonl` is versioned and manifest-hashed for the pinned ToolSandbox commit:

```json
{
  "canonical_tool_name": "...",
  "effect": "sandbox_read | sandbox_write | external_read | external_write | conversation_control",
  "risk": "low | medium | high",
  "prerequisites": [],
  "parallel_safe": false,
  "read_resources": [],
  "write_resources": [],
  "critic_required": false
}
```

The 34 official tools contain 17 `sandbox_read`, 11 episode-local `sandbox_write`, 5 `external_read`, 0 `external_write`, and 1 User-only `conversation_control` records. Contact, message, reminder, and setting mutations are episode-local `sandbox_write`; RapidAPI tools are `external_read`; `end_conversation` is User-only `conversation_control`. If a later commit introduces an `external_write`, it defaults to `risk: high`, `parallel_safe: false`, and `critic_required: true`, and must be blocked with `UNAUTHORIZED_EXTERNAL_SIDE_EFFECT` unless a separately specified structured authorization contract is implemented. Selecting `official_live` is not such authorization.

Resource templates may use only `{arg:/json/pointer}`. Missing or non-scalar values and unknown metadata cannot establish parallel independence.

## 10. Argument Grounding

The Controller recursively checks every argument leaf. A leaf is grounded only when it:

- is strictly equal to a provenance-bearing value in compact state;
- exactly matches a character span in a visible User message;
- comes from `const`, `enum`, or an explicit default in the current augmented schema;
- is produced deterministically by a versioned allowlisted pure transform from one of those values;
- comes from a validated public field in a previous real tool result.

Every pure transform records its input source, character offsets, transform name, version, and output. When ToolSandbox provides a tool for date parsing, unit conversion, or geographic lookup, the Policy should use that tool rather than hidden scenario time or evaluator targets.

For variants with scrambled argument types or descriptions, the Controller cannot inspect the unaugmented signature to fill values for the Policy. Controller-only canonical metadata may determine risk, dependencies, and resource conflicts but cannot restore parameter semantics intentionally removed by the benchmark.

## 11. Policy Action Contract

The Policy returns exactly one JSON envelope:

```json
{
  "action": {
    "type": "function_call",
    "call_id": "c1",
    "selected_skill_id": null,
    "name": "agent_facing_tool_name",
    "arguments": {}
  }
}
```

```json
{
  "action": {
    "type": "parallel_batch",
    "calls": [
      {
        "call_id": "c1",
        "selected_skill_id": null,
        "name": "agent_facing_tool_name",
        "arguments": {}
      }
    ]
  }
}
```

```json
{"action": {"type": "assistant_message", "content": "..."}}
```

`selected_skill_id` is either `null` or the ID of an active skill retrieved for this state. Each call in a batch binds its own skill. A dependent call must wait until the real prerequisite result appears in the next state.

## 12. Controller and Execution Routing

The Controller is deterministic rule-based code. It cannot generate, modify, or execute an action. It returns:

```json
{
  "blocking_codes": [],
  "critic_trigger_codes": [],
  "evidence": [
    {
      "code": "...",
      "source_kind": "state | schema | skill | tool_metadata | action_history",
      "source_ref": "..."
    }
  ]
}
```

Blocking codes are:

```text
INVALID_FUNCTION
INVALID_PARAMETER
INVALID_ARGUMENT_TYPE
INVALID_ARGUMENT_VALUE
UNGROUNDED_ARGUMENT
MISSING_REQUIRED_ARGUMENT
MISSING_DEPENDENCY
DEPENDENT_PARALLEL_CALLS
UNAUTHORIZED_EXTERNAL_SIDE_EFFECT
CONSTRAINT_VIOLATION
REPEATED_FAILED_ACTION
AGENT_FORBIDDEN_TOOL
```

Critic triggers are:

```text
CRITIC_REQUIRED_TOOL
MEDIUM_OR_HIGH_RISK
PARALLEL_BATCH_REVIEW
ASSISTANT_MESSAGE_REVIEW
STRUCTURED_CONSTRAINT_TENSION
EXTERNAL_READ_REVIEW
```

The native ToolSandbox ExecutionEnvironment executes every ordering of parallel calls to test independence, so execution count grows factorially. The Controller performs an all-or-nothing independence precheck before entering the native environment. If any call is invalid, no call in the batch executes. In `strict_replay`, external reads can only use fixtures so permutation checks cannot create live network requests.

Routing is fixed:

1. If both code lists are empty, execute the original action.
2. Otherwise, call the Critic.
3. Execute the original action only when the blocking list is empty and the Critic returns `accept`.
4. If any blocking code exists or the Critic returns `revise/uncertain`, call Revision exactly once.
5. Revision receives the same state, schemas, Policy memory, and skills.
6. After Revision, repeat only blocking checks. Execute if they pass; otherwise emit a safe failure or clarification. Do not call Critic or Revision again.

The Critic can never override a blocking code.

## 13. Critic Contract

The Critic cannot execute tools, inspect hidden world state, or provide a complete replacement action. It returns exactly:

```json
{
  "verdict": "accept | revise | uncertain",
  "predicted_outcome": "success | failure | uncertain",
  "predicted_effect": "brief hypothetical immediate observable effect",
  "error_codes": [],
  "correction": "brief constraint-level correction"
}
```

The error-code enum is the blocking-code enum from Section 12 plus:

```text
PREMATURE_ACTION
UNNECESSARY_RISK
NO_RELEVANT_TOOL
INSUFFICIENT_CONTEXT
LIKELY_MINEFIELD_BEHAVIOR
```

`predicted_effect` and `correction` are each limited to 40 whitespace-delimited words. `accept` requires an empty error list and correction. `revise` requires at least one code and a non-empty correction. `uncertain` requires `predicted_outcome: uncertain`, at least one code, and a correction describing the unresolved constraint.

The Critic cannot see milestone or minefield definitions. `LIKELY_MINEFIELD_BEHAVIOR` may rely only on visible user intent, schemas, and heuristic memory; it cannot claim that a native minefield was actually matched.

## 14. Online Prompts

### Initial Policy

```text
You are the policy model of an interactive ToolSandbox agent.

Select exactly one executable next action from the current agent-visible state.
Do not predict tool results or produce a multi-step plan. Do not output analysis,
reasoning, confidence, or summaries.

Return exactly one supplied JSON action envelope: one function call, one mutually
independent parallel batch, or one assistant message. Use only the current
agent-facing tool names and augmented schemas. Never infer information removed by a
ToolSandbox scrambling condition.

Treat only visible user messages, actual execution-environment results, and
provenance-bearing verified facts as established. Memories and skills are heuristic.
Ground every argument. Ask for clarification only when no safe available tool can
obtain essential information. Do not repeat an unchanged failed action.

For each call, bind one retrieved active skill that directly guided it, or null.
Never invent a skill ID. Output the envelope and nothing else.
```

### Critic

```text
You are the critic-style world model of an interactive ToolSandbox agent.

Evaluate the proposed next action using only agent-visible state, the current
augmented schemas, Controller evidence, and heuristic World memory. Predict only its
immediate externally observable response. Do not execute tools, inspect hidden
databases or evaluation criteria, output a replacement action, or produce a plan.

Check function, parameter, type, grounding, prerequisites, parallel independence,
risk, and whether an assistant response is premature. Unknown external data alone
does not invalidate a grounded external-read call.

Return only JSON matching the Critic schema and fixed error enum.
```

The Critic receives a prompt-safe projection of Controller evidence: each item
contains only its code, source kind, and a canonical hash of the host-only
`source_ref`. Raw Controller `source_ref` values remain in audit artifacts and
must never enter Critic or Revision prompts because they may contain canonical tool
names under a scrambled Agent-facing mapping.

### Revision

```text
You are revising one proposed action for the same ToolSandbox state.

Reuse the exact state, augmented schemas, Policy memory, and retrieved skills from
Initial Policy. Retrieval must not run again. Controller evidence is deterministic;
Critic feedback is hypothetical and cannot become a fact or ground an argument.

Correct only the identified constraint-level problem. If essential information is
missing, choose one safe information-gathering action; ask one concise clarification
only when no tool can obtain it. Do not expand scope or output a plan. No second
revision is allowed.

Return exactly one JSON action envelope and nothing else.
```

Revision input is one canonical JSON envelope containing the exact Initial Policy
state, Policy memory, and Skill views plus `proposed_action`,
`controller_feedback`, and `critic_feedback` fields. The
`controller_feedback` field uses the same prompt-safe evidence projection; the
raw host-only Controller decision is not serialized to the model. No online role
uses XML or Markdown delimiters for runtime data.

## 15. Hard Validation

Every role output uses JSON Schema generated from the same strict Pydantic v2 models for constrained decoding, followed by local validation. Every model uses `extra="forbid"`, and coercion is forbidden.

- The action union is discriminated by `action.type`.
- A batch contains at least one call; call IDs are non-empty and unique within the batch.
- `arguments` is a JSON object.
- Assistant content is non-empty.
- Every Controller code has evidence and cannot occur in both code lists.
- Tool arguments are validated against the current **augmented agent-facing schema**.
- After conversion, the Adapter validates against the execution-facing callable signature. A failure may block execution but cannot reveal schema information hidden by augmentation to the model.

An unparseable Initial Policy response is not sent to the Critic. If any online or offline output still fails its hard check, write a checkpoint and stop. Do not execute a tool or modify an artifact.

The following values are bootstrap ceilings for implementation tests and the first
train-only token-calibration pass, not immutable formal-run parameters:

```yaml
policy_max_tokens_bootstrap: 256
critic_max_tokens_bootstrap: 384
revision_max_tokens_bootstrap: 256
memory_candidate_max_tokens_bootstrap: 512
memory_review_max_tokens_bootstrap: 256
failure_mode_update_max_tokens_bootstrap: 512
skill_candidate_max_tokens_bootstrap: 2048
```

Before a role may enter a formal train round or evaluation, run its versioned
token-limit calibration using only deterministic train inputs. Observe actual
`completion_tokens`, `finish_reason`, and strict-schema validity. A
`finish_reason: length` authorizes a new calibration request with a larger
candidate ceiling and a new request identity; it is never an in-request retry.
Non-length malformed output is a model/prompt failure and cannot be fixed by
silently increasing the ceiling.

After every calibration input completes without length truncation, define:

```text
observed_max = maximum actual completion_tokens for the role
recommended_max_tokens =
  round_up_to_multiple_of_64(ceil(1.25 * observed_max), minimum=64)
```

Rerun the entire calibration set at the recommended value. Freeze the smallest
tested multiple of 64 that has zero length finishes, complete actual usage, and
100% strict-schema validity, subject to the manifest-pinned server/context hard
ceiling. Failure to find one stops the protocol. The committed calibrated config
records sample IDs and hashes, model/server identity, prompt and schema hashes,
bootstrap and final ceilings, every attempted ceiling, observed maxima,
finish-reason counts, schema-valid counts, latency, and actual tokens.

Formal runs use one frozen role-specific value for every request; they never adapt
per scenario or after seeing dev/test data. A model, server, prompt, output schema,
or calibration-selection change invalidates the config and requires a new
train-only calibration. A length finish during a formal run checkpoints and stops;
it never changes the parameter in place. Calibration time and tokens are setup
evidence and are excluded from round/evaluation headline totals.

## 16. Native Evaluator and Trusted Labels

At the end of each episode, call the native evaluator:

```python
scenario.evaluation.evaluate(
    execution_context=ending_context,
    max_turn_count=scenario.max_messages,
)
```

Do not rewrite milestone matching, minefield matching, guardrails, column similarities, or effective turn count. Trusted results are:

```text
milestone_similarity
minefield_similarity
similarity
turn_count
milestone_mapping
minefield_mapping
```

Preserve the native total-score rule: if `minefield_similarity != 0`, then `similarity = 0`; otherwise it equals `milestone_similarity`. Define `fully_successful := similarity == 1.0`, but keep mean native `similarity` as the primary metric. A custom binary success metric cannot replace it.

Evaluator results enter offline records only after episode completion. The online Policy, Controller, Critic, retrieval query, and user simulator cannot read them.

ToolSandbox has no BFCL-style call-level correctness label. Use conservative task-level skill attribution:

1. At least one actually executed call in the final trajectory explicitly binds the skill.
2. The episode has a trusted native evaluator result.
3. One skill receives at most one `evaluated_use` per episode.
4. Count success only when the episode is `fully_successful`; otherwise count failure.
5. Calls that did not execute, were blocked by the Controller, or lack an evaluator result do not update skill statistics.

Failure modes may cite only real exceptions, fixture misses, non-perfect milestones, minefield hits, or visible final state. Critic predictions are not evidence.

## 17. Online and Offline Rounds

The three-round lifecycle is:

```text
Run pinned generation G online on train shard G
→ current-round trajectory buffer
→ native evaluation
→ offline updates using only the current buffer
→ atomic publication of generation G+1
→ archive logs
→ clear the working buffer
```

Each scenario starts from a deep copy of its starting ExecutionContext recorded in the manifest and cannot inherit world state from another scenario. Scenario execution order is fixed. Even when execution is parallel, offline evidence is processed in manifest order.

## 18. Memory Updates

Generation 0 requires:

```text
seed_policy_memory.jsonl
seed_world_memory.jsonl
seed_skills.jsonl
seed_source_manifest.json
```

The source manifest must attest that no dev/test user message, starting database, milestone, minefield, trajectory, or evaluator label was used. Missing files or failed attestation stop the run.

```json
{
  "seed_files": [
    {
      "path": "seed_policy_memory.jsonl",
      "sha256": "...",
      "sources": [{"type": "toolsandbox_public_tool_schema | manual_design", "identifier": "..."}]
    }
  ],
  "dev_test_artifacts_used": false,
  "attested_by": "..."
}
```

All four seed files must appear exactly once. Paths are workspace-relative, sources are non-empty, and actual hashes match the manifest.

Policy memory has this shape:

```yaml
memory_id:
scope:
applicability:
action_guidance:
avoid:
evidence_trajectory_ids:
support_count:
success_rate:
confidence:
created_version:
status:
```

World memory has this shape:

```yaml
memory_id:
action_pattern:
state_conditions:
schema_conditions:
likely_error_codes:
outcome_calibration:
correction_principle:
evidence_trajectory_ids:
support_count:
empirical_failure_rate:
confidence:
created_version:
status:
```

Policy memory contains reusable action-selection, grounding, parallel-applicability, and failure-avoidance guidance. World memory contains critic calibration, common immediate errors, and evidence about whether Revision worked; it cannot supply a complete replacement action.

- The Policy updater processes at most the 50 most recent evaluated train trajectories from the current round.
- The World updater processes only current-round train trajectories where the Critic was called.
- Each trajectory may produce at most one candidate per role, or `NONE`.
- Compare each candidate against the top three similar memories from the matching store; the reviewer returns only `ADD`, `MERGE`, or `SKIP`.
- `MERGE` adds evidence and host-owned statistics without rewriting semantic guidance.
- The first implementation does not automatically delete memories.

Updater output is restricted to:

```json
{"result": "NONE"}
```

```json
{
  "result": "CANDIDATE",
  "role": "policy",
  "candidate": {
    "scope": "...",
    "applicability": [],
    "action_guidance": "...",
    "avoid": []
  }
}
```

```json
{
  "result": "CANDIDATE",
  "role": "world",
  "candidate": {
    "action_pattern": "...",
    "state_conditions": [],
    "schema_conditions": [],
    "likely_error_codes": [],
    "outcome_calibration": "...",
    "correction_principle": "..."
  }
}
```

Natural-language fields contain 1–512 characters. Lists contain at most five unique items. Predicates and error codes use the enums in this specification. Model output cannot contain memory IDs, evidence IDs, statistics, confidence, generation, or status.

Reviewer output is a strict discriminated union:

```json
{"decision": "ADD", "reason": "..."}
```

```json
{"decision": "MERGE", "target_memory_id": "...", "reason": "..."}
```

```json
{"decision": "SKIP", "reason": "..."}
```

Only `MERGE` may contain `target_memory_id`, which must identify one of the supplied top-three memories. `reason` contains 1–240 characters.

Updater prompt contract:

```text
Read exactly one eligible completed train trajectory and its trusted native
ToolSandbox evaluator result. Return at most one short reusable role-appropriate
memory candidate, or NONE. Never preserve concrete scenario answers, temporary
entities, hidden evaluator content, critic predictions as facts, or unsupported
hypotheses. Return only valid JSON matching the supplied schema.
```

Reviewer prompt contract:

```text
Return ADD only for a non-duplicate, reusable candidate grounded in trusted outcomes;
MERGE when one supplied memory already expresses the same rule; otherwise SKIP. Do
not rewrite existing guidance or output an action. Return only schema-valid JSON.
```

For `ADD`, host code assigns `pm_<sha256>` or `wm_<sha256>` from canonical candidate content, sets `support_count=1`, and computes `confidence = support_count / (support_count + 2)`. For `MERGE`, it rejects duplicate evidence IDs and updates rates from trusted task labels. Models cannot assign IDs, evidence, generation, status, rates, or confidence.

The Policy binary label is `fully_successful`. A World binary failure label means that the draft action has a directly verifiable real exception, a selected related milestone with less than perfect similarity, or a minefield hit. If a trusted failure label cannot be attributed to the draft action, the World updater must return `NONE`. For `MERGE`, update the rate as `(old_rate * old_support_count + binary_label) / new_support_count`. Seed records must satisfy `rate * support_count` being an integer within floating-point tolerance. Hash collisions, unknown targets, repeated evidence, or invalid conditional fields fail hard validation.

## 19. Failure Modes and Skill Rewrites

Process each actual failure in manifest order. The Skill Updater compares it with the skill's currently staged buffer and returns `ADD`, `MERGE`, or `SKIP`:

- each skill stores at most five modes;
- `ADD` creates a stable `mode_id`;
- `MERGE` increments support and refreshes a host-owned global sequence number without rewriting text;
- when over capacity, retain higher-support modes; ties retain the more recent `last_observed_seq`;
- the staged buffer becomes visible only in the next generation.

Failure-mode output is a strict discriminated union:

```json
{"decision": "ADD", "task_condition": "...", "failure_mode": "..."}
```

```json
{"decision": "MERGE", "mode_id": "..."}
```

```json
{"decision": "SKIP", "reason": "..."}
```

Only `MERGE` may contain `mode_id`, which must identify a supplied mode. `ADD` text fields contain 1–240 characters. Stable IDs, support counts, and observation sequence numbers are assigned by host code.

```text
Given one actual native-evaluated train failure attributed to this skill and its
current failure-mode buffer, return ADD, MERGE, or SKIP. Add only concise reusable
patterns, merge only semantic equivalents, and skip case-specific or unsupported
observations. Never use critic predictions or hidden evaluator definitions as
evidence. Return only schema-valid JSON.
```

Attempt a rewrite only when all conditions hold:

```text
evaluated_uses >= 10
failures / evaluated_uses > 0.25
evaluated_uses > last_update_attempt_at_use_count
```

The updater receives only the current skill, relevant public canonical tool schemas, accumulated statistics and failure modes, and relevant current-round train trajectories. It cannot read Policy or World memory, add/split/merge skills, or modify another skill. It must preserve `skill_id`.

A rewrite returns only `{"candidate": <SkillContent>}`. `SkillContent` is the Section 9 record without `failure_mode_buffer`, `online_statistics`, `validation`, `version`, or `status`; its `skill_id` must match the triggered skill. Host code assigns `candidate_id` and initializes the candidate's buffer and statistics to empty and zero.

```text
Rewrite exactly one triggered skill from its trusted statistics, accumulated failure
modes, relevant current-round train trajectories, and public ToolSandbox tool schemas.
Correct reusable failure patterns without encoding scenario-specific answers or hidden
evaluation content. Keep the same skill_id; do not create, split, merge, or modify
another skill. Return only the semantic SkillContent candidate JSON.
```

### Dev Mini-Bench

Determine relevant dev scenarios by intersecting the canonical necessary tools of their nominal no-distraction variant, excluding `end_conversation`, with the current skill's `tool_dependencies`. This evaluation metadata is available only to the offline selector and never enters a model prompt.

Sort expanded dev scenarios by the following key and take at most 20:

```text
(SHA256(data_seed + NUL + skill_id + NUL + scenario_id), UTF-8 scenario_id)
```

The A/B branches pin the user simulator, world clock, fixtures, all other skills, staged memories, prompts, models, decoding, schemas, and scenario order. The evaluated skill version is the only difference.

Accept a candidate only when:

1. its number of `fully_successful` scenarios is higher; or
2. the full-success count is tied, its sum of native `similarity` is higher, and the number of minefield-hit scenarios does not increase.

Reject exact ties and improvements in turn count alone. An accepted candidate increments the patch version `v1.n` and deprecates the old version. A rejection does not consume a version number, and the skill cannot retry until it receives a new evaluated use.

Accepted versions store:

```json
{
  "validation": {
    "scenario_count": 20,
    "previous_full_success_count": 0,
    "candidate_full_success_count": 0,
    "previous_similarity_sum": 0.0,
    "candidate_similarity_sum": 0.0,
    "previous_minefield_hit_count": 0,
    "candidate_minefield_hit_count": 0
  }
}
```

When multiple skills trigger in one round, process them in sorted `skill_id` order. Later comparisons include earlier accepted staged updates in both A/B branches.

## 20. ToolSandbox Time, Network, and Parallel Semantics

### World Time

The controlled clock must cover:

- initial message and reminder timestamps in `base_scenarios.py`;
- current-year values used by scenarios;
- `get_current_timestamp` and message/reminder creation timestamps;
- date-canonicalization helpers.

Do not hard-code dates by modifying upstream scenario source. The compatibility layer injects the clock before scenario construction. The manifest records epoch, timezone, and clock-adapter version.

### RapidAPI

These tools are external reads:

```text
convert_currency
search_lat_lon
search_location_around_lat_lon
search_stock
search_weather_around_lat_lon
```

The fixture key is the SHA-256 of canonical JSON containing `{tool_name, arguments, backend_version}`. The value records status code, normalized response body, capture time, source, and body hash. Never store secrets or headers. A replay miss cannot access the network.

### User Simulator

The user simulator is part of the environment and cannot read pipeline memories, skills, Critic output, or evaluator state. Real-data train calibration, Dev Mini-Benches, and `official_live` use the fixed snapshot `gpt-4o-mini-2024-07-18` through the official OpenAI Chat Completions endpoint. This deliberately changes only the upstream concrete model selection; the upstream User-role message conversion, tool schema, stop behavior, and omitted `temperature`, `top_p`, and `seed` contract remain unchanged. The User Simulator shares `OPENAI_API_KEY` with `text-embedding-3-small`, while host code keeps their clients or client roles, request IDs, and usage separate. Its model ID, prompt/few-shot hash, tool schema, deliberately omitted decoding fields, and stop condition are pinned in the manifest. All three evaluated systems use the same configuration. An unavailable or rejected exact snapshot stops the run, with no alias, GPT-4o, or other-model fallback. Reports must identify this as a deviation from the upstream ToolSandbox default. A remote OpenAI User Simulator is not valid for trajectory-level `strict_replay`, which requires a separately selected frozen local checkpoint and deterministic decoding.

### Parallelism

Scenario-level parallelism and tool-call parallelism are distinct. Scenario-level parallelism may affect throughput only; it cannot change seeds, serialized artifact order, or aggregation. Tool-call parallelism retains ToolSandbox's native rule that all call orderings must succeed. The Controller's independence check is an early rejection layer, not a replacement for the native check.

## 21. Transactions, Checkpoints, and Resume

Generation layout is:

```text
artifacts/generations/g000/
  manifest.json
  policy_memory.jsonl
  world_memory.jsonl
  skills.jsonl
  retrieval_indexes/
```

Write every offline change to staging. Publish a generation atomically only after all schema checks, hashes, and mini-benches complete. Online readers use only the last complete generation.

Checkpoints include at least:

```yaml
run_id:
profile:
phase:
round_index:
shard_id:
family_id:
scenario_id:
state_id:
completed_scenario_ids:
serialized_execution_context:
pinned_generation_ids:
retrieval_results:
controller_and_critic_state:
trajectory_buffer:
staged_memory_updates:
staged_skill_updates:
skill_statistics:
failure_mode_buffers:
config_prompt_fixture_hashes:
random_states:
metrics_accumulators:
llm_request_ledger:
```

Write checkpoints atomically after every real Agent-visible message; before and after each tool action; before and after each LLM request; on timeout or hard-check failure; after every offline unit; and before and after generation publication.

Each logical LLM request has a stable `logical_request_id` derived from its role, phase, state ID, canonical input hash, model, and decoding configuration. Before dispatch, atomically record the logical request and its input fingerprint as `prepared`. Each physical dispatch has a new `attempt_id`. After a valid response arrives, atomically store the raw response, response hash, usage, and `completed` status before applying the response to pipeline state. Recovery follows these rules:

- if a completed response exists, reuse it without another physical dispatch;
- if only `prepared` or `in_flight` state exists, the outcome is unknown and the same logical request may be dispatched again with a new `attempt_id` and `replayed_after_unknown_outcome: true`;
- every physical attempt, including an attempt with unknown outcome, remains separately recorded for actual usage accounting; missing usage makes the containing `total_tokens` aggregate incomplete;
- the response and all downstream state transitions are committed at most once per `logical_request_id`;
- reusing a `logical_request_id` with a different input fingerprint is a hard error.

ToolSandbox writes mutate only the in-memory ExecutionContext. Before an action, store the pre-action context; after its result, store the complete committed context. Recovery follows these rules:

- if a committed context exists, reuse it and never re-execute;
- if only a pre-action context exists, local sandbox reads/writes may be replayed from it;
- external reads replay only through fixtures;
- in `official_live`, an external read with an unknown outcome is not retried automatically and requires safe reconciliation;
- reusing a call ID with a different canonical action fingerprint is a hard error.

Completed scenario effects, committed tool effects, applied logical LLM responses, and offline updates must never be committed twice. A physical LLM attempt may repeat only after an unknown outcome under the rules above.

## 22. Logging, Usage, and Effective Qwen Output Cost

Every state replay record includes:

- run, round, family, scenario, state, and message IDs;
- agent-visible compact state and augmented schemas;
- hash of the agent-facing/canonical tool-name mapping;
- generation and retrieved IDs, versions, and scores;
- Initial Policy, Controller, Critic, Revision, and final action;
- actual ToolSandbox messages, tool exception/result, and context hash;
- the complete native evaluator result after episode completion;
- profile, world clock, fixture hit/miss, logical request IDs, attempt IDs, and revision count.

Hidden scenario/evaluator content may appear only in access-restricted evaluator audit artifacts and cannot be copied into online replay prompt artifacts.

Assign one stable `logical_request_id` to each logical LLM request and a unique `attempt_id` to every real physical attempt. Record both IDs, the canonical input fingerprint, replay-after-unknown-outcome flag, role, phase, model, UTC timestamps, monotonic latency, actual input/output/cache tokens, whether the response was durably applied, and `completed/timeout/failed/unknown_outcome` status. Token estimates cannot be reported as actual usage. When actual usage is missing, affected token fields are `null`, and all containing token aggregates set `usage_complete: false`.

The usage adapter enforces:

```text
input_tokens = uncached_input_tokens + cache_read_input_tokens + cache_write_input_tokens
total_tokens = input_tokens + output_tokens
```

`total_cost` is not money. Its fixed unit is
`qwen_effective_output_tokens`: sum actual `output_tokens` only for strictly
validated Qwen responses linked to a durable committed substantive-effect record.
A logical-response `applied` marker or audit/checkpoint write alone is not cost
eligibility.

For an online turn, the substantive effect is the final agent action committed to
the episode. Count each Qwen Policy/Critic/Revision output actually consumed by
that committed decision chain, including an Initial Policy later superseded by
Revision. For an offline update, the substantive effect must be a persistent
semantic Policy/World/Skill or failure-mode mutation published or staged for the
next generation. Count the candidate/reviewer/validation decision chain only when
that mutation commits. `NONE`, `SKIP`, duplicate/no-op decisions, and rejected
Skill candidates or Mini-Bench branches that produce no accepted Skill mutation
contribute zero to `total_cost`, although every physical attempt remains in
`total_tokens`. A metrics/checkpoint/audit record is not itself a substantive
effect.

Exclude Qwen input/cache tokens, invalid/truncated/unapplied/unknown outputs,
unused retry attempts, embeddings, the User Simulator, tools, evaluators, and host
computation. Deduplicate each contributing logical response by its single source
`attempt_id` and require a committed effect-to-application link. If a contributing
Qwen output lacks actual output usage, set `total_cost: null` and
`cost_complete: false`; otherwise report the integer token count and
`cost_complete: true`. No pricing manifest, currency, provider price, or monetary
conversion is used.

`task_metrics.jsonl` records wall time from immediately before the first state until after evaluator output and the final checkpoint, plus actual tokens, effective Qwen output cost, and Policy/Critic/Revision/User-simulator call counts. `round_metrics.jsonl` separately records online, offline, publication, and direct total round time and request/task aggregates. `request_metrics.jsonl`, `task_metrics.jsonl`, `round_metrics.jsonl`, and `run_metrics.json` are immutable artifacts. Actual usage aggregates by unique physical `attempt_id`, including replay attempts; effective Qwen output cost deduplicates through committed substantive-effect links, while logical call counts deduplicate by `logical_request_id`. Offline requests use `task_id: null`.

Every training round and every Vanilla, Generation-0, or Updated evaluation run
has three mandatory headline metrics:

```text
total_running_time_seconds
total_tokens
total_cost
```

`total_cost` always carries
`cost_unit: "qwen_effective_output_tokens"` and the independent
`cost_complete` flag defined above.

For a training round, `total_running_time_seconds` is monotonic wall-clock elapsed time from immediately before its first online scenario dispatch through evaluator completion, all offline updates, generation publication, and the durable final round checkpoint. For an evaluation run, it is monotonic wall-clock elapsed time from immediately before the first scenario dispatch through the last evaluator result and durable final run checkpoint. It includes model and tool waits, retries, checkpointing, and orchestration overhead, but excludes operator setup and explicit preflight commands. It is measured directly from start and end timestamps; it is not the sum of per-request or per-task latency, particularly when work overlaps.

The three training rounds must each emit a separate immutable headline row, even
when one later fails:

```text
round_index
input_generation_id
train_shard_id
published_generation_id | null
started_at_utc
ended_at_utc
total_running_time_seconds
total_tokens
usage_complete
total_cost
cost_unit
cost_complete
completion_status
```

`round_index` 0, 1, and 2 may never be combined into one average or replaced by
run-level totals. A run summary may additionally sum completed round costs and
report end-to-end training time, but it must preserve all three direct per-round
wall-clock values and their independent completeness/status fields.

`total_tokens` is the sum of actual `input_tokens + output_tokens` across every physical model attempt within that boundary, including Qwen pipeline roles, `text-embedding-3-small`, the `gpt-4o-mini-2024-07-18` User Simulator, offline generation/review calls, retries, and replay-after-unknown-outcome attempts when usage is returned. Embedding requests use their provider-reported input/total usage with zero output tokens when that is the provider contract. The artifacts also report input, output, cache-read, and cache-write token counts broken down by role, model, phase, and provider.

Each aggregate contains `usage_complete`. If any physical attempt lacks actual provider or server usage, `usage_complete: false`, the affected counts and `total_tokens` are `null`, and any estimate is stored under an explicitly named diagnostic field that cannot appear as an experimental result. Wall-clock timing remains reportable even when token usage is incomplete.

User-simulator usage is recorded separately as `role: user_simulator`. It is excluded from Policy/Critic/Revision call counts and `total_cost`, but included in `total_tokens`.

Logs are immutable. API keys, RapidAPI keys, authorization headers, and other secrets must never be logged.

## 23. Three-System Evaluation and Reporting

Evaluate these systems on the same test manifest:

1. **Vanilla**: frozen Qwen3-32B as the direct ToolSandbox Agent, without memories, skills, Controller, Critic, or offline updates.
2. **Generation-0**: the complete online pipeline with manual seed artifacts and no offline update.
3. **Updated**: the final generation after three train/offline rounds.

The run also publishes a query-only checkpoint observation registry containing
G000-G003 identities and the train-round metrics associated with producing each
new generation. It may expose deterministic diagnostic pointers such as highest
observed train mean similarity and highest observed train fully-successful rate,
with explicit different-shard/non-comparable labels. These pointers are for later
human lookup only: they cannot select the formal Updated system, which remains
G003, and cannot consume dev/test results.

All three systems share the ToolSandbox commit, test scenario order, starting contexts, tool augmentations, world clock, fixture store, user simulator, Agent checkpoint, tokenizer, maximum message count, and native evaluator.

The primary metric is mean native `similarity` on the test split.

Also report:

- `milestone_similarity`, `minefield_similarity`, and `fully_successful` rate;
- macro and micro averages for every native `ScenarioCategories` value;
- mean and median effective turn count;
- Critic trigger/accept/revise/uncertain rates;
- Revision rate and post-revision episode score;
- fixture hit/miss and external-tool exception rates;
- `total_running_time_seconds`, `total_tokens`, and `total_cost` for each individual training round and for each Vanilla, Generation-0, and Updated evaluation run;
- per-request, per-task, per-phase, per-role, per-model, per-round, and per-run latency, input/output/cache tokens, call counts, and effective Qwen output cost;
- `usage_complete`, with incomplete token aggregates reported as unavailable rather than estimated;
- reproducibility profile, every manifest hash, and failure/timeout counts.

The post-evaluation analysis must also identify test cases deterministically
related to failure modes learned during the three training rounds. Training
creates an immutable failure-mode lineage record from each trusted generalized
failure signature to its committed `mode_id` and, when applicable, the accepted
evolved Skill version caused by that mode. After all three systems finish, an
offline analyzer may match only sanitized trusted trajectory features: Skill ID,
public canonical tool dependency, evidence kind, and sanitized exception/
controller outcome class. It must not use a model, prompt text, raw tool data,
hidden evaluator definitions, or a post-hoc semantic judgment.

The primary evolution repair is a paired test case that Generation-0 did not
fully solve, Updated fully solved, has a deterministic failure-mode lineage match,
and used the linked evolved Skill in the Updated committed decision chain. Report
the related-case count, repaired-case count/rate, unmatched count, ambiguous-match
count, overall Updated-minus-Generation-0 native-similarity point difference, and
the same paired difference on the related subset. Also report Vanilla comparisons
as secondary context. These are mechanism-linked observational attributions, not
causal proof; incomplete or ambiguous evidence is never counted as repaired.
Because ToolSandbox's eight variants within one `scenario_family_id` measure
stability under perturbation, also report per system the all-eight-variants-success
family count/rate, family-minimum native similarity, and within-family similarity
range. For related families, report how many failing variants were repaired and
whether every related failing variant was repaired. Never treat variants as
independent samples for uncertainty or relabel an unrelated score change as a
failure-mode repair.

Do not treat the eight augmentation variants as independent families when computing confidence intervals. Statistical tests and bootstrap sampling use `scenario_family_id` as the cluster.

## 24. Implementation Acceptance and Reproduction Validation

Before a formal Qwen run, all of the following must pass:

1. All original upstream evaluator tests pass, and the compatibility layer does not change evaluator source or hashes.
2. Two fixed-clock builds produce identical manifests and starting-context hashes for all 129 families and 1,032 scenarios.
3. The family split is 79/25/25, and no family crosses a split or training shard.
4. Tests fail if an online component attempts to read a hidden database, milestone, minefield, or target DataFrame.
5. The 34 tools and all agent-facing augmentation schemas match the pinned-commit manifest.
6. In scrambled variants, the Policy and Critic cannot see canonical names or removed descriptions/types.
7. Rebuilding the same state payload twice produces the same `state_id`, and removing `state_id` and recomputing the digest verifies every stored state. Replaying the same verified state/configuration twice produces the same Policy JSON. If the underlying vLLM cannot guarantee this, mark the run non-bitwise-reproducible and store raw response hashes.
8. In `strict_replay`, every external-tool network request without a matching fixture is blocked. The configured OpenAI embedding endpoint is permitted only for `text-embedding-3-small`; no embedding or retrieval fallback is permitted.
9. Dependent parallel batches are blocked before execution; independent batches still pass the native permutation check.
10. Evaluator labels are inaccessible before episode completion, and dev/test artifacts never enter update inputs.
11. Kill-and-resume tests cover points before/after LLM calls, before/after tool calls, during offline updates, and during generation publication. They must show at-most-once logical response application, tool-effect commitment, offline update commitment, and substantive-effect cost linkage. An injected unknown LLM outcome may cause a new physical attempt with the same `logical_request_id` and a distinct `attempt_id`; every attempt is retained in actual usage accounting, while only Qwen responses linked to one committed substantive effect may contribute output tokens to `total_cost`. Applied `NONE`/`SKIP` and rejected/no-op offline chains contribute zero.
12. Vanilla, Generation-0, and Updated use identical test starting-state, environment-configuration, and evaluator hashes.
13. Two `strict_replay` smoke runs have identical final-context hashes, native scores, retrieval IDs, and request outputs.
14. Execution-environment parallel tests pass in the pinned Linux container. Current Windows response ordering is not an acceptable substitute for formal validation.

If any item fails, the conclusion must be "partially reproducible" and list the failed items.

## 25. Directory Boundaries

```text
toolsandbox/
  pipeline.md
  AGENTS.md
  pyproject.toml
  uv.lock
  configs/
  prompts/
  docs/
  tasks/
  src/toolsandbox_pipeline/
    online/
    offline/
    memory/
    skills/
    retrieval/
    toolsandbox_adapter/
    reproducibility/
    checkpointing/
    schemas/
  tests/
  artifacts/
  runs/
```

- `src/toolsandbox_pipeline/online/`: State Builder, Policy, Controller, Critic, Revision, and routing.
- `src/toolsandbox_pipeline/offline/`: memory and skill update orchestration using only the current round.
- `src/toolsandbox_pipeline/toolsandbox_adapter/`: `PipelineAgent`, action-to-Message conversion, agent/execution tool-name mapping, and the native evaluator adapter.
- `src/toolsandbox_pipeline/reproducibility/`: scenario-family manifest, split logic, clock, fixture replay, seeds, and environment hashes.
- `src/toolsandbox_pipeline/memory/`, `skills/`, and `retrieval/`: generation-scoped stores, statistics, and indexes.
- `src/toolsandbox_pipeline/checkpointing/`: atomic checkpoints, ExecutionContext serialization, and idempotent recovery.
- `src/toolsandbox_pipeline/schemas/`: Pydantic v2 contracts and JSON Schemas.
- `configs/` and `prompts/` contain committed, versioned experiment inputs.
- `artifacts/` and `runs/` contain generated outputs and are excluded from Git except for explicit documentation placeholders.

The upstream ToolSandbox Git dependency remains pinned and read-only; it is not a submodule or copied source tree. No module may bypass the Agent-visibility boundary, hard validation, single-revision limit, family-level split, generation pinning, native evaluator, or transactional publication.
