# Task 003: Compact Verified State and Deterministic Reducer

Status: `approved`

## Objective

Implement the strict Compact Verified State schema and a deterministic event reducer that turns already-filtered Agent-visible messages, current agent-facing tool schemas, and committed tool outcomes into one immutable online state plus a separate Controller/offline provenance sidecar.

The State Builder is a trust-boundary component. It must preserve visible evidence without summarization and make it structurally impossible for hidden ToolSandbox state or canonical tool-name mappings to enter Policy, Critic, Revision, retrieval, or embedding inputs.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3, 4, 7-10, 15, 16, 21, and 24;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `tasks/002_core_contracts.md`;
5. this task file.

## Access

```text
Access class: offline
Setup network: none
Data splits: none
Secrets: none
```

Use literal test events and public-schema fixtures only. Do not instantiate an upstream scenario or ExecutionContext and do not access any model, embedding, dataset, tool implementation, credential, or network service.

## Preconditions

- Tasks 001 and 002 are complete and accepted.
- The public canonicalization helpers from Task 002 are unchanged.
- The later ToolSandbox Adapter will supply messages only after upstream Agent-role visibility filtering. This task defines and tests the sanitized input boundary; it does not implement that upstream integration.

## Owned Files

The assigned Agent may create or edit only:

```text
src/toolsandbox_pipeline/schemas/state.py
src/toolsandbox_pipeline/online/__init__.py
src/toolsandbox_pipeline/online/state_builder.py
tests/schemas/test_state.py
tests/online/test_state_builder.py
tests/online/test_state_visibility.py
```

Do not edit Task 002 contracts, ToolSandbox Adapter code, tool metadata files, retrieval, Controller, prompts, checkpoints, dependency files, or generated artifacts.

## Input Trust Boundary

The reducer accepts project-owned strict input models rather than a raw upstream `ExecutionContext` or database:

```python
VisibleMessageInput
CommittedToolOutcomeInput
AgentFacingToolInput
PendingDependencyInput
StateBuildInput
```

### Visible messages

`VisibleMessageInput` contains only:

```text
source_message_index
sender
recipient
content
openai_tool_call_id
openai_function_name
tool_call_exception
```

Requirements:

- `source_message_index` is the original non-negative upstream sandbox message index and is used only to establish order and stable identity.
- Input indices are unique and strictly increasing.
- Sender and recipient use exactly `SYSTEM`, `USER`, `AGENT`, and `EXECUTION_ENVIRONMENT`.
- `content` is preserved byte-for-byte as a Python string; do not trim, normalize, summarize, or reinterpret it.
- `openai_function_name`, when present, is the agent-facing name visible in that scenario.
- The model deliberately has no `visible_to`, `conversation_active`, `tool_trace`, database snapshot, evaluator, milestone, minefield, target, or canonical-name field. Strict validation rejects such fields instead of silently dropping them.
- The upstream Adapter is responsible for applying `BaseRole.filter_messages()` for `RoleType.AGENT` before constructing this input. Task 003 must not reimplement visibility from unfiltered messages.

### Tool outcomes and schemas

`CommittedToolOutcomeInput` binds one visible `EXECUTION_ENVIRONMENT -> AGENT` result message to:

```text
call_id
agent_facing_tool_name
arguments
result_source_message_index
public_return_contract_id
canonical_tool_name
tool_mapping_manifest_hash
```

The last two fields are Controller/offline-only source data. They may enter only the provenance sidecar, never the Agent-visible state. Every outcome must match a visible result message by source index, call ID, and agent-facing function name. A mismatch is a hard error.

`AgentFacingToolInput` contains an agent-facing name and its exact augmented OpenAI-compatible schema. It has no canonical name or unaugmented callable signature. Preserve every supplied schema and its input order; do not truncate.

`PendingDependencyInput` is machine-readable and provenance-bearing. It identifies the agent-facing tool/call, the unsatisfied prerequisite code, and a Controller-only metadata record reference. The Agent-visible state retains only the agent-facing dependency description and opaque record hash; canonical metadata remains in the sidecar. Free-form inferred dependencies are forbidden.

## Public Output Schemas

`schemas/state.py` provides and exports at least:

```python
VisibleRole
VisibleMessage
CurrentObservation
VerifiedFact
CompletedToolCall
FailedAction
PendingDependency
AgentFacingTool
CompactVerifiedState
ControllerFactProvenance
ControllerProvenanceSidecar
StateBuildResult
```

All are `StrictModel` subclasses where applicable.

### CompactVerifiedState

Its wire fields exactly match Section 7:

```text
episode_id
scenario_id
scenario_family_id
state_id
agent_turn_index
visible_messages
current_observation
verified_facts
completed_tool_calls
failed_actions
pending_dependencies
available_tools
conversation_status
```

Requirements:

- IDs are non-empty strings; `agent_turn_index` is a non-negative integer.
- `conversation_status` is exactly `agent_turn` in this task.
- `visible_messages` contains only stable message ID, sender, recipient, and verbatim content.
- `current_observation` contains the message ID and content of the last visible message addressed to `AGENT`; absence of such a message is a hard error.
- `available_tools` contains only agent-facing names and exact augmented schemas.
- The model has no extension dictionary or generic metadata escape hatch.
- `state_id` is computed after every other field using Task 002's `compute_state_id`; caller-supplied state IDs are not trusted.

### Stable message IDs

The reducer deterministically hashes this exact payload for each visible message:

```json
{
  "episode_id": "...",
  "source_message_index": 0,
  "sender": "USER",
  "recipient": "AGENT",
  "content": "...",
  "openai_tool_call_id": null,
  "openai_function_name": null,
  "tool_call_exception": null
}
```

The ID is `"message:"` followed by the hexadecimal portion of Task 002's canonical SHA-256 result. Reusing one `(episode_id, source_message_index)` with different content or metadata in the same build is a hard error.

### Agent turn index

`agent_turn_index` is zero-based and equals the number of completed contiguous runs of visible messages whose sender is `AGENT` before `current_observation`. Consecutive Agent tool-call messages from one parallel batch therefore count as one Agent turn.

### Completed calls, failures, and fact promotion

- A committed outcome with a null `tool_call_exception` becomes one `completed_tool_calls` entry.
- A visible outcome with a non-null `tool_call_exception`, or a deterministic `EXTERNAL_FIXTURE_MISS`, becomes one `failed_actions` entry and contributes no verified fact.
- Failure text must already be visible in the result message. Do not supplement it from traceback objects or `tool_trace`.
- Apply `ast.literal_eval` only to the visible result `content` of a successful committed outcome.
- Validate the parsed object strictly against the registered public return contract selected by `public_return_contract_id`.
- If parsing or validation fails, keep the original visible string, mark the completed call result as unstructured, and promote no facts. Do not fall back to JSON guessing, `eval`, internal trace data, or callable inspection.
- If validation succeeds, convert the validated public result to JSON-compatible data and enumerate its scalar leaves in deterministic RFC 6901 pointer order.
- A fact slot is determined only by the agent-facing tool name, canonical hash of the visible call arguments, and result JSON Pointer. The slot ID must not include or hash the canonical tool name.
- A later successful outcome for the same slot replaces the active `verified_facts` entry. Emit all versions in deterministic `fact_events` within the non-prompt `StateBuildResult` for later immutable audit logging.
- Each visible `VerifiedFact` contains value, source message ID, call ID, result JSON Pointer, and agent-facing tool name. It never contains a canonical name or mapping.

Public return contracts are supplied as an explicit registry of already-constructed strict validators. Missing contract IDs are hard errors; the reducer must not import or inspect the upstream callable to infer a return type.

## Controller/Offline Provenance Sidecar

`ControllerProvenanceSidecar` is bound to the finished `state_id`. For every promoted fact it records:

```text
fact_pointer
call_id
canonical_tool_name
tool_mapping_manifest_hash
```

It may also retain canonical references required by machine-readable pending dependencies. Requirements:

- The sidecar is returned separately from `CompactVerifiedState`.
- Neither `CompactVerifiedState.model_dump()` nor its JSON Schema contains a sidecar, canonical tool name, mapping, hidden database field, or evaluator field.
- Sidecar values do not participate in `state_id`.
- Policy/Critic/Revision/retrieval-facing serialization helpers accept only `CompactVerifiedState`, not `StateBuildResult` or the sidecar.
- Use the same split when agent-facing and canonical names are identical.

## Determinism and Mutation Rules

- The same validated input produces byte-identical state JSON, `state_id`, sidecar, and fact-event order.
- Caller-owned lists, dictionaries, schemas, and message inputs are never mutated.
- Output ordering follows input message/tool order except fact leaves, which use deterministic RFC 6901 pointer order.
- No wall clock, randomness, process ID, object identity, filesystem state, locale, or environment variable may affect output.
- The reducer performs no I/O, logging, model calls, tool calls, or checkpoint writes.

## Required Tests

Tests must cover at least:

1. deterministic reduction of USER, AGENT, and EXECUTION_ENVIRONMENT message sequences;
2. stable IDs and identical states across repeated builds and fresh equivalent input objects;
3. distinct IDs after any visible message or source-index change;
4. zero-based turn counting and one-turn treatment of consecutive parallel-call messages;
5. current-observation selection and failure when no visible message addresses the Agent;
6. successful strict result parsing and deterministic scalar-leaf promotion;
7. later replacement of an active fact slot while preserving both fact events;
8. parse failure, public-contract failure, visible tool exception, and fixture miss behavior;
9. rejection of mismatched result index/call ID/agent-facing name and missing return-contract IDs;
10. preservation of every augmented agent-facing schema without truncation;
11. rejection of `tool_trace`, `conversation_active`, `visible_to`, hidden databases, evaluator fields, milestone/minefield fields, target data, canonical names in Agent-visible input models, and unknown fields generally;
12. proof that canonical names and mapping hashes appear only in the sidecar and do not affect state bytes or `state_id`;
13. no mutation of caller input;
14. Unicode content, null/boolean/numeric/list/object results, RFC 6901 escaping, and non-finite-value rejection;
15. focused JSON Schema assertions showing `additionalProperties: false` and absence of forbidden visibility fields.

Use synthetic names and values. Do not inspect or encode actual scenario prompts, hidden databases, milestones, minefields, or evaluator targets in fixtures.

## Acceptance Commands

Run, in order:

```bash
uv sync --frozen
uv run pytest -q tests/schemas/test_state.py tests/online/test_state_builder.py tests/online/test_state_visibility.py
uv run python -c "from toolsandbox_pipeline.online import StateBuilder; from toolsandbox_pipeline.schemas.state import CompactVerifiedState, ControllerProvenanceSidecar"
git diff --check
git status --short
```

No acceptance command may contact a network service or load an upstream scenario.

## Acceptance Criteria

- All tests pass deterministically without external access.
- The Agent-visible state contains only permitted visible data and agent-facing names.
- Canonical tool identity exists only in the separate sidecar.
- `state_id` hashes only the Agent-visible state without its own `state_id` field.
- Result parsing never uses unsafe evaluation or hidden trace data.
- The reducer is pure apart from constructing and returning new objects.
- No unowned file is modified.

## Completion Report Additions

Include:

```text
State input/output public symbols:
Stable message-ID fixture:
State-ID fixture:
Visibility rejection tests:
Fact promotion/replacement tests:
Dependency/interface change requests:
```

This task performs no model calls or experiment run. Report `Total tokens: 0` and `Usage complete: true`; do not report unit-test duration as experimental `total_running_time_seconds`.
