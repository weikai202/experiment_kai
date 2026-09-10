# Task 004: Pinned ToolSandbox Adapter and PipelineAgent Shell

Status: `approved`

## Objective

Implement the only integration boundary between the project and the pinned Apple ToolSandbox package. The adapter must:

- expose Agent-visible upstream messages as Task 003's sanitized input models while retaining stable upstream message indices;
- expose the exact current agent-facing augmented tool schemas;
- keep the agent-facing-to-canonical name mapping in a separate Controller-only context;
- convert validated Task 002 actions into native upstream `Message` objects;
- provide a `PipelineAgent(BaseRole)` shell that delegates one Agent turn to an injected project-owned responder.

This task preserves upstream roles and execution semantics. It does not implement State reduction, retrieval, Policy, Controller, Critic, Revision, checkpointing, scenario selection, or evaluation.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3, 4, 7, 8, 11, 12, 15, 16, 20, 21, 24, and 25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `tasks/002_core_contracts.md`;
5. `tasks/003_compact_state.md`;
6. this task file;
7. only the public integration files required from the pinned `tool_sandbox` package.

Do not inspect upstream scenario prompts, starting databases, milestones, minefields, target DataFrames, or evaluator targets while implementing this task.

## Access

```text
Access class: offline
Setup network: none
Data splits: none
Secrets: none
```

Use the already-installed pinned dependency and synthetic ExecutionContext/message/tool fixtures. No model, embedding, user simulator, dataset split, RapidAPI, credential, or network access is permitted.

## Preconditions

- Tasks 001-003 are complete and accepted.
- The installed `tool-sandbox` distribution resolves to commit `165848b9a78cead7ca7fe7c89c688b58e6501219`.
- If an upstream API differs from this task, report the exact import/symbol mismatch. Do not patch upstream or silently implement alternate semantics.

## Owned Files

The assigned Agent may create or edit only:

```text
src/toolsandbox_pipeline/toolsandbox_adapter/__init__.py
src/toolsandbox_pipeline/toolsandbox_adapter/contracts.py
src/toolsandbox_pipeline/toolsandbox_adapter/messages.py
src/toolsandbox_pipeline/toolsandbox_adapter/tools.py
src/toolsandbox_pipeline/toolsandbox_adapter/pipeline_agent.py
tests/toolsandbox_adapter/conftest.py
tests/toolsandbox_adapter/test_visibility.py
tests/toolsandbox_adapter/test_tools.py
tests/toolsandbox_adapter/test_messages.py
tests/toolsandbox_adapter/test_pipeline_agent.py
```

Do not edit upstream source, Task 002/003 schemas, dependency files, scenario/evaluator code, other roles, orchestration, prompts, or generated artifacts.

## Pinned Upstream Interfaces

Use the upstream interfaces rather than duplicating their behavior:

```python
tool_sandbox.common.execution_context.DatabaseNamespace
tool_sandbox.common.execution_context.ExecutionContext
tool_sandbox.common.execution_context.RoleType
tool_sandbox.common.execution_context.get_current_context
tool_sandbox.common.message_conversion.Message
tool_sandbox.common.message_conversion.openai_tool_call_to_python_code
tool_sandbox.common.tool_conversion.convert_to_openai_tools
tool_sandbox.roles.base_role.BaseRole
```

The adapter may call documented current-context name-mapping methods. It must not import scenario definitions or evaluator internals.

## Public Adapter Contracts

`contracts.py` defines project-owned typed interfaces for:

```python
AgentTurnView
ControllerToolContext
AdapterTurn
PipelineResponder
```

### AgentTurnView

Contains only:

- ordered `VisibleMessageInput` values from Task 003;
- ordered `AgentFacingToolInput` values from Task 003.

It must have no canonical tool name, mapping, raw callable, unaugmented schema, hidden context, database, `tool_trace`, `conversation_active`, or evaluator field.

### ControllerToolContext

Contains:

- a one-to-one mapping from current agent-facing names to canonical execution-facing names;
- a canonical manifest hash of that complete mapping;
- opaque references needed to associate current callable/public return contracts without exposing callable objects to prompt-facing data.

This object is Controller/host-only. It is never nested inside or serializable through `AgentTurnView`.

### AdapterTurn

Holds `agent_view` and `controller_context` as distinct fields for trusted host orchestration. It is not a Policy/Critic prompt model and must not provide a convenience serializer that combines both objects.

### PipelineResponder

A synchronous injected protocol:

```python
class PipelineResponder(Protocol):
    def respond(self, turn: AdapterTurn) -> ActionEnvelope: ...
```

The Task 004 test responder is deterministic and local. Later orchestration implements this protocol. `PipelineAgent` must not construct a model client or read environment credentials.

## Visible Message Extraction

Implement one adapter function that reads the current sandbox message history with `sandbox_message_index` retained and then applies upstream Agent-role visibility semantics.

Requirements:

- Use `PipelineAgent.filter_messages()`, inherited unchanged from `BaseRole` with `PipelineAgent.role_type = RoleType.AGENT`; do not invent a different visibility rule.
- Respect `ending_index` exactly as upstream `BaseRole.get_messages()` does.
- Validate upstream message ordering and preserve each original sandbox message index as `source_message_index`.
- Convert only fields accepted by `VisibleMessageInput`.
- Never copy `visible_to`, `conversation_active`, or `tool_trace` into project inputs.
- `openai_function_name` remains the agent-facing name returned to the Agent.
- Do not query SETTING, CONTACT, MESSAGING, REMINDER, evaluator, or any non-SANDBOX database.
- Do not use `BaseRole.get_messages()` alone because it drops `sandbox_message_index`; retain the index without weakening upstream filtering.
- Do not log or expose messages that failed the Agent visibility filter.

The adapter may validate that the final raw message is addressed to `AGENT`, matching `BaseRole.messages_validation()`. A System-to-Agent message follows upstream behavior: `PipelineAgent.respond()` returns without calling the injected responder or appending a response.

## Agent-Facing Tool Schemas and Mapping

At each Agent turn:

1. obtain tools through `PipelineAgent.get_available_tools()`;
2. preserve dictionary iteration order;
3. generate prompt-facing schemas through upstream `convert_to_openai_tools()` under the active augmentation context;
4. validate and wrap each result as `AgentFacingToolInput` without truncation or repair;
5. obtain agent-facing-to-execution-facing names through the current `ExecutionContext` mapping API;
6. verify mapping keys exactly equal the available agent-facing tool names and values are unique;
7. hash the complete mapping using Task 002 canonical JSON helpers;
8. place schemas only in `AgentTurnView` and mapping only in `ControllerToolContext`.

Do not regenerate an unaugmented signature for the Policy or Critic. Do not expose callable `__name__`, mapping values, or canonical metadata in prompt-facing objects. `end_conversation` must remain unavailable to the Agent because upstream tool visibility assigns it to the User role.

## Action-to-Message Conversion

Implement conversion from a locally validated `ActionEnvelope` to a non-empty ordered list of native upstream `Message` objects.

### `assistant_message`

Produce exactly one:

```text
sender=AGENT
recipient=USER
content=<verbatim validated content>
```

Do not set `conversation_active`; the Agent cannot end the conversation directly.

### `function_call`

- Confirm the action name is a currently available agent-facing tool.
- Resolve the execution-facing name only through `ControllerToolContext`/current ExecutionContext mapping.
- Build the execution code with upstream `openai_tool_call_to_python_code()` rather than reimplementing its wire format.
- Produce exactly one `AGENT -> EXECUTION_ENVIRONMENT` message.
- Set `openai_tool_call_id` to `call_id` and `openai_function_name` to the agent-facing name.
- Do not place the selected skill ID, canonical name, Controller evidence, or Critic output in the native Message.

### `parallel_batch`

- Preserve declared call order.
- Convert every call using the same rules as `function_call`.
- Return consecutive `AGENT -> EXECUTION_ENVIRONMENT` messages in one Agent response.
- Conversion is all-or-nothing: validate every current tool name and mapping before constructing or appending any message.
- This adapter does not decide independence. It assumes the trusted responder has completed Controller checks, but it must not execute the messages itself.

An unknown or stale agent-facing name, missing mapping, duplicate mapping value, or upstream conversion error is a hard failure. Do not fall back to a canonical name supplied by model output.

## PipelineAgent Behavior

`PipelineAgent` must:

- subclass upstream `BaseRole` and set `role_type = RoleType.AGENT`;
- require an injected `PipelineResponder` in its constructor;
- use upstream current-context storage rather than maintaining a second conversation database;
- on `respond(ending_index)`, validate the current last recipient, extract the `AdapterTurn`, call the responder exactly once, locally revalidate the returned value as `ActionEnvelope`, convert it, and append native messages with `BaseRole.add_messages()`;
- append no messages if extraction, response, or conversion fails;
- return without responder invocation for System-sender messages, matching upstream behavior;
- implement `reset()` and `teardown()` only by delegating to explicitly optional responder lifecycle methods when present; no model/global resource ownership is invented;
- never invoke ExecutionEnvironment, User, evaluator, scenario loop, or tool callable directly.

The shell is intentionally thin. Transactional LLM/tool commitment and resume behavior belong to later orchestration/checkpoint tasks.

## Required Tests

Use synthetic contexts, messages, and decorated test tools. Tests must cover at least:

1. `PipelineAgent` is a `BaseRole` with `RoleType.AGENT`;
2. visible SYSTEM/USER/AGENT/EXECUTION_ENVIRONMENT messages retain correct original indices and verbatim content;
3. messages not visible to the Agent never reach `AgentTurnView` or test logs;
4. `ending_index` matches upstream truncation behavior;
5. `tool_trace` and `conversation_active` are absent from every project-owned prompt-facing object;
6. upstream augmented schema conversion is used and schema order is preserved;
7. identity and scrambled mappings are complete, one-to-one, hash-stable, and Controller-only;
8. `end_conversation` is absent from Agent tools;
9. exact assistant-message conversion;
10. exact function-call conversion, including agent-facing `openai_function_name` and execution-facing code;
11. ordered parallel conversion and all-or-nothing failure;
12. rejection of unavailable/stale names, missing mappings, and invalid responder output;
13. responder invoked exactly once on an ordinary Agent turn and zero times for System messages;
14. no native messages appended on any pre-append failure;
15. import and execution cause no model/API/network access and do not modify upstream files.

Fixtures must not use real scenario prompts, hidden state, milestones, minefields, target data, evaluator definitions, or live tools.

## Acceptance Commands

Run, in order:

```bash
uv sync --frozen
uv run pytest -q tests/toolsandbox_adapter
uv run python -c "from tool_sandbox.roles.base_role import BaseRole; from toolsandbox_pipeline.toolsandbox_adapter import PipelineAgent; assert issubclass(PipelineAgent, BaseRole)"
uv run python -c "from importlib.metadata import distribution; import json; d=json.loads(distribution('tool-sandbox').read_text('direct_url.json')); assert d['vcs_info']['commit_id']=='165848b9a78cead7ca7fe7c89c688b58e6501219'"
git diff --check
git status --short
```

No acceptance command may contact a model, API, dataset, or live tool.

## Acceptance Criteria

- All focused tests pass against the exact pinned upstream dependency.
- The adapter changes only `RoleType.AGENT`; upstream User, ExecutionEnvironment, scenario loop, ExecutionContext, and evaluator remain untouched.
- Prompt-facing objects contain agent-facing augmented information only.
- Canonical names and mapping hashes remain Controller/host-only.
- Native Message output matches upstream semantics and never sets Agent-owned conversation termination.
- No unowned or upstream file is modified.

## Completion Report Additions

Include:

```text
Pinned upstream interfaces used:
Synthetic visibility cases tested:
Augmentation/mapping cases tested:
Native Message conversion cases tested:
Upstream files changed: none | list unexpected changes
Dependency/interface change requests:
```

This task performs no model calls or experiment run. Report `Total tokens: 0` and `Usage complete: true`; do not report unit-test duration as experimental `total_running_time_seconds`.
