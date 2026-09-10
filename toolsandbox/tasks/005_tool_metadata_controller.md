# Task 005: Tool Metadata, Argument Grounding, and Deterministic Controller

Status: `approved`

## Objective

Implement the deterministic safety layer between a validated Policy action and routing to the Critic or ToolSandbox Adapter:

- a strict, versioned metadata record for all 34 public tools in the pinned upstream commit;
- a manifest-verified metadata loader;
- recursive argument grounding using only Agent-visible evidence and current augmented schemas;
- deterministic prerequisite, risk, repeated-failure, forbidden-tool, and parallel-independence checks;
- a rule-based Controller that returns Task 002's `ControllerDecision` without generating, modifying, or executing an action.

This task contains no LLM logic and no heuristic natural-language inference.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3, 4, 7, 9-13, 15, 20, 21, and 24;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `tasks/002_core_contracts.md`;
5. `tasks/003_compact_state.md`;
6. `tasks/004_toolsandbox_adapter.md`;
7. this task file;
8. only the pinned upstream public tool modules, decorators, and tool-conversion helpers needed to verify public names and schemas.

Do not inspect scenario definitions, prompts, starting databases, milestones, minefields, target DataFrames, similarity functions, evaluator mappings, or test data.

## Access

```text
Access class: offline
Setup network: none
Data splits: none
Secrets: none
```

Use the installed pinned dependency, synthetic state/action fixtures, and public tool definitions. Do not call any tool, model, embedding, user simulator, API, dataset loader, or network service.

## Preconditions

- Tasks 001-004 are complete and accepted.
- The locked environment directly contains `pydantic==2.7.4` and `jsonschema==4.19.2`.
- The Task 003 Agent-visible state/Controller sidecar split is unchanged.
- If a required shared contract is missing, submit an interface change request; do not edit another task's schema file.

## Owned Files

The assigned Agent may create or edit only:

```text
configs/tool_metadata/controller_tool_metadata.jsonl
configs/tool_metadata/manifest.json
configs/grounding_transforms.json
src/toolsandbox_pipeline/schemas/tool_metadata.py
src/toolsandbox_pipeline/online/controller_inputs.py
src/toolsandbox_pipeline/online/tool_metadata.py
src/toolsandbox_pipeline/online/grounding.py
src/toolsandbox_pipeline/online/resources.py
src/toolsandbox_pipeline/online/controller.py
tests/tool_metadata/test_inventory.py
tests/tool_metadata/test_manifest.py
tests/online/test_grounding.py
tests/online/test_resources.py
tests/online/test_controller.py
tests/online/test_controller_visibility.py
```

Do not edit Task 002/003 schemas, Adapter code, dependency files, Policy/Critic code, prompts, tools, upstream source, checkpoints, or generated artifacts.

## Fixed Public Tool Inventory

The metadata file contains exactly one active record for each of these 34 canonical upstream tools, sorted lexicographically by `canonical_tool_name`:

```text
add_contact
add_reminder
calculate_lat_lon_distance
convert_currency
datetime_info_to_timestamp
end_conversation
get_cellular_service_status
get_current_location
get_current_timestamp
get_location_service_status
get_low_battery_mode_status
get_wifi_status
modify_contact
modify_reminder
remove_contact
remove_reminder
search_contacts
search_holiday
search_lat_lon
search_location_around_lat_lon
search_messages
search_reminder
search_stock
search_weather_around_lat_lon
seconds_to_hours_minutes_seconds
send_message_with_phone_number
set_cellular_service_status
set_location_service_status
set_low_battery_mode_status
set_wifi_status
shift_timestamp
timestamp_diff
timestamp_to_datetime_info
unit_conversion
```

The inventory test derives the public registered-tool set from the pinned tool package without importing scenario definitions. A missing, extra, or renamed tool is a hard compatibility failure.

## Tool Metadata Contract

`schemas/tool_metadata.py` defines strict models/enums for:

```python
ToolEffect
ToolRisk
MetadataPredicate
ResourceTemplate
ControllerToolMetadata
ToolMetadataManifest
```

`ToolEffect` contains exactly:

```text
sandbox_read
sandbox_write
external_read
external_write
conversation_control
```

Each JSONL record contains the Section 9 fields:

```text
canonical_tool_name
effect
risk
prerequisites
parallel_safe
read_resources
write_resources
critic_required
```

Requirements:

- Current effect counts are exactly 17/11/5/0/1 in the enum order above.
- All contact, messaging, reminder, and setting mutations are `sandbox_write`, even though they model real-world concepts; their effects remain episode-local.
- The five functions in `rapid_api_search_tools.py` are `external_read`.
- `end_conversation` is `conversation_control`, `risk: high`, `parallel_safe: false`, `critic_required: true`, and Agent-forbidden.
- Any future `external_write` defaults to `risk: high`, `parallel_safe: false`, and `critic_required: true`. With the current project contracts it is always blocked as unauthorized; `official_live` alone never authorizes it.
- Baseline risk follows effect: `sandbox_read: low`, `sandbox_write: medium`, `external_read: medium`, `external_write: high`, and `conversation_control: high`. A stricter per-tool risk requires an explicit documented rationale in the manifest.
- Metadata comes only from public tool names, decorators, docstrings, signatures, and public behavior. Scenario/evaluator evidence is forbidden.
- Unknown fields, duplicate tool names, invalid templates, unsorted records, or count mismatches fail loading.

### Manifest

`manifest.json` records at least:

```text
schema_version
upstream_repository
upstream_commit
record_count
effect_counts
metadata_sha256
public_inventory_sha256
risk_policy
generated_at_utc
```

`generated_at_utc` is provenance only and is excluded from content identity. The loader verifies the upstream commit, canonical JSONL content hash, inventory hash, record count, unique/sorted names, and effect counts before returning any metadata.

Runtime code never silently regenerates or repairs the committed metadata.

## Metadata Predicates and Resource Templates

### Prerequisites

Prerequisites are machine-readable records with:

```text
code
path
op
value, when required
```

Paths are RFC 6901 pointers into Agent-visible `CompactVerifiedState`. Operators follow the Section 9 strict predicate semantics. `exists/not_exists` omit `value`; `eq/neq/in/contains` require it. Missing pointers do not satisfy a positive prerequisite.

The Controller may use Controller-side canonical metadata to select the record, but it cannot inspect a hidden database or infer missing state.

### Resources

Read/write resource templates are literal strings containing zero or more placeholders of exactly this form:

```text
{arg:/json/pointer}
```

Resolve placeholders only from the validated action's visible arguments. Values must be JSON scalar values; missing, container, or non-finite values make the resource unresolved. Do not stringify arbitrary objects.

Two calls are independent only when:

- both metadata records have `parallel_safe: true`;
- every resource template resolves;
- neither call writes a resource read or written by the other;
- no prerequisite of either call depends on the other call's future result;
- neither call is `conversation_control` or `external_write`.

An unresolved resource never proves independence. A dependent or unprovable batch receives `DEPENDENT_PARALLEL_CALLS` before native execution.

## Argument Grounding Contract

Recursively inspect every scalar leaf in each function call's `arguments`. Use strict JSON type-and-value equality; Python's `True == 1` must not count as equality.

A leaf is grounded only by one or more structured `GroundingEvidence` records from these sources:

1. an exactly equal provenance-bearing `VerifiedFact.value` in the current Agent-visible state;
2. for a string leaf only, an exact character span in a visible `USER -> AGENT` message, with message ID and start/end offsets recorded;
3. `const`, `enum`, or explicit `default` at the same argument pointer in the current augmented agent-facing schema;
4. a versioned allowlisted pure transform applied to evidence from items 1-3;
5. a validated public-result field already represented by a current `VerifiedFact`.

Requirements:

- The grounding engine receives augmented agent-facing schemas, never unaugmented callable signatures.
- Argument names, descriptions, and types removed by an augmentation cannot be reconstructed from canonical source.
- Evidence records the argument pointer, source kind/reference, source offsets or fact pointer, transform name/version if used, and result value hash.
- The committed `configs/grounding_transforms.json` starts with an empty transform list. No transform is added based on guessed convenience or scenario answers.
- The implementation may support injecting a versioned transform registry for synthetic tests, but production loading allows only entries committed in the reviewed registry.
- `eval`, arbitrary import paths, lambdas from configuration, locale-dependent conversion, current time, and network/filesystem transforms are forbidden.
- If ToolSandbox provides a visible utility tool for conversion, the Policy should call that tool; the Controller cannot perform the conversion invisibly.
- A container is accepted only when all of its scalar leaves are grounded. An empty object/array still must be permitted by the current augmented schema.

## Augmented Schema Validation

Validate tool arguments only against the current agent-facing OpenAI function schema using the locked `jsonschema` implementation.

Map validation failures deterministically:

- unknown property -> `INVALID_PARAMETER`;
- missing required property -> `MISSING_REQUIRED_ARGUMENT`;
- exposed JSON type mismatch -> `INVALID_ARGUMENT_TYPE`;
- exposed enum/const/range/length/pattern or other value constraint -> `INVALID_ARGUMENT_VALUE`.

If an augmentation removed a type or description, do not recover it from the callable. Final execution-facing signature validation belongs to the Adapter boundary after conversion and cannot leak details back into model prompts.

## Controller Input and Skill View

`controller_inputs.py` defines strict project-owned inputs, including:

```python
ControllerInput
RetrievedSkillControllerView
ActionHistoryEntry
```

`ControllerInput` keeps these distinct:

- Agent-visible `CompactVerifiedState`;
- Controller-only provenance/name mapping sidecars;
- validated `ActionEnvelope`;
- manifest-verified tool metadata;
- currently retrieved active-skill views;
- committed action/failure history;
- reproducibility profile.

`RetrievedSkillControllerView` contains only fields needed for deterministic checks: skill ID/version/status, canonical tool dependencies, structured applicability/required-input predicates, and generation ID. It contains no natural-language instruction. The later Skill task must adapt its full records to this view.

The Controller must not serialize its combined input as a Policy/Critic prompt.

## Deterministic Controller Rules

The Controller returns Task 002's `ControllerDecision` and never returns a replacement action.

For every action, evaluate applicable checks and emit unique codes in the exact Section 12 enum order. Evidence for each code is mandatory and contains only an allowed `ControllerSourceKind` and opaque reference.

### Blocking checks

- Unknown non-forbidden tool -> `INVALID_FUNCTION`.
- `end_conversation` or another metadata-marked Agent-forbidden tool -> `AGENT_FORBIDDEN_TOOL`.
- Agent-facing schema errors use the mapping above.
- Any ungrounded scalar leaf -> `UNGROUNDED_ARGUMENT`.
- Unsatisfied metadata or selected-skill prerequisite -> `MISSING_DEPENDENCY` or `CONSTRAINT_VIOLATION`, according to the failed record type.
- Selected skill ID not present in the current retrieved active-skill views, wrong generation, failed applicability, or tool dependency mismatch -> `CONSTRAINT_VIOLATION`.
- Unprovable or dependent parallel calls -> `DEPENDENT_PARALLEL_CALLS`.
- Any `external_write` without a future separately specified structured authorization object -> `UNAUTHORIZED_EXTERNAL_SIDE_EFFECT`.
- A canonical action fingerprint matching a committed visible failed action -> `REPEATED_FAILED_ACTION`.

The action fingerprint excludes `call_id` and `selected_skill_id` and hashes the canonical execution-facing tool name plus canonical JSON arguments. It is Controller-only. Assistant messages use a separate fingerprint over their visible content only when needed for history; they have no tool arguments.

If a function is unavailable, skip schema/grounding/resource checks that require its missing schema or mapping, but continue independent forbidden/history checks. Never manufacture evidence after an upstream lookup failure.

### Critic triggers

- metadata `critic_required: true` -> `CRITIC_REQUIRED_TOOL`;
- any medium/high-risk selected tool -> `MEDIUM_OR_HIGH_RISK`;
- every parallel batch -> `PARALLEL_BATCH_REVIEW`;
- every assistant message -> `ASSISTANT_MESSAGE_REVIEW`;
- a valid structured constraint with competing safe choices -> `STRUCTURED_CONSTRAINT_TENSION` only when represented explicitly in input; no natural-language inference;
- every `external_read` -> `EXTERNAL_READ_REVIEW`.

Collect triggers even when blocking codes exist, because Section 12 routes either list to the Critic. The Critic cannot override a blocking code.

## Forbidden Behavior

The implementation must not:

- call an LLM, embedding service, tool, ExecutionEnvironment, evaluator, or dataset loader;
- modify, repair, replace, or execute the action;
- inspect hidden databases, `tool_trace`, evaluator labels, scenario construction, or unaugmented schemas;
- treat Critic output, Skill prose, or memory text as verified fact;
- ground a value through fuzzy matching, semantic similarity, case folding, numeric coercion, or guessed defaults;
- infer resource independence from tool names or natural language;
- turn a missing metadata/schema record into a warning or permissive decision.

## Required Tests

Tests must cover at least:

1. exact 34-tool inventory, alphabetical uniqueness, pinned commit, manifest hashes, and 17/11/5/0/1 effect counts;
2. representative public tools in every current effect category and all mutation families;
3. hard failure for missing/extra/duplicate/unsorted/tampered metadata and a synthetic future `external_write`;
4. every argument-grounding source, strict bool/int separation, nested arrays/objects, empty containers, span offsets, and ungrounded leaves;
5. explicit proof that removed augmented types/descriptions are not recovered from canonical signatures;
6. empty production transform registry and safe versioned synthetic transform behavior;
7. resource template parsing, RFC 6901 escaping, scalar resolution, conflict detection, unresolved resources, and dependent batches;
8. every blocking code and every Critic trigger code;
9. fixed code ordering, deduplication, and mandatory evidence;
10. unavailable functions, Agent-forbidden `end_conversation`, stale mappings, missing schemas, and future external writes;
11. selected-skill active/retrieved/generation/applicability/tool-dependency checks without reading Skill prose;
12. repeated-failure fingerprint behavior across different call/skill IDs and changed arguments;
13. assistant, single-call, and parallel-batch decisions;
14. input immutability and identical output for repeated equivalent inputs;
15. visibility tests proving no hidden database, evaluator, `tool_trace`, canonical signature, or sidecar value enters prompt-facing evidence.

Inventory tests may inspect registered public tools but must not instantiate or inspect scenario definitions.

## Acceptance Commands

Run, in order:

```bash
uv sync --frozen
uv run pytest -q tests/tool_metadata tests/online/test_grounding.py tests/online/test_resources.py tests/online/test_controller.py tests/online/test_controller_visibility.py
uv run python -c "from toolsandbox_pipeline.online.controller import Controller; from toolsandbox_pipeline.online.tool_metadata import load_tool_metadata"
git diff --check
git status --short
```

No acceptance command may call a tool or external service.

## Acceptance Criteria

- Metadata exactly matches the 34-tool pinned public inventory and all manifest validations pass.
- Grounding accepts only structured permitted evidence and fails closed otherwise.
- Parallel independence is proven structurally or blocked.
- Every Controller code is deterministic, ordered, unique, and evidence-backed.
- The Controller never alters or executes an action.
- Hidden or unaugmented information is inaccessible to Agent-facing validation and evidence.
- No unowned or upstream file is modified.

## Completion Report Additions

Include:

```text
Metadata manifest hash:
Public inventory hash and count:
Effect counts:
Grounding cases tested:
Controller codes/triggers tested:
Future external-write behavior tested:
Dependency/interface change requests:
```

This task performs no model calls or experiment run. Report `Total tokens: 0` and `Usage complete: true`; do not report unit-test duration as experimental `total_running_time_seconds`.
