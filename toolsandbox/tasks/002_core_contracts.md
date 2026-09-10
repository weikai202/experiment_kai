# Task 002: Core JSON Contracts and Canonical Identity

Status: `approved`

## Objective

Implement the smallest shared contract layer needed by later online pipeline tasks:

- one strict Pydantic v2 base model;
- the Policy/Revision action envelope;
- Controller decision and evidence models;
- Critic output models and cross-field invariants;
- canonical JSON bytes and SHA-256 helpers, including non-self-referential `state_id` computation and verification.

This task freezes interfaces only. It does not build compact state, call a model, execute a tool, route an action, write a checkpoint, or implement Skill/Memory records.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3, 7, 11-15, 21, and 24;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `tasks/001_project_scaffold.md`;
5. this task file.

## Access

```text
Access class: offline
Setup network: none
Data splits: none
Secrets: none
```

All tests use literal fixtures. No model, embedding, upstream scenario, dataset, tool, credential, or network access is permitted.

## Preconditions

- Task 001 is complete and its acceptance commands pass.
- The locked environment contains `pydantic==2.7.4` as a direct dependency.
- If the required contract cannot be expressed with the locked dependencies and Python standard library, request an interface/dependency change; do not edit `pyproject.toml` or `uv.lock`.

## Owned Files

The assigned Agent may create or edit only:

```text
src/toolsandbox_pipeline/schemas/__init__.py
src/toolsandbox_pipeline/schemas/base.py
src/toolsandbox_pipeline/schemas/action.py
src/toolsandbox_pipeline/schemas/controller.py
src/toolsandbox_pipeline/schemas/critic.py
src/toolsandbox_pipeline/reproducibility/__init__.py
src/toolsandbox_pipeline/reproducibility/canonical.py
tests/schemas/test_action.py
tests/schemas/test_controller.py
tests/schemas/test_critic.py
tests/reproducibility/test_canonical.py
```

Do not edit dependency files, State/Skill/Memory/checkpoint schemas, orchestration code, generated run artifacts, or another task's files.

## Public Interfaces

Names below are part of the shared project interface. Export them from the indicated package `__init__.py` files without triggering external access or filesystem writes.

### Strict base and JSON values

`schemas/base.py` provides:

```python
class StrictModel(BaseModel): ...

JsonValue = ...
JsonObject = dict[str, JsonValue]
```

Requirements:

- Pydantic configuration uses `extra="forbid"` and strict validation.
- Unknown fields and implicit scalar coercion fail validation.
- JSON values support only null, boolean, integer, finite float, string, arrays, and string-keyed objects recursively.
- NaN and positive/negative infinity are rejected before persistence or hashing.
- Models do not silently normalize, truncate, or discard input.

### Policy and Revision action contract

`schemas/action.py` provides these public models and enums:

```python
ActionType
FunctionCall
FunctionCallAction
ParallelBatchAction
AssistantMessageAction
Action
ActionEnvelope
```

The wire forms must match Section 11 of `pipeline.md` exactly:

- `Action` is a discriminated union on `type`.
- `FunctionCallAction` contains `type`, `call_id`, required `selected_skill_id`, `name`, and `arguments`.
- A batch contains `type` and `calls`; each call contains `call_id`, required `selected_skill_id`, `name`, and `arguments`.
- `selected_skill_id` is explicitly either a non-empty string or JSON null; omitting the field is invalid.
- `call_id`, tool `name`, and assistant `content` are non-empty after validation. Do not trim or rewrite them.
- `arguments` is a JSON object and contains no non-finite float.
- A parallel batch contains at least one call and its `call_id` values are unique.
- The envelope contains exactly one field named `action`.

Do not validate current tool availability, augmented argument schemas, grounding, dependencies, risk, or parallel independence here. Those checks belong to the Controller/Adapter tasks and require runtime context.

### Controller decision contract

`schemas/controller.py` provides:

```python
BlockingCode
CriticTriggerCode
ControllerSourceKind
ControllerEvidence
ControllerDecision
```

The code enums exactly match Section 12 of `pipeline.md`. `ControllerSourceKind` contains exactly:

```text
state
schema
skill
tool_metadata
action_history
```

Requirements:

- `blocking_codes` and `critic_trigger_codes` preserve input order and contain no duplicates.
- A code cannot occur in both lists.
- Every code in either list has at least one evidence item whose `code` matches it.
- Evidence may not cite a code absent from both lists.
- `source_ref` is non-empty and remains an opaque reference; the schema layer does not resolve it.

Because the two enums contain different values in the current specification, cross-list collision is structurally unlikely; retain the explicit invariant so a later enum change fails safely.

### Critic contract

`schemas/critic.py` provides:

```python
CriticVerdict
PredictedOutcome
CriticErrorCode
CriticOutput
```

`CriticErrorCode` contains every blocking code plus the five additional Section 13 codes. Requirements:

- `predicted_effect` and `correction` contain at most 40 whitespace-delimited words each.
- Empty strings are allowed only where the cross-field rules explicitly require them.
- `accept` requires an empty error list and an empty correction.
- `revise` requires at least one unique error code and a non-empty correction.
- `uncertain` requires `predicted_outcome="uncertain"`, at least one unique error code, and a non-empty correction.
- Do not add a replacement action field or native evaluator label.

The model validates only the contract. It does not judge whether the Critic prediction is factually correct.

### Canonical JSON and state identity

`reproducibility/canonical.py` provides:

```python
def canonical_json_bytes(payload: JsonValue) -> bytes: ...
def canonical_sha256(payload: JsonValue) -> str: ...
def compute_state_id(payload_without_state_id: JsonObject) -> str: ...
def verify_state_id(full_state_payload: JsonObject) -> bool: ...
```

`canonical_json_bytes` must be exactly equivalent to Python 3.10:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

Additional requirements:

- `canonical_sha256` returns `"sha256:" + hashlib.sha256(bytes).hexdigest()`.
- `compute_state_id` rejects a payload that already contains the top-level key `state_id`; the hashed payload can never contain its own ID.
- `verify_state_id` requires a string `state_id`, removes only that top-level field from a copy, recomputes the digest, and uses a timing-safe equality comparison.
- None of the functions mutate the caller's payload.
- Unsupported Python objects, non-string object keys, bytes, Decimal, datetime objects, and non-finite floats fail rather than being stringified or coerced.
- Do not introduce a second serialization algorithm or a fallback representation.

This task does not define how the compact State Builder chooses or orders its fields. It only defines the canonical identity primitive that the later State task must call.

## JSON Schema Contract

- `ActionEnvelope.model_json_schema()`, `ControllerDecision.model_json_schema()`, and `CriticOutput.model_json_schema()` are the only authoritative JSON Schemas for constrained decoding.
- Do not hand-maintain a second schema copy.
- Generated schemas must reject unknown object properties through the strict model definitions.
- Local Pydantic validation remains mandatory after constrained decoding; JSON Schema generation is not a substitute for runtime validation.

## Required Tests

Tests must cover at least:

1. every valid action variant and its exact round-trip wire representation;
2. missing/extra fields, scalar coercion, empty identifiers/content, invalid arguments, empty batches, and duplicate batch call IDs;
3. every Controller enum value, missing evidence, orphan evidence, duplicate codes, and cross-list collision protection;
4. every valid Critic verdict combination and every invalid cross-field combination;
5. the 40-word boundary at exactly 40 and 41 whitespace-delimited words;
6. canonical key ordering, compact separators, UTF-8 Unicode preservation, nested values, and input immutability;
7. rejection of NaN, infinity, non-string mapping keys, bytes, Decimal, datetime, and arbitrary objects;
8. exact `sha256:` output against a fixed independently stated fixture;
9. equal IDs for semantically identical mappings with different insertion order;
10. rejection of self-referential state payloads and successful/failed `state_id` verification after payload mutation;
11. JSON Schema snapshots or focused assertions proving discriminators and `additionalProperties: false` behavior without duplicating an entire generated schema file.

Tests must not call implementation-private helpers when a public interface can prove the same property.

## Acceptance Commands

Run, in order:

```bash
uv sync --frozen
uv run pytest -q tests/schemas tests/reproducibility/test_canonical.py
uv run python -c "from toolsandbox_pipeline.schemas import ActionEnvelope, ControllerDecision, CriticOutput; from toolsandbox_pipeline.reproducibility import compute_state_id"
git diff --check
git status --short
```

No acceptance command may contact a network service.

## Acceptance Criteria

- All required valid fixtures pass and invalid fixtures fail closed.
- Wire field names and enum values exactly match `pipeline.md`.
- Canonical bytes and hashes are deterministic across repeated calls and mapping insertion order.
- `state_id` is never self-referential.
- Importing either public package performs no I/O or external access.
- No unowned file is modified.

## Completion Report Additions

Include:

```text
Public symbols added:
Schema invariants tested:
Canonical fixture and expected SHA-256:
Dependency/interface change requests:
```

This task performs no model calls or experiment run. Report `Total tokens: 0` and `Usage complete: true`; do not report unit-test duration as experimental `total_running_time_seconds`.
