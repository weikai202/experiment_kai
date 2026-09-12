# Critic argument evidence projection repair

The user approved repairing the Controller-to-Critic evidence loss on 2026-09-12.
The coordinator explicitly authorized this narrow Task 008 / Task 013 interface
exception to the previous hash-only projection in pipeline section 14. The task
specification and pipeline document were not otherwise modified.

`online/prompt_contracts.py` adds optional `action_pointer` to
`PromptSafeControllerEvidence`. It is an RFC 6901 pointer rooted at the **proposed
ActionEnvelope**, for example `/action/arguments/days`. The existing code,
source kind, and evidence hash remain unchanged. Absent locations are omitted
from serialization so existing feedback artifacts retain their original hashes.

`safe_controller(decision, proposed_action=None)` enumerates argument paths in
the validated proposed action and matches exact Controller reference formats.
It does not expose or generically parse arbitrary host references. Locations are
emitted only for schema/state evidence and only for arguments already present in
the proposed action. Multi-call references include the exact batch index; an
unindexed reference in a multi-call batch has no location. A single-call batch
uses its actual `/action/calls/0/arguments/...` path. Nested properties escape
`~` and `/` according to RFC 6901. No parameter values are duplicated.

Critic context validation rejects any supplied location outside the proposed
arguments. Critic and Revision receive identical safe feedback. Tool metadata,
canonical mappings, raw source refs, missing paths, and ambiguous batch refs
remain hash-only. There is no tool execution, argument repair, grounding change,
new revision loop, or change to deterministic Controller decisions.

Validation:

- Focused evidence, prompt, and visibility tests: 15 passed (1.39 seconds).
- `uv run --frozen pytest -q tests/online tests/checkpointing tests/retrieval`:
  155 passed (5.59 seconds), one pre-existing upstream holidays warning.
- Tests cover `days` versus `timestamp` attribution, exact Critic/Revision request
  payloads, old serialization and hashes, multi-call and one-call batches,
  malicious/missing references, escaped Unicode nested properties, and rejection
  of a forged correctly hashed hidden pointer.
- External calls, model tokens and effective Qwen output cost: zero. These are
  offline development tests, not experiment metrics. Real Qwen behavior must be
  validated by the coordinator-owned train smoke before performance claims.

The implementation is in Task 008's actual prompt contract file
`online/prompt_contracts.py`; no `schemas/online_prompts.py` exists in this tree.

Follow-up validation on 2026-09-12:

- `PreparedRoleRequest` now applies the same visible-location validation during
  direct construction / checkpoint restoration. Six regressions cover Critic and
  Revision requests with corrected envelope hashes but missing argument, tool-name,
  or private-sidecar pointers. Unmodified older requests still restore.
- Full targeted suite after this addition: 161 passed in 5.64 seconds.
- Coordinator-owned real Qwen probe result:
  `/root/toolsandbox-runtime/critic-evidence-probe-20260912T091003/result.json`.
  The previous Critic incorrectly attributed UNGROUNDED_ARGUMENT to `timestamp`;
  with `/action/arguments/days`, the new Critic correctly identified `days`.
  Finish reason `stop`; input tokens 4,155, output tokens 124, total tokens 4,279;
  usage complete. Direct probe elapsed 3.9100672136992216 seconds.
  Effective output cost is zero because this standalone probe committed no
  pipeline effect. This one-case diagnostic does not establish end-to-end task
  success or aggregate performance improvement.

Critic repeated-code contract repair (2026-09-12):

The subsequent real smoke exhausted 768 output tokens repeating
`CONSTRAINT_VIOLATION` in `error_codes`. The user/coordinator approved a bounded
array and a clarification of existing Controller semantics. The schema now limits
non-accept error arrays to the 17 distinct enum members; accept retains maxItems 0.
Local duplicate rejection remains strict. The bound prevents an unbounded array
but does not itself enforce uniqueness in the constrained decoder.

The Critic prompt now explicitly distinguishes blocking codes from review triggers,
clarifies that CRITIC_REQUIRED_TOOL does not mean another tool is missing, and
requires each error code at most once. No criteria, temperature, reasoning mode,
output fields, or backend were changed. The manifest labels this prompt v2 while
retaining the compatibility path `prompts/critic_v1.txt`.

New prompt SHA-256:
`71d9c4c5191d92b7413eb15b851060a27f193b8555965ff0d6a7d735fdc41b95`.
The previous exact prompt and manifest are preserved outside the repository in
`/root/toolsandbox-runtime/proposed-fixes/critic-repeat-contract/before-*`.

Prompt loading and prepared requests accept legacy v1 and Critic-only v2.
`prepare_request` propagates the loaded prompt version. Existing calibrated token
limits are invalidated by the changed prompt and output-schema hashes through
`select_limit`; provisional/development ceilings are not promoted by this change.

Validation: `uv run --frozen pytest -q tests/schemas/test_critic_conditional.py
 tests/online tests/checkpointing tests/retrieval` — 173 passed in 5.71 seconds.
A new real fixed-input probe is still required to establish model compatibility.

Superseding v2 wording after fixed-input GPU comparison:

The initial v2 wording (SHA `71d9c4c5...`) is rejected. It incorrectly accepted
both historical blocking cases. The controlled comparison showed old prompt plus
new schema preserved the correct days revision, whereas new prompt plus old
schema still incorrectly accepted both cases. Therefore the wording, rather than
maxItems eliminating a grammar branch, caused this observed regression. The old
prompt with maxItems stopped the duplicate loop at 17 items but still failed strict
uniqueness validation.

The current v2 prompt preserves the original prompt verbatim and only appends
blocking-priority, review-trigger meaning, and unique-code requirements. Current
SHA-256: `359442e8aa7d11e86d285ccc86df8a4b98668a3cac4e7958b30da429eaa398ca`.
No other source, routing, decoding, or settings changes accompany this replacement.
The rejected v2 hash above remains historical evidence, not the active manifest.

Coordinator GPU result:
`/root/toolsandbox-runtime/critic-blocking-priority-probes-20260912T092257/results.json`.
All three fixed requests passed strict validation: days correctly revised (140
output tokens); prior duplicate-array case revised with one CONSTRAINT_VIOLATION
(193 output tokens); known acceptable action remained accepted (42 output tokens).
The duplicate-array case may still speculate about the precise constraint cause;
these probes do not establish full reasoning correctness or task-level success.

Skill tool mismatch evidence repair (2026-09-12):

The subsequent smoke's turn 11 selected a Skill for get_current_timestamp while
calling search_messages. The Controller's actual failure was tool/Skill mismatch,
but the hash-only skill evidence led Critic to speculate about search specificity.
The coordinator authorized a minimal Task 008 interface extension; deterministic
Controller checks remain unchanged.

`safe_controller(..., retrieved_skills=...)` may now add
`reason: selected_skill_tool_mismatch` and the corresponding selected_skill_id
pointer. It requires an exact host reference to an already retrieved Skill id and
version, one uniquely identifiable bound call, nonempty visible dependencies, and
a call name outside those dependencies. Unknown, matching, unbound, or ambiguous
batch references retain only the original hash. No canonical reference is exposed.

When this diagnosis is present, Critic and Revision envelopes contain the same
minimal `retrieved_skill_bindings` array: only skill_id and agent-facing
 tool_dependencies for the referenced Skills. No complete Skill prose is added to
Critic. `feedback_skill_bindings` derives this projection from immutable Initial
Policy views. CriticContext validates against those views; PreparedRoleRequest
validates pointer/call/id/dependency association and availability on direct
construction/restoration, and Revision additionally checks equivalence with its
original Skill views. All bound dependencies must be currently Agent-visible.
The normal request/ledger hashes retain provenance; these structural checks do
not claim to authenticate arbitrary jointly forged input data outside the ledger.
Old hash-only feedback and old envelopes remain supported unchanged.

Validation: `uv run --frozen pytest -q tests/online tests/checkpointing
 tests/retrieval tests/providers/test_critic_grammar.py` — 174 passed in 6.22
seconds, one upstream holidays warning. Tests cover actual Controller wrong-tool
binding, matching/unknown/ambiguous Skills, precise batch index, scrambled-name
non-disclosure, identical Critic/Revision minimal evidence, and tampered direct
request/context/restoration rejection. No GPU/model calls were made by the repair
agent. A coordinator-owned exact-input probe remains required.
