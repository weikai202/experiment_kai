# BFCL v4 Multi-Turn Evolution Protocol

Status: implemented and offline-tested. This is a pipeline-aligned research
split, not an official BFCL leaderboard protocol.

## Scope and invariants

Use only the four BFCL v4 multi-turn categories: `multi_turn_base`,
`multi_turn_miss_func`, `multi_turn_miss_param`, and
`multi_turn_long_context`, pinned to BFCL revision
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. There are exactly 200 records in
each category. Index-aligned records form an immutable family only if their
canonical tool path, involved simulator classes, and excluded functions match.
All turns, simulator state, delayed-function events, and four variants stay
together.

Sort family IDs by UTF-8 bytes and shuffle exactly once with Python 3.10
`random.Random(0)`. Allocate the first 40 families to sealed test, the next 40
to dev, and the remaining 120 to train. Divide train's shuffled order into three
contiguous 40-family shards. Each shard therefore contains 160 cases.

BFCL supplies fixed dataset user turns; this adaptation has no model-backed User
Simulator and therefore no User Simulator requests or token accounting.

## Online method

For every user turn, Initial Policy creates a BFCL Python-text action from only
the visible conversation, currently exposed function schemas, retrieved Policy
and World memories, and Skills. Controller parses direct calls with `ast` and
`ast.literal_eval`, rejects unavailable functions, and never uses `eval`.
Critic reviews the draft. Revision runs only after `REVISE`; its output passes
the same Controller. Calls execute in the official stateful simulator. A Miss
Func function is exposed only at its recorded turn. All calls in one output are
one parallel BFCL action batch. The official checker and irrelevance checker
remain scoring authorities.

All Qwen roles use frozen `Qwen/Qwen3-32B`, temperature 0, seed 0, and explicit
`chat_template_kwargs.enable_thinking=false`. A formal request has no default
output limit: it must bind a hash-verified calibration receipt created by a
real train-only smoke for the same dataset manifest, model, and base decoding
configuration. The receipt records ordered train case IDs, observed output
lengths and finish reasons, and the deterministic selected limit. Synthetic
receipts are accepted only by offline tests and are rejected by formal plans.
No role can see reference calls,
ground truth, checker internals, future delayed functions, or another split.
Memory and Skill retrieval uses only `text-embedding-3-small`; model mismatch,
missing vectors, invalid dimensions, or provider failure stops the run with no
fallback. Every stored vector binds the resource content hash, exact embedding
model, client configuration, dimension, and vector hash. Eligibility requires
all Skill tool dependencies to be currently visible. A host-issued retrieval
receipt binds the selected resource, its creation effect, query, visible tools,
rank, and embedding configuration into the online decision lineage.

## Offline evolution

G000 contains seed memories and Skills. Its Skill library is compiled
deterministically from only the public function schemas of the pinned four
BFCL files, including delayed public schemas. Every resource binds the pinned
source revision and canonical schema hash; scenario text, answers, evaluator
data, and private fields are excluded. The canonical library file and manifest
hash bind G000 and the final plan. Round 0 runs shard 0 and publishes
G001; round 1 runs shard 1 and publishes G002; round 2 runs shard 2 and
publishes G003. Only sanitized, committed train evidence may update failure
modes, memories, or Skills. Each Skill candidate is evaluated on up to five
relevant dev families selected by a deterministic SHA-256 order. All four
variants are included, giving at most 20 paired cases. Accept only a strict
increase in official valid cases. Dev never creates candidates or selects the
final generation.

Each round durably claims the active case before dispatch and checkpoints
completed case IDs, committed effects, published
generation, durable completion, and the complete accounting ledger. Logical
request identity binds run, round, case, turn, provider, model, role, canonical
input, and request configuration; physical attempts, accepted response
applications, and substantive effects are separately deduplicated. Resume uses
the latest hash-verified stage. An unknown in-flight case is reconciled through
the operation's prepare/complete recovery seam rather than blindly
redispatched. Direct latency begins before round work and ends only after the
final complete state and accounting checkpoint are durably fsynced. Record all
provider input and output
tokens. Non-monetary `qwen_effective_output_tokens` includes only an
accepted Qwen output linked to a substantive committed effect; `NONE`, `SKIP`,
rejected, retry-only, and no-op output contributes zero.

## Final evaluation and reporting

Freeze the exact 40-family / 160-case test manifest, G000, G003, model contract,
calibration receipt, and run plan before acquiring the one-use test authority.
The plan hash binds
the exact non-thinking Qwen decoding configuration, embedding configuration,
G000/G003 generation hashes, BFCL revision, simulator environment, official
evaluator, generation resource manifest, experiment configuration, and
single-process executor. The authority is resumable only by the same owner and
same frozen plan. Before each manifest-ordered unit, the runner durably records
its dispatch identity; an unknown in-flight unit must be reconciled through the
executor recovery seam. Each official result, trusted trajectory/effect proof,
complete-or-explicitly-missing usage snapshot, and runner-owned direct latency
is stored in
a hash-verified per-unit artifact bound to its plan/environment/evaluator.
The runner persists same-boot monotonic and cross-boot wall-clock anchors before
the first dispatch of each Vanilla, G000, and G003 system run. It closes that
run only after the last evaluator result and a durable system-completion
checkpoint are fsynced. Reporting uses these three authoritative run receipts;
the sum of per-unit durations is diagnostic only and is never reported as total
experiment latency. Unit timing stops after its durable result/accounting
completion checkpoint. Executor-reported timing is never trusted.
Every G000/G003 dispatch, artifact, receipt, and trusted trajectory binds the
exact generation SHA-256 and generation-scoped resource-manifest SHA-256 frozen
in the plan; matching a generation ID alone is insufficient. G000 must be a
root generation and fixed G003 must have G002 as its direct parent.
Persisted prefixes are never rerun, and completion binds the immutable final
report hash into the authority marker. Run, in order, Vanilla, G000, and G003.
Never select a checkpoint from test results.

Report category accuracy, overall official validity, all-four-variants family
success, variant-count distribution, and failure-mode repair attribution.
Lineage matching is exact and host-derived. A repair requires G000 failure,
G003 success, one matching lineage, and a committed G003 decision chain proving
use of the accepted evolved Skill. Preserve unmatched, ambiguous, and incomplete
counts. The analysis is observational, not causal.

The Skill-use proof is constructed by the host from a persisted retrieval
receipt, final decision, accepted online response application, exact executed
call identity, and the substantive committed effect for that call. Reports do
not accept caller-authored proof triples.

## Prohibited actions

Do not call models in offline tests, include secrets, commit generated runs,
execute sealed test during development, import sibling project code, modify
BFCL, use unrestricted Python evaluation, use dev to generate updates, or call
this an official leaderboard split.
