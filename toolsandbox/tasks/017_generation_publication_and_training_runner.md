# Task 017: Generation Publication and Training Runner

Status: `approved`

## Objective

Implement the top-level train-side coordinator that:

- validates one fully resolved immutable run manifest;
- constructs and publishes Generation 0 from attested seed artifacts;
- runs exactly three ordered train shards against generations G000/G001/G002;
- invokes Task 015 memory updates and Task 016 Skill updates;
- builds and atomically publishes G001/G002/G003;
- clears/archives raw round trajectories so later rounds cannot read them;
- records direct per-round timing, all physical tokens, substantive-effect Qwen
  `total_cost`, and immutable metrics;
- resumes safely from Task 011 checkpoints;
- exposes allowlisted module commands for offline validation, calibration,
  train-only smoke, and formal training.

It does not run final test evaluation, compare Vanilla/Generation-0/Updated,
perform statistical reporting, modify model weights, or tune from dev/test results.

## Approved Feedback Decisions

- Generation-0 Skill seed content is generated deterministically from pinned
  public ToolSandbox tool schemas by the script owned below, not authored from
  scenario/evaluator evidence.
- Formal training is authorized for the exact immutable three-round protocol in
  this task after all stated preconditions, preflights, calibrations, and smoke pass.
  Any manifest/config/hash/profile change invalidates this authorization.
- Updated remains fixed to G003 during the formal protocol.
- G000-G003 and diagnostic best-observed checkpoint pointers must be retained for
  later human query, but those pointers cannot select the formal Updated system.
- Each round 0/1/2 must independently record direct total wall-clock latency and
  substantive-effect `total_cost`, including terminal/incomplete status.

## Required Reading

Read, in order:

1. `pipeline.md` in full;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. Tasks 001-016 in numeric order;
6. this task file.

Inspect upstream only through the pinned public scenario registry/role/evaluator
interfaces already authorized by Tasks 009 and 014. Do not inspect test outcomes
or bypass dataset access guards.

## Access

```text
Development access class: offline
Development data: synthetic only
Development secrets: none

Deferred calibration/smoke access class: real_data
Deferred data: deterministic train-only IDs
Deferred secrets: QWEN_BASE_URL, QWEN_API_KEY, OPENAI_API_KEY
Deferred external tools: fixture replay only

Formal training access class: coordinator-owned formal run
Formal data: three train shards plus selector-only dev Mini-Benches
Formal secrets: profile-selected model/tool credentials
```

The development Agent implements and tests with fakes only. Only the
project-external secret launcher may run deferred commands.

## Preconditions

- Tasks 001-016 are implemented and accepted.
- Task 006 User-Simulator durable seam required by Task 014 is present.
- All online/offline Qwen role token limits have completed train-only calibration
  and are coordinator-approved before formal training.
- Dataset, fixture, runtime, prompt, seed, checkpoint, and environment manifests
  are complete and hash-pinned.
- No formal run directory already exists under the requested `run_id`.

## Owned Files

The assigned Agent may create or edit only:

```text
configs/run/run_manifest.schema.json
configs/run/train_smoke_v1.json
scripts/build_seed_skill_library.py
src/toolsandbox_pipeline/schemas/run.py
src/toolsandbox_pipeline/orchestration/__init__.py
src/toolsandbox_pipeline/orchestration/run_manifest.py
src/toolsandbox_pipeline/orchestration/preflight_gate.py
src/toolsandbox_pipeline/orchestration/seed_skill_builder.py
src/toolsandbox_pipeline/orchestration/generation_builder.py
src/toolsandbox_pipeline/orchestration/generation_publisher.py
src/toolsandbox_pipeline/orchestration/checkpoint_registry.py
src/toolsandbox_pipeline/orchestration/round_runner.py
src/toolsandbox_pipeline/orchestration/training_runner.py
src/toolsandbox_pipeline/orchestration/resume.py
src/toolsandbox_pipeline/orchestration/cli.py
tests/orchestration/test_run_manifest.py
tests/orchestration/test_preflight_gate.py
tests/orchestration/test_seed_skill_builder.py
tests/orchestration/test_generation_builder.py
tests/orchestration/test_generation_publisher.py
tests/orchestration/test_checkpoint_registry.py
tests/orchestration/test_round_runner.py
tests/orchestration/test_training_runner.py
tests/orchestration/test_resume.py
tests/orchestration/test_cli.py
tests/orchestration/test_train_smoke.py
tests/orchestration/test_leakage.py
```

Do not edit component implementations from earlier tasks, upstream ToolSandbox,
dependency files, prompt/manual seed contents, dataset/fixture manifests, secret
launcher, or generated real run/generation artifacts. The owned builder may create
`seed_skills.jsonl` only at an explicit new staging/output path; it never edits an
existing seed file.

Commands use `python -m toolsandbox_pipeline.orchestration.cli`; this task does
not need to edit `pyproject.toml`.

## Immutable Run Manifest

`schemas/run.py` and the JSON Schema define a frozen, strict, non-secret
`ResolvedRunManifest` containing:

```text
schema/protocol versions
run_id and purpose
profile
data_seed and dataset manifest path/hash
ordered train shard IDs and dev/test access policy
ToolSandbox commit/source/dependency/image/environment hashes
Python patch, timezone, locale, world clock
scenario/tool/evaluator/schema hashes
Qwen exact model/base-URL identity hash/container/server/decoding/wire mode
embedding exact model/client config hash
User Simulator profile/model/prompt/few-shot/tool-schema/stop config hashes
online/offline prompt manifests and calibrated token-limit hashes
Generation-0 seed paths/attestation hashes
public tool-schema inventory/generator/version/output hashes
fixture mode/manifest/backend hashes
checkpoint config
query-only checkpoint observation registry schema/hash
metrics schemas and cost unit
process count and deterministic ordering policy
run root
```

Secrets and raw base URLs are forbidden. Endpoint identity is a sanitized
configured hash/label. Unknown fields, relative run roots, symlinks, provisional
formal token limits, unapproved profile combinations, missing hashes, and mismatched
component identities fail before credential lookup or dataset materialization.

The canonical manifest hash is the root identity for every checkpoint, request,
trajectory, generation, and metric artifact. Resume requires byte-identical
canonical content.

## Preflight Gate

Formal training requires separate successful, hash-bound preflight evidence for:

```text
qwen
embedding
user-simulator or strict local user
dataset manifest
fixture/live tool mode
checkpoint filesystem
```

Evidence records mode, exact returned model identity where applicable, configuration
hash, timestamp, sanitized status, actual setup latency/tokens, and response hash.
Preflight responses/tokens are setup evidence and never enter experiment totals.

One failed/missing/stale/mismatched preflight stops before the first train
scenario. The gate never launches preflights implicitly and never reads secrets.

## Generation 0

### Deterministic public-schema Skill Library builder

`scripts/build_seed_skill_library.py` is a thin fixed CLI over
`seed_skill_builder.py`. It reads only the Task 005 verified public tool-schema
inventory and writes a new canonical `seed_skills.jsonl` plus provenance report.
It performs no provider/model/network/dataset/scenario/evaluator access.

The builder:

1. verifies the pinned ToolSandbox commit and complete public inventory hash;
2. excludes only `end_conversation`;
3. emits exactly one `v1.0` active zero-statistics Skill per remaining actionable
   public canonical tool;
4. derives a stable opaque `skill_id` from generator version and canonical schema
   bytes;
5. places canonical identity only in the singleton sorted
   `tool_dependencies` field;
6. derives semantic fields deterministically from the intersection of information
   visible under every allowed augmentation variant, using fixed templates and no
   LLM;
7. rejects any canonical tool token in prohibited prose, any field dependent on
   removed description/type information, duplicate/collision, invalid dependency,
   or Task 007 schema violation;
8. writes canonical UTF-8 JSONL in `skill_id` order and records input/generator/
   algorithm/output hashes plus
   `dev_test_or_scenario_artifacts_used: false`.

Running the builder twice on identical input must produce byte-identical Skill
Library and provenance bytes. It may not fall back to hand-authored, empty,
synthetic, or model-generated Skills.

Generation 0 requires:

```text
seed_policy_memory.jsonl
seed_world_memory.jsonl
seed_skills.jsonl
seed_source_manifest.json
```

Validate exact paths/hashes, schemas, tool dependencies, sorted IDs, version/status
invariants, and the attestation:

```text
dev_test_artifacts_used = false
every seed source type is toolsandbox_public_tool_schema or manual_design
every seed file appears exactly once
```

For `seed_skills.jsonl`, every source entry must be
`toolsandbox_public_tool_schema` and must match the builder provenance; its
source list cannot contain `manual_design`. Manual design remains allowed only
for the separately attested Policy/World seed files.

Build all Task 007 retrieval documents/embeddings/indexes, validate the complete
generation snapshot with its public loader, then publish G000 atomically. No empty
or synthetic fallback generation is permitted.

Seed attestation is human/project provenance, not a model output. A failed
attestation stops the run.

## Next-Generation Builder

For round G:

1. load exactly generation G;
2. apply Task 015 staged Policy/World mutations in deterministic record order;
3. apply Task 016 staged Skill histories/active versions in deterministic order;
4. preserve unchanged records byte-for-byte where their schema allows;
5. validate evidence, statistics, statuses, versions, dependencies, and validation
   summaries;
6. build retrieval text and embeddings for changed/new active records;
7. reuse an existing embedding only when exact key/model/input hash matches;
8. rebuild complete deterministic Policy/World/Skill indexes;
9. create a complete G+1 manifest with all file/count/hash/config/source-generation
   identities;
10. load/validate the staged directory as a complete Task 007 generation.

No online reader can see staging. Memory/Skill outputs are never written directly
into the current generation.

## Atomic Generation Publication

Publication uses a new same-filesystem staging directory and Task 011 publication
transaction:

1. prepare expected destination/content hashes and pre-publication checkpoint;
2. write new files mode `0600`, directories `0700`;
3. fsync files and directories;
4. validate complete snapshot and exact generation ID;
5. atomically rename staging to the previously nonexistent destination;
6. fsync parent;
7. commit publication marker and post-publication checkpoint;
8. update an atomic derived `latest` pointer only after the committed generation
   exists.

Existing destination with identical committed identity is reused on recovery.
Existing conflicting/incomplete destination fails. No overwrite, partial merge,
in-place edit, delete, or best-effort repair is allowed.

## One Training Round

Round index G uses:

```text
online generation: g00G
train shard: G
published generation: g00(G+1)
```

The `RoundRunner`:

1. validates round/generation/shard/checkpoint high-water identity;
2. starts direct `round_total` monotonic timing immediately before first online
   scenario dispatch;
3. runs Task 014 EpisodeRunner for every shard scenario in manifest order from a
   fresh deep copy;
4. seals the complete current-round train trajectory buffer;
5. invokes Task 015 memory updates;
6. invokes Task 016 Skill statistics/failure/rewrite/Mini-Bench;
7. builds and publishes G+1;
8. writes final round checkpoint;
9. stops direct round timing;
10. materializes request/task/round metric records;
11. archives the raw trajectory buffer and clears the working reference.

Dev Mini-Bench work used for Skill selection is inside the training-round time,
`total_tokens`, and `total_cost` boundary with phase
`dev_minibench`. It remains separately attributable and cannot update train
statistics/memory.

No scenario failure is silently skipped. A typed terminal failure stops the round
before publication unless the manifest explicitly defines a pre-reviewed
fail-fast result policy; the initial protocol is fail-fast.

## Three-Round Training Run

`TrainingRunner` requires exactly:

```text
round 0: g000 + train shard 0 -> g001
round 1: g001 + train shard 1 -> g002
round 2: g002 + train shard 2 -> g003
```

Before each round:

- previous generation publication is committed and loadable;
- prior raw trajectory working buffer is empty/inaccessible;
- only aggregate prior statistics and the published generation persist;
- shard has not run and no family crosses shards;
- all component hashes still match the run manifest.

After round 2, pin G003 as `updated_generation_id`. Do not inspect test data or
choose among G001/G002/G003 based on dev/test score. The protocol's Updated system
is always G003 if all rounds complete.

## Query-Only Checkpoint Observation Registry

After each publication, append an immutable registry entry for G000-G003 with its
generation/manifest/content hashes. For G001-G003 also link the producing round,
train shard, native mean similarity, fully-successful rate, direct
`total_running_time_seconds`, `total_tokens`/`usage_complete`, and
`total_cost`/`cost_complete`.

The registry exposes deterministic diagnostic pointers:

```text
highest_observed_train_mean_similarity
highest_observed_train_fully_successful_rate
```

Ties choose the lower numeric generation ID. Every pointer must state
`selection_allowed: false` and
`comparability: different_train_shards_not_directly_comparable`. G000 has no
producing-round score. No dev/test metric or hidden evaluator content enters the
registry. Querying it is read-only and cannot alter `updated_generation_id = g003`.

## Raw-Trajectory Archive Boundary

At round completion:

- close the current buffer;
- write its immutable archive manifest/hash under the restricted round directory;
- include the Task 016 sanitized failure-mode lineage records and their hashes in
  that archive so Task 018 can analyze them after all systems complete, while raw
  train trajectories remain unavailable;
- revoke the current-buffer capability/token;
- expose only allowed aggregate statistics and published generation to later
  rounds;
- reject any attempt by later prompt/projection/update code to open older raw
  archives.

Archives remain available only to explicit post-run restricted audit, never to
online/offline model inputs. Moving a path is not sufficient; access is enforced
by phase/round capability checks.

## Resume Coordinator

`resume.py` opens only an explicit existing run root, validates the canonical
manifest/environment/config, and asks Task 011 for a recovery plan.

It can resume:

- the current scenario/role/tool boundary;
- sealed-buffer offline memory/Skill unit;
- Dev Mini-Bench branch/scenario;
- generation build/publication;
- metric materialization;
- transition to the next round.

It cannot change profile, split/order, generation, token limits, prompts, model
identity, fixtures, clock, process count, or failure policy. A live external-read
`reconciliation_required` plan exits with a sanitized operator-action request and
never retries automatically.

Completed runs are read-only; resume returns their recorded result without opening
models/datasets for execution.

## Commands and Allowlist

The module CLI has explicit subcommands and no arbitrary passthrough:

```text
validate-config
validate-seeds
build-seed-skills
build-generation-zero
query-checkpoints
calibrate-online
calibrate-offline-memory
calibrate-offline-skill
train-smoke
dev-mini-bench
train
resume
```

Rules:

- no default command, `all`, shell string, Python expression, or unknown option;
- `validate-*` are offline and retrieve no secrets;
- `build-seed-skills` accepts only a pinned verified public-schema inventory and
  a new staging output root;
- `query-checkpoints` is read-only, returns G000-G003 identities/metrics and the
  explicitly non-selecting diagnostic pointers, and cannot open a dataset or
  modify a run;
- calibration accepts deterministic train IDs only and never publishes a
  generation;
- `train-smoke` accepts only the committed smoke config and train split;
- `dev-mini-bench` accepts only a Task 016 prepared candidate/selection;
- `train` requires all formal gates and exact three-round plan;
- `resume` accepts only an existing run root and identical manifest;
- output is one sanitized JSON result with status/artifact hashes/metrics or error
  class, never prompts/responses/keys/headers/stack traces.

The project-external secret launcher maps its reviewed modes to these fixed
subcommands. The CLI never invokes `secret-tool` itself.

## Train-Only Integration Smoke

`configs/run/train_smoke_v1.json` is non-formal setup evidence:

- dataset source is the real validated manifest but split is train only;
- scenario IDs are selected deterministically before outcome inspection;
- use the smallest fixed set that exercises assistant, tool, and evaluator flow;
- external reads are fixture-backed; no live RapidAPI;
- Qwen, `text-embedding-3-small`, and GPT-4o mini User Simulator use real
  configured endpoints through the launcher;
- all retries remain explicit/durable;
- no output modifies formal generations or calibrated configs;
- smoke artifacts live under a separate `runs/smoke/` namespace.

The smoke validates wiring and actual usage/latency, not model quality. It must not
be expanded or reselected based on scores. Report actual
`total_running_time_seconds`, `total_tokens`, `usage_complete`,
`total_cost` with its fixed unit, native score, final context hash, physical
attempt counts, and sanitized artifact hashes.

If a fixed smoke case reveals `finish_reason: length`, do not silently raise its
limit. Return a calibration-required failure; only the explicit train-only
calibration commands may produce a replacement config.

## Metrics

Training metrics include every physical Qwen/embedding/User request in the round,
including offline updates, Dev Mini-Bench, retries, and recovery attempts.
`total_cost` includes only Qwen outputs linked to a committed substantive effect
under Task 012 rules. NONE/SKIP/no-op and rejected Skill candidate/Mini-Bench
units contribute zero.

Round wall time is direct from immediately before first scenario dispatch through
offline work, generation publication, and durable final checkpoint. It is not a
sum of tasks/requests. Training-run time is separately direct across all three
rounds. Preflights/calibration/setup are separately reported and excluded.

Each of round 0, 1, and 2 emits exactly one immutable headline metric row with
input/output generation, shard, UTC endpoints, direct
`total_running_time_seconds`, `total_tokens`/`usage_complete`,
`total_cost`/`cost_unit`/`cost_complete`, and completion status. A failed
round still emits its terminal partial row. Do not replace these rows with an
average or only a run total.

Incomplete physical usage makes containing `total_tokens` null. Missing output
usage on a linked substantive-effect Qwen response independently makes
`total_cost` null. Do not estimate either.

## Required Offline Tests

Cover at least:

1. strict run-manifest/JSON-Schema parity, canonical hash, and every drift check;
2. secret/raw-URL/provisional config/symlink/relative root/unknown profile denial;
3. preflight evidence completeness/freshness/hash/model/mode and no implicit calls;
4. deterministic public-schema Skill builder coverage/exclusion/visibility,
   byte-identical output, provenance, and forbidden-access tests, plus all seed
   file/attestation/source/hash/order/schema/dependency validation;
5. G000 complete build/index/load and no empty fallback;
6. next-generation memory/Skill overlay, unchanged preservation, embeddings,
   indexes, and complete loader validation;
7. atomic publication permissions/fsync/rename/marker/pointer order;
8. crash before/after every publication step, identical recovery, and conflict
   denial;
9. exact round generation/shard mapping and full manifest scenario order;
10. episode fail-fast, sealed buffer, memory-before-Skill deterministic offline
    order, and G+1 publication;
11. Dev Mini-Bench inside round boundary but isolated from train updates;
12. separate direct round-0/1/2 timing and physical-token/substantive-effect cost
    aggregation, terminal partial rows, and no average/run-total substitution;
13. archive capability revocation and later-round raw trajectory denial;
14. exact three-round lifecycle, G003 final pin, and query-only G000-G003
    checkpoint registry/best-observed pointers without checkpoint selection;
15. resume at every scenario/offline/dev/publication/metric/round boundary;
16. official-live reconciliation stop and immutable completed-run behavior;
17. CLI command/argument allowlist, read-only checkpoint query, no
    default/all/passthrough, secret isolation, and sanitized JSON;
18. train-smoke fixed selection/profile restrictions/artifact namespace and no
    tuning/publication;
19. output-length smoke failure requiring separate calibration;
20. no test split opened by any Task 017 command;
21. secret/prompt/raw-response/hidden evaluator/dev trajectory/old train archive
    sentinel scans;
22. imports/construction perform no filesystem, environment, dataset, provider,
    role, evaluator, or network access.

## Deferred Runner Validation

After offline tests pass, the coordinator executes in order:

```text
validate-config
validate-seeds
separate Qwen/embedding/User preflights
required train-only token calibrations
train-smoke
optional prepared-candidate dev-mini-bench smoke
```

Each external action is separately allowlisted and returns sanitized artifacts.
The user has explicitly authorized formal `train` for the exact manifest-bound
three-round protocol recorded here. Execution still cannot begin until every
precondition and preceding validation step passes. A changed manifest, profile,
prompt, calibrated limit, split, generation seed, model identity, or environment
voids the authorization and requires a new approval.

## Acceptance Commands

```bash
uv sync --frozen
uv run pytest -q tests/orchestration tests/offline_memory tests/offline_skill tests/episode
uv run python -m toolsandbox_pipeline.orchestration.cli validate-config --help
uv run python -m toolsandbox_pipeline.orchestration.cli query-checkpoints --help
git diff --check
git status --short
```

## Acceptance Criteria

- One immutable manifest pins every formal train dependency and identity.
- G000 and each G+1 are complete, validated, and atomically published.
- Three rounds use exact generation/shard mapping and no checkpoint selection.
- G000-G003 remain queryable with diagnostic best-observed pointers that are
  explicitly non-selecting and different-shard/non-comparable.
- Every round has its own direct total latency and substantive-effect total cost
  record; none may be replaced by an average.
- Later rounds cannot read earlier raw trajectories.
- Resume is deterministic, idempotent, and fail-closed.
- CLI/secret/data access is explicitly allowlisted.
- Real smoke is train-only, non-formal, fixed, and reports actual usage/timing.
- Task 017 never opens final test data or runs final evaluation.
- Offline tests pass with zero external access and no unowned files are modified.

## Completion Report Additions

```text
Run-manifest path/SHA-256:
Preflight gate cases:
Seed/G000 cases:
Public-schema Skill builder/provenance:
G+1 build/publication cases:
Checkpoint observation registry and diagnostic pointers:
Round and three-round lifecycle cases:
Round 0 total_running_time_seconds / total_cost / completeness:
Round 1 total_running_time_seconds / total_cost / completeness:
Round 2 total_running_time_seconds / total_cost / completeness:
Archive access-denial cases:
Resume/crash boundaries:
CLI allowlist cases:
Offline tests:
External preflights: not run | results
Train-only calibration: not run | results
Train smoke: not run | results
Smoke total_running_time_seconds:
Smoke total_tokens / usage_complete:
Smoke total_cost / cost_unit / cost_complete:
Formal training authorization: granted for exact protocol, pending gates
Dependency/interface change requests:
```

Development tests and setup preflights are not formal experiment results.
