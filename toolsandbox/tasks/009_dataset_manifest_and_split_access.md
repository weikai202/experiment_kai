# Task 009: Deterministic Dataset Manifest and Split Access

Status: `approved`

## Objective

Implement the trusted setup boundary that turns the pinned upstream ToolSandbox
scenario registry into a deterministic, hash-verified dataset without exposing
sealed data to ordinary development code. This task owns:

- fixed-world-time injection for scenario construction and deterministic runtime
  helpers;
- the 129-family registry built before augmentation;
- forward expansion and validation of all eight upstream augmentation variants;
- the fixed family-level train/dev/test split and three train shards;
- canonical hashes for starting ExecutionContexts, evaluator definitions, ordered
  Agent-facing tool schemas, and environment identity;
- immutable split manifests and a deny-by-default train/dev access API;
- a privileged, coordinator-only manifest-build command that seals test metadata
  without running any scenario.

This task does not play scenarios, invoke a User Simulator, call Qwen or OpenAI,
execute a tool, evaluate a trajectory, capture/replay RapidAPI fixtures, build
retrieval indexes, update generations, implement final-test authorization, or
write experimental metrics.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3-6, 8, 16-17, 20-25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. `tasks/002_core_contracts.md`;
6. `tasks/004_toolsandbox_adapter.md`;
7. this task file;
8. only the pinned upstream public scenario registry, `Scenario`,
   `ExecutionContext`, `ScenarioCategories`, evaluator record definitions, tool
   discovery, and datetime call sites required by this task.

Do not inspect concrete scenario prompts, database contents, milestones,
minefields, target DataFrames, or test scenario IDs manually. The privileged
builder may process those values mechanically inside canonical hash functions, but
must never print or copy them into development-visible output.

## Access

```text
Access class: real_data
Setup network: none
Implementation-test data: synthetic factories only
Deferred external validation: complete pinned upstream registry, hash-only setup pass
Secrets: none
Allowed external operations: coordinator-owned build-dataset-manifest only
```

The development Agent runs unit tests with synthetic `Scenario` factories and
temporary directories. It receives no model endpoint, API key, live-tool access,
or real scenario content.

After offline tests pass, the coordinator runs the privileged manifest builder in
an isolated process. That process may construct the full upstream registry only to
assign families, validate counts, and compute hashes. It must not play or evaluate
a scenario. Its sanitized console report contains counts, manifest paths/hashes,
and pass/fail categories only—never scenario IDs, messages, databases, evaluator
targets, or tool arguments.

## Preconditions

- Tasks 001-006 are complete and accepted.
- The installed `tool-sandbox` dependency resolves exactly to commit
  `165848b9a78cead7ca7fe7c89c688b58e6501219`.
- Task 004 exposes the upstream-derived ordered Agent-facing tool schemas without
  leaking the execution-facing mapping into prompt data.
- Python 3.10, dependency lock, upstream source hashes, and project canonical JSON
  helpers are available.
- The proposed fixed world epoch below must be approved before this task changes
  from `draft-for-review` to `approved`.

## Owned Files

The assigned Agent may create or edit only:

```text
configs/reproducibility/dataset_build_v1.json
src/toolsandbox_pipeline/schemas/dataset.py
src/toolsandbox_pipeline/reproducibility/clock.py
src/toolsandbox_pipeline/reproducibility/scenario_hashes.py
src/toolsandbox_pipeline/reproducibility/splits.py
src/toolsandbox_pipeline/reproducibility/dataset_manifest.py
src/toolsandbox_pipeline/reproducibility/dataset_access.py
src/toolsandbox_pipeline/reproducibility/dataset_manifest_cli.py
tests/reproducibility/test_clock.py
tests/reproducibility/test_scenario_hashes.py
tests/reproducibility/test_splits.py
tests/reproducibility/test_dataset_manifest.py
tests/reproducibility/test_dataset_access.py
tests/reproducibility/test_dataset_manifest_cli.py
```

Do not edit the upstream package, dependency files, prompts, generation records,
retrieval, provider gateways, online runners, Controller, Adapter, fixture store,
checkpointing, experiment CLI, or generated real-data manifests.

## Versioned Build Configuration

`configs/reproducibility/dataset_build_v1.json` is strict, hashable, and contains:

```yaml
schema_version: 1
upstream_commit: 165848b9a78cead7ca7fe7c89c688b58e6501219
preferred_tool_backend: DEFAULT
data_seed: 0
test_fraction: 0.20
dev_fraction: 0.20
num_update_rounds: 3
world_epoch_iso: "2024-05-01T12:00:00Z"
world_epoch_unix: 1714564800.0
timezone: UTC
locale: C.UTF-8
clock_adapter_version: toolsandbox-fixed-clock-v1
family_registry_order:
  - single_tool_call
  - multiple_tool_call
  - multiple_user_turn
  - insufficient_information
variant_order:
  - no_distraction
  - three_distraction_tools
  - ten_distraction_tools
  - all_tools
  - three_distraction_tool_description_scrambled
  - three_distraction_argument_type_scrambled
  - three_distraction_argument_description_scrambled
  - three_distraction_tool_name_scrambled
```

The two epoch representations must describe the identical instant. Unknown fields,
numeric coercion, a non-UTC timezone, a different locale/backend, reordered registry
or variants, nonzero seed, or changed split constants fail validation. The build
records the configuration path and exact file SHA-256.

The fixed epoch is environment state visible to date-dependent tools and scenario
construction. It is not telemetry time: request latency and total running time use
the real monotonic clock and real UTC audit timestamps.

## Fixed World Clock

`clock.py` implements an explicit context manager; importing the module has no
effect. Entering it must:

1. require the exact build configuration;
2. require the process timezone and locale selected by the coordinator to match the
   config before scenario construction;
3. patch the pinned upstream datetime references used by base scenario creation,
   current timestamp, current year/date canonicalization, messaging timestamps,
   and reminder timestamps;
4. return the same naive UTC wall time wherever upstream code calls
   `datetime.datetime.now()` and preserve ordinary `timedelta`, `fromtimestamp`,
   and explicit timestamp conversion semantics;
5. leave `time.monotonic()`, `perf_counter()`, real audit UTC timestamps, provider
   SDKs, and project metrics unpatched;
6. restore every patched object even when construction fails;
7. reject nested entry, wrong upstream module identity, missing/extra audited clock
   site, or concurrent use from another thread.

The implementation maintains an explicit allowlist of upstream module attributes
and verifies their identities before patching. It may not globally monkeypatch the
standard-library `datetime` module or modify upstream source files. A pinned-source
clock-site audit test fails if the upstream files contain an unaccounted direct
`datetime.now()`/timestamp/current-year use relevant to scenario construction or
ToolSandbox tools.

## Family Registry and Variant Expansion

Build the nominal registry before calling upstream `named_scenarios()`:

1. enter the fixed-clock context;
2. invoke the four `named_*_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)`
   factories directly exactly once each in configured registry order;
3. preserve each returned mapping's insertion order;
4. reject duplicate, empty, non-UTF-8, or non-string family IDs;
5. require exactly 129 unique families;
6. record each family ID and source-registry label internally.

Then construct the augmented registry under the same clock and a protected global
random-state context:

1. save the caller's complete `random` state;
2. seed Python 3.10 module-level `random` with `data_seed=0` immediately before the
   single upstream `named_scenarios()` call; that upstream function necessarily
   invokes the four factories internally a second time, but those results cannot
   redefine the already-recorded family registry;
3. restore the caller state in `finally`;
4. forward-generate the exact eight expected scenario IDs from each already-known
   family ID and explicit variant suffix table;
5. validate the upstream insertion order exactly: all 129 unaugmented IDs in family
   registry order, followed by each family's seven derived IDs in upstream
   augmentation order;
6. build each split's canonical scenario order separately as shuffled manifest
   family order multiplied by the configured eight-entry `variant_order`;
7. compare identities and orders with upstream output and reject missing, extra,
   duplicated, or reordered variants.

Forward generation from the pre-augmentation registry is required. Recovering a
family by stripping or heuristically parsing an augmented scenario ID is forbidden.
Every scenario must carry the variant's exact required augmentation category while
retaining its native task categories. Family and scenario order are immutable.

## Fixed Family Split and Train Shards

Implement the Section 5 algorithm literally with Python 3.10 semantics:

1. sort family IDs by UTF-8 bytes;
2. copy the sequence and shuffle it once with a private
   `random.Random(data_seed).shuffle`;
3. assign the first `floor(0.20 * N)` families to test;
4. assign the next `floor(0.20 * N)` families to dev;
5. assign the rest to train;
6. split shuffled train families into contiguous `27/26/26` family shards;
7. expand variants only from the forward mapping after family assignment.

Hard requirements for the audited commit are:

```yaml
families: 129
train_families: 79
dev_families: 25
test_families: 25
scenarios: 1032
train_scenarios: 632
dev_scenarios: 200
test_scenarios: 200
train_round_scenarios: [216, 208, 208]
```

No family may cross a split or train shard. Sorting or shuffling scenarios instead
of families is a hard error.

## Canonical Scenario Hashes

`scenario_hashes.py` produces hash-only identities and never logs its input.

### Starting ExecutionContext

Deep-copy the starting context and serialize through the pinned upstream
`ExecutionContext.to_dict(serialize_console=False)`. Canonicalize explicitly:

- database namespaces in enum order;
- each Polars table with column names, exact dtypes, and rows in existing order;
- role/category/backend enums by their pinned string value;
- tool allow/deny/augmentation lists in existing order;
- `trace_tool` and all other serialized context flags;
- JSON scalar types without float coercion or non-finite values;
- `interactive_console` as the explicit excluded/null sentinel.

Hash canonical JSON bytes. Round-tripping through
`ExecutionContext.from_dict()` and hashing again must match. Do not use `repr`,
pickle/dill bytes, object IDs, or unordered dataframe conversion as an identity.

### Evaluator definition

Hash a canonical private projection of both native matcher definitions, preserving:

- milestone versus minefield type and list order;
- edge lists exactly;
- ordered snapshot constraints;
- database namespace and reference milestone index;
- target DataFrame columns, dtypes, and ordered rows;
- snapshot-constraint and column-similarity callable identities as
  `module:qualname`, restricted to the pinned upstream package;
- guardrail database lists in their stored order.

Reject lambdas, closures, partials, builtins, external callables, unknown attrs
fields, unqualified callables, non-finite data, or values that canonical JSON cannot
represent. The upstream commit/source manifest binds callable implementations; no
callable source, target DataFrame, milestone, or minefield content is written to a
development-visible manifest.

### Agent-facing tools and environment

For each deep-copied starting context, use Task 004's upstream-backed schema path
under the active context to hash the complete ordered Agent-facing tool schema
list. Separately hash the ordered Agent-facing tool names. Never derive schemas
from canonical metadata or unaugmented signatures.

The environment identity hashes the upstream commit, dependency-lock hash, Python
patch version, platform, timezone, locale, build-config hash, clock adapter version,
and preferred tool backend. Formal Linux/container image identity may initially be
unresolved, but unresolved identity marks the manifest `setup_only` and blocks a
formal run.

## Manifest Layout and Visibility

The privileged builder atomically writes a private manifest bundle supplied by an
explicit absolute output directory:

```text
dataset_index.json
train_manifest.json
dev_manifest.json
sealed/test_manifest.json
```

It creates no default output path and never writes into `configs/` implicitly.
Every file uses canonical JSON, mode `0600`, and an fsync-before-rename atomic
write. `dataset_index.json` records each split-manifest SHA-256. The index's own
SHA-256 is computed after writing and returned to the coordinator/run manifest; it
cannot be embedded in itself. Existing files are never overwritten; an identical
rerun verifies and reuses them, while any byte mismatch fails.

`dataset_index.json` contains configuration/environment identities, counts, and
the three split-manifest hashes. It contains no scenario or family IDs. Each split
manifest contains ordered family/scenario records with:

```text
family_id
source_registry
split
train_shard, nullable
ordered_variant_scenario_ids

scenario_id
scenario_family_id
variant
categories
max_messages
starting_context_sha256
evaluation_definition_sha256
agent_facing_tool_names_sha256
agent_facing_tool_schema_sha256
```

Manifests contain hashes, IDs, enum labels, counts, and ordering only. They contain
no message text, database row, target DataFrame, evaluator rule, tool argument,
tool result, callable source, or secret.

The sanitized builder report exposes only overall/split/shard counts, index hash,
per-file hash, environment-complete flag, duration, and status. It must not expose
scenario IDs or per-scenario hashes in stdout/stderr.

## Dataset Access Gate

`dataset_access.py` defines strict purposes:

```text
development
retrieval_smoke
online_token_calibration
train_round
skill_ab_validation
```

Rules:

- train permits `development`, `retrieval_smoke`, `online_token_calibration`, and
  `train_round`;
- dev permits only `skill_ab_validation`;
- no ordinary public loader accepts `test`;
- callers must provide an exact manifest hash, non-empty ordered requested IDs,
  run/phase identity, and—for train rounds—the exact shard;
- every requested ID must belong to the manifest and purpose-allowed shard;
- selection preserves manifest order and rejects duplicate, unknown, reordered,
  cross-family, cross-split, or cross-shard input;
- hashes are recomputed from freshly constructed deep copies before a scenario is
  returned;
- any mismatch fails before returning the first scenario;
- a returned trusted-host lease is immutable, has no JSON/model serializer, and
  contains a deep-copied native Scenario plus its manifest record;
- the lease must never be passed to a Policy, Critic, Revision, retrieval query, or
  User Simulator. Later orchestration passes only Task 004/008 projections.

Every successful or rejected request emits a sanitized audit record containing
timestamp, split, purpose, run/phase, manifest hash, requested-count, and a hash of
the ordered requested-ID list. It does not log raw scenario IDs or scenario
content. Audit emission is injected; this task does not choose a run directory.

Test manifest loading is deliberately absent. A later final-evaluation task may
consume `sealed/test_manifest.json` only after the durable one-time ledger and
three-system configuration equality checks are implemented. Do not add a hidden
flag, environment override, or private convenience function that bypasses this
boundary.

## Privileged Manifest CLI

The only command is:

```bash
uv run python -m toolsandbox_pipeline.reproducibility.dataset_manifest_cli \
  build-dataset-manifest \
  --config /absolute/path/to/dataset_build_v1.json \
  --output-dir /absolute/new/private/manifest-directory
```

It rejects relative paths, symlinks, an existing non-identical bundle, additional
arguments, arbitrary callable/factory names, split selection, scenario selection,
and output inside the repository's tracked source/config/task directories. It has
no `show`, `list`, `dump`, `cat`, or test-content inspection command.

The project-external launcher may later allowlist this exact setup command without
retrieving any credential. It must execute in a fresh process with pinned `TZ` and
locale.

## Required Tests

Use only synthetic scenario factories and temporary directories. Cover at least:

1. exact config constants, epoch equivalence, strict types, and frozen models;
2. fixed datetime/current-year behavior plus untouched monotonic/telemetry clocks;
3. restoration on success/failure and rejection of nested/concurrent clock use;
4. audited clock-site allowlist drift detection;
5. four-registry order, duplicate detection, and family construction before
   augmentation;
6. forward variant mapping and rejection of suffix-based/extra/missing variants;
7. Python 3.10 family shuffle golden vector and exact 79/25/25 assignment;
8. exact 27/26/26 train shards and 216/208/208 expanded sizes;
9. no family crossing split/shard and stable manifest order;
10. context hash stability, round-trip verification, dtype/row/list-order
    sensitivity, and interactive-console exclusion;
11. evaluator hash stability and sensitivity to every permitted field;
12. rejection of unsafe callables, hidden-content serialization, non-finite values,
    and unsupported upstream shape drift;
13. Task 004 Agent-facing augmented schema hashing, including scrambled variants;
14. atomic bundle creation, permissions, idempotent identical rerun, and
    non-overwrite on mismatch;
15. sanitized CLI output and absence of IDs, messages, rows, targets, arguments, or
    evaluator definitions;
16. train/dev purpose matrix and absence of any ordinary test loader;
17. rejection before return for wrong hash, purpose, split, shard, order, duplicate,
    unknown ID, or recomputed scenario mismatch;
18. deep-copy isolation and sanitized access-audit records;
19. imports perform no scenario construction, clock patch, random mutation,
    filesystem write, environment read, network access, or model/client creation.

## Deferred Real-Registry Validation

After offline tests pass, submit:

```text
Runner request:
Task: 009
Access class: real_data
Allowlisted mode: build-dataset-manifest
Versioned config: configs/reproducibility/dataset_build_v1.json
Scenario split and IDs: full registry; IDs hidden from returned report
Expected artifacts: private immutable index plus train/dev/sealed-test manifests
Reason: validate pinned upstream counts, hashes, split, and deterministic rebuild
```

The coordinator runs two fresh-process builds into separate new private
directories. Validation passes only if:

- both builds report 129 families and 1,032 scenarios;
- split/scenario/shard counts match Section 5 exactly;
- all four bundle files are byte-identical between builds;
- all corresponding hashes match;
- no scenario is played or evaluated;
- sanitized stdout/stderr contains no raw scenario/family ID or hidden content;
- no network, model, embedding, User Simulator, or tool call occurs.

If these prerequisites or the approved world epoch are unavailable, code may be
reported offline-complete but Task 009 remains externally unvalidated.

## Acceptance Commands

The development Agent runs:

```bash
uv sync --frozen
uv run pytest -q tests/reproducibility/test_clock.py tests/reproducibility/test_scenario_hashes.py tests/reproducibility/test_splits.py tests/reproducibility/test_dataset_manifest.py tests/reproducibility/test_dataset_access.py tests/reproducibility/test_dataset_manifest_cli.py
uv run python -c "from toolsandbox_pipeline.reproducibility.dataset_access import DatasetAccessGate"
git diff --check
git status --short
```

No development-Agent acceptance command may construct the real upstream scenario
registry, read a real split manifest, access a model/API/network, execute a tool,
or run the evaluator.

## Acceptance Criteria

- Fixed-time construction is explicit, scoped, restored, and separate from real
  monotonic experiment timing.
- Family identity is registered before augmentation and never inferred by suffix.
- Counts, family-level split, train shards, variant order, and hashes exactly match
  the approved algorithm.
- Manifests preserve upstream identities without containing hidden dataset content.
- Ordinary code can access only purpose-allowed train/dev scenarios and has no test
  loader.
- Two deferred real-registry builds are byte-identical or the outstanding external
  validation is reported explicitly.
- No scenario is played/evaluated and no model, API, live tool, or secret is used.
- No unowned or upstream file is modified.

## Completion Report Additions

Include:

```text
Build-config path and SHA-256:
World epoch/timezone/clock adapter:
Audited upstream clock sites:
Synthetic family/variant/split cases:
Context/evaluator/tool-schema hash cases:
Dataset access denial cases:
Real registry validation: pass | fail | not run
Family/scenario/split/shard counts:
Rebuild byte identity: pass | fail | not run
Private manifest paths and SHA-256 hashes:
Environment identity complete:
Sanitized-output check:
External validation blockers:
Dependency/interface change requests:
```

This is setup validation, not a training round or evaluation. Report its wall time
separately as setup duration. It makes no model calls, so `Total tokens: 0` and
`Usage complete: true` when the builder itself completes.
