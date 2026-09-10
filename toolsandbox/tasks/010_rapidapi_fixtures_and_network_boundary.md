# Task 010: RapidAPI Fixtures and External-Read Network Boundary

Status: `approved`

## Objective

Implement the only permitted network boundary for ToolSandbox's five native
RapidAPI-backed `external_read` tools:

```text
convert_currency
search_lat_lon
search_location_around_lat_lon
search_stock
search_weather_around_lat_lon
```

This task owns:

- strict identification of the five pinned upstream backend requests;
- canonical fixture keys from effective backend arguments;
- immutable, hash-pinned fixture bundles;
- replay with zero credential/environment/network access;
- a separate coordinator-authorized capture path;
- audited live dispatch for a later `official_live` profile;
- deterministic fixture-miss behavior and sanitized external-read evidence;
- scoped installation/restoration of the boundary around native tool execution.

The five public tool functions, validation, current-location resolution, response
post-processing, decorators, schemas, and native ExecutionEnvironment remain
unchanged. This task replaces only the internal raw JSON fetch performed through
the pinned upstream `rapid_api_get_request()` function.

It does not select or play scenarios, build dataset manifests, route Agent actions,
run tools itself, capture fixtures automatically on a miss, implement checkpoints,
retry unknown live outcomes, call models, calculate model tokens, or authorize
the one-time final evaluation.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 3-6, 9, 12, 17, and 20-25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. `docs/SECRET_INJECTION.md`;
5. `tasks/002_core_contracts.md`;
6. `tasks/004_toolsandbox_adapter.md`;
7. `tasks/005_tool_metadata_controller.md`;
8. `tasks/009_dataset_manifest_and_split_access.md`;
9. this task file;
10. only the pinned upstream `rapid_api_search_tools.py`, tool registration, and
    ExecutionEnvironment exception behavior required for this boundary.

Do not inspect scenario definitions, prompts, databases, evaluator definitions,
milestones, minefields, target DataFrames, dev/test IDs, or prior trajectories.

## Access

```text
Access class: official_live
Setup network: none
Implementation-test data: synthetic request/response fixtures only
Default tests: no environment, credential, DNS, socket, or HTTP access
Deferred external validation: coordinator-authorized RapidAPI capture probe only
Secrets: RAPID_API_KEY in external runner only
```

The development Agent receives no real key and injects fake transports. Task code
must be offline-complete and fully tested without a credential.

Live capture is a separate operator-authorized setup action. Selecting
`strict_replay`, encountering a fixture miss, running a train smoke, or importing a
module can never retrieve `RAPID_API_KEY` or contact RapidAPI. `official_live`
dispatch is available only to a later coordinator-owned episode runner with the
exact reviewed profile and run manifest.

## Preconditions

- Tasks 001-006 and 009 are complete and accepted.
- Task 005 identifies exactly these five tools as `external_read` and no tool as
  `external_write`.
- The installed upstream dependency is commit
  `165848b9a78cead7ca7fe7c89c688b58e6501219`.
- The upstream module exposes one callable `rapid_api_get_request(url, params,
  headers)` used by all five external-read tools.
- The later checkpoint/orchestrator task will inject run/scenario/state/call
  identity and decide recovery; this task does not invent a second execution
  ledger.

## Owned Files

The assigned Agent may create or edit only:

```text
configs/reproducibility/rapidapi_backends_v1.json
src/toolsandbox_pipeline/schemas/fixtures.py
src/toolsandbox_pipeline/reproducibility/fixture_store.py
src/toolsandbox_pipeline/reproducibility/rapidapi_boundary.py
src/toolsandbox_pipeline/reproducibility/fixture_capture.py
src/toolsandbox_pipeline/reproducibility/fixture_cli.py
tests/reproducibility/test_fixture_schemas.py
tests/reproducibility/test_fixture_store.py
tests/reproducibility/test_rapidapi_boundary.py
tests/reproducibility/test_fixture_capture.py
tests/reproducibility/test_fixture_cli.py
```

Do not edit upstream code, dependencies, dataset manifests, Task 004 adapter,
Controller metadata, provider gateways, secret launcher, prompts, online/offline
runners, checkpointing, metrics aggregation, or generated fixture bundles.

## Versioned Backend Manifest

`configs/reproducibility/rapidapi_backends_v1.json` is a committed non-secret
contract containing exactly:

```text
schema_version
adapter_version
upstream_commit
request_timeout_seconds
allow_redirects = false
backend records in canonical tool-name order
```

Each backend record contains:

```text
canonical_tool_name
backend_version
method = GET
exact HTTPS URL
exact X-RapidAPI-Host value
ordered permitted parameter names
```

The five records must exactly match the pinned upstream endpoints and request
shapes. URLs, hosts, method, timeout, redirects, backend versions, or parameter
sets cannot be overridden by an Agent action, environment variable, fixture, or
per-call option. Unknown/duplicate tools, hosts, endpoints, or parameters fail
before network or fixture lookup.

The manifest contains no API key, authorization header, response, scenario ID, or
capture request. Runtime verifies its exact file SHA-256 and upstream commit before
installing any boundary.

## Interception Boundary

`rapidapi_boundary.py` installs an explicit context manager around native
ExecutionEnvironment tool execution. Importing or constructing it does not patch
anything.

On entry it must:

1. verify the exact upstream module and `rapid_api_get_request` callable identity;
2. verify through a pinned-source audit that all five external-read tools call that
   boundary and no additional Agent-visible tool does;
3. replace only the upstream module's `rapid_api_get_request` attribute with the
   selected dispatcher;
4. replace only that module's `requests` reference with a rejecting proxy so a new
   direct request call cannot bypass the dispatcher;
5. reject nested entry, concurrent thread use, wrong process/profile, unknown mode,
   or upstream source/identity drift;
6. restore both upstream module attributes in `finally` without changing the
   process-global `requests` package used by model clients.

The context is process-scoped and non-reentrant. Scenario-level parallelism may
use separate processes only. It must never globally monkeypatch sockets,
`requests.Session`, OpenAI transports, or the standard library.

The installed native-tool boundary supports exactly these strict modes:

```text
replay
official_live
```

Capture is not a native-tool boundary mode; it exists only as the standalone
approved-request CLI below. No `auto`, `fallback`, `prefer_fixture`, episode
capture, or combined capture-on-miss mode exists.

## Effective Backend Request and Fixture Key

The dispatcher receives the upstream `url`, `params`, and non-secret host header
after native tool validation and current-location/default resolution. It maps the
exact URL/host pair to one canonical tool and validates the complete parameter
shape against the backend manifest.

The fixture `arguments` field is the canonical JSON representation of those
effective backend parameters, not the model-produced action envelope. This ensures
that omitted latitude/longitude resolved from ToolSandbox settings and fixed-world
date inputs are part of fixture identity.

Compute:

```text
fixture_key = "sha256:" + SHA256(canonical JSON of
  {
    "tool_name": canonical_tool_name,
    "arguments": effective_backend_parameters,
    "backend_version": backend_version
  }
).hexdigest()
```

Preserve exact JSON scalar types, Unicode, list order, and object keys. Reject
bytes, tuples, sets, enums not explicitly normalized, NaN/infinity, duplicate JSON
keys, unsupported nested values, secret/header-like keys, and values outside the
backend contract. `1` and `1.0` remain different canonical arguments.

The dispatcher obtains an injected immutable `ExternalReadContext` immediately
before work. It contains run/profile/phase/family/scenario/state/call identity and
the pinned fixture/backend manifest hashes. Missing, stale, or changed context
fails before lookup/dispatch.

## Fixture Entry and Bundle

`schemas/fixtures.py` defines frozen strict models for:

```python
ExternalReadContext
FixtureRequest
FixtureEntry
FixtureManifest
FixtureMissEvidence
ExternalReadAttempt
```

A fixture entry contains exactly:

```text
schema_version
fixture_key
canonical_tool_name
effective_arguments
backend_version
request_url_identity
status_code
normalized_response_body
response_body_sha256
captured_at_utc
source = rapidapi_capture
```

`captured_at_utc` is provenance and excluded from fixture content identity. The
response identity hashes canonical normalized JSON, not formatting-dependent HTTP
bytes. A fixture never contains headers, API keys, cookies, request IDs from the
provider, environment values, or raw exception messages.

Bundle layout is:

```text
fixtures/<fixture_generation_id>/
  manifest.json
  entries/
    <64 lowercase hex fixture key>.json
```

The manifest records schema/adapter/backend versions, upstream commit, backend
manifest hash, entry count, ordered fixture keys, each entry file hash, canonical
entry-set hash, creation timestamp, and publication status. The generation ID is
derived from the canonical entry-set identity, never a mutable label.

Load requires a complete immutable manifest, exact hashes, filename/key agreement,
mode `0600`, no symlink/path escape, sorted unique keys, and no unexpected file.
Any malformed/missing/extra entry, body hash mismatch, backend drift, or partial
publication stops before returning a store.

Capture writes a new staging bundle and publishes it atomically only after every
entry validates. It never modifies an existing bundle in place. Identical content
reuses the same generation only after byte verification; conflicting bytes under
the same identity are a hard error.

## Replay Mode

Replay mode:

1. does not read `RAPID_API_KEY`, any environment variable, DNS, socket, or HTTP
   client;
2. computes the exact fixture key and performs one lookup in the pinned immutable
   store;
3. on a hit, returns a deep copy of `normalized_response_body` to the unchanged
   upstream tool function for native post-processing;
4. records one `ExternalReadAttempt` with status `fixture_hit`, fixture key,
   manifest identity, real monotonic lookup latency, and no response content;
5. on a miss, records status `fixture_miss`, emits injected host-only
   `FixtureMissEvidence`, and raises an exception whose only public message is
   `EXTERNAL_FIXTURE_MISS`.

Miss evidence contains the external-read context, fixture key, canonical tool,
backend version, and effective arguments so a coordinator may create a separate
capture request. It is restricted host evidence and must never enter an Agent
message, Policy/Critic prompt, ordinary log, completion report, or sanitized CLI
output. The ExecutionEnvironment-visible exception contains only the fixed public
code.

A fixture miss never contacts a network, changes mode, reads a key, creates an
entry, or retries. Later orchestration checkpoints and stops the setup/formal run.

## Explicit Capture Mode

Capture is a standalone setup operation, not an episode fallback. It consumes an
immutable coordinator-reviewed `FixtureCaptureRequestManifest` from an absolute
path. The manifest contains exact effective request envelopes and their hashes; the
CLI accepts no tool name, URL, host, parameter, or argument directly.

Before retrieving a credential, capture validates:

- input manifest schema and content hash;
- the exact expected request-manifest SHA-256 supplied by the project-external
  launcher from its operator-maintained approval allowlist;
- backend/config/upstream identities;
- an exact capture purpose of `synthetic_contract_probe` or
  `train_fixture_preparation`, with the latter bound to a train-manifest hash and
  restricted fixture-miss audit hash;
- unique ordered requests and exact expected fixture keys;
- a new absolute staging/output directory;
- maximum request count and configured rate/concurrency limits.

The external runner then injects `RAPID_API_KEY`. The capture transport:

1. sends exactly one HTTPS GET per approved request with the manifest URL/host,
   effective params, key header, finite timeout, and redirects disabled;
2. uses no SDK/HTTP automatic retry;
3. records real monotonic latency, sanitized exception class, status code, and
   response byte count;
4. parses the body once as strict JSON with duplicate-key/non-finite rejection;
5. normalizes and validates it without applying tool-specific post-processing;
6. requires HTTP status 200 for a publishable fixture, then builds the immutable
   entry and response-body hash;
7. publishes a complete new fixture generation only if every approved request
   succeeds and validates.

Timeout, connection loss, or an exception after dispatch is `unknown_outcome` and
is never retried automatically. No partial fixture bundle is published. The
staging audit retains attempt metadata without secret/header/body logging.

The sanitized capture report contains request/success/failure counts, fixture
generation/hash, total running time, per-status counts, latency aggregates, and
sanitized error classes. It contains no effective arguments, URLs, bodies, keys,
headers, raw exception text, scenario IDs, or fixture-miss evidence.

Capture authorization is intentionally unresolved until the operator approves the
RapidAPI account, allowed request manifest, rate limits, and data-retention policy.
Offline implementation does not grant that authorization.

## Audited `official_live` Mode

Later coordinator-owned experiment orchestration may select `official_live` only
with a complete run manifest and injected `RAPID_API_KEY`. It uses the same exact
backend validation and one-dispatch transport as capture, but returns normalized
JSON directly to the current native tool and does not add or update a fixture
bundle.

Each call emits an immutable `ExternalReadAttempt` with logical tool call identity,
status, status code, latency, response-body hash when available, and sanitized
exception class. Bodies and arguments remain in restricted trajectory/context
artifacts only where later specifications permit them.

An `official_live` call never reads or falls back to a fixture. An unknown outcome
is returned to later checkpoint orchestration and is not automatically retried or
reconciled by this task.

## Fixture Preparation Workflow

The approved workflow for a missing replay entry is:

```text
replay run reaches an external read
→ deterministic fixture miss, restricted evidence, checkpoint, stop
→ coordinator reviews a separate capture-request manifest
→ explicitly authorized capture publishes a new fixture generation
→ original setup run is restarted from its beginning with the new pinned store
```

Do not resume the same run under a changed fixture manifest. A train smoke or token
calibration is setup evidence and may be restarted. A formal strict-replay test
cannot capture after seeing a miss and cannot be rerun; fixture completeness must
be established before the one-time test begins.

This task guarantees fixture identity and enforcement, not dataset-wide fixture
completeness. A complete strict-replay test remains blocked until the coordinator
can prove that its pinned fixture store covers every external request that may
occur without inspecting or tuning on test trajectories.

## CLI Contracts

Local validation:

```bash
uv run python -m toolsandbox_pipeline.reproducibility.fixture_cli \
  preflight-fixtures \
  --backend-config /absolute/path/to/rapidapi_backends_v1.json \
  --fixture-manifest /absolute/path/to/manifest.json
```

Authorized capture:

```bash
uv run python -m toolsandbox_pipeline.reproducibility.fixture_cli \
  capture-fixtures \
  --backend-config /absolute/path/to/rapidapi_backends_v1.json \
  --request-manifest /absolute/path/to/approved_capture_requests.json \
  --expected-request-manifest-sha256 sha256:<operator-approved-digest> \
  --output-dir /absolute/path/to/new_fixture_bundle
```

The CLI has no arbitrary command passthrough, direct URL/argument flags, fixture
editing, deletion, body display, key lookup, replay-to-network fallback, or
`capture-missing` operation. `preflight-fixtures` never retrieves a credential.
Only the project-external launcher may invoke `capture-fixtures` with a key. Its
local allowlist maps an approval identity to the exact request-manifest hash,
backend-manifest hash, purpose, maximum request count, and permitted output root;
the capture process must not trust an approval claim embedded only in its input
manifest.

## Required Tests

Use synthetic parameters, bodies, fake transports, fake context providers, and
temporary directories. Cover at least:

1. exact five-tool backend manifest, endpoint/host/parameter identity, strict
   types, and unknown/override rejection;
2. fixture-key golden vectors, exact numeric/Unicode/list semantics, and key
   sensitivity to tool, arguments, and backend version;
3. rejection of secrets/header-like arguments, duplicate JSON keys, non-finite or
   unsupported values;
4. strict fixture/manifest validation, body and file hashes, permissions, path
   safety, unexpected files, and immutable publication;
5. explicit scoped installation, upstream callable/source audit, restoration on
   success/failure, and nested/concurrent rejection;
6. proof that replacing the upstream module's `requests` reference does not change
   the process-global module or provider transports;
7. all five unchanged upstream public functions receiving replay JSON and applying
   their native post-processing correctly using synthetic bodies;
8. current-location/default resolution influencing the effective fixture key;
9. replay hit with exactly one lookup, deep-copy isolation, and zero environment,
   credential, DNS, socket, or HTTP access;
10. replay miss fixed public exception, host-only evidence, no fallback/write, and
    no sensitive console/log content;
11. one fake capture/live dispatch with exact method/timeout/redirect/header
    contract and no retry;
12. status code/body/latency/hash attempt records and sanitized transport failures;
13. timeout/connection unknown outcomes and rejection-before-dispatch distinction;
14. all-or-nothing capture publication and no partial bundle after any failure;
15. CLI mode isolation, absolute paths, sanitized reports, and absence of body,
    arguments, URL, headers, or key material;
16. imports/construction perform no patch, file write, environment read, credential
    lookup, network access, dataset load, tool call, or model/client creation.

## Deferred External Validation

After offline tests pass, two validations are separate:

### Fixture-store preflight

The coordinator may run `preflight-fixtures` against any prepared bundle. It uses
no credential or network and reports only status, entry count, bundle/backend
hashes, validation time, and sanitized error class.

### Capture contract probe

Run only after explicit operator authorization:

```text
Runner request:
Task: 010
Access class: official_live
Allowlisted mode: capture-fixtures
Secret: RAPID_API_KEY only
Request manifest: coordinator-approved bounded synthetic probe
Expected artifacts: one new immutable fixture bundle and sanitized capture audit
Reason: validate the real RapidAPI transport/capture contract
```

The probe must not use a train/dev/test scenario, user-derived data, or a request
chosen from sealed test behavior. A missing key or capture authorization is
reported as external validation outstanding; it does not weaken replay tests.

## Acceptance Commands

The development Agent runs:

```bash
uv sync --frozen
uv run pytest -q tests/reproducibility/test_fixture_schemas.py tests/reproducibility/test_fixture_store.py tests/reproducibility/test_rapidapi_boundary.py tests/reproducibility/test_fixture_capture.py tests/reproducibility/test_fixture_cli.py
uv run python -c "from toolsandbox_pipeline.reproducibility.rapidapi_boundary import RapidAPIBoundary"
git diff --check
git status --short
```

No development-Agent acceptance command may access a real environment credential,
network, scenario, split manifest, tool execution, evaluator, or model API.

## Acceptance Criteria

- Exactly five pinned native external reads cross one reviewed raw-JSON boundary.
- Replay is credential-free and network-impossible, with deterministic miss
  behavior and no automatic capture/fallback.
- Native ToolSandbox argument validation, default resolution, tool schemas, and
  response post-processing remain unchanged.
- Fixture keys, entries, bundles, attempts, and publication are strict, immutable,
  canonical, and hash-verified.
- Capture/live make one finite, non-retried request and never expose secrets,
  arguments, bodies, URLs, headers, or raw provider errors in sanitized output.
- Unknown live outcome is preserved for later checkpoint recovery rather than
  retried here.
- Fixture completeness limitations are explicit and test data cannot be used to
  fill a store after evaluation begins.
- Offline tests pass; unavailable store/capture validation is reported as
  outstanding rather than simulated.
- No unowned or upstream file is modified.

## Completion Report Additions

Include:

```text
Backend-config path and SHA-256:
Upstream boundary/source audit:
Fixture-key golden vectors:
Fixture bundle generation/path/SHA-256:
Fixture entries validated:
Replay hit/miss cases:
Network-bypass denial cases:
Fake capture/live attempts:
Fixture preflight: pass | fail | not run
Real capture probe: pass | fail | not run
Capture request/success/failure counts:
Capture total running time seconds:
Capture tokens: 0
Capture usage complete: true
Sanitized-output check:
Fixture completeness status:
External validation blockers:
Dependency/interface change requests:
```

Fixture preparation and capture timing are setup evidence, not a training round or
evaluation result. RapidAPI uses no model tokens; do not combine its request counts
with LLM call counts.
