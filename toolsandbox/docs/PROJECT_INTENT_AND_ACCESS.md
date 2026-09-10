# Project Intent, Boundaries, and Runtime Access

Status: `approved`

This document defines what the standalone `toolsandbox/` project is allowed to build and which external systems each development or experiment phase may access. It is subordinate to `pipeline.md`. If this document, an Agent task file, and `pipeline.md` disagree, implementation stops and the coordinator resolves the specification conflict before code changes continue.

## 1. Project Intent

Build an auditable implementation of the ToolSandbox Policy-Controller-Critic pipeline specified in `pipeline.md` while preserving Apple ToolSandbox's native scenarios, role visibility, execution environment, state semantics, tool augmentations, and evaluator.

The implementation must:

- run as an independent Python project rooted at `toolsandbox/`;
- use frozen `Qwen/Qwen3-32B` weights for every pipeline LLM role;
- use only OpenAI `text-embedding-3-small` for semantic retrieval;
- support deterministic local tests without model, GPU, API-key, or network access;
- keep per-round total running time, total token usage, non-monetary
  substantive-effect Qwen-output cost, hashes, recovery decisions, and
  experimental artifacts auditable;
- preserve train/dev/test isolation and the online visibility boundary in `pipeline.md`.

## 2. Explicit Non-Goals

This project does not:

- train, fine-tune, merge, quantize, or modify model weights;
- modify the pinned upstream Apple ToolSandbox source;
- replace or reinterpret ToolSandbox's native evaluator;
- import implementation code or runtime state from `bfcl-online` or another parent/sibling project;
- expose hidden databases, milestones, minefields, evaluator mappings, or target DataFrames to any online component;
- provide a general production agent service;
- require live external services for default development tests.

## 3. Standalone Project Boundary

The project root is the directory containing this file's parent `toolsandbox/`. All package imports, configuration paths, test paths, caches, checkpoints, artifacts, and run outputs must resolve from this project root or from an explicit absolute path supplied by configuration. `uv` is the only project environment and dependency manager. It creates `toolsandbox/.venv` from the committed `pyproject.toml` and `uv.lock`; project commands use `uv run`, and installation uses `uv sync --frozen`.

The implementation package is named `toolsandbox_pipeline`. The upstream dependency keeps its own `tool_sandbox` package name. Code must not modify `sys.path`, depend on the current shell directory, or use `PYTHONPATH` to import code from outside this project.

Apple ToolSandbox is installed as a non-editable Git dependency pinned and locked at commit:

```text
165848b9a78cead7ca7fe7c89c688b58e6501219
```

It is not a Git submodule and its source is not copied into this project. All integration and compatibility code belongs under `src/toolsandbox_pipeline/toolsandbox_adapter/`.

## 4. Access Matrix

| Activity | Qwen endpoint | OpenAI embeddings | User simulator | RapidAPI/live tools | GPU |
| --- | --- | --- | --- | --- | --- |
| Documentation and static review | No | No | No | No | No |
| Unit tests | Fake only | Fake only | Fake only | Fixtures only | No |
| Default integration tests | Fake only | Fake only | Fake only | Fixtures only | No |
| Qwen contract test | Required | No | Fake only | Fixtures only | Yes, on model server |
| Embedding contract test | No | Required | Fake only | Fixtures only | No |
| Real-data train smoke/calibration | Required | Required | OpenAI `gpt-4o-mini-2024-07-18` | Fixture replay only | Yes, on model server |
| Dev Skill Mini-Bench | Required | Required | OpenAI `gpt-4o-mini-2024-07-18` | Fixture replay only | Yes, on model server |
| `strict_replay` dataset run | Required | Required | Frozen local checkpoint required | Fixture replay only | Yes, on model server |
| `official_live` dataset run | Required | Required | OpenAI `gpt-4o-mini-2024-07-18` | Live access as configured | Yes, on model server |

Every module must have deterministic fake-client tests, but fake clients are not the only development path. Tasks responsible for model behavior, retrieval quality, ToolSandbox integration, or experiment orchestration must also request the task's explicitly assigned real-data checks with real Qwen, OpenAI embeddings, and the audited User Simulator. The coordinator-owned real-data runner performs those checks after offline tests pass; ordinary development Agent windows do not receive the credentials. Credentials are never copied into task files or source code.

### 4.1 Real-data development boundary

"Real-data development" means executing the pinned upstream ToolSandbox scenarios, starting ExecutionContexts, tool schemas, native tools, user-simulator protocol, and native evaluator rather than synthetic replacements.

- Iterative debugging, prompt adjustment, controller calibration, memory generation, and retrieval inspection use only the train split.
- Dev scenarios are used only by the deterministic Skill Dev Mini-Bench to accept or reject an already-generated skill candidate, as specified in `pipeline.md`.
- Test scenarios remain sealed until the one-time Vanilla, Generation-0, and Updated evaluation. A development task must not open, sample, inspect, or execute the test split.
- Start with a small, manifest-recorded train smoke subset. Expand to the assigned train shard only after unit tests and the small smoke subset pass.
- Output-token ceilings begin as provisional bootstrap values. A coordinator-owned
  train smoke records actual completion tokens, finish reasons, and strict-schema
  validity from complete real trajectories, adds the specified safety margin, and
  promotes one versioned role-specific config before formal use. Dev/test examples
  may never select or enlarge those ceilings; a formal length truncation stops the
  run instead of adapting the parameter.
- Real-data development runs use the same visibility guards, Qwen model, embedding model, tool fixtures, and evaluator semantics as the target profile.
- Development-run artifacts are labeled `development_only` and cannot be reported as formal test results.

The dataset access layer enforces these rules. Its ordinary loader defaults to train/development; dev requires the exact `skill_ab_validation` purpose; test is unavailable through ordinary library and development commands. Only the coordinator-owned `run-final-test` entry point may load the sealed test manifest after validating its hash, system configurations, and durable one-time execution ledger. Every access records split, purpose, phase, manifest hash, and scenario IDs. Invalid split/purpose combinations fail before returning scenario content.

## 5. Qwen Runtime Contract

### 5.1 Model identity and roles

The only pipeline generation model is:

```text
Qwen/Qwen3-32B
```

One OpenAI-compatible vLLM endpoint serves all of these roles:

- Initial Policy;
- Critic/World Model;
- Revision;
- Policy-memory candidate generation and review;
- World-memory candidate generation and review;
- failure-mode update;
- skill rewrite.

Roles differ only through validated prompts, inputs, output schemas, and generation-scoped external artifacts. They must not use different model weights.

### 5.2 Required decoding configuration

Every Qwen request uses:

```yaml
model: Qwen/Qwen3-32B
temperature: 0.0
seed: 0
enable_thinking: false
top_p: omitted
```

The vLLM request carries `chat_template_kwargs: {"enable_thinking": false}`. Structured decoding uses an explicitly configured `guided_json` or `structured_outputs_json` wire mode compatible with the manifest-pinned vLLM version. Runtime auto-detection and fallback are forbidden. The formal server manifest pins its container, launch arguments, structured-output backend, and generation-config policy; unrecorded model-repository sampling overrides and reasoning-mode output are not accepted.

The deployment adapter must verify that `enable_thinking: false` reaches the Qwen chat template. It must not assume that an unsupported or ignored request field disabled thinking. A preflight probe fails if the returned content or reasoning fields show thinking-mode output.

The upstream Qwen model card recommends different sampling values for normal non-thinking use. This project intentionally retains the audited pipeline's zero-temperature configuration for its reproducibility experiment; this deviation and any observed quality or repetition failure must be reported rather than silently changing decoding.

### 5.3 Endpoint and weight access

The runtime receives the endpoint and secret separately:

```text
QWEN_BASE_URL
QWEN_API_KEY
```

`QWEN_API_KEY` may be a deployment-local placeholder such as `EMPTY` only when the endpoint requires no authentication. The formal model ID, weight revision, tokenizer revision, file hashes, vLLM version, and container digest come from the versioned experiment configuration, not mutable environment-variable overrides.

`HF_TOKEN` is permitted only during an explicit model-download/setup phase when required by the selected artifact source. Dataset runs load an already resolved local model artifact and must not download or update weights.

Before a real run, preflight must verify endpoint reachability, served-model identity, structured-output behavior, token-usage fields, non-thinking behavior, and the configured maximum context/output limits.

## 6. OpenAI Embedding Contract

The only semantic retrieval model is:

```text
text-embedding-3-small
```

Requests use the OpenAI embeddings endpoint and require:

```text
OPENAI_API_KEY
```

The default base URL is `https://api.openai.com/v1`. A different `OPENAI_BASE_URL` may be used only when the run configuration explicitly records it; it must still serve the exact `text-embedding-3-small` model contract. Formal requests use `encoding_format: float` and omit the optional `dimensions` parameter. The first valid vector dimension is recorded in the manifest, and every vector in the run and generation must match it.

No alternate embedding model, local embedding model, BM25 retrieval, or lexical fallback is permitted. An unavailable API, rejected request, empty input, over-limit input, unexpected returned model name, non-finite vector value, inconsistent dimension, missing vector, or invalid response writes a checkpoint and stops the run.

Cache keys include provider, base-URL identity without credentials, exact model name, encoding format, dimensions setting, and exact input hash. Cached vectors record their response hash and usage metadata.

Embedding inputs may contain only the compact Agent-visible state and the permitted generation-scoped memory or skill content. Hidden evaluator or database content must never be sent to OpenAI.

## 7. User-Simulator and External-Tool Access

The default test user simulator is deterministic and local to the test process. It requires no API.

Real-data train calibration, the Dev Skill Mini-Bench, and the main `official_live` benchmark use this project-selected User Simulator:

```yaml
provider: openai
model: gpt-4o-mini-2024-07-18
endpoint: https://api.openai.com/v1/chat/completions
credential: OPENAI_API_KEY
temperature: omitted
top_p: omitted
seed: omitted
```

The User Simulator and `text-embedding-3-small` explicitly share the same `OPENAI_API_KEY`. They use separate clients or client roles, logical request IDs, attempt IDs, usage records, cost records, rate-limit handling, and preflight checks. Accounting labels User Simulator calls as `role: user_simulator` and embedding calls as `role: embedding`; sharing a credential must never merge their metrics.

The exact User Simulator model snapshot, upstream User-role behavior and prompt/few-shot content hash, tool schema, omitted decoding fields, and stop condition are pinned in every real-data run manifest. Vanilla, Generation-0, and Updated use the same User Simulator configuration. Selecting GPT-4o mini instead of the upstream `gpt-4o-2024-05-13` concrete class is an intentional experiment-level deviation and must be disclosed in reports; it is not presented as the untouched upstream benchmark configuration.

A remote OpenAI User Simulator is valid for real-data development and `official_live`, but not for trajectory-level `strict_replay`. The fixed `gpt-4o-mini-2024-07-18` snapshot is used instead of the moving `gpt-4o-mini` alias. If that exact snapshot is unavailable, preflight stops; no alias, GPT-4o snapshot, or other model may replace it. A `strict_replay` dataset run still requires a frozen local user-simulator checkpoint, tokenizer, deterministic decoding, pinned prompt/few-shot hash, and stop condition. That local checkpoint is not yet selected.

RapidAPI credentials are forbidden in `strict_replay`. `RAPID_API_KEY` is read only for an explicitly selected `official_live` run. Fixture capture is a separate authorized preparation operation and is never triggered automatically by a fixture miss.

## 8. Secret Boundary

On the current Linux host, real secrets are stored in Secret Service and retrieved with `/usr/bin/secret-tool`. A project-external, owner-only launcher injects them into the coordinator-owned `real-data-runner` process immediately before an allowlisted project command. The launcher and its local host configuration are not committed. `QWEN_BASE_URL` is non-secret configuration; only a non-placeholder `QWEN_API_KEY` belongs in Secret Service.

Secrets are never stored in:

- `pipeline.md`, `AGENTS.md`, or task files;
- committed configuration;
- `.env.example`;
- prompts, checkpoints, trajectories, fixtures, request/response logs, or test snapshots;
- Agent completion reports or Git commits.

`.env.example` contains variable names and descriptions only. A real `.env` file must not be created even if Git ignores it. Logs may record that a credential was configured, but never its value, prefix, suffix, hash, header, or resolved environment dump.

Normal development and review Agent windows never receive real secrets. A task whose access class requires an external contract test or real-data train/dev run submits the exact requested command, configuration manifest, and expected artifact paths to the coordinator. The dedicated runner injects only the minimum credentials, executes the allowlisted entry point, and returns sanitized status, usage, cost, and artifact metadata. Test-split and `official_live` execution remain coordinator-only.

Because all Agent windows currently run as the same Linux user, Secret Service plus written Agent policy prevents accidental propagation but cannot stop an arbitrary same-user process from querying the keyring. If hard isolation is required, the runner must move behind a separate OS account, container boundary, or narrow credential-holding proxy. See `docs/SECRET_INJECTION.md` for the setup and handoff contract.

## 9. Required Preflight Modes

The CLI must eventually expose these non-mutating checks:

```text
validate-config        # no external access
preflight-qwen         # Qwen endpoint only
preflight-embedding    # OpenAI embeddings endpoint only
preflight-user         # configured user simulator only
preflight-fixtures     # local fixture store only
```

No generic preflight command may contact every configured service implicitly. Each external preflight is explicit so credentials, cost, and network access remain visible to the operator.

## 10. Open Decisions Blocking Formal Runs

The following decisions do not block scaffolding, fake-client development, unit tests, or default integration tests, but they block a formal dataset run:

1. the exact Qwen weight and tokenizer revisions and their file hashes;
2. the pinned vLLM and container-image versions;
3. the strict-replay local user-simulator model/checkpoint;
4. whether formal benchmark runs require stronger isolation than the approved same-user Secret Service runner;
5. the OpenAI project/account whose access and rate limits cover both the `gpt-4o-mini-2024-07-18` User Simulator and embedding workloads;
6. the fixture-capture authorization and provenance for the five RapidAPI tools.

These values must be resolved in versioned configuration and manifests. They must not be guessed independently by implementation Agents.

## 11. Primary References

- Qwen3-32B model card: <https://huggingface.co/Qwen/Qwen3-32B>
- Qwen OpenAI-compatible vLLM guidance: <https://qwen.readthedocs.io/en/stable/deployment/vllm.html>
- OpenAI `text-embedding-3-small`: <https://developers.openai.com/api/docs/models/text-embedding-3-small>
- OpenAI embeddings API: <https://developers.openai.com/api/reference/resources/embeddings/methods/create>
