# Task 001: Standalone Project Scaffold and Frozen Dependency Lock

Status: `approved`

## Objective

Create the minimal installable and testable Python 3.10 project that every later ToolSandbox pipeline task will use. Establish the project-local `uv` environment, lock the upstream Apple ToolSandbox dependency at the audited commit, and prove that both package namespaces import without relying on a parent or sibling project.

This task builds infrastructure only. It does not implement pipeline behavior.

## Required Reading

Read, in order:

1. `pipeline.md`, especially Sections 1-6, 24, and 25;
2. `AGENTS.md`;
3. `docs/PROJECT_INTENT_AND_ACCESS.md`;
4. this task file.

## Access

```text
Access class: offline
Setup network: locked_dependency_fetch
Data splits: none
Secrets: none
```

The setup-network permission is limited to `uv` fetching declared Python dependencies and this exact Git dependency:

```text
https://github.com/apple/ToolSandbox.git@165848b9a78cead7ca7fe7c89c688b58e6501219
```

It does not authorize model endpoints, OpenAI, datasets, RapidAPI, or general web requests. If the coordinator supplies an already-populated `uv` cache, prefer the cache.

## Preconditions

- The coordinator must make a reviewed `uv` executable available on the host. `uv` was not installed when this task was drafted.
- Python 3.10 must be available directly or installable through the coordinator-approved `uv` setup.
- Run all commands from the standalone `toolsandbox/` project root.

Do not install `uv` through an unreviewed shell pipe. If either prerequisite is missing, report it as a blocker rather than switching to `pip`, Poetry, Conda, or the system environment.

## Owned Files

The assigned Agent may create or edit only:

```text
.gitignore
.python-version
README.md
pyproject.toml
uv.lock
src/toolsandbox_pipeline/__init__.py
src/toolsandbox_pipeline/py.typed
tests/test_project_scaffold.py
```

The Agent must not edit `pipeline.md`, `AGENTS.md`, `docs/`, `tasks/`, or create placeholder modules owned by future tasks.

## Required Project Contract

### Packaging

- Distribution name: `toolsandbox-pipeline`.
- Import package: `toolsandbox_pipeline`.
- Source layout: `src/toolsandbox_pipeline/`.
- Python requirement: `>=3.10,<3.11`.
- Build backend must support the `src` layout without modifying `sys.path`.
- Mark the package as typed with `py.typed`.
- The base package must expose a static package version without importing upstream ToolSandbox or initiating external access at import time.

### Dependency management

- `uv` is the only environment and dependency manager.
- Commit both `pyproject.toml` and `uv.lock`.
- Add Apple ToolSandbox as a non-editable Git dependency pinned to commit `165848b9a78cead7ca7fe7c89c688b58e6501219`.
- Declare `pydantic==2.7.4` as a direct runtime dependency for the project's strict shared contracts; this matches the pinned upstream version.
- Declare `jsonschema==4.19.2` as a direct runtime dependency for validation against current augmented agent-facing tool schemas; this matches the pinned upstream version.
- Declare `openai==1.17.0` as a direct runtime dependency for the project's OpenAI-compatible Qwen transport, OpenAI embeddings, and instrumented upstream User Simulator; this matches the pinned upstream version.
- Use the upstream distribution/import identity as published; do not rename, vendor, patch, fork, or add it as a submodule.
- Declare every package imported directly by scaffold code or tests as a direct runtime or development dependency. Do not rely accidentally on an upstream transitive dependency.
- Keep development dependencies in one documented `dev` dependency group and include the test runner used by the acceptance commands.
- Do not add model-serving, experiment-tracking, notebook, UI, or cloud SDK dependencies unless the scaffold itself directly requires them.
- `uv sync --frozen` must succeed from the committed lock without re-resolution.

### Repository hygiene

`.gitignore` must exclude at least:

```text
.venv/
.env
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.py[cod]
artifacts/**
runs/**
```

Do not ignore `.env.example`, source, configuration, prompts, tests, task files, or documentation. Do not create a real `.env`, generated run, artifact, or dataset copy.

### README

Document only verified scaffold operations:

- supported Python version;
- `uv sync --frozen`;
- `uv run python -c "import toolsandbox_pipeline; import tool_sandbox"`;
- `uv run pytest tests/test_project_scaffold.py`;
- the fact that model/data setup is intentionally deferred to later tasks.

Do not claim that the pipeline, model access, real-data runner, or formal evaluation is implemented.

## Required Tests

`tests/test_project_scaffold.py` must verify without network, API keys, models, or datasets that:

1. `toolsandbox_pipeline` imports from this project's `src/` tree;
2. `tool_sandbox` imports from the installed pinned dependency rather than a copied repository directory;
3. the installed ToolSandbox distribution metadata contains the exact requested Git commit in its direct-URL record;
4. importing `toolsandbox_pipeline` has no external-access or filesystem-write side effects;
5. the exposed project version is a non-empty static string.

Tests must not inspect upstream scenario definitions or test data.

## Acceptance Commands

Run, in this order:

```bash
uv sync --frozen
uv run python -c "import toolsandbox_pipeline; import tool_sandbox"
uv run pytest -q tests/test_project_scaffold.py
git diff --check
git status --short
```

The first command may use the approved locked-dependency setup network. After synchronization, rerun the import and test commands with no model, API, data, or live-tool access.

## Acceptance Criteria

- Every acceptance command succeeds.
- `.venv/` is located under this project root and is not tracked.
- `uv.lock` resolves the exact ToolSandbox Git commit.
- No import depends on a parent/sibling repository, `PYTHONPATH`, user-site package, Conda environment, or manual activation.
- The task creates no network-capable application code and accesses no secrets or data split.
- `git status --short` shows only the task's owned files plus pre-existing coordinator changes.

## Completion Report Additions

In addition to the global completion report, include:

```text
Python version:
uv version:
Resolved ToolSandbox commit:
Project .venv path:
Setup network used: yes | no
```

For this scaffold task, `Total running time seconds` and `Total tokens` refer only to tests that implement those metrics. Since this task performs no experiment or model calls, report `Total tokens: 0` and `Usage complete: true`; do not present dependency-download duration as an experiment result.
