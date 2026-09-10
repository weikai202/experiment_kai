# ToolSandbox Pipeline Agent Instructions

Status: `approved`

These instructions apply to every coding or review Agent working anywhere under this `toolsandbox/` project.

## 1. Authority and Required Reading

Read these files before editing:

1. `pipeline.md` as the authoritative system and experiment specification;
2. `docs/PROJECT_INTENT_AND_ACCESS.md` for project, data, model, network, and secret boundaries;
3. the one assigned file under `tasks/` for scope, owned files, interfaces, and acceptance commands;
4. any interface document explicitly listed by that task.

An assigned task narrows the global specification; it cannot override it. If two instructions conflict or an implementation requires weakening a rule, stop and report the exact conflict to the coordinator. Do not choose an interpretation silently.

## 2. Standalone Environment

- Treat this directory as the complete project root.
- Use `uv` as the only project environment and dependency manager. `uv sync --frozen` creates or updates this project's `.venv` strictly from `pyproject.toml` and `uv.lock`; it must fail rather than re-resolve a stale lock.
- Run Python and project commands from this directory through `uv run`; do not depend on manual virtual-environment activation.
- Do not use a parent, sibling, user-site, Conda, or system Python environment for project commands.
- Do not import code or runtime state from a parent or sibling project.
- Do not modify `sys.path` or rely on `PYTHONPATH` or the caller's current working directory.
- Use the `toolsandbox_pipeline` package for new code. The `tool_sandbox` package belongs to the pinned upstream dependency.
- Install upstream Apple ToolSandbox as a non-editable Git dependency pinned to commit `165848b9a78cead7ca7fe7c89c688b58e6501219` and locked by `uv.lock`. Treat it as read-only.
- Do not add Apple ToolSandbox as a Git submodule or copy its source tree into this project.
- Put all compatibility behavior in `src/toolsandbox_pipeline/toolsandbox_adapter/`; never patch upstream evaluator, scenarios, roles, databases, or tools.

## 3. Task and File Ownership

- Tasks 001-005 are manually reviewed baseline contracts. Later task files may
  extend them but cannot silently reinterpret or override them. If a later task
  appears to conflict with Tasks 001-005, stop the conflicting part, notify the
  coordinator and the affected task owner/Agent, and agree on an explicit
  interface resolution before editing or continuing.
- Work on exactly one reviewed `tasks/*.md` assignment at a time.
- Edit only files listed under that task's `Owned files` section.
- Do not edit `pipeline.md`, this `AGENTS.md`, another task file, or another Agent's owned files.
- Do not perform opportunistic refactors outside the assignment.
- Preserve unrelated and pre-existing worktree changes.
- Do not change a shared schema, public interface, dependency, prompt contract, or artifact format unless the task explicitly owns it.
- If an unowned change is required, stop and send the coordinator the requested file, interface change, rationale, and affected tasks.

The default development mode is multiple Agent windows sharing this one project directory and Git worktree. Changes are immediately visible across windows, so reviewed task files must assign non-overlapping owned file sets before parallel work starts. Subagents do not stage, commit, merge, rebase, or switch branches. The coordinator alone reviews the combined worktree, resolves approved interface changes, runs integration checks, and creates commits.

If a task requires an unowned interface change, include this request in the handoff instead of editing the file:

```text
Requested interface change:
File:
Symbol or schema:
Requested change:
Reason:
Affected tasks:
Blocking: yes | no
```

## 4. Data and Evaluation Boundary

- Train is the only split used for iterative debugging, prompt adjustment, controller calibration, retrieval inspection, and update generation.
- Dev is used only by the deterministic Skill Dev Mini-Bench to accept or reject an already-generated skill candidate.
- Test remains sealed until the coordinator starts the one-time Vanilla, Generation-0, and Updated evaluation.
- Do not list, inspect, sample, open, embed, execute, or summarize test scenarios during development.
- Never expose hidden databases, scenario-construction code, milestones, minefields, target DataFrames, similarity functions, evaluator mappings, or post-episode labels to an online component.
- Native evaluator results become update evidence only after the episode completes and only where `pipeline.md` permits them.
- Real-data development artifacts are marked `development_only` and are not formal test results.

These boundaries must be enforced by the dataset access layer, not only by Agent instructions:

- the ordinary loader defaults to `split="train"` and `purpose="development"`;
- loading dev requires exactly `split="dev"` and `purpose="skill_ab_validation"` from the Skill Dev Mini-Bench;
- no ordinary library or development CLI exposes test loading;
- test is available only through the coordinator-owned `run-final-test` entry point;
- final-test startup verifies the sealed test-manifest hash, the three system configurations, and a durable execution ledger;
- the ledger rejects a repeated completed Vanilla, Generation-0, or Updated test evaluation;
- every dataset access records split, purpose, caller phase, manifest hash, and scenario IDs for audit;
- tests must prove that invalid split/purpose combinations and development-time test access fail before scenario content is returned.

Upstream scenario definitions are present in the installed dependency and therefore cannot be made physically unreadable to a developer. Agents must not inspect them to learn test content, and project code must still enforce and audit the runtime split boundary above.

## 5. Model, API, and Network Access

Every task declares one access class:

```text
offline
qwen
embedding
user_simulator
real_data
official_live
```

Each task also declares `Setup network: none | locked_dependency_fetch`. This field is separate from runtime/model access. `locked_dependency_fetch` permits only `uv` to obtain the dependency sources named in `pyproject.toml` or `uv.lock`, including the exact pinned ToolSandbox Git commit. It does not permit model, dataset, arbitrary web, or live-tool access. Only a task that owns dependency files may request it.

- `offline`: no network, API key, model server, or GPU; use fake clients and fixtures.
- `qwen`: may access only the configured frozen `Qwen/Qwen3-32B` vLLM endpoint.
- `embedding`: may access only OpenAI `text-embedding-3-small`.
- `user_simulator`: may access only OpenAI `gpt-4o-mini-2024-07-18` for the ToolSandbox User role.
- `real_data`: may request an assigned train run or Dev Mini-Bench through the coordinator-owned real-data runner; it does not grant a normal development window direct credential access.
- `official_live`: coordinator-only; the dedicated runner may additionally use explicitly configured live external tools for the assigned run.

Access is deny-by-default. Do not contact a service that is not allowed by the assigned task. Do not convert a failed fixture lookup into a live request. Do not run an external preflight implicitly.

`gpt-4o-mini-2024-07-18` and `text-embedding-3-small` share the process's `OPENAI_API_KEY`, but their clients or client roles, logical request IDs, attempt IDs, usage, retry state, and cost records remain separate. Qwen uses `QWEN_BASE_URL` and `QWEN_API_KEY` and must never be routed through the OpenAI credential by accident.

## 6. Secret Handling

- Normal coding and review Agent windows receive no real credentials. Only the coordinator-owned `real-data-runner` process may receive runtime credentials.
- On this Linux host, store `OPENAI_API_KEY` and any non-placeholder `QWEN_API_KEY` in Secret Service through `/usr/bin/secret-tool`. Keep `QWEN_BASE_URL` in non-secret runtime configuration.
- Inject secrets only through the project-external, owner-only runner launcher described in `docs/SECRET_INJECTION.md`. Do not accept an arbitrary command in that launcher; it exposes only reviewed, allowlisted project entry points.
- A task with `qwen`, `embedding`, `user_simulator`, `real_data`, or `official_live` access submits the required command and manifest to the coordinator. The dedicated runner executes it and returns sanitized results.
- Read credentials only through the approved runtime client configuration.
- Never print environment variables, authentication headers, credential presence details beyond a boolean, or secret values in full or in part.
- Never store a secret, its hash, prefix, suffix, or transformed form in source, configuration, prompts, fixtures, tests, snapshots, checkpoints, artifacts, logs, reports, or commits.
- Do not create a real `.env` file. `.env.example` contains names and descriptions only.
- Do not ask another Agent to send a key through chat or a task file.
- If an authorized credential is missing, report only the variable name and blocked command.
- This is an operational guard against accidental exposure, not a hard security boundary between processes owned by the same Linux user. Strong isolation requires a separate OS account, container boundary, or narrow credential-holding proxy.

## 7. Implementation Rules

- Use Python 3.10 and the versions resolved by the locked project environment.
- Use strict Pydantic v2 models and JSON Schemas for model and artifact contracts; forbid unknown fields and implicit coercion where `pipeline.md` requires it.
- Keep Controller decisions deterministic and rule-based. Controller code cannot invent or modify model actions.
- Preserve Agent-facing augmented schemas and names for Policy/Critic inputs. Canonical tool metadata remains Controller/offline-only.
- Preserve provenance for verified facts and argument grounding.
- Do not add an embedding or retrieval fallback. Embedding failure checkpoints and stops the run.
- Do not add revision loops. One real state permits at most one Revision.
- Do not silently truncate schemas, messages, retrieval results, or model outputs. If a defined limit is exceeded, follow the specified hard-failure behavior or report a missing specification.
- Use atomic writes, stable IDs, canonical serialization, manifest hashes, and idempotent logical commits where assigned.
- Do not weaken validation, catch-and-ignore a hard failure, or replace a required assertion with a warning to make a test pass.

## 8. Testing Rules

- Add or update the focused tests required by the task.
- Run the task's targeted tests before broader tests.
- Default unit and integration tests use fake model clients and deterministic fixtures.
- Tasks marked for real-data validation must also run their assigned train subset or Dev Mini-Bench after offline tests pass.
- Never run the test split as a development test.
- External tests must be individually selected and must contact only the access class declared by the task.
- Do not remove, skip, loosen, or rewrite an existing failing test unless the task explicitly changes the tested contract.
- Record exact commands, pass/fail/skip counts, external profiles, total running
  time, actual total token usage, and substantive-effect Qwen-output
  `total_cost` in the completion report. Never substitute estimates for missing
  provider usage.
- Formal training reports must preserve separate direct
  `total_running_time_seconds`, `total_tokens`/`usage_complete`, and
  `total_cost`/`cost_complete` rows for round 0, round 1, and round 2. A run
  total or average cannot replace any round row.
- A passing fake-client test does not prove real-model compatibility; a passing real-model smoke test does not replace deterministic unit tests.

## 9. Worktree and Git Safety

- Inspect `git status --short` before editing and before handoff.
- Do not discard, overwrite, reset, clean, or revert changes you did not create.
- Do not modify `.git`, remotes, branches, tags, or worktrees unless the task explicitly requests it.
- Do not commit generated runs, artifacts, model files, caches, `.venv`, `.env`, or secrets.
- Subagents do not stage or commit in the shared worktree. Only the coordinator creates project commits after reviewing the combined changes and test results.

## 10. Required Completion Report

Return a concise handoff containing:

```text
Task:
Status: complete | blocked
Access class actually used:
Changed files:
Interfaces added or changed:
Tests run and results:
Real-data scenarios run:
Total running time seconds:
Total tokens:
Usage complete: true | false
Total cost:
Cost unit: qwen_effective_output_tokens
Cost complete: true | false
Per-round latency/cost rows, when applicable:
Artifacts produced:
Assumptions:
Remaining blockers or risks:
Commit SHA, if requested:
```

Do not claim completion when required tests were not run. If external access was unavailable, distinguish code completion from external validation and report the exact remaining preflight or test.
