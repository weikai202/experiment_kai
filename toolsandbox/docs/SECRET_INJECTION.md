# Secret Injection and Real-Data Runner

Status: `approved`

This document defines how the standalone ToolSandbox project uses real model credentials without placing them in the repository or ordinary Agent windows. It implements the secret boundary in `AGENTS.md` and `docs/PROJECT_INTENT_AND_ACCESS.md`.

## 1. Approved Host Design

- Linux Secret Service stores `OPENAI_API_KEY` and, when needed, a non-placeholder `QWEN_API_KEY`.
- `/usr/bin/secret-tool` is the approved local client. It is installed on the current host.
- `QWEN_BASE_URL` is not a secret and belongs in the selected runtime configuration.
- `OPENAI_API_KEY` is shared by the `text-embedding-3-small` client and the `gpt-4o-mini-2024-07-18` User Simulator, while their clients, request IDs, usage, and retry state remain separate. Neither contributes to the non-monetary substantive-effect Qwen-output `total_cost`.
- A project-external launcher owned by the operator injects credentials only into the coordinator-owned `real-data-runner` process.
- The launcher exposes reviewed project modes, not arbitrary shell commands.

Normal coding and review Agent windows run without these credentials. A task's access class authorizes a requested service or real-data check; it does not automatically give that Agent the key.

An operator may instead inject an already-rotated credential directly into the
owner-only launcher process environment. That environment must never be created
from a command shown in an Agent transcript, inherited by development windows,
dumped for debugging, or persisted in a shell file. The repository-facing
contract is the same: clients read only the fixed variable names and artifacts
never contain their values.

## 2. One-Time Secret Storage

The operator performs these commands in an interactive terminal, outside an Agent transcript. Each command prompts for the secret on standard input; never append the secret to the command line:

```bash
secret-tool store \
  --label='ToolSandbox OpenAI API key' \
  service toolsandbox-pipeline \
  key OPENAI_API_KEY
```

Only when the Qwen endpoint requires a real credential:

```bash
secret-tool store \
  --label='ToolSandbox Qwen API key' \
  service toolsandbox-pipeline \
  key QWEN_API_KEY
```

Do not run a standalone `secret-tool lookup` in an Agent-visible terminal because its normal output is the secret. Do not store `EMPTY` in Secret Service; select the documented placeholder in local runtime configuration when the Qwen endpoint requires no authentication.

## 3. External Launcher Contract

The launcher is installed outside the Git repository, for example at:

```text
/home/weik/.local/bin/toolsandbox-with-secrets
```

It must be owned by the operator with mode `0700`. It must:

1. accept only a fixed mode such as `preflight-qwen`, `preflight-embedding`,
   `preflight-user`, `train-smoke`, `dev-mini-bench`, or `final-test`; a
   `train-smoke` request must also carry an allowlisted purpose such as
   `online-token-calibration` rather than an arbitrary command;
2. reject unknown modes before retrieving a secret;
3. retrieve only the credentials required by the selected mode;
4. fail with only the missing variable name when a lookup returns empty;
5. disable shell tracing and never run `env`, `printenv`, or diagnostic dumps;
6. change to the exact project root and execute the corresponding `uv run` entry point;
7. pass the selected versioned configuration or manifest explicitly;
8. replace itself with the project process so credentials exist in only that process tree;
9. never accept an arbitrary command string, shell fragment, Python expression, or unrestricted passthrough executable.

The executable launcher is installed only after the project CLI names and arguments are implemented and reviewed. Until then, this document is the contract; no placeholder launcher should pretend that real-data execution is available.

## 4. Runner Request and Result Handoff

A development Agent that needs an external check returns this request to the coordinator:

```text
Runner request:
Task:
Access class:
Allowlisted mode:
Versioned config or manifest:
Scenario split and IDs:
Expected artifacts:
Reason:
```

The coordinator verifies that offline tests have passed, the requested split and purpose are allowed, and the mode matches the task. The real-data runner then returns only:

```text
Runner result:
Mode:
Status: pass | fail | blocked
Manifest hash:
Scenario IDs:
Test counts:
Total running time seconds:
Total tokens:
Usage complete: true | false
Actual usage by role:
Total cost (qwen_effective_output_tokens):
Cost complete: true | false
Substantive-effect Qwen output cost by role:
Sanitized error class, if any:
Artifact paths and hashes:
```

The result must not contain secret values, environment dumps, authentication headers, request headers, or credential-derived identifiers.

## 5. Access Mapping

| Access class | Runner behavior |
| --- | --- |
| `offline` | No launcher or secret lookup |
| `qwen` | Explicit Qwen preflight only |
| `embedding` | Explicit `text-embedding-3-small` preflight only |
| `user_simulator` | Explicit `gpt-4o-mini-2024-07-18` User Simulator preflight only |
| `real_data` | Assigned train smoke/shard or Dev Mini-Bench with fixture-replayed tools |
| `official_live` | Coordinator-only formal run with separately authorized live-tool credentials |

No mode contacts every configured service implicitly. Fixture misses never trigger live network calls.

## 6. Revocation and Rotation

The operator may remove an entry interactively:

```bash
secret-tool clear service toolsandbox-pipeline key OPENAI_API_KEY
secret-tool clear service toolsandbox-pipeline key QWEN_API_KEY
```

Rotation means clearing or replacing the Secret Service entry and rerunning only the relevant explicit preflight. No committed file, manifest, task document, or fixture changes because of key rotation.

## 7. Security Limit

This arrangement reduces accidental disclosure and keeps secrets out of repository files and normal process environments. It is not a hardened boundary between Agent windows that execute as the same Linux user: a same-user process may be able to query the unlocked keyring.

If formal runs require protection from arbitrary same-user processes, use a separate Unix account, isolated container or VM, or a narrow local/remote proxy that holds credentials and exposes only allowlisted model operations. That decision must be resolved before formal benchmark execution.
