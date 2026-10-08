# Prompt Appendix for ToolSandbox, tau3, and BFCL

This supplement records every fixed prompt text and prompt-composition rule that is relevant to the three benchmark adaptations. It is designed to support a paper appendix without conflating method prompts, benchmark-native prompts, and per-instance dataset content.

For direct paper drafting, `ALL_PROMPTS.md` is the consolidated single-file appendix, and `ALL_PROMPTS.pdf` is its typeset reading copy. The files under each benchmark directory remain the authoritative byte-level artifacts.

## Interpretation Rules

The inventory uses four status labels:

- **Runtime-bound**: the checked-in implementation loads or constructs this prompt on the experiment path.
- **Benchmark-native**: the prompt belongs to the pinned upstream benchmark and is preserved here as part of the environment or simulator definition.
- **Dynamic dataset input**: task-specific user text, tool schemas, policies, memories, skills, or conversation history inserted at runtime. The template is documented, but sealed test instances are not copied.
- **Missing binding**: the local adaptation defines an interface or role but does not yet provide a concrete production prompt. Such an item must not be described as an executed experimental prompt.

## Source Revisions

| Component | Pinned revision |
|---|---|
| Experiment repository snapshot used for this inventory | `d2b5404330a83ed9be982034a897e67d968771a8` |
| ToolSandbox upstream | `165848b9a78cead7ca7fe7c89c688b58e6501219` |
| tau2-bench upstream used for tau3 | `17e07b1da2bbc0cadfddeea36412686e0604127b` |
| Gorilla/BFCL upstream | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` |

The appendix files added after the experiment snapshot do not change model behavior. `SHA256SUMS` hashes every copied or extracted prompt artifact.

## Complete Prompt Inventory

| Dataset | Prompt or input | Status | Exact artifact |
|---|---|---|---|
| ToolSandbox | Initial Policy system prompt, version v3 | Runtime-bound | `toolsandbox/method/policy_v3.txt` |
| ToolSandbox | Critic system prompt, version v3 | Runtime-bound | `toolsandbox/method/critic_v3.txt` |
| ToolSandbox | Revision system prompt, version v3 | Runtime-bound | `toolsandbox/method/revision_v3.txt` |
| ToolSandbox | Vanilla-agent system prompt, version v1 | Runtime-bound | `toolsandbox/method/vanilla_v1.txt` |
| ToolSandbox | Memory-candidate system prompt, version v1 | Runtime-bound | `toolsandbox/method/memory_candidate_v1.txt` |
| ToolSandbox | Memory-review system prompt, version v1 | Runtime-bound | `toolsandbox/method/memory_review_v1.txt` |
| ToolSandbox | Failure-mode-update system prompt, version v1 | Runtime-bound | `toolsandbox/method/failure_mode_update_v1.txt` |
| ToolSandbox | Skill-candidate system prompt, version v1 | Runtime-bound | `toolsandbox/method/skill_candidate_v1.txt` |
| ToolSandbox | User-simulator base instruction | Benchmark-native and runtime-bound | `toolsandbox/native/user_instruction.txt` |
| ToolSandbox | Six user-simulator few-shot dialogues | Benchmark-native and runtime-bound when selected by the scenario | `toolsandbox/native/user_simulator_few_shots.json` |
| ToolSandbox | Per-scenario task instruction and current dialogue | Dynamic dataset input | Template only; sealed/evaluation instances are not duplicated |
| tau3 | Native customer-service agent instruction and system template | Benchmark-native | `tau3/native/agent_instruction.txt`, `tau3/native/agent_system_template.txt` |
| tau3 | Airline policy | Benchmark-native dynamic insertion | `tau3/native/airline_policy.md` |
| tau3 | Retail policy | Benchmark-native dynamic insertion | `tau3/native/retail_policy.md` |
| tau3 | Telecom main policy and technical-support manual | Benchmark-native dynamic insertion | `tau3/native/telecom_main_policy.md`, `tau3/native/telecom_tech_support_manual.md` |
| tau3 | User-simulator system template | Benchmark-native | `tau3/native/user_system_template.txt` |
| tau3 | User-simulator guidelines without tools | Benchmark-native | `tau3/native/user_simulation_guidelines.md` |
| tau3 | User-simulator guidelines with tools | Benchmark-native | `tau3/native/user_simulation_guidelines_tools.md` |
| tau3 | Scenario instructions and optional persona guidelines | Dynamic dataset input | Inserted into the documented user-system template |
| tau3 | Evolution Policy/Critic/Revision role strings | Runtime-bound but placeholder-level | `tau3/current_evolution_role_prompts.txt` |
| tau3 | Offline memory/skill updater prompts | Missing binding | The code exposes `OfflineUpdater.propose` and `OfflineUpdater.review` only |
| BFCL | Official classic prompting-mode system prompt | Benchmark-native | `bfcl/native/official_classic_system_prompt.txt` |
| BFCL | Delayed-function user prompt | Benchmark-native and runtime-bound by the adapter | `bfcl/native/additional_function_prompt.txt` |
| BFCL | Function schemas and benchmark user turns | Dynamic dataset input | Loaded from each BFCL case; individual evaluation cases are not duplicated |
| BFCL | Evolution Policy/Critic/Revision prompts | Missing binding | The code defines `PolicyRole`, `CriticRole`, and `RevisionRole` protocols only |
| BFCL | Offline memory/skill updater prompts | Missing binding | No concrete production prompt is checked in |

### Frozen ToolSandbox Method Prompt Hashes

| Role | Version | SHA-256 |
|---|---|---|
| Policy | v3 | `05fb1c48a1b4764e68edc14296a0dcd45e27b7480b3513ee392b9b190bc54e66` |
| Critic | v3 | `8345d104461234dd4a3db763622082811c0ee3f8c6d71d5f75afe84f337d211b` |
| Revision | v3 | `d6614d8e8e09c8162287157522f34ed6d5b4344aad450f707bef2c624f3985df` |
| Vanilla | v1 | `5fac4ea2e3c7773ef866a978d70f3d386bd9757edebe27f12cd9af9bdd8db970` |
| Memory candidate | v1 | `5dba47b4f1ad2d6302417de251835b6a89e1887303a67ff78d00946d510f9718` |
| Memory review | v1 | `915b11b2b9582d2e4f3a4b3624f2b3b2273e82540238345a1cda2a06341840fd` |
| Failure-mode update | v1 | `0f5384b4b08148c760ed9163b515b02c1c8a5d4ad46fd504b9436b94b7ed8ddc` |
| Skill candidate | v1 | `c76213b25ac75cfe6560d2e2beadd1b65cad39caf92ee293ae259a17a17338ba` |

## Experimental Claim Boundary

Only ToolSandbox currently has a complete, versioned, hash-verified set of method prompts for online Policy/Controller/Critic/Revision operation and offline memory/skill evolution. The tau3 port currently sends the minimal system strings `tau3 evolution role: policy`, `tau3 evolution role: critic`, and `tau3 evolution role: revision`; these strings identify roles but do not encode the full method behavior. The BFCL port currently injects Python role implementations and has no checked-in concrete Policy/Critic/Revision prompt text.

Therefore, a paper based on the current code may accurately report:

1. the complete ToolSandbox method prompts;
2. the native tau3 and BFCL benchmark templates preserved by the adapters; and
3. the prompt interfaces and dynamic inputs of the tau3/BFCL ports.

It must not claim that the full ToolSandbox evolution prompts were already executed unchanged on tau3 or BFCL. Before model-backed tau3/BFCL experiments, those method prompts should be explicitly adapted, frozen, hashed, and bound in code.

## Dataset Content and Leakage Boundary

This appendix intentionally does not copy individual train, Dev Mini-Bench, or sealed test task messages. Those messages are benchmark data, not fixed method prompts. For reproducibility, the runtime composition is documented using named placeholders, and the public fixed domain policies and simulator guidelines are copied verbatim. This keeps the appendix complete at the prompt-template level without making sealed evaluation content part of the method specification.

## Operational Prompts Excluded from Experimental Reporting

ToolSandbox contains two setup probes:

```text
Return {"action":{"type":"assistant_message","content":"ok"}} only.
Reply with ok.
```

They are used only for Qwen and user-simulator service preflight checks. They do not process benchmark examples, do not affect learned artifacts, and should not be listed as experimental method prompts.

## Paper-Ready Summary

The following wording is safe for the appendix:

> For ToolSandbox, each model call consisted of a frozen role-specific system prompt and a canonical JSON user envelope containing only agent-visible state and generation-scoped retrieved artifacts. The Policy proposed one action, the Critic evaluated the proposal using controller evidence and world memory, and an optional single Revision produced the final action. Offline prompts separately generated and reviewed memory candidates and updated skill failure modes and skill content. Prompt files were versioned and verified by SHA-256. The ToolSandbox user simulator retained the benchmark-native instruction and few-shot demonstrations and used GPT-4o-mini through the same OpenAI credential boundary. For tau3 and BFCL, benchmark-native policy, simulator, function-schema, and user-turn templates were preserved. At the captured implementation revision, the full evolution-role prompt bindings for tau3 and BFCL were not yet frozen; this is reported as an implementation limitation rather than an executed condition.

## Per-Benchmark Composition Notes

- `toolsandbox/README.md` documents the exact canonical JSON envelopes for all runtime-bound method prompts.
- `tau3/README.md` documents the native agent/user composition and the current placeholder evolution strings.
- `bfcl/README.md` documents the official classic system prompt, dynamic functions insertion, delayed-function prompt, and missing evolution-role bindings.
