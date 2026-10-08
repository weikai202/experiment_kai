# tau3 Prompt Composition

## Benchmark-Native Agent Prompt

The native agent system message is formed by substituting `agent_instruction.txt` and the selected domain policy into `agent_system_template.txt`:

```text
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
```

The domain insertion is:

- Airline: `native/airline_policy.md`.
- Retail: `native/retail_policy.md`.
- Telecom: `<main_policy>` containing `native/telecom_main_policy.md`, followed by `<tech_support_policy>` containing `native/telecom_tech_support_manual.md`.

Tool definitions and the visible conversation are supplied through the native model API rather than interpolated into the fixed text.

## Benchmark-Native User Simulator

The user-simulator system message follows `native/user_system_template.txt`:

```text
{global_user_sim_guidelines_with_persona}

<scenario>
{instructions}
</scenario>
```

The global block is `native/user_simulation_guidelines.md` when the simulator has no user-side tools and `native/user_simulation_guidelines_tools.md` when tools are present. `<PERSONA_GUIDELINES>` is replaced by optional runtime persona text. `{instructions}` is the task-specific scenario instruction. The configured simulator model in the experiment design is GPT-4o-mini.

## Current Evolution Port

The current Qwen request constructor sends one of the three exact system strings in `current_evolution_role_prompts.txt` plus a JSON user payload serialized with `sort_keys=True`.

Policy payload fields:

```text
domain_policy, tools, visible_history,
policy_memory_records, world_memory_records, skill_records,
selected_policy_memory_ids, selected_world_memory_ids,
selected_policy_memory_versions, selected_world_memory_versions,
selected_skill_versions, retrieval_sha256
```

The Critic receives the same fields plus `initial_action`. Revision receives those fields plus `initial_action` and `critic_reason`.

Policy and Revision must return either a message object or a tool-call object. Critic must return `{"decision":"KEEP"|"REVISE","reason":"..."}`. These output contracts exist in code, but the three current system strings are role labels rather than full method prompts. The offline updater is an injected protocol and has no concrete prompt text. These are missing production bindings, not omitted appendix material.

