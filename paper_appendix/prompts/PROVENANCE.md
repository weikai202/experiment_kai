# Prompt Source Provenance

All source paths below are relative to the named pinned upstream repository or the experiment repository. Source-file hashes cover the complete source file, while `SHA256SUMS` covers each appendix artifact after exact copying or constant extraction.

## ToolSandbox

Pinned upstream revision: `165848b9a78cead7ca7fe7c89c688b58e6501219`.

| Prompt family | Source | Source SHA-256 |
|---|---|---|
| User instruction and six few-shot dialogues | `tool_sandbox/scenarios/user_simulator_few_shot_examples.py` | `3ae184a73c6f6bed0d00d7c14f644331ab97a6189f32adc8184698fb1b8d3731` |
| User-simulator role mapping and native model-call behavior | `tool_sandbox/roles/openai_api_user.py` | `1bc99a6afb9e752e4e06d346f038cbb686ff12d994cfe1d97a688d2c0dcaef35` |

The eight method prompt files are copied from `toolsandbox/prompts/` in the experiment repository. Their runtime manifests provide the role-level hashes documented in the top-level inventory and verified again in `SHA256SUMS`.

## tau3

Pinned tau2-bench revision: `17e07b1da2bbc0cadfddeea36412686e0604127b`.

| Prompt family | Source | Source SHA-256 |
|---|---|---|
| Native agent instruction and system template | `src/tau2/agent/llm_agent.py` | `d13412ff502859f0ef0492af0128f1b2666be32640057ba8e076d2602af084d8` |
| Native user-simulator composition | `src/tau2/user/user_simulator.py` | `9cb7656c4ca6345634fab445e6cd07e16dd9388746c781c022756f0d6ec59dc6` |
| User guidelines without tools | `data/tau2/user_simulator/simulation_guidelines.md` | `740a29dfa64d7bc08eea3bf7493575b914a63f744acbaf7f199ee07eddaf72d3` |
| User guidelines with tools | `data/tau2/user_simulator/simulation_guidelines_tools.md` | `cbf3d8a4d8642fd04e559862f1afef55d7dd4e6a7e727ca49e239023c599de0c` |
| Airline policy | `data/tau2/domains/airline/policy.md` | `10dc0525421521208be39cee235bba84a16e2bcba9899eb93d92cd81d2f62fc4` |
| Retail policy | `data/tau2/domains/retail/policy.md` | `2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467` |
| Telecom main policy | `data/tau2/domains/telecom/main_policy.md` | `95943844a0cf11fdf0e2842b483b81d9f9338aa53235712ed131db075d086ed7` |
| Telecom technical-support manual | `data/tau2/domains/telecom/tech_support_manual.md` | `015a3ee49ec8c199c5e1a059c922081937b92cf3a5ea74fedbc4357816f389ba` |

The current evolution role strings are taken from `tau3/src/tau3_evolution/online_pipeline.py` in the experiment repository.

## BFCL

Pinned Gorilla/BFCL revision: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.

| Prompt family | Source | Source SHA-256 |
|---|---|---|
| Official classic system prompt and delayed-function prompt | `berkeley-function-call-leaderboard/bfcl_eval/constants/default_prompts.py` | `4cc033be99ae2b10bbde30e7751ddcac5468cb45841b6dc7b3a97b92adf38205` |

The exact upstream constants were evaluated from their Python string expressions before being written to the `.txt` artifacts. This preserves literal escape sequences and trailing-newline behavior that would be lost by manually copying the displayed source.

