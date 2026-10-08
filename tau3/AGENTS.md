# tau3 Evolution Development Rules

- Treat this directory as a standalone Python 3.10 project. Do not import sibling projects.
- The pinned tau3 source is commit `17e07b1da2bbc0cadfddeea36412686e0604127b`.
- Formal domains are Airline, Retail, and Telecom. Mock, voice, and Banking Knowledge are excluded.
- Evolution uses only the `evolution144_dev34_seed0_v1` manifest. The `official178_no_dev_v1`
  manifest exists only for compatibility and results from the two protocols must not be compared.
- Train generates updates. Dev is used only for paired acceptance of an already-generated Skill.
  Test is available only through the frozen final plan and one-time guard.
- A tau3 task is the cluster. Simulator trials or seeds are execution variants of that same task
  and must never be placed in different data splits.
- Never expose native evaluator state, reference actions, private user tools, or simulator-private
  messages to the Agent or an online update role.
- Qwen is fixed to `Qwen/Qwen3-32B` with `enable_thinking=false`, temperature 0, and seed 0.
  Embeddings are `text-embedding-3-small` without fallback. The simulator is
  `gpt-4o-mini-2024-07-18`.
- Record each round's direct wall-clock duration, all physical provider tokens, and only Qwen
  output tokens causally bound to a committed substantive effect as `total_cost`.
- No source file, fixture, log, artifact, or checkpoint may contain a credential.
- Offline tests do not prove live model, simulator, or official benchmark execution.
