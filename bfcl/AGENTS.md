# BFCL Pipeline Development Rules

This file applies to the complete `bfcl/` project.

- `pipeline.md` is the experiment authority.
- Keep this package standalone. Never import sibling ToolSandbox, CoMAP, or
  EarlyExperience implementation code.
- Keep all source, tests, configuration, and documentation in English.
- Unit tests must not call a model, execute sealed test cases, access secrets,
  or write generated run artifacts into the repository.
- Train evidence may update resources. Dev may only evaluate an already-created
  Skill candidate. Test requires the frozen plan and one-use authority.
- Preserve fixed BFCL user turns, multi-step turn shape, simulator state,
  delayed functions, literal-only execution, and official checker authority.
- Do not claim official BFCL leaderboard comparability for this family split.
- Use `Qwen/Qwen3-32B` with explicit non-thinking mode for every pipeline role.
- Record direct round latency, complete provider usage, and only
  substantive-effect-linked Qwen output tokens as non-monetary cost.
