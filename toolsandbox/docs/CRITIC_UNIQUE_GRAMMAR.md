# Explicit Critic unique-code grammar

The user/coordinator approved the `ordered_unique_grammar_v1` Critic-only wire
option after real Qwen responses repeatedly emitted duplicate error codes despite
prompt instructions and a finite maxItems bound. This is an explicit protocol
change from `structured_outputs.json` to `structured_outputs.grammar` for Critic.
It is not automatic fallback and does not change the structured backend.

`providers/critic_grammar.py` exports:

- `CRITIC_GRAMMAR_VERSION = "ordered_unique_grammar_v1"`
- `build_critic_grammar() -> str`
- `critic_grammar_sha256() -> str` (including the `sha256:` prefix)

The grammar uses the declaration order of `CriticErrorCode` to emit each selected
code once. Every one of the 131,072 possible error sets remains expressible; the
empty set belongs to accept, nonempty sets to revise/uncertain. Permutations of a
set have one canonical order. No generated output is silently sorted, repaired,
or deduplicated. The grammar itself is a compact DAG with 17 suffix states and
O(n²) alternatives; it does not enumerate the subsets.

All original output fields and verdict-dependent constraints are preserved:
accept has no errors and an empty correction, revise has errors and a nonempty
correction, uncertain additionally requires an uncertain outcome. Predicted effect
is nonempty. JSON escapes and Unicode are supported; raw control characters are
rejected. Output uses deterministic single-space JSON separators, consistent with
the configured server's whitespace restriction. Python strict validation remains
mandatory, including the existing 128-word bound and duplicate rejection. The
grammar does not claim to enforce word limits or semantic correctness.

The caller must pin version and grammar hash in configuration, request identity,
and run manifests. Previous JSON-wire calibration cannot be reused as grammar-wire
calibration. Other roles retain their configured JSON schema wire. Config/provider
integration is separately coordinated; this module does not modify routing or
select its own wire mode.

Validation:

- Project tests: `uv run --frozen pytest -q tests/providers/test_critic_grammar.py
  tests/schemas/test_critic_conditional.py` — 13 passed, 0.57 seconds.
- Deployment compatibility uses the **already installed** vLLM environment's
  xgrammar on CPU only, explicitly authorized as a deployment diagnostic rather
  than project Python execution. No dependency install, sys.path modification,
  network access, model request or GPU work occurs.
- Actual xgrammar matcher exhaustively accepted all 131,071 nonempty sorted
  subsets; empty arrays were rejected by the nonempty array production and
  accepted in accept outputs. It accepted 36 full outputs covering verdicts,
  escaped Unicode/quotes/backslashes/control escapes and native Unicode, and
  rejected 11 malformed/duplicate/reversed/unknown-code/invalid-verdict cases.
- CPU matching elapsed 1.1967089897952974 seconds. Artifacts and standalone
  compatibility script: `/root/toolsandbox-runtime/proposed-fixes/critic-unique-grammar/`.
- Initial grammar hash:
  `sha256:2ea927db043ebde93c5a645fb4c42f790a9f524e8ba321429aa33459e112dee4`.

These checks establish the generated language, not real-model task performance.
