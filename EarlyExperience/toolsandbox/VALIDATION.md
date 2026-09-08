# Standalone EarlyExperience validation — 2026-09-08

Scope: IL, IWM→IL, SR, direct native ToolSandbox evaluation. No dependency on the user's pipeline or G000/G003.

- uv sync removed toolsandbox-pipeline; pyproject.toml, uv.lock and active source no longer reference it.
- Ruff passed.
- Final standalone tests: **11 passed**, one upstream holidays warning; **4.60 seconds**.
- Real native tools and evaluator tested with synthetic fixtures: branch isolation, runtime errors, native episode+dataset construction, visibility, ID normalization, split leakage prevention, completion loss masking.
- Additional checks: installed package dependencies contain no pipeline; Qwen non-thinking/seed request shape and failed usage retention; explicit expert-selection exclusion; full standalone evaluate CLI function through native evaluator (synthetic case).
- collect/evaluate/prepare CLI and IL/IWM/SR training help entrypoints passed.
- Prior real local Qwen3 tokenizer verification remains in validation/qwen_tokenizer_validation.json; full GPU training was not run.
- No live model calls, formal data collection or benchmark evaluation were executed in this turn. Generated model/API tokens: 0.

Historical pipeline alignment code, tests, and reports are not included in this standalone distribution and are not prerequisites.

## Publication verification

The standalone copy in `EarlyExperience/toolsandbox` was checked on 2026-09-08 using the existing Python 3.10 runtime environment with `PYTHONPATH=src`. All 11 tests passed in 2.53 seconds with one upstream holidays warning, and Ruff passed. Import resolution was checked to confirm that the published source was exercised. This is offline validation, not a benchmark run.
