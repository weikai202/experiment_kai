# Experimental protocol and reproduction record

Recorded on 2026-09-07.

- User-selected policy and data generator: Qwen3-32B in non-thinking mode.
- Original standalone project: `/home/weik/early-experience-bfcl`. It reuses the existing Gorilla installation without modifying upstream files. The published copy is under `EarlyExperience/bfcl`.
- Validated BFCL commit: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, dated 2026-03-23.
- The EarlyExperience reference notes specify a December 2025 BFCL version. This adaptation records and uses the installed version. Successful compatibility checks do not establish an exact reproduction of the paper's experiments.
- Expert inputs are the Base result and score files for `claude-opus-4-5-20251101-FC` under `HuanzhiMao/BFCL-Result/main/2025-12-16`. Exact SHA256 values are recorded in the locally generated `data/prepared/manifest.json`.
- Only the 162 officially successful Base cases are used. Numeric case IDs are sorted, then shuffled with an isolated `random.Random(42)`. The first 121 cases form the training set and the remaining 41 form the held-out set. Exact IDs are saved. The reference notes do not publish the full split ID list, so equality with their original assignment is not claimed.
- Each of the three OOD categories contains 200 cases and is excluded from training. The 38 failed expert Base cases are excluded from training and held-out reporting under this protocol.
- Observations come from the actual simulator. K=10 alternative method names are sampled structurally, arguments are generated in one request, and each candidate executes on a separate deepcopy. SR uses K=3 of those alternatives.
- IWM predicts descriptive observation summaries. Raw outputs and post-action states remain in intermediate records and are not exposed as additional policy inputs.
- SR uses explicit `<reflection>` text followed by a fixed expert action. The prompt targets about 200 words with a soft limit of about 500 words. Qwen's built-in thinking mode remains disabled.
- The API output limit controls serving resources. Any finish reason other than `stop` is treated as incomplete and raises an error; truncated examples are not accepted.
- IWM retains execution errors. SR audits vocabulary leakage by default; `--filter-leaks` enables the reference vocabulary filter and records its impact.
- Natural-language expert closing responses are normalized to `[]`, with corresponding stop examples added consistently across all three baselines.
- Policy prompts use the official prompting-mode system instructions and tool documentation. Tool observations are converted to equivalent `<tool_response>` user text. Equivalence was checked with the official Qwen tokenizer.
- Scoring uses the official state/response and irrelevance checkers. Direct method calls with AST literal arguments replace upstream `eval`. Syntax compatibility was checked for all 4,625 reference calls across the four supported categories.
- Training defaults to full-parameter SFT, one epoch per stage, and learning rate 1e-5. These are configurable adaptation defaults. LoRA is optional and should be used consistently across baselines and reported explicitly.

The original offline validation passed 22 tests. Replaying all successful expert trajectories produced zero tool-observation mismatches. The expert training corpus contains 1,196 examples and 8,461,337 tokens, including 15,929 supervised tokens. The maximum sequence length is 10,197 tokens. Tokenizer validation used official Qwen3 files with the installed Transformers 5.16.1. Optional training dependencies are separately constrained to Transformers 4.51 through 4.x; full GPU training has not been run.

Local validation artifacts are `outputs/validation/tests.xml` and `outputs/validation/data_and_tokenizer.json`. They are not committed to the repository. Test generators and oracles validate the implementation only and do not represent model benchmark results. The recorded numbers of real model API requests and GPU training runs are both zero.
