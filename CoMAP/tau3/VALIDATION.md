> Historical validation of the earlier 144/34 protocol. The current confirmed
> protocol is official178; see configs/pipeline_aligned.yaml and README.md.

# Implementation validation

Validated on 2026-09-07. This is not a core benchmark performance report.

- τ³ source HEAD: `17e07b1da2bbc0cadfddeea36412686e0604127b`.
- No tracked τ³ source files were changed by this implementation.
- Python 3.13.9; torch 2.14.0+cpu; transformers 4.57.6; pytest 9.1.1.
- 22 tests passed across the final relevant invocations:
  - 16 agent, split, Mock integration, and training tests;
  - 3 runner artifact/usage tests;
  - 3 public core domain interface tests.
- `ruff check src tests`: passed.
- `ruff format --check src tests`: 14 files already formatted.
- CLI help and official split fingerprint/count validation passed.
- Mock integration called real native Mock tools and received native reward 1.0
  with a scripted policy/world model/user. This does not measure LLM capability.
- Tiny locally initialized GPT2 models completed real WM and policy optimizer
  updates; saved student and EMA teacher checkpoints reloaded successfully.
- No real model API calls, large-model training, or core test evaluation ran.

Candidate split SHA256:
`ef1ecee5a963a9952d9333ecdc24d26a14ed30ffa33036d46c3d7aee612aec0f`

This candidate has 144 evolution tasks, 34 dev tasks, and 100 test tasks. Its task
assignments have not been verified against the user's previously fixed pipeline
manifest. Model/checkpoint identifiers and simulator configuration remain unset.

For formal reporting, describe this as a CoMAP adaptation with a single-device
full-parameter HF trainer and JSON action protocol. Consult README.md for the
source mapping and differences from the public ALFWorld/verl implementation.
