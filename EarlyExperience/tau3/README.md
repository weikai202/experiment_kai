# Early Experience baseline for tau3-bench

This directory contains data generation, training, and native tau3 evaluation
adapters for the IL, IWM -> IL, and SR baselines. It supports text half-duplex
tasks in retail, airline, and telecom.

## Use with the pinned tau3 environment

The experiment module follows the upstream tau3 directory structure. Copy it into
the pinned upstream checkout, then follow the full guide from that checkout's
root to install dependencies, generate data, train models, and evaluate them.

```bash
# Run from the experiment_kai repository root; use a new upstream directory.
git clone https://github.com/sierra-research/tau2-bench.git ../tau3-early-experience
git -C ../tau3-early-experience checkout 17e07b1da2bbc0cadfddeea36412686e0604127b
cp -R EarlyExperience/tau3/src/experiments/early_experience \
  ../tau3-early-experience/src/experiments/
cd ../tau3-early-experience
```

Continue with the [full workflow guide](src/experiments/early_experience/README.md).
The native Python package and CLI retain the upstream name `tau2`.

## Files

Paths below are relative to `src/experiments/early_experience/`.

- `core.py`: isolated environment branches, training formats, and split validation.
- `pipeline.py`: real branch sampling, reflection generation, resume support, and SFT export.
- `train.py`: IL, two-stage IWM, and mixed SR training.
- `agent.py` and `__main__.py`: inference adapter and collection/evaluation commands.
- `tests/`: data, real domain tools, native evaluator, and tiny-model CPU training tests.

Validation: 14 data/environment/evaluator tests and 3 CPU training tests passed,
along with Ruff checks. No full-scale base-model training or official benchmark
scoring has been run.

Method reference: [OSU-NLP-Group/EarlyExperience](https://github.com/OSU-NLP-Group/EarlyExperience).
