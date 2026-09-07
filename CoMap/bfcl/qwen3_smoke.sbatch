#!/usr/bin/env bash
#SBATCH --job-name=comap-qwen32-smoke
#SBATCH --partition=gpu_small
#SBATCH --gres=gpu:A100_80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=comap-qwen3-%j.log

set -euo pipefail
# Submit from the experiment_kai repository root; override paths through env.
cd "${COMAP_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
eval_python="${COMAP_EVAL_PYTHON:-$PWD/CoMap/bfcl/.venv/bin/python}"
export VLLM_BIN="${VLLM_BIN:-vllm}"
export COMAP_PORT="${COMAP_PORT:-18000}"
export OPENAI_BASE_URL="http://127.0.0.1:$COMAP_PORT/v1"
source CoMap/bfcl/qwen3_32b.env.sh
run_dir="$PWD/CoMap/bfcl/results/qwen3-smoke-$SLURM_JOB_ID"
mkdir -p "$run_dir"
bash CoMap/bfcl/serve_qwen3_32b.sh > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
ready=false
for attempt in $(seq 1 240); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 60 "$run_dir/server.log"
    exit 1
  fi
  if curl --fail --silent --max-time 2 "$OPENAI_BASE_URL/models" > /dev/null; then
    ready=true
    break
  fi
  sleep 5
done
if [[ "$ready" != true ]]; then
  echo 'Model server failed to become ready within 20 minutes.' >&2
  exit 1
fi
"$eval_python" -m CoMap.bfcl.smoke --result-dir "$run_dir/results"
"$eval_python" -m CoMap.bfcl.run evaluate \
  --model "$COMAP_REGISTRY_NAME" --test-category simple_python,multi_turn_base \
  --result-dir "$run_dir/results" --score-dir "$run_dir/scores" --partial-eval
