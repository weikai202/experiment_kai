#!/usr/bin/env bash
set -euo pipefail
exec "${VLLM_BIN:-vllm}" serve "${COMAP_MODEL_PATH:-Qwen/Qwen3-32B}" \
  --served-model-name Qwen/Qwen3-32B \
  --host 127.0.0.1 --port "${COMAP_PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size "${COMAP_TP_SIZE:-1}" \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config vllm "$@"
