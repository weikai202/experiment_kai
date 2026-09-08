#!/usr/bin/env bash
set -euo pipefail
# Run on allocated GPUs in an environment with vLLM installed.
# Initial model defaults to Hugging Face; a trained checkpoint can be passed as argument 1.
model_path="${1:-Qwen/Qwen3-32B}"
served_name="${SERVED_MODEL_NAME:-Qwen/Qwen3-32B}"
revision_args=()
if [[ "$model_path" == "Qwen/Qwen3-32B" ]]; then
  revision_args=(--revision 9216db5781bf21249d130ec9da846c4624c16137 --tokenizer-revision 9216db5781bf21249d130ec9da846c4624c16137)
fi
exec "${VLLM_EXECUTABLE:-vllm}" serve "$model_path" \
  "${revision_args[@]}" \
  --served-model-name "$served_name" \
  --host "${VLLM_HOST:-127.0.0.1}" --port "${VLLM_PORT:-8000}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
  --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
