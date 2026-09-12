#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
unset OPENAI_API_KEY QWEN_API_KEY HF_TOKEN
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
exec /root/toolsandbox-runtime/vllm-env/bin/vllm serve /root/toolsandbox-runtime/models/Qwen3-32B --served-model-name Qwen/Qwen3-32B --host 127.0.0.1 --port 18080 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 2 --generation-config vllm --structured-outputs-config '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}'  --enforce-eager --disable-log-requests
