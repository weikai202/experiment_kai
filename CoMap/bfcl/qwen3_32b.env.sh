# Source from bash before invoking the CoMAP BFCL runner.
export COMAP_POLICY_MODEL=Qwen/Qwen3-32B
export COMAP_WM_MODEL=Qwen/Qwen3-32B
export COMAP_REGISTRY_NAME=CoMAP-Qwen3-32B-NoThink-FC
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export COMAP_WM_BASE_URL="${COMAP_WM_BASE_URL:-$OPENAI_BASE_URL}"
export COMAP_WM_API_KEY="${COMAP_WM_API_KEY:-$OPENAI_API_KEY}"
export COMAP_POLICY_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
export COMAP_WM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
export COMAP_MAX_TOKENS=2048
export COMAP_WM_MAX_TOKENS=1024
