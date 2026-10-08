# Model Access Boundary

Every online and offline pipeline role uses the same frozen
`Qwen/Qwen3-32B` endpoint. Requests use temperature 0, seed 0, and explicit
`chat_template_kwargs={"enable_thinking": false}`. Runtime configuration is
injected with `QWEN_BASE_URL` and `QWEN_API_KEY`; credentials are never stored in
the repository, prompts, checkpoints, or reports.

Formal Qwen requests fail closed without a canonical calibration receipt from a
real smoke over train IDs only. The receipt binds the dataset manifest, model,
base decoding configuration, ordered cases, observed output-token lengths,
finish reasons, selected `max_tokens`, and its own hash. The same receipt and
selected limit are bound into the final plan. Offline synthetic receipts are
explicitly labeled and cannot authorize a formal run.

BFCL supplies fixed user turns, so this project does not call an OpenAI User
Simulator. Offline tests inject fake Policy, Critic, Revision, and updater roles
and make no network calls. A fake-role pass is infrastructure validation, not a
model-backed result. Before a live train smoke, an external coordinator must
verify served-model identity, non-thinking output, token usage fields, context
limits, and the pinned BFCL runtime. Test remains sealed during all smoke and
training operations.

Memory and Skill retrieval uses only OpenAI `text-embedding-3-small`, injected
through `OPENAI_API_KEY`. The store validates exact model identity, vector
dimension, finite values, and nonzero cosine inputs. There is no lexical, hash,
local-model, or alternate-embedding fallback.
