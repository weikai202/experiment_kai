"""Standalone evolution pipeline contracts for tau3 core text domains."""

TAU_COMMIT = "17e07b1da2bbc0cadfddeea36412686e0604127b"
QWEN_MODEL = "Qwen/Qwen3-32B"
USER_SIMULATOR_MODEL = "gpt-4o-mini-2024-07-18"
EMBEDDING_MODEL = "text-embedding-3-small"
DOMAINS = ("airline", "retail", "telecom")
SYSTEM_ORDER = ("vanilla", "generation_0", "updated")

__all__ = [
    "DOMAINS",
    "EMBEDDING_MODEL",
    "QWEN_MODEL",
    "SYSTEM_ORDER",
    "TAU_COMMIT",
    "USER_SIMULATOR_MODEL",
]
