"""Normalize provider usage without tokenization or estimation."""
from collections.abc import Mapping
from toolsandbox_pipeline.schemas.usage import TokenUsage


def normalize_usage(raw, *, embedding: bool = False) -> TokenUsage:
    if raw is None:
        return TokenUsage(output_tokens=0, cache_read_input_tokens=0,
                          cache_write_input_tokens=0) if embedding else TokenUsage()
    if not isinstance(raw, Mapping):
        raise ValueError("invalid usage")

    def count(source, key, default=None):
        value = source.get(key, default)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("invalid token count")
        return value

    prompt = count(raw, "prompt_tokens")
    total = count(raw, "total_tokens")
    output = 0 if embedding else count(raw, "completion_tokens")
    details = raw.get("prompt_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, Mapping):
        raise ValueError("invalid cache details")
    read = 0 if embedding else count(details, "cached_tokens", 0)
    write = 0 if embedding else count(details, "cache_write_tokens", 0)
    # Some compatible servers report cache counts directly.
    if not embedding:
        for key, old in (("cache_read_input_tokens", read), ("cache_write_input_tokens", write)):
            if key in raw:
                new = count(raw, key)
                detail_key = "cached_tokens" if key.startswith("cache_read") else "cache_write_tokens"
                if detail_key in details and new != old:
                    raise ValueError("conflicting cache usage")
                if key.startswith("cache_read"):
                    read = new
                else:
                    write = new
    uncached = prompt - read - write if all(v is not None for v in (prompt, read, write)) else None
    counts = dict(input_tokens=prompt, output_tokens=output, total_tokens=total,
                  uncached_input_tokens=uncached, cache_read_input_tokens=read,
                  cache_write_input_tokens=write)
    return TokenUsage(**counts, usage_complete=all(v is not None for v in counts.values()))


class UsageRecorder:
    """Per-role in-memory sink; persistent request identity belongs to the ledger.

    Identical re-recordings are idempotent. Distinct replay attempt IDs count
    separately. No run latency or applied-Qwen-output cost is computed here.
    """
    def __init__(self, role):
        self.role = role
        self._attempts = {}

    @property
    def attempts(self):
        return tuple(self._attempts.values())

    def record(self, attempt):
        if attempt.context.role != self.role:
            raise ValueError("recorder role mismatch")
        key = attempt.context.attempt_id
        if key in self._attempts and self._attempts[key] != attempt:
            raise ValueError("conflicting attempt ID")
        self._attempts[key] = attempt

    def token_totals(self) -> TokenUsage:
        names = ("input_tokens", "output_tokens", "total_tokens", "uncached_input_tokens",
                 "cache_read_input_tokens", "cache_write_input_tokens")
        values = {}
        for name in names:
            counts = [getattr(a.metrics.usage, name) for a in self.attempts]
            values[name] = None if None in counts else sum(counts)
        return TokenUsage(**values, usage_complete=all(v is not None for v in values.values()))
