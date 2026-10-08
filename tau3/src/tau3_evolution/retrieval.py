"""Strict text-embedding-3-small cache and deterministic cosine index."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from . import EMBEDDING_MODEL
from .canonical import atomic_write_json, canonical_sha256, read_json
from .ledger import LogicalRequest, PhysicalAttempt


@dataclass(frozen=True)
class EmbeddingResponse:
    model: str
    vector: tuple[float, ...]
    attempt_id: str
    input_tokens: int | None
    output_tokens: int | None
    response_sha256: str


class EmbeddingProvider(Protocol):
    def embed(self, text: str, *, model: str, logical_request_id: str) -> EmbeddingResponse: ...

    def recover(self, *, logical_request_id: str) -> EmbeddingResponse | None: ...


@dataclass(frozen=True)
class StoredVector:
    record_kind: str
    record_id: str
    version: int
    content_sha256: str
    model: str
    dimension: int
    vector: tuple[float, ...]


def validate_vector(vector: tuple[float, ...], *, dimension: int | None = None) -> None:
    if not vector or any(type(value) is not float or not math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain finite floats")
    if dimension is not None and len(vector) != dimension:
        raise ValueError("embedding dimension mismatch")
    if math.fsum(value * value for value in vector) == 0.0:
        raise ValueError("zero embedding vector is not allowed")


class EmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._values = (
            read_json(path) if self.path.exists() else {"schema_version": 1, "values": {}}
        )
        if set(self._values) != {"schema_version", "values"} or self._values["schema_version"] != 1:
            raise ValueError("invalid embedding cache")

    def embed(
        self,
        text: str,
        *,
        provider: EmbeddingProvider,
        scope_id: str,
        purpose: str,
    ) -> tuple[tuple[float, ...], tuple[object, ...], str]:
        if type(text) is not str or not text:
            raise ValueError("non-empty embedding text required")
        content_sha = canonical_sha256(text)
        key = canonical_sha256({"model": EMBEDDING_MODEL, "content_sha256": content_sha})
        cached = self._values["values"].get(key)
        if cached is not None:
            if cached["model"] != EMBEDDING_MODEL or cached["content_sha256"] != content_sha:
                raise ValueError("embedding cache provenance mismatch")
            vector = tuple(cached["vector"])
            validate_vector(vector, dimension=cached["dimension"])
            return vector, (), content_sha
        logical_id = canonical_sha256(
            {
                "scope_id": scope_id,
                "purpose": purpose,
                "model": EMBEDDING_MODEL,
                "content": content_sha,
            }
        )
        request_payload = {
            "model": EMBEDDING_MODEL,
            "input_sha256": content_sha,
            "encoding_format": "float",
        }
        response = provider.recover(logical_request_id=logical_id)
        if response is None:
            response = provider.embed(text, model=EMBEDDING_MODEL, logical_request_id=logical_id)
        if response.model != EMBEDDING_MODEL:
            raise ValueError("embedding provider model drift; no fallback is allowed")
        if not response.response_sha256.startswith("sha256:") or not response.attempt_id:
            raise ValueError("embedding response lacks exact physical-attempt provenance")
        for count in (response.input_tokens, response.output_tokens):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError("invalid embedding usage")
        validate_vector(response.vector)
        request = LogicalRequest(
            logical_id,
            scope_id,
            "embedding",
            purpose,
            canonical_sha256(request_payload),
            EMBEDDING_MODEL,
            canonical_sha256({"encoding_format": "float"}),
        )
        attempt = PhysicalAttempt(
            response.attempt_id,
            logical_id,
            response.input_tokens,
            response.output_tokens,
            True,
            response.response_sha256,
        )
        self._values["values"][key] = {
            "content_sha256": content_sha,
            "model": EMBEDDING_MODEL,
            "dimension": len(response.vector),
            "vector": list(response.vector),
            "origin": {
                "logical_request_id": logical_id,
                "scope_id": scope_id,
                "purpose": purpose,
                "request": asdict(request),
                "attempt": asdict(attempt),
            },
        }
        atomic_write_json(self.path, self._values)
        return response.vector, (request, attempt), content_sha

    def recover_operation(
        self, text: str, *, provider: EmbeddingProvider, scope_id: str, purpose: str
    ) -> tuple[tuple[float, ...], tuple[object, ...], str] | None:
        content_sha = canonical_sha256(text)
        key = canonical_sha256({"model": EMBEDDING_MODEL, "content_sha256": content_sha})
        logical_id = canonical_sha256(
            {
                "scope_id": scope_id,
                "purpose": purpose,
                "model": EMBEDDING_MODEL,
                "content": content_sha,
            }
        )
        cached = self._values["values"].get(key)
        if cached is None:
            if provider.recover(logical_request_id=logical_id) is None:
                return None
            return self.embed(text, provider=provider, scope_id=scope_id, purpose=purpose)
        if cached["model"] != EMBEDDING_MODEL or cached["content_sha256"] != content_sha:
            raise ValueError("embedding cache provenance mismatch")
        vector = tuple(cached["vector"])
        validate_vector(vector, dimension=cached["dimension"])
        origin = cached.get("origin")
        if origin is not None and origin.get("logical_request_id") == logical_id:
            events = (
                LogicalRequest(**origin["request"]),
                PhysicalAttempt(**origin["attempt"]),
            )
        else:
            events = ()
        return vector, events, content_sha


def rank_vectors(query: tuple[float, ...], records: tuple[StoredVector, ...], maximum: int = 3):
    validate_vector(query)
    if maximum != 3:
        raise ValueError("formal retrieval is fixed at top three")
    scored = []
    q_norm = math.sqrt(math.fsum(value * value for value in query))
    for record in records:
        if record.model != EMBEDDING_MODEL or record.dimension != len(query):
            raise ValueError("stored embedding model or dimension mismatch")
        validate_vector(record.vector, dimension=record.dimension)
        score = math.fsum(left * right for left, right in zip(query, record.vector, strict=True))
        score /= q_norm * math.sqrt(math.fsum(value * value for value in record.vector))
        scored.append((score, record))
    return tuple(
        record
        for _, record in sorted(
            scored,
            key=lambda item: (
                -item[0],
                item[1].record_id.encode("utf-8"),
                -item[1].version,
                item[1].record_kind.encode("utf-8"),
            ),
        )[:maximum]
    )
