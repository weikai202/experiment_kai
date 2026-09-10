"""Deterministic matching-role candidate retrieval with staged-round visibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from toolsandbox_pipeline.retrieval.contracts import ResolvedEmbedding, RetrievalError, validate_vector
from toolsandbox_pipeline.retrieval.index import cosine
from toolsandbox_pipeline.retrieval.queries import document, validate_input
from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.generation import RetrievalIndexEntry
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory
from toolsandbox_pipeline.schemas.offline_memory import PolicyMemoryCandidate, WorldMemoryCandidate


class CandidateEmbeddingResolver(Protocol):
    def resolve_one(self, text: str) -> ResolvedEmbedding: ...


@dataclass(frozen=True)
class CandidateMatches:
    records: tuple[PolicyMemory | WorldMemory, ...]
    embedding: ResolvedEmbedding


def candidate_document(candidate: PolicyMemoryCandidate | WorldMemoryCandidate) -> str:
    if type(candidate) not in (PolicyMemoryCandidate, WorldMemoryCandidate):
        raise TypeError("concrete memory candidate required")
    return validate_input(
        canonical_json_bytes(candidate.candidate.model_dump(mode="json")).decode("utf-8")
    )


class MemoryCandidateRetriever:
    def __init__(
        self,
        *,
        generation_id: str,
        policy_records: tuple[PolicyMemory, ...],
        world_records: tuple[WorldMemory, ...],
        indexes: tuple[RetrievalIndexEntry, ...],
        embedding_resolver: CandidateEmbeddingResolver,
    ) -> None:
        self.generation_id = generation_id
        self.embedding_resolver = embedding_resolver
        self._current = {
            "policy": {record.memory_id: record for record in policy_records if record.status == "active"},
            "world": {record.memory_id: record for record in world_records if record.status == "active"},
        }
        self._vectors: dict[str, tuple[float, ...]] = {}
        for entry in indexes:
            if entry.generation_id != generation_id or entry.record_kind not in {"policy", "world"}:
                continue
            if entry.record_id not in self._current[entry.record_kind]:
                raise RetrievalError("offline index contains stale active memory")
            self._vectors[entry.record_id] = validate_vector(entry.vector)
        missing = set(self._current["policy"]) | set(self._current["world"])
        if missing != set(self._vectors):
            raise RetrievalError("offline candidate retrieval requires complete active indexes")
        self._staged: dict[str, dict[str, PolicyMemory | WorldMemory]] = {
            "policy": {}, "world": {}
        }

    def retrieve(
        self, candidate: PolicyMemoryCandidate | WorldMemoryCandidate
    ) -> CandidateMatches:
        if type(candidate) not in (PolicyMemoryCandidate, WorldMemoryCandidate):
            raise TypeError("concrete memory candidate required")
        resolved = self.embedding_resolver.resolve_one(candidate_document(candidate))
        vector = validate_vector(resolved.vector)
        role = candidate.role
        visible = {**self._current[role], **self._staged[role]}
        scored = []
        for memory_id, record in visible.items():
            stored = self._vectors.get(memory_id)
            if stored is None:
                raise RetrievalError("staged memory vector is missing")
            scored.append((record, cosine(stored, vector)))
        ranked = tuple(
            record
            for record, _ in sorted(
                scored,
                key=lambda item: (-item[1], item[0].memory_id.encode("utf-8")),
            )[:3]
        )
        return CandidateMatches(records=ranked, embedding=resolved)

    def remember_staged(
        self,
        record: PolicyMemory | WorldMemory,
        *,
        semantic_vector: tuple[float, ...],
    ) -> None:
        if type(record) is PolicyMemory:
            role = "policy"
        elif type(record) is WorldMemory:
            role = "world"
        else:
            raise TypeError("strict memory record required")
        if record.status != "active":
            raise RetrievalError("only active staged memory is searchable")
        vector = validate_vector(semantic_vector)
        existing = self._staged[role].get(record.memory_id)
        if existing is not None and existing != record:
            raise RetrievalError("conflicting staged memory identity")
        self._staged[role][record.memory_id] = record
        self._vectors[record.memory_id] = vector

    def remember_recovered(self, record: PolicyMemory | WorldMemory) -> None:
        """Restore a committed staged record through the audited embedding resolver."""

        resolved = self.embedding_resolver.resolve_one(document(record))
        self.remember_staged(record, semantic_vector=resolved.vector)

    def vector_for(self, memory_id: str) -> tuple[float, ...]:
        try:
            return self._vectors[memory_id]
        except KeyError as error:
            raise RetrievalError("unknown memory vector") from error


__all__ = [
    "CandidateEmbeddingResolver",
    "CandidateMatches",
    "MemoryCandidateRetriever",
    "candidate_document",
]
