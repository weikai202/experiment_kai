from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol

from .canonical import sha256_json
from .sanitization import SanitizedEpisodeEvidence

EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class GenerationResource:
    resource_id: str
    generation_id: str
    kind: str
    content: str
    canonical_tool_dependencies: tuple[str, ...]
    creation_effect_id: str
    resource_sha256: str

    @classmethod
    def build(
        cls,
        resource_id: str,
        generation_id: str,
        kind: str,
        content: str,
        dependencies: Iterable[str],
        creation_effect_id: str = "seed",
    ) -> "GenerationResource":
        if kind not in {"policy_memory", "world_memory", "skill"}:
            raise ValueError("Unknown generation resource kind")
        if not creation_effect_id:
            raise ValueError("Generation resource requires creation-effect provenance")
        deps = tuple(sorted(set(dependencies)))
        core = {
            "resource_id": resource_id,
            "generation_id": generation_id,
            "kind": kind,
            "content": content,
            "canonical_tool_dependencies": deps,
            "creation_effect_id": creation_effect_id,
        }
        return cls(**core, resource_sha256=sha256_json(core))

    def validate(self) -> None:
        core = {
            "resource_id": self.resource_id,
            "generation_id": self.generation_id,
            "kind": self.kind,
            "content": self.content,
            "canonical_tool_dependencies": self.canonical_tool_dependencies,
            "creation_effect_id": self.creation_effect_id,
        }
        if self.resource_sha256 != sha256_json(core):
            raise ValueError("Generation resource content hash is invalid")


@dataclass(frozen=True)
class GenerationResourceManifest:
    generation_id: str
    skill_records: tuple[tuple[str, str, str], ...]
    manifest_sha256: str

    @classmethod
    def build(cls, generation_id: str, resources: Iterable[GenerationResource]) -> "GenerationResourceManifest":
        rows = tuple(resources)
        for resource in rows:
            resource.validate()
            if resource.generation_id != generation_id:
                raise ValueError("Resource manifest contains another generation")
        records = tuple(sorted(
            (
                (resource.resource_id, resource.resource_sha256, resource.creation_effect_id)
                for resource in rows
                if resource.kind == "skill"
            ),
            key=lambda row: row[0].encode("utf-8"),
        ))
        if len({row[0] for row in records}) != len(records):
            raise ValueError("Resource manifest contains duplicate Skill versions")
        core = {"generation_id": generation_id, "skill_records": records}
        return cls(generation_id, records, sha256_json(core))

    def validate(self) -> None:
        core = {"generation_id": self.generation_id, "skill_records": self.skill_records}
        if self.manifest_sha256 != sha256_json(core):
            raise ValueError("Generation resource manifest hash is invalid")


@dataclass(frozen=True)
class ResourceEmbedding:
    resource_id: str
    resource_sha256: str
    model: str
    client_config_sha256: str
    vector: tuple[float, ...]
    vector_sha256: str

    @classmethod
    def build(cls, resource: GenerationResource, client_config_sha256: str, vector: tuple[float, ...]) -> "ResourceEmbedding":
        return cls(resource.resource_id, resource.resource_sha256, EMBEDDING_MODEL, client_config_sha256, vector, sha256_json(vector))


@dataclass(frozen=True)
class QueryEmbedding:
    input_sha256: str
    model: str
    client_config_sha256: str
    vector: tuple[float, ...]
    vector_sha256: str


@dataclass(frozen=True)
class RetrievalReceipt:
    resource_id: str
    resource_sha256: str
    generation_id: str
    kind: str
    creation_effect_id: str
    query_sha256: str
    visible_tools_sha256: str
    rank: int
    embedding_model: str
    embedding_client_config_sha256: str
    receipt_sha256: str

    @classmethod
    def build(
        cls,
        resource: GenerationResource,
        query: str,
        visible_tools: tuple[str, ...],
        rank: int,
        client_config_sha256: str,
    ) -> "RetrievalReceipt":
        core = {
            "resource_id": resource.resource_id,
            "resource_sha256": resource.resource_sha256,
            "generation_id": resource.generation_id,
            "kind": resource.kind,
            "creation_effect_id": resource.creation_effect_id,
            "query_sha256": sha256_json(query),
            "visible_tools_sha256": sha256_json(visible_tools),
            "rank": rank,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_client_config_sha256": client_config_sha256,
        }
        return cls(**core, receipt_sha256=sha256_json(core))

    def validate(self) -> None:
        core = {key: value for key, value in self.__dict__.items() if key != "receipt_sha256"}
        if self.embedding_model != EMBEDDING_MODEL or self.rank < 0 or self.receipt_sha256 != sha256_json(core):
            raise ValueError("Retrieval receipt provenance is invalid")


class EmbeddingProvider(Protocol):
    model: str
    client_config_sha256: str

    def embed(self, text: str) -> QueryEmbedding: ...


def _validate_vector(vector: tuple[float, ...]) -> None:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding vector must be non-empty and finite")
    if math.fsum(value * value for value in vector) == 0:
        raise ValueError("Zero-norm embeddings are forbidden")


class GenerationResourceStore:
    def __init__(self, generation_id: str, resources: Iterable[GenerationResource], embeddings: Iterable[ResourceEmbedding], provider: EmbeddingProvider):
        self.generation_id = generation_id
        self._resources = tuple(resources)
        self._provider = provider
        if any(x.generation_id != generation_id for x in self._resources):
            raise ValueError("Cross-generation resources are forbidden")
        if len({x.resource_id for x in self._resources}) != len(self._resources):
            raise ValueError("Duplicate generation resource ID")
        if provider.model != EMBEDDING_MODEL:
            raise ValueError("Only text-embedding-3-small is allowed")
        embedding_rows = tuple(embeddings)
        records = {x.resource_id: x for x in embedding_rows}
        if len(records) != len(embedding_rows) or set(records) != {x.resource_id for x in self._resources}:
            raise ValueError("Every resource requires one unique embedding record")
        dimensions = set()
        for resource in self._resources:
            resource.validate()
            record = records[resource.resource_id]
            if (
                record.resource_sha256 != resource.resource_sha256
                or record.model != EMBEDDING_MODEL
                or record.client_config_sha256 != provider.client_config_sha256
                or record.vector_sha256 != sha256_json(record.vector)
            ):
                raise ValueError("Resource embedding provenance mismatch")
            _validate_vector(record.vector)
            dimensions.add(len(record.vector))
        if len(dimensions) != 1:
            raise ValueError("Resource embedding dimensions are inconsistent")
        self._embeddings = records
        self._dimension = next(iter(dimensions))

    def retrieve(self, visible_tools: Iterable[str], query: str, limit: int) -> tuple[GenerationResource, ...]:
        if limit < 0 or not query:
            raise ValueError("Retrieval query and limit are invalid")
        tools = set(visible_tools)
        eligible = [x for x in self._resources if set(x.canonical_tool_dependencies) <= tools]
        if not eligible or limit == 0:
            return ()
        result = self._provider.embed(query)
        if (
            result.input_sha256 != sha256_json(query)
            or result.model != EMBEDDING_MODEL
            or result.client_config_sha256 != self._provider.client_config_sha256
            or result.vector_sha256 != sha256_json(result.vector)
        ):
            raise ValueError("Query embedding provenance mismatch")
        _validate_vector(result.vector)
        if len(result.vector) != self._dimension:
            raise ValueError("Query embedding dimension mismatch")

        def cosine(resource: GenerationResource) -> float:
            vector = self._embeddings[resource.resource_id].vector
            denominator = math.sqrt(math.fsum(x * x for x in vector)) * math.sqrt(math.fsum(x * x for x in result.vector))
            return math.fsum(x * y for x, y in zip(vector, result.vector)) / denominator

        eligible.sort(key=lambda x: (-cosine(x), x.resource_id.encode("utf-8")))
        return tuple(eligible[:limit])

    def retrieve_with_receipts(self, visible_tools: Iterable[str], query: str, limit: int) -> tuple[RetrievalReceipt, ...]:
        tools = tuple(sorted(set(visible_tools), key=lambda x: x.encode("utf-8")))
        resources = self.retrieve(tools, query, limit)
        receipts = tuple(
            RetrievalReceipt.build(resource, query, tools, rank, self._provider.client_config_sha256)
            for rank, resource in enumerate(resources)
        )
        for receipt, resource in zip(receipts, resources):
            receipt.validate()
            if receipt.resource_sha256 != resource.resource_sha256:
                raise ValueError("Retrieval receipt does not match the ranked resource")
        return receipts


class OfflineUpdater(Protocol):
    def propose(self, generation_id: str, evidence: SanitizedEpisodeEvidence) -> tuple[GenerationResource, ...]: ...
