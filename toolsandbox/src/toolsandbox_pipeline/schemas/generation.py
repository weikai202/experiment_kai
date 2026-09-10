"""Complete generation and index manifests."""
from typing import Annotated, Literal
from pydantic import Field, model_validator

from .memory import Count, Digest, FrozenRecord, GenerationId, Identifier, PolicyMemory, WorldMemory, unique
from .skill import SkillRecord
from toolsandbox_pipeline.retrieval.contracts import EmbeddingBatchConfig, EmbeddingCacheKey, EmbeddingIdentity, validate_vector

UPSTREAM_COMMIT = "165848b9a78cead7ca7fe7c89c688b58e6501219"
STORE_PATHS = ("policy_memory.jsonl", "world_memory.jsonl", "skills.jsonl")
INDEX_PATHS = ("retrieval_indexes/policy.jsonl", "retrieval_indexes/world.jsonl", "retrieval_indexes/skill.jsonl", "retrieval_indexes/manifest.json")


class GenerationFileEntry(FrozenRecord):
    path: Literal["policy_memory.jsonl", "world_memory.jsonl", "skills.jsonl", "retrieval_indexes/policy.jsonl", "retrieval_indexes/world.jsonl", "retrieval_indexes/skill.jsonl", "retrieval_indexes/manifest.json"]
    sha256: Digest
    record_count: Count


class RetrievalIndexEntry(FrozenRecord):
    generation_id: GenerationId
    record_kind: Literal["policy", "world", "skill"]
    record_id: Identifier
    record_version: Annotated[str, Field(pattern=r"^v1\.(0|[1-9][0-9]*)$")] | None = None
    document_sha256: Digest
    embedding_cache_key: EmbeddingCacheKey
    vector_dimension: Annotated[int, Field(gt=0)]
    vector: tuple[float, ...]

    @model_validator(mode="after")
    def invariants(self):
        if (self.record_kind == "skill") != (self.record_version is not None):
            raise ValueError("only Skill entries have a version")
        if self.document_sha256 != self.embedding_cache_key.input_sha256:
            raise ValueError("document/cache identity mismatch")
        validate_vector(self.vector, self.vector_dimension)
        return self


class RetrievalIndexManifest(FrozenRecord):
    schema_version: Literal[1]
    generation_id: GenerationId
    embedding: EmbeddingIdentity
    vector_dimension: Annotated[int, Field(gt=0)]
    batching: EmbeddingBatchConfig
    state_serialization: Literal["retrieval-state-v1"]
    action_serialization: Literal["retrieval-action-v1"]
    document_serialization: Literal["retrieval-document-v1"]
    stores: tuple[GenerationFileEntry, ...]
    indexes: tuple[GenerationFileEntry, ...]

    @model_validator(mode="after")
    def layout(self):
        if tuple(e.path for e in self.stores) != STORE_PATHS or tuple(e.path for e in self.indexes) != INDEX_PATHS[:3]:
            raise ValueError("unexpected index manifest layout")
        return self


class GenerationManifest(FrozenRecord):
    schema_version: Literal[1]
    generation_id: GenerationId
    parent_generation_id: GenerationId | None
    publication_status: Literal["complete"]
    upstream_commit: Literal["165848b9a78cead7ca7fe7c89c688b58e6501219"]
    tool_inventory_sha256: Digest
    embedding: EmbeddingIdentity
    vector_dimension: Annotated[int, Field(gt=0)]
    files: tuple[GenerationFileEntry, ...]

    @model_validator(mode="after")
    def invariants(self):
        number = int(self.generation_id[1:])
        expected = f"g{number - 1:03d}" if number else None
        if self.parent_generation_id != expected:
            raise ValueError("incorrect parent generation")
        if tuple(entry.path for entry in self.files) != STORE_PATHS + INDEX_PATHS:
            raise ValueError("exact ordered complete file set required")
        if self.files[-1].record_count != 1:
            raise ValueError("one index manifest required")
        return self


class GenerationSnapshot(FrozenRecord):
    manifest: GenerationManifest
    policy_memory: tuple[PolicyMemory, ...]
    world_memory: tuple[WorldMemory, ...]
    skills: tuple[SkillRecord, ...]
    index_manifest: RetrievalIndexManifest
    indexes: tuple[RetrievalIndexEntry, ...]
