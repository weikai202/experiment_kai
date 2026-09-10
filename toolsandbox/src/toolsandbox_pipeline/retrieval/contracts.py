"""Immutable retrieval identities and results."""
import math
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from toolsandbox_pipeline.schemas.memory import Digest, FrozenRecord, GenerationId, Identifier
from toolsandbox_pipeline.schemas.runtime import validate_endpoint


class RetrievalError(ValueError):
    """A corrupt or incompatible retrieval artifact; callers must stop."""


class EmbeddingIdentity(FrozenRecord):
    provider: Literal["openai"] = "openai"
    base_url_identity: str = "https://api.openai.com/v1"
    model: Literal["text-embedding-3-small"] = "text-embedding-3-small"
    encoding_format: Literal["float"] = "float"
    dimensions_setting: Literal["omitted"] = "omitted"

    _endpoint = field_validator("base_url_identity")(validate_endpoint)


class EmbeddingCacheKey(EmbeddingIdentity):
    input_sha256: Digest


class EmbeddingBatchConfig(FrozenRecord):
    max_items: Annotated[int, Field(ge=1, le=2048)] = 2048
    max_bytes: Annotated[int, Field(ge=8000, le=280000)] = 280000


def validate_vector(vector, dimension=None):
    if not vector or any(type(v) is not float or not math.isfinite(v) for v in vector):
        raise RetrievalError("invalid vector")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0 or (dimension is not None and len(vector) != dimension):
        raise RetrievalError("zero vector or dimension mismatch")
    return tuple(vector)


class ResolvedEmbedding(FrozenRecord):
    key: EmbeddingCacheKey
    vector: tuple[float, ...]
    cache_hit: bool
    source_attempt_id: Identifier

    @model_validator(mode="after")
    def vector_valid(self):
        validate_vector(self.vector)
        return self


class RetrievalHit(FrozenRecord):
    record_id: Identifier
    record_version: str | None = None
    generation_id: GenerationId
    rank: Annotated[int, Field(ge=1, le=3)]
    score: Annotated[float, Field(ge=-1, le=1)]
    document_sha256: Digest
    query_sha256: Digest
    cache_hit: bool
    source_attempt_id: Identifier
