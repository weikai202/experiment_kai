"""Generation-scoped Policy/World Memory and Skill store with strict semantic retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from . import DOMAINS, EMBEDDING_MODEL
from .canonical import atomic_write_json, canonical_sha256, read_json
from .online_pipeline import RetrievedContext
from .retrieval import (
    EmbeddingCache,
    EmbeddingProvider,
    StoredVector,
    rank_vectors,
    validate_vector,
)


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    version: int
    memory_kind: str
    domains: tuple[str, ...]
    tool_dependencies: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    version: int
    domains: tuple[str, ...]
    tool_dependencies: tuple[str, ...]
    content: str
    accepted_effect_id: str | None
    dev_evidence_sha256: str | None


@dataclass(frozen=True)
class GenerationState:
    generation_id: str
    parent_generation_id: str | None
    memories: tuple[MemoryRecord, ...]
    skills: tuple[SkillRecord, ...]
    vectors: tuple[StoredVector, ...]
    seed_library_sha256: str

    @property
    def state_sha256(self) -> str:
        return canonical_sha256(asdict(self))


def generation_state_from_document(document: dict) -> GenerationState:
    expected = {
        "generation_id",
        "parent_generation_id",
        "memories",
        "skills",
        "vectors",
        "seed_library_sha256",
        "state_sha256",
    }
    if set(document) != expected:
        raise ValueError("generation document has missing or extra fields")
    memory_fields = {
        "record_id",
        "version",
        "memory_kind",
        "domains",
        "tool_dependencies",
        "content",
    }
    skill_fields = {
        "skill_id",
        "version",
        "domains",
        "tool_dependencies",
        "content",
        "accepted_effect_id",
        "dev_evidence_sha256",
    }
    vector_fields = {
        "record_kind",
        "record_id",
        "version",
        "content_sha256",
        "model",
        "dimension",
        "vector",
    }
    if any(type(row) is not dict or set(row) != memory_fields for row in document["memories"]):
        raise ValueError("generation memory record schema mismatch")
    if any(type(row) is not dict or set(row) != skill_fields for row in document["skills"]):
        raise ValueError("generation Skill record schema mismatch")
    if any(type(row) is not dict or set(row) != vector_fields for row in document["vectors"]):
        raise ValueError("generation vector record schema mismatch")
    state = GenerationState(
        document["generation_id"],
        document["parent_generation_id"],
        tuple(
            MemoryRecord(
                row["record_id"],
                row["version"],
                row["memory_kind"],
                tuple(row["domains"]),
                tuple(row["tool_dependencies"]),
                row["content"],
            )
            for row in document["memories"]
        ),
        tuple(
            SkillRecord(
                row["skill_id"],
                row["version"],
                tuple(row["domains"]),
                tuple(row["tool_dependencies"]),
                row["content"],
                row["accepted_effect_id"],
                row["dev_evidence_sha256"],
            )
            for row in document["skills"]
        ),
        tuple(
            StoredVector(
                row["record_kind"],
                row["record_id"],
                row["version"],
                row["content_sha256"],
                row["model"],
                row["dimension"],
                tuple(row["vector"]),
            )
            for row in document["vectors"]
        ),
        document["seed_library_sha256"],
    )
    if document["state_sha256"] != state.state_sha256:
        raise ValueError("generation content hash mismatch")
    return state


def validate_generation_state(state: GenerationState, *, parent: GenerationState | None) -> None:
    if not state.seed_library_sha256.startswith("sha256:"):
        raise ValueError("generation requires a seed-library content hash")
    if state.generation_id not in {"g000", "g001", "g002", "g003"}:
        raise ValueError("generation ID outside fixed chain")
    if state.generation_id == "g000":
        if state.parent_generation_id is not None or parent is not None:
            raise ValueError("G000 has no parent")
    else:
        expected_parent = f"g{int(state.generation_id[1:]) - 1:03d}"
        if (
            state.parent_generation_id != expected_parent
            or parent is None
            or parent.generation_id != expected_parent
        ):
            raise ValueError("generation parent mismatch")
        if state.seed_library_sha256 != parent.seed_library_sha256:
            raise ValueError("generation cannot change its G000 seed-library binding")
    if any(row.memory_kind not in {"policy", "world"} for row in state.memories):
        raise ValueError("Memory records must be explicitly Policy or World scoped")
    records = (*state.memories, *state.skills)
    if any(
        type(row.version) is not int
        or row.version < 0
        or not set(row.domains) <= set(DOMAINS)
        or any(type(value) is not str or not value for value in row.tool_dependencies)
        or type(row.content) is not str
        for row in records
    ):
        raise ValueError("generation record fields are invalid")
    memory_ids = [(row.record_id, row.version) for row in state.memories]
    skill_ids = [(row.skill_id, row.version) for row in state.skills]
    if (
        len(memory_ids) != len(set(memory_ids))
        or len({row.record_id for row in state.memories}) != len(state.memories)
        or len(skill_ids) != len(set(skill_ids))
        or len({row.skill_id for row in state.skills}) != len(state.skills)
    ):
        raise ValueError("generation contains duplicate record identities")
    record_keys = [
        ("memory", row.record_id, row.version, canonical_sha256(row.content))
        for row in state.memories
    ] + [
        ("skill", row.skill_id, row.version, canonical_sha256(row.content)) for row in state.skills
    ]
    vector_keys = [
        (row.record_kind, row.record_id, row.version, row.content_sha256) for row in state.vectors
    ]
    if len(vector_keys) != len(set(vector_keys)) or sorted(record_keys) != sorted(vector_keys):
        raise ValueError("every generation record requires one exact content-bound embedding")
    if any(row.model != EMBEDDING_MODEL for row in state.vectors):
        raise ValueError("generation embedding model drift")
    if len({row.dimension for row in state.vectors}) > 1:
        raise ValueError("generation embedding index has mixed dimensions")
    for row in state.vectors:
        if row.dimension != len(row.vector):
            raise ValueError("generation embedding dimension mismatch")
        validate_vector(row.vector, dimension=row.dimension)


@dataclass(frozen=True)
class GenerationRetrieval:
    context: RetrievedContext
    ledger_events: tuple[object, ...]


class GenerationStore:
    def __init__(self, root: str | Path, *, cache: EmbeddingCache | None = None):
        self.root = Path(root)
        self.cache = cache or EmbeddingCache(self.root / "embedding_cache.json")

    def _path(self, generation_id: str) -> Path:
        return self.root / f"{generation_id}.json"

    def _validate_state(self, state: GenerationState) -> None:
        parent = None
        if state.generation_id != "g000":
            expected_parent = f"g{int(state.generation_id[1:]) - 1:03d}"
            if not self._path(expected_parent).exists():
                raise ValueError("generation parent mismatch")
            parent = self.load(expected_parent)
        validate_generation_state(state, parent=parent)

    def publish(self, state: GenerationState) -> str:
        self._validate_state(state)
        payload = asdict(state)
        payload["state_sha256"] = state.state_sha256
        path = self._path(state.generation_id)
        if path.exists():
            if read_json(path) != payload:
                raise ValueError("conflicting immutable generation publication")
            return state.state_sha256
        atomic_write_json(path, payload)
        return state.state_sha256

    def load(self, generation_id: str) -> GenerationState:
        state = generation_state_from_document(read_json(self._path(generation_id)))
        self._validate_state(state)
        return state

    def embed_record(
        self,
        *,
        record_kind: str,
        record_id: str,
        version: int,
        content: str,
        provider: EmbeddingProvider,
        scope_id: str,
    ):
        if record_kind not in {"memory", "skill"}:
            raise ValueError("unknown record kind")
        vector, events, content_sha = self.cache.embed(
            content, provider=provider, scope_id=scope_id, purpose=f"index_{record_kind}"
        )
        return StoredVector(
            record_kind, record_id, version, content_sha, EMBEDDING_MODEL, len(vector), vector
        ), events

    def recover_embedded_record(
        self,
        *,
        record_kind: str,
        record_id: str,
        version: int,
        content: str,
        provider: EmbeddingProvider,
        scope_id: str,
    ):
        recovered = self.cache.recover_operation(
            content, provider=provider, scope_id=scope_id, purpose=f"index_{record_kind}"
        )
        if recovered is None:
            return None
        vector, events, content_sha = recovered
        return (
            StoredVector(
                record_kind, record_id, version, content_sha, EMBEDDING_MODEL, len(vector), vector
            ),
            events,
        )

    def retrieve(
        self,
        generation_id: str,
        *,
        domain: str,
        tool_names: tuple[str, ...],
        query: str,
        provider: EmbeddingProvider,
        scope_id: str,
    ) -> GenerationRetrieval:
        state = self.load(generation_id)
        query_vector, events, _ = self.cache.embed(
            query, provider=provider, scope_id=scope_id, purpose="retrieval_query"
        )
        tools = set(tool_names)

        def relevant(domains, dependencies):
            return (not domains or domain in domains) and (
                not dependencies or set(dependencies) <= tools
            )

        memory_by_key = {(row.record_id, row.version): row for row in state.memories}
        skill_by_key = {(row.skill_id, row.version): row for row in state.skills}
        vectors = {(row.record_kind, row.record_id, row.version): row for row in state.vectors}
        policy_candidates = tuple(
            vectors[("memory", row.record_id, row.version)]
            for row in state.memories
            if row.memory_kind == "policy" and relevant(row.domains, row.tool_dependencies)
        )
        world_candidates = tuple(
            vectors[("memory", row.record_id, row.version)]
            for row in state.memories
            if row.memory_kind == "world" and relevant(row.domains, row.tool_dependencies)
        )
        skill_candidates = tuple(
            vectors[("skill", row.skill_id, row.version)]
            for row in state.skills
            if relevant(row.domains, row.tool_dependencies)
        )
        policy = rank_vectors(query_vector, policy_candidates)
        world = rank_vectors(query_vector, world_candidates)
        skills = rank_vectors(query_vector, skill_candidates)
        payload = {
            "generation_id": generation_id,
            "domain": domain,
            "tool_names": list(tool_names),
            "query_sha256": canonical_sha256(query),
            "policy": [[row.record_id, row.version] for row in policy],
            "world": [[row.record_id, row.version] for row in world],
            "skills": [[row.record_id, row.version] for row in skills],
        }
        return GenerationRetrieval(
            RetrievedContext(
                tuple(memory_by_key[(row.record_id, row.version)].content for row in policy),
                tuple(memory_by_key[(row.record_id, row.version)].content for row in world),
                tuple(skill_by_key[(row.record_id, row.version)].content for row in skills),
                tuple(row.record_id for row in policy),
                tuple(row.record_id for row in world),
                tuple((row.record_id, row.version) for row in policy),
                tuple((row.record_id, row.version) for row in world),
                tuple((row.record_id, row.version) for row in skills),
                canonical_sha256(payload),
            ),
            events,
        )
