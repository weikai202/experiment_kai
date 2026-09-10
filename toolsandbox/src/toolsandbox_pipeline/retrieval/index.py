"""Exact deterministic cosine indexes, bound to complete record stores."""
import math
from pathlib import Path

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes
from toolsandbox_pipeline.schemas.generation import GenerationFileEntry, RetrievalIndexEntry, RetrievalIndexManifest, INDEX_PATHS, STORE_PATHS
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory, unique
from toolsandbox_pipeline.schemas.skill import SkillRecord
from .contracts import RetrievalError, validate_vector
from .queries import ACTION_VERSION, DOCUMENT_VERSION, STATE_VERSION, document, input_hash


def file_hash(data):
    from hashlib import sha256
    return "sha256:" + sha256(data).hexdigest()


def jsonl_bytes(records):
    return b"".join(canonical_json_bytes(r.model_dump(mode="json")) + b"\n" for r in records)


def record_key(record):
    return (record.skill_id, record.version) if type(record) is SkillRecord else (record.memory_id, None)


def index_key(entry):
    return entry.record_id.encode("utf-8"), int(entry.record_version[3:]) if entry.record_version else -1


def validate_indexes(snapshot):
    manifest = snapshot.manifest
    index_manifest = snapshot.index_manifest
    if (index_manifest.generation_id, index_manifest.embedding, index_manifest.vector_dimension) != (
        manifest.generation_id, manifest.embedding, manifest.vector_dimension
    ) or index_manifest.stores != manifest.files[:3] or index_manifest.indexes != manifest.files[3:6]:
        raise RetrievalError("index manifest identity mismatch")
    for kind, records in (("policy", snapshot.policy_memory), ("world", snapshot.world_memory), ("skill", snapshot.skills)):
        expected = {record_key(r): r for r in records if r.status == "active"}
        entries = tuple(e for e in snapshot.indexes if e.record_kind == kind)
        keys = [(e.record_id, e.record_version) for e in entries]
        if len(set(keys)) != len(keys) or set(keys) != set(expected) or list(entries) != sorted(entries, key=index_key):
            raise RetrievalError("missing, extra, duplicate, or unordered index records")
        for entry in entries:
            record = expected[(entry.record_id, entry.record_version)]
            identity = entry.embedding_cache_key.model_dump(exclude={"input_sha256"})
            if entry.generation_id != manifest.generation_id or identity != manifest.embedding.model_dump():
                raise RetrievalError("index generation/embedding mismatch")
            if entry.vector_dimension != manifest.vector_dimension or entry.document_sha256 != input_hash(document(record)):
                raise RetrievalError("stale document or vector dimension")


def cosine(left, right):
    left = validate_vector(left)
    right = validate_vector(right, len(left))
    a, b = math.hypot(*left), math.hypot(*right)
    score = sum((x / a) * (y / b) for x, y in zip(left, right))
    if not math.isfinite(score):
        raise RetrievalError("non-finite cosine")
    return max(-1.0, min(1.0, score))


def rank(entries, vector):
    scored = [(entry, cosine(entry.vector, vector)) for entry in entries]
    return tuple(sorted(scored, key=lambda item: (-item[1], *index_key(item[0])))[:3])


def build_indexes(staging_directory, *, generation_id, policy_memory, world_memory, skills,
                  store_entries, tool_inventory, cache, gateway, context_factory, record_durable):
    root = Path(staging_directory)
    if not root.is_absolute() or ".." in root.parts or not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise RetrievalError("explicit non-symlink staging directory required")
    if root.name == generation_id or (root / "manifest.json").exists():
        raise RetrievalError("cannot build into a published generation")
    index_dir = root / "retrieval_indexes"
    if index_dir.exists() or index_dir.is_symlink():
        raise RetrievalError("index output must be new")
    from toolsandbox_pipeline.skills.store import validate_skills
    unique(tuple(tool_inventory), ordered=True)
    policy_memory = tuple(PolicyMemory.model_validate(r) for r in policy_memory)
    world_memory = tuple(WorldMemory.model_validate(r) for r in world_memory)
    skills = tuple(SkillRecord.model_validate(r) for r in skills)
    for records in (policy_memory, world_memory):
        unique(tuple(r.memory_id for r in records), ordered=True)
        if any(r.created_version > generation_id for r in records):
            raise RetrievalError("record created after index generation")
    validate_skills(skills, tuple(tool_inventory))
    groups = (("policy", policy_memory), ("world", world_memory), ("skill", skills))
    if tuple(e.path for e in store_entries) != STORE_PATHS:
        raise RetrievalError("store entries mismatch")
    for entry, (_, records) in zip(store_entries, groups):
        raw = (root / entry.path).read_bytes()
        if file_hash(raw) != entry.sha256 or len(raw.splitlines()) != entry.record_count or raw != jsonl_bytes(records):
            raise RetrievalError("staging store mismatch")
    records = [(kind, r) for kind, rs in groups for r in rs if r.status == "active"]
    texts = [document(r) for _, r in records]
    embeddings = cache.resolve(texts, gateway=gateway, context_factory=context_factory, record_durable=record_durable) if texts else ()
    entries = tuple(RetrievalIndexEntry(
        generation_id=generation_id, record_kind=kind, record_id=record_key(record)[0], record_version=record_key(record)[1],
        document_sha256=embedding.key.input_sha256, embedding_cache_key=embedding.key,
        vector_dimension=cache.dimension, vector=embedding.vector,
    ) for (kind, record), embedding in zip(records, embeddings))
    blobs = tuple(jsonl_bytes(sorted((e for e in entries if e.record_kind == kind), key=index_key)) for kind, _ in groups)
    files = tuple(GenerationFileEntry(path=path, sha256=file_hash(raw), record_count=len(raw.splitlines())) for path, raw in zip(INDEX_PATHS, blobs))
    manifest = RetrievalIndexManifest(
        schema_version=1, generation_id=generation_id, embedding=cache.identity, vector_dimension=cache.dimension,
        batching=cache.batches, state_serialization=STATE_VERSION, action_serialization=ACTION_VERSION,
        document_serialization=DOCUMENT_VERSION, stores=tuple(store_entries), indexes=files,
    )
    index_dir.mkdir()
    for path, raw in zip(INDEX_PATHS, (*blobs, canonical_json_bytes(manifest.model_dump(mode="json")))):
        with (root / path).open("xb") as stream:
            stream.write(raw)
    return manifest
