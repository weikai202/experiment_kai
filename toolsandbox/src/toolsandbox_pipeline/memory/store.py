"""Fail-closed reader for explicitly selected complete generations."""
from pathlib import Path

from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.schemas.generation import GenerationManifest, GenerationSnapshot, RetrievalIndexEntry, RetrievalIndexManifest, INDEX_PATHS, STORE_PATHS
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory, unique
from toolsandbox_pipeline.schemas.skill import SkillRecord
from toolsandbox_pipeline.skills.store import validate_skills
from toolsandbox_pipeline.retrieval.contracts import RetrievalError
from toolsandbox_pipeline.retrieval.index import file_hash, validate_indexes


def load_generation(directory, *, tool_inventory, tool_inventory_sha256, expected_generation_id=None, expected_embedding=None):
    root = Path(directory)
    if not root.is_absolute() or ".." in root.parts or any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
        raise RetrievalError("explicit absolute generation directory required")
    expected = {"manifest.json", "retrieval_indexes", *STORE_PATHS, *INDEX_PATHS}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RetrievalError("invalid generation file")
        actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise RetrievalError("incomplete or unexpected generation layout")
    manifest = GenerationManifest.model_validate_json((root / "manifest.json").read_bytes())
    if root.name != manifest.generation_id or (expected_generation_id and manifest.generation_id != expected_generation_id):
        raise RetrievalError("generation directory identity mismatch")
    unique(tuple(tool_inventory), ordered=True)
    if canonical_sha256(list(tool_inventory)) != tool_inventory_sha256 or manifest.tool_inventory_sha256 != tool_inventory_sha256:
        raise RetrievalError("tool inventory identity mismatch")
    if expected_embedding is not None and manifest.embedding != expected_embedding:
        raise RetrievalError("embedding identity mismatch")
    raw_files = {}
    for entry in manifest.files:
        raw = (root / entry.path).read_bytes()
        if file_hash(raw) != entry.sha256 or len(raw.splitlines()) != entry.record_count:
            raise RetrievalError("generation file hash/count mismatch")
        raw_files[entry.path] = raw
    stores = []
    for path, model in zip(STORE_PATHS, (PolicyMemory, WorldMemory, SkillRecord)):
        records = tuple(model.model_validate_json(line) for line in raw_files[path].splitlines())
        if model is not SkillRecord:
            unique(tuple(r.memory_id for r in records), ordered=True)
            if any(r.created_version > manifest.generation_id for r in records):
                raise RetrievalError("memory created in a future generation")
        stores.append(records)
    validate_skills(stores[2], tuple(tool_inventory))
    indexes = []
    for kind, path in zip(("policy", "world", "skill"), INDEX_PATHS[:3]):
        entries = tuple(RetrievalIndexEntry.model_validate_json(line) for line in raw_files[path].splitlines())
        if any(entry.record_kind != kind for entry in entries):
            raise RetrievalError("index entry in wrong store file")
        indexes.extend(entries)
    snapshot = GenerationSnapshot(
        manifest=manifest, policy_memory=stores[0], world_memory=stores[1], skills=stores[2],
        index_manifest=RetrievalIndexManifest.model_validate_json(raw_files[INDEX_PATHS[-1]]),
        indexes=tuple(indexes),
    )
    validate_indexes(snapshot)
    return snapshot
