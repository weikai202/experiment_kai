import json
import pytest
from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.generation import GenerationFileEntry, GenerationManifest, STORE_PATHS, INDEX_PATHS, UPSTREAM_COMMIT
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
from toolsandbox_pipeline.retrieval.index import build_indexes, file_hash, jsonl_bytes
from toolsandbox_pipeline.memory.store import load_generation
from tests.memory.test_records import policy
from tests.skills.test_records import skill
from tests.retrieval.test_embedding_cache import dependencies

INVENTORY = ("search_contacts",)
INVENTORY_HASH = canonical_sha256(list(INVENTORY))


def generation(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    stores = ((policy(),), (), (skill(),))
    files = []
    for path, records in zip(STORE_PATHS, stores):
        raw = jsonl_bytes(records)
        (staging / path).write_bytes(raw)
        files.append(GenerationFileEntry(path=path, sha256=file_hash(raw), record_count=len(records)))
    _, _, kwargs = dependencies()
    with EmbeddingCache(tmp_path / "build-cache", EmbeddingIdentity(), 2) as cache:
        index = build_indexes(staging, generation_id="g000", policy_memory=stores[0], world_memory=stores[1], skills=stores[2],
                              store_entries=tuple(files), tool_inventory=INVENTORY, cache=cache, **kwargs)
    files.extend(index.indexes)
    files.append(GenerationFileEntry(path=INDEX_PATHS[-1], sha256=file_hash((staging / INDEX_PATHS[-1]).read_bytes()), record_count=1))
    manifest = GenerationManifest(schema_version=1, generation_id="g000", parent_generation_id=None, publication_status="complete",
                                  upstream_commit=UPSTREAM_COMMIT, tool_inventory_sha256=INVENTORY_HASH,
                                  embedding=EmbeddingIdentity(), vector_dimension=2, files=tuple(files))
    (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    root = tmp_path / "g000"
    staging.rename(root)
    return root


def load(root):
    return load_generation(root, tool_inventory=INVENTORY, tool_inventory_sha256=INVENTORY_HASH)


def test_complete_generation(tmp_path):
    root = generation(tmp_path)
    snapshot = load(root)
    assert snapshot.manifest.generation_id == "g000" and len(snapshot.indexes) == 2
    assert load(root) == snapshot
    with pytest.raises(ValueError):
        snapshot.skills = ()


@pytest.mark.parametrize("damage", ["extra", "missing", "symlink", "hash", "generation", "publication"])
def test_layout_rejection(tmp_path, damage):
    root = generation(tmp_path)
    if damage == "extra":
        (root / "extra").write_text("x")
    elif damage == "missing":
        (root / "skills.jsonl").unlink()
    elif damage == "symlink":
        (root / "skills.jsonl").unlink()
        (root / "skills.jsonl").symlink_to(tmp_path / "build-cache")
    elif damage == "hash":
        (root / "skills.jsonl").write_text("{}")
    else:
        data = json.loads((root / "manifest.json").read_text())
        data["generation_id" if damage == "generation" else "publication_status"] = "g001" if damage == "generation" else "staging"
        (root / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load(root)
