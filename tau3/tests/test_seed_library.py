from copy import deepcopy

import pytest

from tau3_evolution import TAU_COMMIT
from tau3_evolution.canonical import canonical_sha256
from tau3_evolution.generation_store import GenerationStore
from tau3_evolution.retrieval import EmbeddingResponse
from tau3_evolution.seed_library import (
    compile_seed_library,
    publish_g000,
    verify_seed_library,
)


class Provider:
    def __init__(self):
        self.receipts = {}

    def embed(self, text, *, model, logical_request_id):
        response = EmbeddingResponse(
            model,
            (float(len(text)), 1.0),
            f"attempt:{logical_request_id}",
            1,
            0,
            canonical_sha256({"model": model, "text": text}),
        )
        self.receipts[logical_request_id] = response
        return response

    def recover(self, *, logical_request_id):
        return self.receipts.get(logical_request_id)


def sources():
    return {
        domain: {
            "source_path": f"src/tau2/domains/{domain}/tools.py",
            "source_sha256": f"sha256:{domain}-source",
            "schemas": [
                {
                    "type": "function",
                    "function": {
                        "name": f"{domain}_lookup",
                        "description": f"Look up public {domain} records.",
                        "parameters": {
                            "type": "object",
                            "properties": {"record_id": {"type": "string"}},
                            "required": ["record_id"],
                        },
                    },
                }
            ],
        }
        for domain in ("airline", "retail", "telecom")
    }


def test_schema_only_compilation_is_deterministic_and_hash_bound():
    first = compile_seed_library(sources(), source_commit=TAU_COMMIT)
    second = compile_seed_library(deepcopy(sources()), source_commit=TAU_COMMIT)
    assert first == second
    assert first["library_sha256"].startswith("sha256:")
    assert [row["skill_id"] for row in first["skills"]] == [
        "airline.airline_lookup",
        "retail.retail_lookup",
        "telecom.telecom_lookup",
    ]
    assert first["skills"][0]["tool_dependencies"] == ["airline_lookup"]
    verify_seed_library(first)


def test_compiler_rejects_private_or_non_pinned_provenance():
    private = sources()
    private["telecom"]["source_path"] = "src/tau2/domains/telecom/user_tools.py"
    with pytest.raises(ValueError, match="only from"):
        compile_seed_library(private, source_commit=TAU_COMMIT)
    with pytest.raises(ValueError, match="pinned"):
        compile_seed_library(sources(), source_commit="wrong")


def test_tampered_skill_cannot_claim_the_seed_library_hash():
    document = compile_seed_library(sources(), source_commit=TAU_COMMIT)
    document["skills"][0]["content"] = "fabricated"
    with pytest.raises(ValueError, match="library_sha256 mismatch"):
        verify_seed_library(document)


def test_verified_library_publishes_hash_bound_g000_with_embeddings(tmp_path):
    document = compile_seed_library(sources(), source_commit=TAU_COMMIT)
    store = GenerationStore(tmp_path / "generations")
    state, state_sha, events = publish_g000(store, document, Provider(), scope_id="seed:g000")
    loaded = store.load("g000")
    assert state_sha == state.state_sha256 == loaded.state_sha256
    assert loaded.seed_library_sha256 == document["library_sha256"]
    assert tuple(row.skill_id for row in loaded.skills) == tuple(
        row["skill_id"] for row in document["skills"]
    )
    assert len(events) == 2 * len(document["skills"])
