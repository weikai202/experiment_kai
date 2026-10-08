import pytest

from bfcl_pipeline.canonical import sha256_json
from bfcl_pipeline.resources import GenerationResource, GenerationResourceStore, QueryEmbedding, ResourceEmbedding


def test_generation_store_is_scoped_and_retrieval_is_deterministic():
    resources = [GenerationResource.build("m", "g001", "policy_memory", "memory", ["read"]), GenerationResource.build("s", "g001", "skill", "skill", ["write"])]
    class Provider:
        model = "text-embedding-3-small"
        client_config_sha256 = "config"
        def embed(self, text):
            vector = (1.0, 0.0)
            return QueryEmbedding(sha256_json(text), self.model, self.client_config_sha256, vector, sha256_json(vector))
    embeddings = [ResourceEmbedding.build(resources[0], "config", (1.0, 0.0)), ResourceEmbedding.build(resources[1], "config", (0.0, 1.0))]
    store = GenerationResourceStore("g001", resources, embeddings, Provider())
    assert store.retrieve(["read"], "query", 5) == (resources[0],)


def test_retrieval_rejects_every_embedding_fallback():
    resource = GenerationResource.build("m", "g001", "policy_memory", "memory", [])
    class WrongProvider:
        model = "local-embedding"
        client_config_sha256 = "config"
        def embed(self, text): raise AssertionError
    with pytest.raises(ValueError, match="Only text-embedding-3-small"):
        GenerationResourceStore("g001", [resource], [ResourceEmbedding.build(resource, "config", (1.0,))], WrongProvider())


def test_retrieval_requires_every_dependency_visible():
    resource = GenerationResource.build("s", "g001", "skill", "skill", ["read", "write"])
    class Provider:
        model = "text-embedding-3-small"; client_config_sha256 = "config"
        def embed(self, text):
            vector = (1.0,); return QueryEmbedding(sha256_json(text), self.model, self.client_config_sha256, vector, sha256_json(vector))
    store = GenerationResourceStore("g001", [resource], [ResourceEmbedding.build(resource, "config", (1.0,))], Provider())
    assert store.retrieve(["read"], "query", 5) == ()
    assert store.retrieve(["read", "write"], "query", 5) == (resource,)
