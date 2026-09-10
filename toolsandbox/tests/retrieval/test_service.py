import pytest
from toolsandbox_pipeline.retrieval import RetrievalService
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from tests.memory.test_generation_store import generation, load
from tests.retrieval.test_embedding_cache import dependencies
from tests.retrieval.test_queries import state, action


def test_shared_query_and_repeat(tmp_path):
    snapshot = load(generation(tmp_path))
    transport, _, kwargs = dependencies()
    with EmbeddingCache(tmp_path / "query-cache", EmbeddingIdentity(), 2) as cache:
        service = RetrievalService(snapshot, cache=cache, **kwargs)
        first = service.retrieve_policy_skills(state(), canonical_to_agent={"search_contacts": "scrambled"})
        second = service.retrieve_policy_skills(state(), canonical_to_agent={"search_contacts": "scrambled"})
        assert len(transport.calls) == 1
        assert first.policy_hits[0].record_id == second.policy_hits[0].record_id
        assert first.skills == second.skills and second.query.cache_hit
        assert not hasattr(first, "retrieve")
        with pytest.raises(ValueError):
            service.retrieve_world(state(), action(), controller_decision=ControllerDecision(blocking_codes=(), critic_trigger_codes=(), evidence=()))


def test_new_modules_import_without_operational_io():
    import subprocess
    script = '''
import importlib
from pathlib import Path
import sqlite3
from unittest.mock import patch
import toolsandbox_pipeline.online.controller
import toolsandbox_pipeline.providers.qwen
modules = (
    'schemas.memory', 'schemas.skill', 'schemas.generation', 'memory.store',
    'skills.store', 'skills.views', 'retrieval.queries', 'retrieval.embedding_cache',
    'retrieval.index', 'retrieval.service', 'online.prompt_contracts',
    'online.prompt_loader', 'online.prompt_builder', 'online.qwen_roles',
    'online.token_limits', 'online.token_limit_calibration', 'schemas.dataset',
    'reproducibility.clock', 'reproducibility.scenario_hashes',
    'reproducibility.splits', 'reproducibility.dataset_manifest',
    'reproducibility.dataset_access', 'reproducibility.dataset_manifest_cli',
)
def denied(*args, **kwargs):
    raise AssertionError('operational I/O at import')
with patch.object(Path, 'read_bytes', denied), patch.object(Path, 'read_text', denied), patch.object(Path, 'write_bytes', denied), patch.object(Path, 'write_text', denied), patch.object(sqlite3, 'connect', denied):
    for module in modules:
        importlib.import_module('toolsandbox_pipeline.' + module)
'''
    result = subprocess.run(["uv", "run", "--offline", "python", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
