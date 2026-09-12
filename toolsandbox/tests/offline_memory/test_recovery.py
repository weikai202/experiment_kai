from tests.offline_memory.test_orchestrator import (
    Durability,
    Executor,
    buffer,
    reference,
    subject,
)
from toolsandbox_pipeline.schemas.offline_memory import PolicyMemoryCandidateDecision
from toolsandbox_pipeline.schemas.offline_memory import MemoryReviewOutput


def test_committed_none_unit_is_reused_without_second_model_call():
    durability = Durability()
    first_executor = Executor((PolicyMemoryCandidateDecision(result="NONE"),))
    sealed = buffer((reference(),))
    first = subject(first_executor, durability).run(sealed)
    second_executor = Executor(())
    second = subject(second_executor, durability).run(sealed)
    assert second == first
    assert len(first_executor.calls) == 1 and second_executor.calls == []
    assert durability.effects == []


def test_committed_add_restores_staged_record_without_repeating_model_calls():
    durability = Durability()
    candidate = PolicyMemoryCandidateDecision(
        result="CANDIDATE",
        role="policy",
        candidate={
            "scope": "General prerequisites",
            "applicability": [],
            "action_guidance": "Validate required arguments before execution",
            "avoid": [],
        },
    )
    first_executor = Executor(
        (candidate, MemoryReviewOutput(decision="ADD", reason="Reusable new rule"))
    )
    sealed = buffer((reference(),))
    first = subject(first_executor, durability).run(sealed)

    second_executor = Executor(())
    second = subject(second_executor, durability).run(sealed)

    assert second == first
    assert len(first_executor.calls) == 2 and second_executor.calls == []
    assert second.staged_policy_memory == first.staged_policy_memory
    assert len(durability.effects) == 1


def test_packed_candidate_add_merge_disk_roundtrip_and_next_generation(tmp_path):
    from toolsandbox_pipeline.offline.memory_updates import memory_id
    from toolsandbox_pipeline.schemas.offline_memory import MemoryRoundResult, MemoryUpdateUnitResult
    from toolsandbox_pipeline.retrieval.index import build_indexes, file_hash, jsonl_bytes
    from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
    from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
    from toolsandbox_pipeline.schemas.generation import GenerationFileEntry, GenerationManifest, STORE_PATHS, INDEX_PATHS, UPSTREAM_COMMIT
    from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
    from toolsandbox_pipeline.memory.store import load_generation
    from tests.retrieval.test_embedding_cache import dependencies
    from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
    from tests.offline_memory.test_orchestrator import Resolver

    candidate = PolicyMemoryCandidateDecision(result='CANDIDATE', role='policy', candidate={
        'scope': 'Tool prerequisites', 'applicability': [],
        'action_guidance': 'Validate required arguments before execution', 'avoid': []})
    target = memory_id(candidate.decision())
    outputs = (candidate, MemoryReviewOutput(decision='ADD', reason='Reusable new rule'),
               candidate, MemoryReviewOutput(decision='MERGE', target_memory_id=target, reason='Same general guidance'))

    class DiskDurability(Durability):
        def load_unit(self, unit_reference):
            path = tmp_path / (unit_reference + '.json')
            return MemoryUpdateUnitResult.model_validate_json(path.read_bytes()) if path.exists() else None
        def commit_unit(self, result):
            (tmp_path / (result.unit_id + '.json')).write_text(result.model_dump_json())
            return super().commit_unit(result)

    class SerializedExecutor(Executor):
        def execute_and_apply(self, prepared):
            # Simulate storage/recovery at the validated role-output boundary.
            self.outputs[0] = type(self.outputs[0]).model_validate_json(self.outputs[0].model_dump_json())
            return super().execute_and_apply(prepared)

    durability = DiskDurability()
    executor = SerializedExecutor(outputs)
    sealed = buffer((reference(0, 'a'), reference(1, 'b')))
    runner = subject(executor, durability)
    runner.input_representation = 'packed-v2'
    result = runner.run(sealed)
    assert result.policy_counts == {'ADD': 1, 'MERGE': 1, 'NONE': 0, 'SKIP': 0}
    assert [request.prompt_version for request in executor.calls] == ['v2', 'v1', 'v2', 'v1']
    assert len(durability.effects) == 2
    record = result.staged_policy_memory[0]
    assert record.support_count == 2 and len(record.evidence_trajectory_ids) == 2
    assert record.created_version == 'g001'
    assert MemoryRoundResult.model_validate_json(result.model_dump_json()) == result
    recovered_executor = SerializedExecutor(())
    recovered = subject(recovered_executor, durability).run(sealed)
    assert recovered == result and not recovered_executor.calls
    assert len(durability.effects) == 2

    # Exercise actual generation indexing/loading with fake embedding transport.
    root = tmp_path / 'staging'
    root.mkdir()
    stores = (result.staged_policy_memory, result.staged_world_memory, ())
    files = []
    for path, records in zip(STORE_PATHS, stores):
        raw = jsonl_bytes(records)
        (root / path).write_bytes(raw)
        files.append(GenerationFileEntry(path=path, sha256=file_hash(raw), record_count=len(records)))
    _, _, kwargs = dependencies()
    with EmbeddingCache(tmp_path / 'index-cache', EmbeddingIdentity(), 2) as cache:
        indexes = build_indexes(root, generation_id='g001', policy_memory=stores[0], world_memory=stores[1], skills=(),
                                store_entries=tuple(files), tool_inventory=(), cache=cache, **kwargs)
    files.extend(indexes.indexes)
    files.append(GenerationFileEntry(path=INDEX_PATHS[-1], sha256=file_hash((root / INDEX_PATHS[-1]).read_bytes()), record_count=1))
    inventory_hash = canonical_sha256([])
    manifest = GenerationManifest(schema_version=1, generation_id='g001', parent_generation_id='g000', publication_status='complete',
        upstream_commit=UPSTREAM_COMMIT, tool_inventory_sha256=inventory_hash, embedding=EmbeddingIdentity(), vector_dimension=2, files=tuple(files))
    (root / 'manifest.json').write_bytes(canonical_json_bytes(manifest.model_dump(mode='json')))
    published = tmp_path / 'g001'
    root.rename(published)
    snapshot = load_generation(published, tool_inventory=(), tool_inventory_sha256=inventory_hash)
    assert snapshot.policy_memory == (record,)
    retriever = MemoryCandidateRetriever(generation_id='g001', policy_records=snapshot.policy_memory,
        world_records=(), indexes=snapshot.indexes, embedding_resolver=Resolver())
    assert retriever.retrieve(candidate.decision()).records == (record,)
