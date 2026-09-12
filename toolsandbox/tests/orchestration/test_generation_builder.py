import json
import stat

import pytest

from toolsandbox_pipeline.offline.skill_orchestrator import (
    SealedSkillTrajectoryBuffer,
    SkillUpdateOrchestrator,
)
from toolsandbox_pipeline.orchestration.generation_builder import (
    apply_generation_overlays,
    apply_round_results,
    build_generation,
)
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
from toolsandbox_pipeline.schemas.offline_memory import MemoryRoundResult, MemoryUpdateIdentity
from toolsandbox_pipeline.schemas.offline_skill import SkillRoundIdentity

from tests.memory.test_records import policy
from tests.offline_skill.test_orchestrator import Never, Sink
from tests.retrieval.test_embedding_cache import dependencies
from tests.skills.test_records import skill


INVENTORY = ("search_contacts",)
INVENTORY_HASH = canonical_sha256(list(INVENTORY))


def build(tmp_path, generation_id="g000", parent=None):
    stage = tmp_path / ("stage-" + generation_id)
    stage.mkdir()
    _, _, kwargs = dependencies()
    with EmbeddingCache(tmp_path / ("cache-" + generation_id), EmbeddingIdentity(), 2) as cache:
        snapshot = build_generation(
            stage,
            generation_id=generation_id,
            parent_generation_id=parent,
            policy_memory=(policy(),),
            world_memory=(),
            skills=(skill(),),
            tool_inventory=INVENTORY,
            tool_inventory_sha256=INVENTORY_HASH,
            cache=cache,
            **kwargs,
        )
    return snapshot, stage / generation_id


def test_builds_complete_loadable_private_generation_zero(tmp_path):
    snapshot, root = build(tmp_path)
    assert snapshot.manifest.generation_id == "g000"
    assert len(snapshot.indexes) == 2
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in root.rglob("*") if path.is_file()
    )


def test_overlay_replaces_authoritative_memory_and_skill_keys_but_preserves_others(tmp_path):
    snapshot, _ = build(tmp_path)
    updated_policy = policy().model_copy(update={"support_count": 2, "success_rate": 0.5, "confidence": 0.5})
    updated_skill = skill().model_copy(update={
        "online_statistics": skill().online_statistics.model_copy(update={
            "evaluated_uses": 1, "successes": 1, "success_rate": 1.0,
        })
    })
    policies, worlds, skills = apply_generation_overlays(
        snapshot,
        policy_updates=(updated_policy,),
        skill_updates=(updated_skill,),
    )
    assert policies == (updated_policy,)
    assert worlds == ()
    assert skills == (updated_skill,)


def test_round_results_validate_identities_and_flatten_skill_histories(tmp_path):
    snapshot, _ = build(tmp_path)
    updated_policy = policy().model_copy(
        update={"support_count": 2, "success_rate": 0.5, "confidence": 0.5}
    )
    memory = MemoryRoundResult.build(
        identity=MemoryUpdateIdentity(
            run_id="run",
            round_index=0,
            shard_id="train-shard-0",
            current_generation_id="g000",
            next_generation_id="g001",
            dataset_manifest_sha256=INVENTORY_HASH,
            config_manifest_sha256=INVENTORY_HASH,
            prompt_manifest_sha256=INVENTORY_HASH,
            sealed_input_buffer_sha256=INVENTORY_HASH,
        ),
        selected_policy_trajectory_ids=(),
        selected_world_trajectory_ids=(),
        ordered_unit_ids=(),
        units=(),
        policy_counts={"ADD": 0, "MERGE": 1, "SKIP": 0, "NONE": 0},
        world_counts={"ADD": 0, "MERGE": 0, "SKIP": 0, "NONE": 0},
        staged_policy_memory=(updated_policy,),
        staged_world_memory=(),
        source_generation_sha256=INVENTORY_HASH,
        last_checkpoint_id=None,
        ledger_high_water_marks={},
        accounting_projections=(),
    )
    skill_identity = SkillRoundIdentity(
        run_id="run",
        round_index=0,
        shard_id="train-shard-0",
        current_generation_id="g000",
        next_generation_id="g001",
        dataset_manifest_sha256=INVENTORY_HASH,
        config_manifest_sha256=INVENTORY_HASH,
        prompt_manifest_sha256=INVENTORY_HASH,
        token_limit_config_sha256=INVENTORY_HASH,
        sealed_input_buffer_sha256=canonical_sha256(
            {
                "protocol": "sealed-skill-buffer-v1",
                "run_id": "run",
                "round_index": 0,
                "shard_id": "train-shard-0",
                "generation_id": "g000",
                "entries": [],
            }
        ),
    )
    skill_result = SkillUpdateOrchestrator(
        failure_executor=Never(),
        candidate_executor=Never(),
        mini_bench_executor=Never(),
        effect_sink=Sink(),
        public_tool_inventory=INVENTORY,
    ).run(
        buffer=SealedSkillTrajectoryBuffer(skill_identity, ()),
        current_skills=snapshot.skills,
    )

    policies, worlds, skills = apply_round_results(
        snapshot,
        run_id="run",
        round_index=0,
        train_shard_id="train-shard-0",
        dataset_manifest_sha256=INVENTORY_HASH,
        config_manifest_sha256=INVENTORY_HASH,
        input_generation_id="g000",
        output_generation_id="g001",
        memory_result=memory,
        skill_result=skill_result,
    )

    assert policies == (updated_policy,)
    assert worlds == ()
    assert skills == skill_result.staged_mutations[0].staged_records
    with pytest.raises(Exception, match="identity mismatch"):
        apply_round_results(
            snapshot,
            run_id="different",
            round_index=0,
            train_shard_id="train-shard-0",
            dataset_manifest_sha256=INVENTORY_HASH,
            config_manifest_sha256=INVENTORY_HASH,
            input_generation_id="g000",
            output_generation_id="g001",
            memory_result=memory,
            skill_result=skill_result,
        )


@pytest.fixture
def coordinator_case(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from toolsandbox_pipeline.checkpointing import CheckpointStore, LLMLedger, RunIdentity, UnitLedger
    from toolsandbox_pipeline.orchestration.generation_builder import RoundGenerationCoordinator
    from toolsandbox_pipeline.orchestration.generation_publisher import publish_generation_zero

    snapshot, staged = build(tmp_path)
    root = tmp_path / 'generations'
    root.mkdir(mode=0o700)
    publish_generation_zero(staged, root)
    staging = tmp_path / 'next'
    staging.mkdir(mode=0o700)
    source_hash = canonical_sha256(snapshot.manifest.model_dump(mode='json'))
    identity = MemoryUpdateIdentity(
        run_id='run', round_index=0, shard_id='train-shard-0',
        current_generation_id='g000', next_generation_id='g001',
        dataset_manifest_sha256=INVENTORY_HASH, config_manifest_sha256=INVENTORY_HASH,
        prompt_manifest_sha256=INVENTORY_HASH, sealed_input_buffer_sha256=INVENTORY_HASH,
    )
    memory = MemoryRoundResult.build(
        identity=identity, selected_policy_trajectory_ids=(), selected_world_trajectory_ids=(),
        ordered_unit_ids=(), units=(),
        policy_counts=dict(ADD=0, MERGE=0, SKIP=0, NONE=0),
        world_counts=dict(ADD=0, MERGE=0, SKIP=0, NONE=0),
        staged_policy_memory=(), staged_world_memory=(), source_generation_sha256=source_hash,
        last_checkpoint_id=None, ledger_high_water_marks={}, accounting_projections=(),
    )
    skill_identity = SkillRoundIdentity(
        **identity.model_dump(exclude={'sealed_input_buffer_sha256'}),
        token_limit_config_sha256=INVENTORY_HASH,
        sealed_input_buffer_sha256=canonical_sha256(dict(
            protocol='sealed-skill-buffer-v1', run_id='run', round_index=0,
            shard_id='train-shard-0', generation_id='g000', entries=[],
        )),
    )
    skills = SkillUpdateOrchestrator(
        failure_executor=Never(), candidate_executor=Never(), mini_bench_executor=Never(),
        effect_sink=Sink(), public_tool_inventory=INVENTORY,
    ).run(buffer=SealedSkillTrajectoryBuffer(skill_identity, ()), current_skills=snapshot.skills)
    run_identity = RunIdentity(
        run_id='run', profile='offline', environment_identity=INVENTORY_HASH,
        dataset_manifest_sha256=INVENTORY_HASH, config_manifest_sha256=INVENTORY_HASH,
        prompt_manifest_sha256=INVENTORY_HASH, generation_manifest_sha256=source_hash,
        fixture_manifest_sha256=INVENTORY_HASH,
    )
    store = CheckpointStore.create(
        tmp_path / 'ledger', run_identity,
        Path(__file__).parents[2] / 'configs/reproducibility/checkpointing_v1.json',
    )
    transport, durable, kwargs = dependencies()
    with EmbeddingCache(tmp_path / 'coordinator-cache', EmbeddingIdentity(), 2) as cache:
        service = RoundGenerationCoordinator(
            run_id='run', round_index=0, dataset_manifest_sha256=INVENTORY_HASH,
            config_manifest_sha256=INVENTORY_HASH, parent_manifest_sha256=source_hash,
            generation_root=root, staging_parent=staging, tool_inventory=INVENTORY,
            tool_inventory_sha256=INVENTORY_HASH, cache=cache,
            ledger=UnitLedger(store), checkpoints=LLMLedger(store), **kwargs,
        )
        yield SimpleNamespace(service=service, memory=memory, skills=skills, root=root,
                              staging=staging, transport=transport, durable=durable, store=store)
    store.close()


def run_coordinator(case):
    return case.service.build_and_publish(**dict(
        round_index=0, input_generation_id='g000', memory_result=case.memory,
        skill_result=case.skills,
    ))


def test_coordinator_builds_commits_reloads_and_reuses_without_duplicate_effects(coordinator_case):
    case = coordinator_case
    parent_bytes = {str(p.relative_to(case.root)): p.read_bytes()
                    for p in (case.root / 'g000').rglob('*') if p.is_file()}
    result = run_coordinator(case)
    assert result.generation_id == 'g001'
    assert result.publication_record.status == 'committed'
    assert result.snapshot.manifest.parent_generation_id == 'g000'
    assert result.snapshot.policy_memory == (policy(),)
    assert result.snapshot.skills == case.skills.staged_mutations[0].staged_records
    calls = len(case.transport.calls)
    assert calls > 0
    high_water_marks = case.store.high_water_marks()
    assert run_coordinator(case) is result
    assert case.store.high_water_marks() == high_water_marks
    assert len(case.transport.calls) == calls
    assert parent_bytes == {str(p.relative_to(case.root)): p.read_bytes()
                            for p in (case.root / 'g000').rglob('*') if p.is_file()}
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600
               for p in (case.root / 'g001').rglob('*') if p.is_file())


@pytest.mark.parametrize('defect', ['round', 'source_hash', 'result_hash', 'config', 'skill_source', 'skill_coverage', 'destination'])
def test_coordinator_rejects_mismatched_inputs_before_embedding_or_staging(coordinator_case, defect):
    case = coordinator_case
    kwargs = dict(round_index=0, input_generation_id='g000', memory_result=case.memory, skill_result=case.skills)
    if defect == 'round':
        kwargs['round_index'] = 1
    elif defect == 'source_hash':
        case.service.parent_manifest_sha256 = 'sha256:' + 'f' * 64
    elif defect == 'result_hash':
        kwargs['memory_result'] = case.memory.model_copy(update={'result_sha256': 'sha256:' + 'f' * 64})
    elif defect == 'config':
        case.service.config_manifest_sha256 = 'sha256:' + 'f' * 64
    elif defect == 'destination':
        (case.root / 'g001').mkdir(mode=0o700)
    else:
        from toolsandbox_pipeline.schemas.offline_skill import SkillRoundResult
        payload = case.skills.model_dump(mode='json', exclude={'result_sha256'})
        if defect == 'skill_coverage':
            payload['staged_mutations'] = []
        else:
            payload['staged_mutations'][0]['previous_record']['instruction'] = 'Changed source instruction.'
        payload['result_sha256'] = canonical_sha256(payload)
        kwargs['skill_result'] = SkillRoundResult.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError if defect == 'result_hash' else RuntimeError):
        case.service.build_and_publish(**kwargs)
    assert not case.transport.calls
    assert not tuple(case.staging.iterdir())


def test_coordinator_stops_after_checkpoint_failure_and_preserves_private_staging(coordinator_case):
    case = coordinator_case
    class FailingCheckpoints:
        def commit_checkpoint(self, *args):
            raise RuntimeError('synthetic checkpoint failure')
    case.service.checkpoints = FailingCheckpoints()
    with pytest.raises(RuntimeError, match='synthetic checkpoint failure'):
        run_coordinator(case)
    calls = len(case.transport.calls)
    assert (case.staging / 'g001').is_dir()
    assert not (case.root / 'g001').exists()
    with pytest.raises(RuntimeError, match='explicit recovery'):
        run_coordinator(case)
    assert len(case.transport.calls) == calls


def test_coordinator_repeated_call_checks_published_bytes(coordinator_case):
    case = coordinator_case
    run_coordinator(case)
    (case.root / 'g001' / 'policy_memory.jsonl').write_text('corrupt')
    calls = len(case.transport.calls)
    with pytest.raises(ValueError):
        run_coordinator(case)
    assert len(case.transport.calls) == calls



def test_coordinator_rejects_changed_results_after_success(coordinator_case):
    case = coordinator_case
    run_coordinator(case)
    calls = len(case.transport.calls)
    changed = case.memory.identity_payload()
    changed["last_checkpoint_id"] = "different-checkpoint"
    changed["result_sha256"] = canonical_sha256(changed)
    memory = MemoryRoundResult.model_validate_json(json.dumps(changed))
    with pytest.raises(RuntimeError, match="identity conflict"):
        case.service.build_and_publish(
            round_index=0, input_generation_id="g000",
            memory_result=memory, skill_result=case.skills,
        )
    assert len(case.transport.calls) == calls
