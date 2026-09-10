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
