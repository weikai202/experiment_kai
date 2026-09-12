"""Synthetic assembly tests; no real data, transport or calibration claim."""
from types import SimpleNamespace
from pathlib import Path
from hashlib import sha256
import pytest
from toolsandbox_pipeline.orchestration.live_training import LiveTrainingRuntime, LiveTrainingAssets
from toolsandbox_pipeline.orchestration.live_episode import LiveEpisodeExecutor
from toolsandbox_pipeline.orchestration.live_offline import LiveMemoryUpdater, LiveSkillUpdater
from toolsandbox_pipeline.orchestration.generation_builder import RoundGenerationCoordinator
from toolsandbox_pipeline.orchestration.live_training_metrics import LiveTrainingMetrics
from toolsandbox_pipeline.online.prompt_loader import load_prompts
from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig
from toolsandbox_pipeline.reproducibility import canonical_sha256
from tests.orchestration.test_live_offline import live, D
from tests.orchestration.test_generation_builder import build, INVENTORY, INVENTORY_HASH
from tests.retrieval.test_embedding_cache import dependencies

ROOT=Path(__file__).parents[2]


def assets(live):
    texts={role:role for role in ('failure_mode_update','skill_candidate')}
    return LiveTrainingAssets(online_prompts={p.entry.role:p for p in load_prompts(ROOT)},
        online_limits=SimpleNamespace(status='calibrated',runtime_inputs_sha256=D), online_limits_sha256=D,
        online_runtime_inputs_sha256=D,memory_prompts=live.prompts,memory_limits=live.limits,
        memory_limits_sha256=live.limits_hash,skill_prompts=texts,
        skill_prompt_hashes={role:'sha256:'+sha256(text.encode()).hexdigest() for role,text in texts.items()},
        skill_prompt_manifest_sha256=D,skill_limits_sha256=D,
        skill_token_limits={role:RoleTokenLimitConfig(version='synthetic-evidence',role=role,stage='calibrated',
            max_tokens=512,evidence_manifest_identity=D) for role in texts},
        public_tool_inventory=INVENTORY,public_tool_schemas=(),metadata=())


def test_provisional_memory_rejected_before_evidence_callback(live):
    with pytest.raises(ValueError,match='actual online and memory calibration'):
        assets(live).verify_formal(SimpleNamespace(),lambda *_:pytest.fail('not ready'))


def test_three_fresh_rounds_construct_actual_services_with_current_generation(live,tmp_path):
    live.requests.gateway.config = live.requests.gateway.config.model_copy(update=dict(
        expected_vllm_version="synthetic",container_digest=D,served_model_id="synthetic",
        structured_output_backend="synthetic",generation_config_policy="vllm",
        server_launch_configuration="synthetic",output_limit=4096))
    snapshot,_=build(tmp_path)
    _,_,deps=dependencies()
    instance=LiveTrainingRuntime.__new__(LiveTrainingRuntime)
    manifest=SimpleNamespace(run_id='run',profile='official_live',dataset_manifest_sha256=D,manifest_sha256=live.ledger.store.identity.config_manifest_sha256,
        online_prompt_manifest_sha256=D,environment_sha256=D,fixture=SimpleNamespace(manifest_sha256=D),
        tool_inventory_sha256=INVENTORY_HASH,run_root=tmp_path/'run',qwen=live.requests.gateway.config)
    instance.manifest,instance.assets,instance.ledger=manifest,assets(live),live.ledger
    instance.cache=SimpleNamespace(identity=SimpleNamespace())
    instance.store=SimpleNamespace()
    instance.train_gate=SimpleNamespace(load=lambda **_:pytest.fail('no data during construction'))
    phases=[]
    provider=SimpleNamespace(ledger=live.ledger,manifest_identity=manifest.manifest_sha256,
        qwen_config=live.requests.gateway.config,qwen_gateway=live.requests.gateway,
        gateways={role:live.requests.gateway for role in ('policy','critic','revision')},
        embedding_gateway=deps['gateway'],embedding_context_factory=lambda *args:None,
        embedding_record_durable=deps['record_durable'],user_factory=lambda *args:None)
    instance.provider_factory=lambda phase:phases.append(phase) or provider
    instance.dev_factory=lambda **_:SimpleNamespace(evaluate=lambda **kwargs:pytest.fail('no Dev during construction'))
    instance.authorization=lambda _:None
    instance.verify_skill_limit=lambda *_:True
    instance.count_tokens=lambda _:10
    instance.attempts=lambda *args:()
    instance.boot_id='boot'
    instance.generation_root=tmp_path
    instance.external_factory=None
    instance.backend_sha=D
    instance.metrics=LiveTrainingMetrics(ledger=live.ledger,manifest=manifest)
    instance.shards=(('a','b'),('c',),('d',))
    instance._built=set()
    def load(gid):
        return SimpleNamespace(**{**snapshot.__dict__,'manifest':snapshot.manifest.model_copy(update={'generation_id':gid}),
            'indexes':tuple(item.model_copy(update={'generation_id':gid}) for item in snapshot.indexes)})
    instance._load=load
    for index in range(3):
        runner=instance.build_round(index)
        assert isinstance(runner.episodes,LiveEpisodeExecutor)
        assert isinstance(runner.memory,LiveMemoryUpdater)
        assert isinstance(runner.skills,LiveSkillUpdater)
        assert isinstance(runner.generations,RoundGenerationCoordinator)
        assert runner.episodes.scope.generation_id==f'g{index:03d}'
        assert dict(runner.episodes.scope.scenario_positions)=={sid:p for p,sid in enumerate(instance.shards[index])}
    assert phases==['train_round','offline_update','generation_publication']*3
    with pytest.raises(ValueError,match='fresh consecutive'):
        instance.build_round(2)
    assert not live.transport.calls
