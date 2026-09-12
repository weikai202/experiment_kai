"""Cross-module runtime construction uses fake transports, no dataset access."""
from types import SimpleNamespace
import pytest
from pathlib import Path

from toolsandbox_pipeline.orchestration.live_native_dev import NativeDevBranchExecutor, _AuthorizedDevGate
from toolsandbox_pipeline.orchestration.live_dev_minibench import NativeDevBranchRequest
from toolsandbox_pipeline.orchestration.live_episode import LiveEpisodeExecutor
from toolsandbox_pipeline.orchestration.live_providers import LiveProviderServices
from toolsandbox_pipeline.online.prompt_loader import load_prompts
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
from toolsandbox_pipeline.schemas.offline_skill import SkillContent
from toolsandbox_pipeline.schemas.runtime import EmbeddingConfig,UserSimulatorConfig
from toolsandbox_pipeline.schemas.trajectory import EpisodeExecutionStatus
from tests.orchestration.test_live_offline import live,D
from tests.orchestration.test_generation_builder import build,INVENTORY,INVENTORY_HASH
from tests.orchestration.test_live_episode import record
from tests.retrieval.test_embedding_cache import Transport

ROOT=Path(__file__).parents[2]


@pytest.mark.parametrize('complete',[False,True])
def test_native_dev_builds_real_branch_generation_providers_and_episode(live,tmp_path,monkeypatch,complete):
    snapshot,_=build(tmp_path)
    qwen=live.requests.gateway.config.model_copy(update=dict(expected_vllm_version='synthetic',
        container_digest=D,served_model_id='synthetic',structured_output_backend='synthetic',
        generation_config_policy='vllm',server_launch_configuration='synthetic',output_limit=4096))
    config_sha=live.ledger.store.identity.config_manifest_sha256
    manifest=SimpleNamespace(run_id='run',run_root=tmp_path/'formal',manifest_sha256=config_sha,
        dataset_manifest_sha256=D,online_prompt_manifest_sha256=D,environment_sha256=D,
        fixture=SimpleNamespace(manifest_sha256=D),profile='official_live',tool_inventory_sha256=INVENTORY_HASH)
    assets=SimpleNamespace(public_tool_inventory=INVENTORY,online_limits_sha256=D,
        online_runtime_inputs_sha256=D,online_limits=SimpleNamespace(status='calibrated',runtime_inputs_sha256=D),
        online_prompts={p.entry.role:p for p in load_prompts(ROOT)},metadata=())
    transport=Transport()
    services=LiveProviderServices(ledger=live.ledger,qwen_config=qwen,
        embedding_config=EmbeddingConfig(expected_dimension=2),user_config=UserSimulatorConfig(),
        manifest_identity=config_sha,phase='dev_minibench',qwen_transport=live.transport,
        embedding_transport=transport,user_transport=object())
    gate=_AuthorizedDevGate(SimpleNamespace())
    gate.records['family']=record()
    from toolsandbox_pipeline.toolsandbox_adapter.trajectory_store import TrajectoryStore
    from toolsandbox_pipeline.schemas.trajectory import TrustedTrajectory,TrustedEvaluatorRecord
    from toolsandbox_pipeline.reproducibility import canonical_sha256
    store=TrajectoryStore(live.ledger.store,environment_identity=live.ledger.store.identity.environment_identity)
    seen=[]
    def native(self,**kwargs):
        seen.append((self,kwargs))
        if not complete:
            return SimpleNamespace(status=EpisodeExecutionStatus.TERMINAL_FAILURE_BEFORE_EVALUATION)
        identity=kwargs['identity']
        evaluator=TrustedEvaluatorRecord(milestone_similarity=1.,minefield_similarity=0.,similarity=1.,
            turn_count=0,milestone_mapping=(),minefield_mapping=(),fully_successful=True,
            evaluation_definition_sha256=identity.evaluation_definition_sha256,ending_context_sha256=D)
        eref,_=store.persist_evaluator(identity,evaluator)
        trajectory=TrustedTrajectory.build(identity=identity,messages=(),online_turns=(),tool_actions=(),
            logical_request_ids=(),physical_attempt_ids=(),ending_context_reference=eref,ending_context_sha256=D,
            evaluator_record_reference=eref,evaluator_record_sha256=canonical_sha256(evaluator.model_dump(mode='json')),
            skill_attributions=(),eligible_for_train_offline_consumption=False)
        tref,_=store.persist_trajectory(identity,trajectory)
        return SimpleNamespace(status=EpisodeExecutionStatus.COMPLETED_EVALUATED,
            evaluator_record_reference=eref,trusted_trajectory_reference=tref)
    monkeypatch.setattr(LiveEpisodeExecutor,'run_dev_lease',native)
    with EmbeddingCache(tmp_path/'branch-cache',EmbeddingIdentity(),2) as cache:
        runtime=SimpleNamespace(manifest=manifest,assets=assets,ledger=live.ledger,cache=cache,
            authorization=lambda _:None,provider_factory=lambda phase:services,count_tokens=lambda _:10,
            store=store,attempts=lambda *args:(),boot_id='boot',metrics=SimpleNamespace(record_episode=lambda _:None),
            external_factory=None,backend_sha=D)
        executor=NativeDevBranchExecutor(runtime=runtime,index=0,snapshot=snapshot,gate=gate,dev_manifest_sha256=D)
        current=snapshot.skills[0]
        content=SkillContent.from_record(current).model_copy(update={'instruction':'Use verified arguments before executing'})
        request=NativeDevBranchRequest(run_id='run',round_index=0,unit_id='unit',scenario_id='family',
            episode_id='dev-candidate',branch='candidate',evaluated_skill=content,evaluated_skill_version='v1.1',
            earlier_accepted_skills=(),shared_configuration_sha256=D,dev_manifest_sha256=D)
        result=executor.run_branch(request=request,scenario=object())
        assert result.result.complete==complete
        if complete:
            assert result.result.fully_successful and result.result.similarity==1.
        assert seen[0][1]['identity'].phase=='dev_minibench'
        assert seen[0][0].generation.skills[0].version=='v1.1'
        assert len(transport.calls)>0 and not live.transport.calls
        assert not live.ledger.applications()
