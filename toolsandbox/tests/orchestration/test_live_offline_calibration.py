from dataclasses import replace
import json
import pytest

from toolsandbox_pipeline.orchestration.live_offline_calibration import (
    CapturedOfflineCalibrationRequest, DurableOfflineCalibrationReplay,
    calibrate_offline_requests, OfflineCalibrationScope,
    collect_offline_calibration_requests, verify_collected_source,
)
from toolsandbox_pipeline.offline.skill_roles import OfflineSkillRoleRunner
from toolsandbox_pipeline.providers.contracts import ProviderRole, ProviderRequestError, TransportResponse
from toolsandbox_pipeline.reproducibility import canonical_sha256
from tests.orchestration.test_live_offline import live, prepared, D
from tests.providers.test_request_identity import chat_response


def replay_for(live):
    return DurableOfflineCalibrationReplay(ledger=live.ledger,gateway=live.requests.gateway,
        manifest_identity=live.requests.manifest_identity,count_tokens=lambda _:10)


def item(live):
    return CapturedOfflineCalibrationRequest('scenario','family','source-request','source-attempt',D,prepared(live))


def test_real_ledger_replay_restores_completed_without_redispatch(live):
    replay = replay_for(live)
    payload = item(live).payload()
    first = replay.invoke(payload,corpus_sha256=D,ordinal=0,ceiling=512)
    assert first['strict_valid'] and first['attempt']['metrics']['usage']['output_tokens'] == 3
    assert replay.invoke(payload,corpus_sha256=D,ordinal=0,ceiling=512) == first
    assert len(live.transport.calls) == 1
    assert not live.ledger.applications()
    assert live.ledger.effective_output_cost() == (0,True)


def test_length_is_durable_and_next_ceiling_has_new_request(live):
    replay = replay_for(live)
    live.transport.payload['choices'][0]['finish_reason'] = 'length'
    first = replay.invoke(item(live).payload(),corpus_sha256=D,ordinal=0,ceiling=512)
    assert not first['strict_valid']
    assert replay.invoke(item(live).payload(),corpus_sha256=D,ordinal=0,ceiling=512) == first
    live.transport.payload = chat_response('{"result":"NONE"}')
    second = replay.invoke(item(live).payload(),corpus_sha256=D,ordinal=1,ceiling=1024)
    assert first['logical_request_id'] != second['logical_request_id']
    assert len(live.transport.calls) == 2


def test_non_length_malformed_stops_and_does_not_increase_limit(live):
    replay = replay_for(live)
    live.transport.payload = chat_response('invalid json')
    with pytest.raises(ProviderRequestError):
        replay.invoke(item(live).payload(),corpus_sha256=D,ordinal=0,ceiling=512)
    with pytest.raises(RuntimeError,match='explicit recovery'):
        replay.invoke(item(live).payload(),corpus_sha256=D,ordinal=0,ceiling=512)
    assert len(live.transport.calls) == 1


def test_corpus_missing_role_or_unverified_source_stops_before_dispatch(live):
    kwargs = dict(captured=(item(live),),replay=replay_for(live),runtime_inputs={'identity':D},hard_ceiling=4096)
    with pytest.raises(ValueError,match='unverified'):
        calibrate_offline_requests(**kwargs,verify_source=lambda _:False)
    with pytest.raises(ValueError,match='nonempty'):
        calibrate_offline_requests(**kwargs,verify_source=lambda _:True)
    assert not live.transport.calls


def test_train_source_label_and_context_boundaries(live):
    with pytest.raises(ValueError,match='train'):
        replace(item(live),split='dev').payload()
    replay = replay_for(live)
    replay.count_tokens = lambda _:32768
    with pytest.raises(ValueError,match='context'):
        replay.invoke(item(live).payload(),corpus_sha256=D,ordinal=0,ceiling=512)
    assert not live.transport.calls


def test_four_role_actual_usage_full_rerun_and_idempotent_evidence(live):
    from toolsandbox_pipeline.offline.memory_roles import prepare_review_request
    from toolsandbox_pipeline.schemas.offline_memory import PolicyMemoryCandidate
    candidate = lambda: PolicyMemoryCandidate(result='CANDIDATE',role='policy',candidate=dict(scope='Prerequisites',applicability=[],action_guidance='Verify arguments',avoid=[]))
    # Reuse production schemas, fake responses only at the transport boundary.
    requests = [prepared(live)]
    requests.append(prepare_review_request(candidate(),(),unit_reference='review',prompts=live.prompts,
        limits=live.limits,limits_sha256=live.limits_hash,structured_output_wire_mode='guided_json'))
    for role,cap in ((ProviderRole.FAILURE_MODE_UPDATE,512),(ProviderRole.SKILL_CANDIDATE,2048)):
        requests.append(OfflineSkillRoleRunner(live.requests.gateway,role=role,max_tokens=cap).prepare(
            unit_id=role.value,messages=[{'role':'user','content':'actual synthetic role input'}]))
    captured = tuple(CapturedOfflineCalibrationRequest('scenario','family',f'source-{i}',f'attempt-{i}',D,p) for i,p in enumerate(requests))
    from tests.offline_skill.test_rewrite import skill
    from toolsandbox_pipeline.schemas.offline_skill import SkillContentCandidate, SkillContent
    skill_candidate = lambda: SkillContentCandidate(candidate=SkillContent.from_record(skill()))
    outputs = ['{"result":"NONE"}', '{"decision":"SKIP","reason":"Not reusable"}',
               '{"decision":"SKIP","reason":"No reusable mode"}', skill_candidate().model_dump_json()]
    class Sequence:
        def __init__(self): self.calls=[]
        def create(self,**kwargs):
            self.calls.append(kwargs)
            response=chat_response(outputs[(len(self.calls)-1)//2])
            return TransportResponse(json.dumps(response).encode(),response)
    transport=Sequence()
    live.requests.gateway._transport=transport
    replay=replay_for(live)
    kwargs=dict(captured=captured,replay=replay,verify_source=lambda _:True,runtime_inputs={'identity':D},hard_ceiling=4096)
    result=calibrate_offline_requests(**kwargs)
    assert len(transport.calls)==8
    from toolsandbox_pipeline.orchestration.live_offline_calibration import offline_limit_recommendations
    recommendation=offline_limit_recommendations(evidence=result,replay=replay,runtime_inputs={'identity':D})
    selected=recommendation['selections']['skill_candidate']
    assert selected.stage=='calibrated'
    assert recommendation['verify_token_limit_evidence'](selected,recommendation['hashes']['offline_skill_token_limits.calibrated.json'])
    assert not recommendation['verify_token_limit_evidence'](selected,D)
    with pytest.raises(ValueError,match='identity'):
        offline_limit_recommendations(evidence=result,replay=replay,runtime_inputs={'changed':D})
    assert result['total_tokens']==64 and result['usage_complete']
    assert all(r['recommended_max_tokens']==64 for r in result['roles'].values())
    assert calibrate_offline_requests(**kwargs)==result
    assert len(transport.calls)==8


def pilot_fixture(position=None, success=True):
    from tests.orchestration.test_live_projections import sample
    from toolsandbox_pipeline.schemas.trajectory import TrustedTrajectory
    from toolsandbox_pipeline.orchestration.live_projections import CapturedEpisodeInputs, CapturedOnlineTurn
    t,c,_=sample(critic=False,exception=False,success=success)
    extra={} if position is None else dict(manifest_position=position,episode_id=f'episode-{position}',scenario_id=f'scenario-{position}')
    identity=type(t.identity).model_validate({**t.identity.model_dump(),**extra,
        'phase':'online_token_calibration_pilot','round_index':None,'shard_id':None})
    decision=c.turns[0].decision
    online_extra={} if position is None else dict(episode_id=f'episode-{position}',scenario_id=f'scenario-{position}')
    online=type(decision.identity).model_validate({**decision.identity.model_dump(),**online_extra,
        'phase':'online_token_calibration_pilot','round_index':None,'shard_id':None})
    decision=type(decision).model_validate({**decision.model_dump(),'identity':online})
    turn=type(t.online_turns[0]).model_validate({**t.online_turns[0].model_dump(),
        'decision_sha256':canonical_sha256(decision.model_dump(mode='json'))})
    body={name:getattr(t,name) for name in type(t).model_fields}
    body.pop('trajectory_id');body.pop('schema_version',None)
    body.update(identity=identity,online_turns=(turn,),eligible_for_train_offline_consumption=False)
    t=TrustedTrajectory.build(**body)
    captured=CapturedEpisodeInputs(c.evaluator,(CapturedOnlineTurn(c.turns[0].policy_envelope,c.turns[0].proposed_action,decision),))
    return t,captured


def test_natural_collection_preserves_noneligible_pilot_and_does_not_force_rare_roles(live):
    from tests.orchestration.test_live_offline import Resolver
    from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
    from tests.offline_skill.test_orchestrator import current_skill
    t,c=pilot_fixture()
    scope=OfflineCalibrationScope('run',D,D,('scenario',))
    scope.validate(t)
    skill=current_skill().model_copy(update={'skill_id':'skill-1','tool_dependencies':('search_messages',)})
    retriever=MemoryCandidateRetriever(generation_id='g000',policy_records=(),world_records=(),indexes=(),embedding_resolver=Resolver())
    result=collect_offline_calibration_requests(trajectories=(t,),capture_loader=lambda _:c,scope=scope,
        replay=replay_for(live),prompts=live.prompts,limits=live.limits,limits_sha256=live.limits_hash,
        retriever=retriever,current_skills=(skill,),skill_prompts={'failure_mode_update':'system','skill_candidate':'system'},
        public_tool_schemas=(),runtime_inputs={'identity':D},hard_ceiling=4096)
    assert len(result)==1 and result[0].prepared.role=='memory_candidate'
    assert not t.eligible_for_train_offline_consumption
    assert len(live.transport.calls)==1 and not live.ledger.applications()
    assert verify_collected_source(live.ledger,result[0])
    assert not verify_collected_source(live.ledger,replace(result[0],scenario_id='different'))


def test_calibration_scope_rejects_formal_round_and_unlisted_scenario():
    from tests.orchestration.test_live_projections import sample
    t,_,_=sample()
    with pytest.raises(ValueError,match='pilot'):
        OfflineCalibrationScope('run',D,D,('scenario',)).validate(t)
    t,_=pilot_fixture()
    with pytest.raises(ValueError,match='pilot'):
        OfflineCalibrationScope('run',D,D,('other',)).validate(t)


def test_pilot_capture_checkpoint_loader_reads_exact_blobs_and_checks_inputs(live):
    from datetime import datetime,timezone
    from toolsandbox_pipeline.schemas.trajectory import EpisodeResult,EpisodeExecutionStatus
    from toolsandbox_pipeline.schemas.accounting import TaskAccountingInput,ScopeTimingInput
    from toolsandbox_pipeline.orchestration.live_offline_calibration import load_pilot_calibration_inputs
    from toolsandbox_pipeline.reproducibility import canonical_json_bytes
    t,c=pilot_fixture()
    def put(value,name):
        return live.ledger.store.blobs.put(canonical_json_bytes(value.model_dump(mode='json')),
            media_type='application/vnd.toolsandbox.canonical+json',schema_name=name,schema_version=1)
    tref,eref=put(t,'TrustedTrajectory'),put(c.evaluator,'TrustedEvaluatorRecord')
    timing=ScopeTimingInput(scope_kind='scenario_task',scope_id='episode',boot_id='boot',
        started_at_utc=datetime.now(timezone.utc),start_monotonic_ns=0)
    accounting=TaskAccountingInput(run_id='run',task_id='episode',scenario_family_id='family',scenario_id='scenario',
        system_variant='generation_0',timing=timing,completion_status='complete')
    result=EpisodeResult(identity=t.identity,status=EpisodeExecutionStatus.COMPLETED_EVALUATED,
        ending_context_reference=t.ending_context_reference,ending_context_sha256=t.ending_context_sha256,
        trusted_trajectory_reference=tref,evaluator_record_reference=eref,task_accounting_input=accounting,last_checkpoint_ordinal=1)
    live.ledger.commit_checkpoint('calibration-episode-capture-'+canonical_sha256('episode')[7:],
        'calibration_episode_capture',dict(identity=t.identity.model_dump(mode='json'),result=result.model_dump(mode='json'),
            initial_envelopes=[[0,json.dumps(c.turns[0].policy_envelope)]],
            decisions=[c.turns[0].decision.model_dump(mode='json')],
            proposed_actions=[c.turns[0].proposed_action.model_dump(mode='json')]))
    restored,loader=load_pilot_calibration_inputs(ledger=live.ledger,episode_ids=('episode',),scope=OfflineCalibrationScope('run',D,D,('scenario',)))
    assert restored==(t,) and loader(t)==c
    with pytest.raises(ValueError,match='missing'):
        load_pilot_calibration_inputs(ledger=live.ledger,episode_ids=('missing',),scope=OfflineCalibrationScope('run',D,D,('scenario',)))
    assert not live.transport.calls


def test_ten_real_attributions_naturally_trigger_isolated_skill_rewrite(live):
    from tests.orchestration.test_live_offline import Resolver
    from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
    from tests.offline_skill.test_orchestrator import current_skill
    from toolsandbox_pipeline.schemas.offline_skill import SkillContent,SkillContentCandidate
    pairs=[pilot_fixture(i,success=False) for i in range(10)]
    trajectories=tuple(t for t,_ in pairs)
    captures={t.trajectory_id:c for t,c in pairs}
    skill=current_skill().model_copy(update={'skill_id':'skill-1','tool_dependencies':('search_messages',)})
    candidate=SkillContentCandidate(candidate=SkillContent.from_record(skill))
    class NaturalTransport:
        def __init__(self):self.calls=[]
        def create(self,**request):
            self.calls.append(request)
            system=request['messages'][0]['content']
            if system=='failure-system':content='{"decision":"SKIP","reason":"No reusable mode"}'
            elif system=='skill-system':content=candidate.model_dump_json()
            else:content='{"result":"NONE"}'
            response=chat_response(content)
            return TransportResponse(json.dumps(response).encode(),response)
    transport=NaturalTransport()
    live.requests.gateway._transport=transport
    result=collect_offline_calibration_requests(trajectories=trajectories,capture_loader=lambda t:captures[t.trajectory_id],
        scope=OfflineCalibrationScope('run',D,D,tuple(t.identity.scenario_id for t in trajectories)),
        replay=replay_for(live),prompts=live.prompts,limits=live.limits,limits_sha256=live.limits_hash,
        retriever=MemoryCandidateRetriever(generation_id='g000',policy_records=(),world_records=(),indexes=(),embedding_resolver=Resolver()),
        current_skills=(skill,),skill_prompts={'failure_mode_update':'failure-system','skill_candidate':'skill-system'},
        public_tool_schemas=(),runtime_inputs={'identity':D},hard_ceiling=4096)
    assert len(result)==21
    roles=[p.payload()['role'] for p in result]
    assert roles.count('failure_mode_update')==10 and roles.count('skill_candidate')==1
    assert roles.count('memory_review')==0
    assert all(verify_collected_source(live.ledger,p) for p in result)
    assert skill.online_statistics.evaluated_uses==0 and not live.ledger.applications()
    assert all(not t.eligible_for_train_offline_consumption for t in trajectories)
