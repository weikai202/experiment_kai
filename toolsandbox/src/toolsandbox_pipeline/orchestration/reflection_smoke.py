"""Development-only fresh G000 -> Policy reflection -> G001 paired smoke.

Run through the owner-only reflection-smoke launcher. No formal training.
"""
import os, json, locale, time, uuid, inspect, typing
from pathlib import Path
from dataclasses import replace
from contextlib import redirect_stdout, redirect_stderr
import io
import httpx
from pydantic import TypeAdapter
from openai import NOT_GIVEN
from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.tool_conversion import convert_to_openai_tool
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from toolsandbox_pipeline.reproducibility import canonical_sha256, canonical_json_bytes
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.checkpointing import CheckpointStore, RunIdentity, LLMLedger, LogicalLLMRequestIdentity
from toolsandbox_pipeline.checkpointing.tool_ledger import ToolLedger
from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig, UserSimulatorConfig
from toolsandbox_pipeline.providers.preflight import run_preflight
from toolsandbox_pipeline.providers.qwen import QwenGateway
from toolsandbox_pipeline.metrics.usage import UsageRecorder
from toolsandbox_pipeline.providers.embedding import EmbeddingGateway
from toolsandbox_pipeline.providers.user_simulator import InstrumentedGPT4oMiniUser
from toolsandbox_pipeline.providers.contracts import ProviderRole, ProviderRequestError
from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
from toolsandbox_pipeline.retrieval.service import RetrievalService
from toolsandbox_pipeline.orchestration.seed_skill_builder import build_seed_skills
from toolsandbox_pipeline.orchestration.generation_builder import build_generation
from toolsandbox_pipeline.online.prompt_loader import load_prompts
from toolsandbox_pipeline.online.token_limits import load_token_limits
from toolsandbox_pipeline.online.prompt_builder import initial_context, critic_context, prepare_request
from toolsandbox_pipeline.online.prompt_contracts import RevisionContext
from toolsandbox_pipeline.online.qwen_roles import InitialPolicyRunner, CriticRunner, RevisionRunner
from toolsandbox_pipeline.online.durable_roles import LedgerQwenResponseSeam, Task011DurableRoleBackend, Task011CheckpointEventSink, DurableRoleExecutor
from toolsandbox_pipeline.online.turn_context import TurnContext, TurnRoleRequestBuilders, PreparedInitialPolicy, PreparedCritic
from toolsandbox_pipeline.online.turn_responder import DurableTurnResponder
from toolsandbox_pipeline.online.state_builder import StateBuilder
from toolsandbox_pipeline.online.controller import Controller
from toolsandbox_pipeline.online.controller_inputs import ReproducibilityProfile
from toolsandbox_pipeline.online.tool_metadata import load_tool_metadata, public_tool_inventory
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnIdentity, OnlineTurnAuditRecord
from toolsandbox_pipeline.schemas.trajectory import EpisodeIdentity
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.state import CommittedToolOutcomeInput
from toolsandbox_pipeline.schemas.dataset import SplitManifest
from toolsandbox_pipeline.toolsandbox_adapter.pipeline_agent import PipelineAgent
from toolsandbox_pipeline.toolsandbox_adapter.messages import extract_visible_messages
from toolsandbox_pipeline.toolsandbox_adapter.tools import build_adapter_turn
from toolsandbox_pipeline.toolsandbox_adapter.trajectory_store import TrajectoryStore
from toolsandbox_pipeline.toolsandbox_adapter.native_evaluator import NativeEvaluator
from toolsandbox_pipeline.toolsandbox_adapter.episode_runner import EpisodeRunner, EpisodeRunInput
from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import TransactionalAgentRole, TransactionalUserRole, TransactionalExecutionEnvironment, ToolActionBinding
from toolsandbox_pipeline.reproducibility.dataset_access import DatasetAccessGate
from toolsandbox_pipeline.reproducibility.dataset_manifest import load_build_config
from toolsandbox_pipeline.reproducibility.clock import FixedWorldClock
from toolsandbox_pipeline.reproducibility.splits import build_augmented_registry

ROOT=Path('/root/toolsandbox-runtime')
PROJECT=Path(__file__).resolve().parents[3]

def save(path, data):
    raw=canonical_json_bytes(data)
    temp=path.with_suffix(path.suffix+'.tmp')
    with open(temp,'wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def main():
    os.umask(0o077)
    os.environ['TZ']='UTC';time.tzset();locale.setlocale(locale.LC_ALL,'C.UTF-8')
    if not os.environ.get('OPENAI_API_KEY'):
        print('OPENAI_API_KEY');return 2
    os.environ['QWEN_BASE_URL']='http://127.0.0.1:18080/v1'
    os.environ['QWEN_API_KEY']='EMPTY'
    run_id='reflection-smoke-'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]
    run=ROOT/'runs'/run_id;run.mkdir(parents=True,mode=0o700)
    save(ROOT/'latest-reflection-run.json',{'run_id':run_id,'run_dir':str(run),'status':'running'})
    campaign_config=json.loads((PROJECT/'configs/run/reflection_smoke_v1.json').read_text())
    if campaign_config['scope']!='policy_memory' or campaign_config['formal'] or campaign_config['split']!='train':raise ValueError('ReflectionConfigMismatch')
    qc=QwenConfig.model_validate_json((ROOT/'qwen-config.json').read_bytes())
    ec=EmbeddingConfig();uc=UserSimulatorConfig()
    preflight=[]
    for mode,cfg in [('qwen',qc),('embedding',ec),('user-simulator',uc)]:
        result=run_preflight(mode,cfg);preflight.append(result)
        save(run/'preflight.json',preflight)
        print(json.dumps({'phase':'preflight',**result}),flush=True)
        if result['status']!='pass':
            save(ROOT/'latest-reflection-run.json',{'run_id':run_id,'run_dir':str(run),'status':'preflight_failed'});return 2
    ec=ec.model_copy(update={'expected_dimension':preflight[1]['vector_dimension']})
    train_path=ROOT/'dataset/train_manifest.json'
    train=SplitManifest.model_validate_json(train_path.read_bytes());train_hash=file_hash(train_path.read_bytes())
    # Fixed before observing any results: first train family, first two variants.
    selection=json.loads((ROOT/'smoke-selection.json').read_text())
    if selection['scenario_ids']!=campaign_config['scenario_ids']:raise ValueError('ReflectionSelectionMismatch')
    if selection['dataset_manifest_sha256']!=train_hash:raise ValueError('SelectionManifestMismatch')
    records=tuple(train.scenarios[i] for i in selection['manifest_positions'])
    if [r.scenario_id for r in records]!=selection['scenario_ids']:raise ValueError('SelectionIdentityMismatch')
    previous={'completed':[]}
    build_cfg,_=load_build_config(PROJECT/'configs/reproducibility/dataset_build_v1.json')
    prompts=dict(zip(('policy','critic','revision'),load_prompts(PROJECT)))
    ph=file_hash((PROJECT/'prompts/manifest.json').read_bytes())
    limitpath=PROJECT/'configs/online_token_limits.smoke.json';lh=file_hash(limitpath.read_bytes())
    limits=load_token_limits(limitpath,expected_sha256=lh)
    metadata_path=PROJECT/campaign_config.get('tool_metadata','configs/tool_metadata/controller_tool_metadata.jsonl')
    metadata_manifest_path=PROJECT/campaign_config.get('tool_metadata_manifest','configs/tool_metadata/manifest.json')
    metadata=load_tool_metadata(metadata_path,metadata_manifest_path);by_name={m.canonical_tool_name:m for m in metadata}
    inventory=public_tool_inventory();ih=canonical_sha256(list(inventory))
    fixture_config={'development_only':True,'external_read_mode':'deny_until_fixture_available'}
    fh=canonical_sha256(fixture_config)
    manifest={'reflection_input_representation':'packed-v2','reflection_packing_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/offline/reflection_packing.py').read_bytes()),'memory_roles_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/offline/memory_roles.py').read_bytes()),'memory_orchestrator_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/offline/memory_orchestrator.py').read_bytes()),'critic_schema_source_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/schemas/critic.py').read_bytes()),'offline_memory_schema_source_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/schemas/offline_memory.py').read_bytes()),'execution_call_identity_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/toolsandbox_adapter/call_identity.py').read_bytes()),'transactional_roles_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/toolsandbox_adapter/transactional_roles.py').read_bytes()),'retrieval_embedding_strategy':'utf8-token-chunks-weighted-l2-v1','retrieval_long_inputs_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/retrieval/long_inputs.py').read_bytes()),'embedding_cache_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/retrieval/embedding_cache.py').read_bytes()),'development_only':True,'formal':False,'pythonhashseed':0,'profile':'official_live','external_read_mode':'fixture_only_no_live_tools',
      'image':json.loads((ROOT/'image-provenance.json').read_text()),'model_revision':'9216db5781bf21249d130ec9da846c4624c16137',
      'qwen':qc.model_dump(mode='json'),'dataset_manifest_sha256':train_hash,'scenario_ids':[r.scenario_id for r in records],'selection':selection,'completed_before_this_run':previous['completed'],
      'prompt_manifest_sha256':ph,'token_limit_config_sha256':lh,'tool_metadata_sha256':file_hash(metadata_path.read_bytes()),'tool_metadata_manifest_sha256':file_hash(metadata_manifest_path.read_bytes()),'prompt_builder_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/online/prompt_builder.py').read_bytes()),'prompt_contracts_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/online/prompt_contracts.py').read_bytes()),'runner_sha256':file_hash(Path(__file__).read_bytes()),
      'dependency_lock_sha256':file_hash((PROJECT/'uv.lock').read_bytes()),'embedding_limit_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/providers/embedding_limits.py').read_bytes()),
      'reflection_config':campaign_config,'reflection_support_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/orchestration/reflection_support.py').read_bytes()),'memory_retrieval_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/offline/memory_retrieval.py').read_bytes()),'generation_builder_sha256':file_hash((PROJECT/'src/toolsandbox_pipeline/orchestration/generation_builder.py').read_bytes()),
      'reflection':{'scope':'policy_memory','rounds':1,'same_train_retest':True,'skill_rewrite':False,'world_reflection':False},
      'offline_prompt_manifest_sha256':file_hash((PROJECT/'prompts/offline/memory_manifest.json').read_bytes()),
      'offline_limits_sha256':file_hash((PROJECT/'configs/offline_memory_token_limits.provisional.json').read_bytes()),
      'notes':['Development remote User with fixed world clock; not trajectory-level strict replay.','One fresh Policy-memory reflection round; same-train diagnostic retest, no formal/test evaluation.']}
    mh=canonical_sha256(manifest);save(run/'manifest.json',manifest)
    environment_hash=canonical_sha256({'image':manifest['image'],'model_revision':manifest['model_revision'],'qwen':manifest['qwen']})
    seed,provenance=build_seed_skills();save(run/'seed-provenance.json',provenance)
    identity=RunIdentity(run_id=run_id,profile='official_live',environment_identity=environment_hash,dataset_manifest_sha256=train_hash,config_manifest_sha256=mh,prompt_manifest_sha256=ph,generation_manifest_sha256=canonical_sha256([s.model_dump(mode='json') for s in seed]),fixture_manifest_sha256=fh)
    store=CheckpointStore.create(run/'checkpoint',identity,PROJECT/'configs/reproducibility/checkpointing_v1.json')
    ledger=LLMLedger(store)
    attempt_reports=[]
    scope=AccountingScope(run_id=run_id,task_id='generation-zero',scenario_family_id='setup',scenario_id='setup',system_variant='generation_0')
    def request(role,payload,unit,model,schema):
        rid=ledger.prepare_request(LogicalLLMRequestIdentity(run_id=run_id,role=role,phase='development_smoke',unit_reference=unit,input_fingerprint=canonical_sha256(payload),model=model,decoding_configuration_sha256=canonical_sha256({'model':model,'role':role.value}),output_schema_sha256=canonical_sha256(schema)))
        ledger.bind_accounting_scope(rid.logical_request_id,scope)
        ctx=ledger.allocate_attempt(rid.logical_request_id,manifest_identity=mh,replayed_after_unknown_outcome=False)
        ledger.mark_in_flight(ctx);return ctx
    def emb_context(ordinal,texts):return request(ProviderRole.EMBEDDING,list(texts),scope.task_id+'-'+canonical_sha256(list(texts))[7:],ec.model,{'type':'embedding'})
    def emb_durable(response):
        ledger.complete_response(response,validated_output=[list(v) for v in response.value],output_schema_name='EmbeddingVectors',output_schema_version=1)
        attempt_reports.append(response.attempt.model_dump(mode='json'));return True
    eg=EmbeddingGateway(ec)
    cache=EmbeddingCache(run/'embedding-cache.sqlite',EmbeddingIdentity(),ec.expected_dimension)
    (run/'generation-staging').mkdir()
    gen=build_generation(run/'generation-staging',generation_id='g000',parent_generation_id=None,policy_memory=(),world_memory=(),skills=seed,tool_inventory=inventory,tool_inventory_sha256=ih,cache=cache,gateway=eg,context_factory=emb_context,record_durable=emb_durable)
    retrieval=RetrievalService(gen,cache=cache,gateway=eg,context_factory=emb_context,record_durable=emb_durable)
    qgs={role:QwenGateway(qc,recorder=UsageRecorder(ProviderRole(role))) for role in ('policy','critic','revision')}
    seam=LedgerQwenResponseSeam(ledger)
    runners={role:cls(qgs[role],manifest_identity=mh,mode='offline',durability_seam=seam) for role,cls in [('policy',InitialPolicyRunner),('critic',CriticRunner),('revision',RevisionRunner)]}
    def count_tokens(messages):
        r=httpx.post('http://127.0.0.1:18080/tokenize',json={'model':qc.model,'messages':[m.model_dump(mode='json') for m in messages],'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}},timeout=60)
        r.raise_for_status();return r.json()['count']
    # Registry reconstruction is host-only; only the access gate returns train leases.
    from tool_sandbox.scenarios import named_scenarios
    from toolsandbox_pipeline.reproducibility.splits import build_family_registry
    from toolsandbox_pipeline.schemas.dataset import REGISTRIES
    from tool_sandbox.scenarios import named_single_tool_call_scenarios,named_multiple_tool_call_scenarios,named_multiple_user_turn_scenarios,named_insufficient_information_scenarios
    episode_reports=[];clock=FixedWorldClock(build_cfg)
    with clock:
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            families=build_family_registry(dict(zip(REGISTRIES,(named_single_tool_call_scenarios,named_multiple_tool_call_scenarios,named_multiple_user_turn_scenarios,named_insufficient_information_scenarios))),ToolBackend.DEFAULT)
            registry=build_augmented_registry(families,named_scenarios,ToolBackend.DEFAULT)
        audits=[]
        gate=DatasetAccessGate(manifest_path=train_path,expected_manifest_sha256=train_hash,reconstruct=lambda rec:registry[rec.scenario_id],audit=audits.append)
        leases=gate.load(requested_ids=[r.scenario_id for r in records],run_id=run_id,phase='development_smoke',manifest_sha256=train_hash)
        save(run/'dataset-access.json',audits)
        from toolsandbox_pipeline.orchestration.reflection_support import policy_projection, LedgerMemoryDurability
        from toolsandbox_pipeline.offline.memory_orchestrator import MemoryTrajectoryReference, SealedMemoryTrajectoryBuffer, MemoryUpdateOrchestrator, AppliedMemoryDecision
        from toolsandbox_pipeline.offline.memory_roles import MemoryRoleRunner, load_token_limits as load_memory_limits
        from toolsandbox_pipeline.offline.memory_prompts import load_memory_prompts
        from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
        from toolsandbox_pipeline.schemas.offline_memory import MemoryUpdateIdentity
        from toolsandbox_pipeline.orchestration.generation_builder import apply_generation_overlays
        phase_metrics=[]
        def metrics_snapshot():
            attempts=attempt_reports+[a.model_dump(mode='json') for g in qgs.values() for a in g.recorder.attempts]
            complete=all(a['metrics']['usage']['usage_complete'] for a in attempts)
            tokens=sum(a['metrics']['usage']['total_tokens'] or 0 for a in attempts) if complete else None
            cost,cost_complete=ledger.effective_output_cost()
            return tokens,complete,cost,cost_complete
        def phase_record(name,started,before):
            after=metrics_snapshot()
            return {'phase':name,'seconds':time.monotonic()-started,
                    'total_tokens':after[0]-before[0] if after[1] and before[1] else None,
                    'usage_complete':after[1] and before[1],
                    'total_cost':after[2]-before[2] if after[3] and before[3] else None,
                    'cost_complete':after[3] and before[3],'cost_unit':'qwen_effective_output_tokens'}
        def rec_position(rec,train):return next(i for i,r in enumerate(train.scenarios) if r.scenario_id==rec.scenario_id)
        entries=[]
        for pass_index in range(2):
            gid=gen.manifest.generation_id
            variant='generation_0' if pass_index==0 else 'updated'
            phase='train_reflection_smoke' if pass_index==0 else 'development_reflection_retest'
            phase_start=time.monotonic()
            phase_before=metrics_snapshot()
            for pos,lease in enumerate(leases):
                rec=lease.record;eid=f'{run_id}-{gid}-episode-{pos}'
                scope=AccountingScope(run_id=run_id,round_index=0 if pass_index==0 else None,task_id=eid,scenario_family_id=rec.scenario_family_id,scenario_id=rec.scenario_id,system_variant=variant)
                ei=EpisodeIdentity(run_id=run_id,profile='official_live',phase=phase,family_id=rec.scenario_family_id,scenario_id=rec.scenario_id,episode_id=eid,manifest_position=next(i for i,r in enumerate(train.scenarios) if r.scenario_id==rec.scenario_id),system_variant=variant,generation_id=gid,starting_context_sha256=rec.starting_context_sha256,evaluation_definition_sha256=rec.evaluation_definition_sha256,agent_tool_schema_sha256=rec.agent_facing_tool_schema_sha256,dataset_manifest_sha256=train_hash,runtime_config_sha256=mh,prompt_manifest_sha256=ph,token_limit_config_sha256=lh,fixture_manifest_sha256=fh,environment_sha256=environment_hash,max_messages=rec.max_messages,round_index=0 if pass_index==0 else None,shard_id='train-shard-0' if pass_index==0 else None)
                ts=TrajectoryStore(store,environment_identity=environment_hash)
                captured_envelopes=[]
                toolsledger=ToolLedger(store);responders={};outcomes=[];contracts={};bindings=[]
                def factory(turn_index):
                    visible=extract_visible_messages(PipelineAgent)
                    adapter=build_adapter_turn(PipelineAgent,visible)
                    mapping=dict(adapter.controller_context.agent_to_execution_name);reverse={v:k for k,v in mapping.items()}
                    for name,fn in adapter.controller_context.tool_objects.items():
                        ret=typing.get_type_hints(fn).get('return')
                        if ret is not None:contracts[mapping[name]]=TypeAdapter(ret)
                    from toolsandbox_pipeline.orchestration.reflection_support import committed_outcomes
                    current_outcomes=committed_outcomes(bindings,environment.records,visible)
                    online_id=OnlineTurnIdentity(run_id=run_id,profile=ReproducibilityProfile.OFFICIAL_LIVE,phase=phase,family_id=rec.scenario_family_id,scenario_id=rec.scenario_id,episode_id=eid,agent_turn_index=turn_index,expected_generation_id=gid,dataset_manifest_sha256=train_hash,runtime_config_sha256=mh,prompt_manifest_sha256=ph,token_limit_config_sha256=lh,fixture_manifest_sha256=fh,environment_identity=environment_hash,round_index=0 if pass_index==0 else None,shard_id='train-shard-0' if pass_index==0 else None)
                    def prepared(ctx,role):return prepare_request(ctx,prompt=prompts[role],token_limits=limits,token_limit_config_sha256=lh,qwen_config=qc,canonical_to_agent=reverse,mode='calibration',count_prompt_tokens=count_tokens)
                    def initial(state,bundle):
                        ctx=initial_context(state,bundle,prompts['policy']);captured_envelopes.append(json.loads(ctx.user_envelope));return PreparedInitialPolicy(ctx,prepared(ctx,'policy'))
                    def critic(initial,action,decision,bundle):
                        ctx=critic_context(initial,action,decision,bundle);return PreparedCritic(ctx,prepared(ctx,'critic'))
                    def revision(initial,critic,output):
                        ctx=RevisionContext(initial=initial,critic=critic,critic_feedback_json=output.model_dump_json())
                        return prepared(ctx,'revision')
                    backend=Task011DurableRoleBackend(ledger=ledger,runners=runners,run_id=run_id,phase=phase,qwen_model=qc.model,decoding_configuration_sha256=canonical_sha256(qc.model_dump(mode='json')),manifest_identity=mh,accounting_scope=scope)
                    responder=DurableTurnResponder(TurnContext(identity=online_id,generation=gen,retrieval=retrieval,state_builder=StateBuilder(contracts),controller=Controller(),controller_tool_metadata=metadata,role_request_builders=TurnRoleRequestBuilders(initial,critic,revision),durable_roles=DurableRoleExecutor(backend),checkpoint_sink=Task011CheckpointEventSink(ledger),committed_tool_outcomes=tuple(current_outcomes)))
                    responders[turn_index]=(responder,mapping);return responder
                def load_audit(decision):
                    responder,_=responders[decision.identity.agent_turn_index]
                    event=ledger.get_checkpoint(responder._final_checkpoint_id(responder._input_fingerprint))
                    return OnlineTurnAuditRecord.model_validate_json(json.dumps(event.payload['audit_record']))
                agent=TransactionalAgentRole(identity=ei,trajectory_store=ts,responder_factory=factory,audit_loader=load_audit)
                from toolsandbox_pipeline.toolsandbox_adapter.call_identity import execution_call_ids
                def binding(messages):
                    ordinal=len(environment.records)+1
                    if messages[0].sender==RoleType.USER:
                        return ToolActionBinding(action_sha256=canonical_sha256([m.content for m in messages]),call_ids=tuple(m.openai_tool_call_id or ('user-control-'+canonical_sha256([eid,ordinal,index,m.content])[7:]) for index,m in enumerate(messages)),selected_skill_ids=(None,)*len(messages),canonical_tool_ids=('end_conversation',)*len(messages),effect_classes=('conversation_control',)*len(messages),action_ordinal=ordinal)
                    decision=agent.decisions[-1];action=decision.final_action.action
                    calls=tuple(action.calls) if hasattr(action,'calls') else (action,)
                    _,mapping=responders[decision.identity.agent_turn_index]
                    effects=tuple(by_name[mapping[c.name]].effect.value for c in calls)
                    if 'external_read' in effects:raise RuntimeError('SmokeRequiresUnavailableExternalFixture')
                    bound=ToolActionBinding(action_sha256=canonical_sha256(decision.final_action.model_dump(mode='json')),call_ids=execution_call_ids(decision),selected_skill_ids=tuple(c.selected_skill_id for c in calls),canonical_tool_ids=tuple(mapping[c.name] for c in calls),effect_classes=effects,action_ordinal=ordinal)
                    bindings.append((bound,calls,mapping));return bound
                environment=TransactionalExecutionEnvironment(ExecutionEnvironment(),identity=ei,tool_ledger=toolsledger,trajectory_store=ts,binding_provider=binding,backend_manifest_sha256=fh)
                class UserSeam:
                    def load_completed_response(self,ctx):return None
                    def persist_completed_response(self,response):
                        ledger.complete_response(response,validated_output=response.value.model_dump(mode='json'),output_schema_name='ChatCompletion',output_schema_version=1)
                        attempt_reports.append(response.attempt.model_dump(mode='json'))
                def user_context():
                    messages=user.filter_messages(user.get_messages())
                    tools=[convert_to_openai_tool(t) for t in user.get_available_tools().values()] if messages[-1].sender==RoleType.AGENT else None
                    return request(ProviderRole.USER_SIMULATOR,{'messages':user.to_openai_messages(messages),'tools':tools,'temperature':'omitted','top_p':'omitted','seed':'omitted'},eid+'-user-'+str(get_current_context().max_sandbox_message_index),uc.model,{'type':'ChatCompletion'})
                user=InstrumentedGPT4oMiniUser(context_provider=user_context,config=uc,durability_seam=UserSeam())
                roles={RoleType.AGENT:agent,RoleType.EXECUTION_ENVIRONMENT:environment,RoleType.USER:TransactionalUserRole(user,identity=ei,trajectory_store=ts)}
                runner=EpisodeRunner(trajectory_store=ts,native_evaluator=NativeEvaluator(ts),physical_attempt_provider=lambda ids:tuple(a for t in agent.records for a in t.source_attempt_ids),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
                print(json.dumps({'phase':'episode_start','scenario_id':rec.scenario_id}),flush=True)
                result=runner.run(EpisodeRunInput(scenario=lease.scenario,manifest_record=rec,identity=ei,roles=roles,skill_versions={s.skill_id:s.version for s in seed},eligible_for_train_offline_consumption=(pass_index==0)))
                visible=extract_visible_messages(PipelineAgent)
                report={'scenario_id':rec.scenario_id,'result':result.model_dump(mode='json'),'visible_messages':[m.model_dump(mode='json') for m in visible if m.sender.value!='SYSTEM'],'decisions':[d.model_dump(mode='json') for d in agent.decisions]}
                if result.evaluator_record_reference:report['evaluation']=ts.load_evaluator(result.evaluator_record_reference).model_dump(mode='json')
                episode_reports.append(report);save(run/f'{gid}-episode-{pos}.json',report)
                print(json.dumps({'phase':'episode_end','scenario_id':rec.scenario_id,'status':result.status.value,'error':result.sanitized_failure_class,'agent_turns':len(agent.decisions),'similarity':report.get('evaluation',{}).get('similarity')}),flush=True)
                if result.sanitized_failure_class:raise RuntimeError('ReflectionEpisodeFailed')
                if pass_index==0:
                    trajectory=ts.load_trajectory(result.trusted_trajectory_reference)
                    evaluator=ts.load_evaluator(result.evaluator_record_reference)
                    from toolsandbox_pipeline.schemas.action import ActionEnvelope
                    proposed=tuple(ActionEnvelope.model_validate_json(json.dumps(ledger.load_completed_output(d.initial_policy_logical_request_id).output)) for d in agent.decisions)
                    projection=policy_projection(trajectory,evaluator,captured_envelopes,proposed,agent.decisions)
                    entries.append(MemoryTrajectoryReference(trajectory_id=trajectory.trajectory_id,trajectory_sha256=result.trusted_trajectory_reference.sha256,
                        run_id=run_id,round_index=0,shard_id='train-shard-0',generation_id=gid,dataset_manifest_sha256=train_hash,
                        config_manifest_sha256=mh,split='train',status='completed_evaluated',eligible_for_train_offline_consumption=True,
                        manifest_position=rec_position(rec,train),policy_projection=projection))
            phase_metrics.append({**phase_record(phase,phase_start,phase_before),'generation_id':gid})
            if pass_index==0:
                scope=AccountingScope(run_id=run_id,round_index=0,task_id='reflection-round-0',scenario_family_id='reflection',scenario_id='reflection',system_variant='generation_0')
                reflection_start=time.monotonic();reflection_before=metrics_snapshot()
                memory_prompts=load_memory_prompts(PROJECT,PROJECT/'prompts/offline/memory_manifest.json')
                memory_limits,memory_limits_hash=load_memory_limits(PROJECT/'configs/offline_memory_token_limits.provisional.json')
                buffer_hash=canonical_sha256({'protocol':'sealed-memory-buffer-v1','run_id':run_id,'round_index':0,'shard_id':'train-shard-0','generation_id':'g000','entries':[{'trajectory_id':e.trajectory_id,'trajectory_sha256':e.trajectory_sha256,'manifest_position':e.manifest_position} for e in entries]})
                update_identity=MemoryUpdateIdentity(run_id=run_id,round_index=0,shard_id='train-shard-0',current_generation_id='g000',next_generation_id='g001',dataset_manifest_sha256=train_hash,config_manifest_sha256=mh,prompt_manifest_sha256=memory_prompts.manifest_sha256,sealed_input_buffer_sha256=buffer_hash)
                buffer=SealedMemoryTrajectoryBuffer(identity=update_identity,entries=tuple(entries))
                save(run/'reflection-inputs.json',buffer.model_dump(mode='json'))
                class MemorySeam:
                    def load_completed_response(self,ctx):return None
                    def persist_completed_response(self,response):
                        ledger.complete_response(response,validated_output=response.value.model_dump(mode='json'),output_schema_name=type(response.value).__name__,output_schema_version=1)
                        attempt_reports.append(response.attempt.model_dump(mode='json'))
                memory_runner=MemoryRoleRunner(QwenGateway(qc),manifest_identity=mh,durability_seam=MemorySeam())
                class Executor:
                    def execute_and_apply(self,prepared):
                        prompt_tokens=count_tokens(prepared.messages)
                        save(run/('reflection-request-'+prepared.unit_reference+'-'+prepared.role+'.json'), {'prompt_version':prepared.prompt_version,'input_fingerprint':prepared.canonical_input_fingerprint,'prompt_tokens':prompt_tokens,'max_output_tokens':prepared.max_tokens,'context_limit':qc.context_limit})
                        if prompt_tokens+prepared.max_tokens>qc.context_limit:
                            raise RuntimeError('ReflectionContextLimitExceededBeforeDispatch')
                        rid=ledger.prepare_request(LogicalLLMRequestIdentity(run_id=run_id,role=ProviderRole(prepared.role),phase='offline_reflection',unit_reference=prepared.unit_reference,input_fingerprint=prepared.canonical_input_fingerprint,model=qc.model,decoding_configuration_sha256=canonical_sha256({'qwen':qc.model_dump(mode='json'),'max_tokens':prepared.max_tokens}),output_schema_sha256=prepared.output_schema_sha256))
                        ledger.bind_accounting_scope(rid.logical_request_id,scope)
                        ctx=ledger.allocate_attempt(rid.logical_request_id,manifest_identity=mh,replayed_after_unknown_outcome=False);ledger.mark_in_flight(ctx)
                        try:call=memory_runner.run(prepared,ctx)
                        except ProviderRequestError as error:
                            ledger.record_failure(error);attempt_reports.append(error.attempt.model_dump(mode='json'));raise
                        output=call.output.model_dump(mode='json');digest=canonical_sha256(output);cid='reflection-applied-'+ctx.attempt_id
                        app=ledger.commit_checkpoint_and_apply(checkpoint_id=cid,event_kind='reflection_response_applied',checkpoint_payload={'prepared':prepared.model_dump(mode='json'),'output':output},logical_request_id=ctx.logical_request_id,source_attempt_id=ctx.attempt_id,application_artifact_id=prepared.unit_reference+'-'+prepared.role,application_artifact_sha256=digest)
                        return AppliedMemoryDecision(output=call.output,logical_request_id=ctx.logical_request_id,source_attempt_id=ctx.attempt_id,application_id=app.application_id,checkpoint_id=cid)
                class Resolver:
                    def resolve_one(self,text):return cache.resolve((text,),gateway=eg,context_factory=emb_context,record_durable=emb_durable)[0]
                retriever=MemoryCandidateRetriever(generation_id='g000',policy_records=gen.policy_memory,world_records=gen.world_memory,indexes=gen.indexes,embedding_resolver=Resolver())
                update=MemoryUpdateOrchestrator(prompts=memory_prompts,limits=memory_limits,limits_sha256=memory_limits_hash,structured_output_wire_mode=qc.structured_output_wire_mode,role_executor=Executor(),retriever=retriever,durability=LedgerMemoryDurability(ledger),source_generation_sha256=canonical_sha256(gen.manifest.model_dump(mode='json')),current_policy_memory=gen.policy_memory,current_world_memory=gen.world_memory,input_representation='packed-v2').run(buffer)
                save(run/'reflection-result.json',update.model_dump(mode='json'))
                policy,world,skills=apply_generation_overlays(gen,policy_updates=update.staged_policy_memory)
                (run/'g001-staging').mkdir(mode=0o700)
                gen=build_generation(run/'g001-staging',generation_id='g001',parent_generation_id='g000',policy_memory=policy,world_memory=world,skills=skills,tool_inventory=inventory,tool_inventory_sha256=ih,cache=cache,gateway=eg,context_factory=emb_context,record_durable=emb_durable)
                retrieval=RetrievalService(gen,cache=cache,gateway=eg,context_factory=emb_context,record_durable=emb_durable)
                save(run/'g001-ready.json',{'generation_manifest':gen.manifest.model_dump(mode='json'),'policy_memory':[m.model_dump(mode='json') for m in gen.policy_memory],'world_memory_changed':False,'skills_changed':False})
                phase_metrics.append({**phase_record('offline_reflection_and_build',reflection_start,reflection_before),'policy_counts':update.policy_counts})
                phase_metrics.append({**phase_record('round_0',phase_start,phase_before),'input_generation':'g000','output_generation':'g001'})
                save(run/'phase-metrics.json',phase_metrics)
                print(json.dumps({'phase':'reflection_complete','policy_counts':update.policy_counts,'generation_id':'g001'}),flush=True)
                # Online retest receives only published generation records, not raw training projections.
                entries.clear();captured_envelopes.clear();buffer=None
        save(run/'phase-metrics.json',phase_metrics)
    # Native provider usage stays separate by role, with actual values only.
    attempt_reports.extend(a.model_dump(mode='json') for g in qgs.values() for a in g.recorder.attempts)
    save(run/'provider-attempts.json',attempt_reports)
    total_tokens=sum(a['metrics']['usage']['total_tokens'] or 0 for a in attempt_reports)
    usage_complete=all(a['metrics']['usage']['usage_complete'] for a in attempt_reports)
    cost,complete=ledger.effective_output_cost()
    save(run/'summary.json',{'development_only':True,'episodes':len(episode_reports),'total_tokens':total_tokens if usage_complete else None,'usage_complete':usage_complete,'statuses':[r['result']['status'] for r in episode_reports],'total_cost':cost,'cost_complete':complete,'cost_unit':'qwen_effective_output_tokens'})
    save(ROOT/'latest-reflection-run.json',{'run_id':run_id,'run_dir':str(run),'status':'finished'})
    store.close();cache.close();return 0

if __name__=='__main__':
    import sys
    if len(sys.argv)!=1:raise SystemExit('No arguments accepted')
    try:raise SystemExit(main())
    except Exception as error:
        # Never print exception text or local variables from credential-holding process.
        import traceback
        frames=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(error.__traceback__)]
        root=ROOT/'latest-reflection-run.json'
        info=json.loads(root.read_text()) if root.exists() else {}
        error_report={'status':'failed','error_class':type(error).__name__,'frames':frames}
        if info.get('run_dir'):
            save(Path(info['run_dir'])/'failure.json',error_report)
            save(root,{**info,'status':'failed'})
            # Preserve sanitized usage even when an episode/reflection stops early.
            import sqlite3
            dbpath=Path(info['run_dir'])/'checkpoint/checkpointing/ledger.sqlite3'
            if dbpath.exists():
                with sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True) as db:
                    attempts=[json.loads(row[0]) for row in db.execute('SELECT result FROM physical_llm_attempts WHERE result IS NOT NULL')]
                    unfinished=db.execute('SELECT COUNT(*) FROM physical_llm_attempts WHERE result IS NULL').fetchone()[0]
                save(Path(info['run_dir'])/'failed-attempts.json',{'attempts':attempts,'unfinished_attempts':unfinished})
        print(json.dumps(error_report),flush=True)
        raise SystemExit(2)
