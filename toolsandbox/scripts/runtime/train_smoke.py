"""Fixed, operator-launched development smoke; never accepts shell commands."""
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
PROJECT=Path('/root/toolsandbox')

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
    run_id='smoke-'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]
    run=ROOT/'runs'/run_id;run.mkdir(parents=True,mode=0o700)
    save(ROOT/'latest-run.json',{'run_id':run_id,'run_dir':str(run),'status':'running'})
    qc=QwenConfig.model_validate_json((ROOT/'qwen-config.json').read_bytes())
    ec=EmbeddingConfig();uc=UserSimulatorConfig()
    preflight=[]
    for mode,cfg in [('qwen',qc),('embedding',ec),('user-simulator',uc)]:
        result=run_preflight(mode,cfg);preflight.append(result)
        save(run/'preflight.json',preflight)
        print(json.dumps({'phase':'preflight',**result}),flush=True)
        if result['status']!='pass':
            save(ROOT/'latest-run.json',{'run_id':run_id,'run_dir':str(run),'status':'preflight_failed'});return 2
    ec=ec.model_copy(update={'expected_dimension':preflight[1]['vector_dimension']})
    train_path=ROOT/'dataset/train_manifest.json'
    train=SplitManifest.model_validate_json(train_path.read_bytes());train_hash=file_hash(train_path.read_bytes())
    # Fixed before observing any results: first train family, first two variants.
    selection=json.loads((ROOT/'smoke-selection.json').read_text())
    if selection['dataset_manifest_sha256']!=train_hash:raise ValueError('SelectionManifestMismatch')
    records=tuple(train.scenarios[i] for i in selection['manifest_positions'])
    if [r.scenario_id for r in records]!=selection['scenario_ids']:raise ValueError('SelectionIdentityMismatch')
    # Resume this length-smoke campaign without repeating its completed first case.
    progress=ROOT/'length-smoke-progress.json'
    previous=json.loads(progress.read_text()) if progress.exists() else {'completed':[]}
    completed_ids=set()
    for done in previous['completed']:
        recovery=json.loads(Path(done['recovery_report']).read_text())
        if recovery['status']!='completed_evaluated' or recovery['scenario_id']!=done['scenario_id']:
            raise ValueError('PreviousCompletionMismatch')
        completed_ids.add(done['scenario_id'])
    records=tuple(r for r in records if r.scenario_id not in completed_ids)
    if not records:raise ValueError('SmokeSelectionAlreadyCompleted')
    build_cfg,_=load_build_config(PROJECT/'configs/reproducibility/dataset_build_v1.json')
    prompts=dict(zip(('policy','critic','revision'),load_prompts(PROJECT)))
    ph=file_hash((PROJECT/'prompts/manifest.json').read_bytes())
    limitpath=PROJECT/'configs/online_token_limits.smoke.json';lh=file_hash(limitpath.read_bytes())
    limits=load_token_limits(limitpath,expected_sha256=lh)
    metadata=load_tool_metadata();by_name={m.canonical_tool_name:m for m in metadata}
    inventory=public_tool_inventory();ih=canonical_sha256(list(inventory))
    fixture_config={'development_only':True,'external_read_mode':'deny_until_fixture_available'}
    fh=canonical_sha256(fixture_config)
    manifest={'development_only':True,'formal':False,'pythonhashseed':0,'profile':'official_live','external_read_mode':'fixture_only_no_live_tools',
      'image':json.loads((ROOT/'image-provenance.json').read_text()),'model_revision':'9216db5781bf21249d130ec9da846c4624c16137',
      'qwen':qc.model_dump(mode='json'),'dataset_manifest_sha256':train_hash,'scenario_ids':[r.scenario_id for r in records],'selection':selection,'completed_before_this_run':previous['completed'],
      'prompt_manifest_sha256':ph,'token_limit_config_sha256':lh,'runner_sha256':file_hash(Path(__file__).read_bytes()),
      'notes':['Development remote User with fixed world clock; not trajectory-level strict replay.','No offline evolution or final test in this smoke.']}
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
    runners={role:cls(qgs[role],manifest_identity=mh,mode='calibration',durability_seam=seam) for role,cls in [('policy',InitialPolicyRunner),('critic',CriticRunner),('revision',RevisionRunner)]}
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
        for pos,lease in enumerate(leases):
            rec=lease.record;eid=f'{run_id}-episode-{pos}'
            scope=AccountingScope(run_id=run_id,task_id=eid,scenario_family_id=rec.scenario_family_id,scenario_id=rec.scenario_id,system_variant='generation_0')
            ei=EpisodeIdentity(run_id=run_id,profile='official_live',phase='development_smoke',family_id=rec.scenario_family_id,scenario_id=rec.scenario_id,episode_id=eid,manifest_position=next(i for i,r in enumerate(train.scenarios) if r.scenario_id==rec.scenario_id),system_variant='generation_0',generation_id='g000',starting_context_sha256=rec.starting_context_sha256,evaluation_definition_sha256=rec.evaluation_definition_sha256,agent_tool_schema_sha256=rec.agent_facing_tool_schema_sha256,dataset_manifest_sha256=train_hash,runtime_config_sha256=mh,prompt_manifest_sha256=ph,token_limit_config_sha256=lh,fixture_manifest_sha256=fh,environment_sha256=environment_hash,max_messages=rec.max_messages)
            ts=TrajectoryStore(store,environment_identity=environment_hash)
            toolsledger=ToolLedger(store);responders={};outcomes=[];contracts={};bindings=[]
            def factory(turn_index):
                visible=extract_visible_messages(PipelineAgent)
                adapter=build_adapter_turn(PipelineAgent,visible)
                mapping=dict(adapter.controller_context.agent_to_execution_name);reverse={v:k for k,v in mapping.items()}
                for name,fn in adapter.controller_context.tool_objects.items():
                    ret=typing.get_type_hints(fn).get('return')
                    if ret is not None:contracts[mapping[name]]=TypeAdapter(ret)
                current_outcomes=[]
                for bound,calls,bmap in bindings:
                    for call in calls:
                        for message in visible:
                            if message.sender.value=='EXECUTION_ENVIRONMENT' and message.openai_tool_call_id==call.call_id:
                                current_outcomes.append(CommittedToolOutcomeInput(call_id=call.call_id,agent_facing_tool_name=call.name,arguments=call.arguments,result_source_message_index=message.source_message_index,public_return_contract_id=bmap[call.name],canonical_tool_name=bmap[call.name],tool_mapping_manifest_hash=canonical_sha256(bmap)))
                online_id=OnlineTurnIdentity(run_id=run_id,profile=ReproducibilityProfile.OFFICIAL_LIVE,phase='development_smoke',family_id=rec.scenario_family_id,scenario_id=rec.scenario_id,episode_id=eid,agent_turn_index=turn_index,expected_generation_id='g000',dataset_manifest_sha256=train_hash,runtime_config_sha256=mh,prompt_manifest_sha256=ph,token_limit_config_sha256=lh,fixture_manifest_sha256=fh,environment_identity=environment_hash)
                def prepared(ctx,role):return prepare_request(ctx,prompt=prompts[role],token_limits=limits,token_limit_config_sha256=lh,qwen_config=qc,canonical_to_agent=reverse,mode='calibration',count_prompt_tokens=count_tokens)
                def initial(state,bundle):
                    ctx=initial_context(state,bundle,prompts['policy']);return PreparedInitialPolicy(ctx,prepared(ctx,'policy'))
                def critic(initial,action,decision,bundle):
                    ctx=critic_context(initial,action,decision,bundle);return PreparedCritic(ctx,prepared(ctx,'critic'))
                def revision(initial,critic,output):
                    ctx=RevisionContext(initial=initial,critic=critic,critic_feedback_json=output.model_dump_json())
                    return prepared(ctx,'revision')
                backend=Task011DurableRoleBackend(ledger=ledger,runners=runners,run_id=run_id,phase='development_smoke',qwen_model=qc.model,decoding_configuration_sha256=canonical_sha256(qc.model_dump(mode='json')),manifest_identity=mh,accounting_scope=scope)
                responder=DurableTurnResponder(TurnContext(identity=online_id,generation=gen,retrieval=retrieval,state_builder=StateBuilder(contracts),controller=Controller(),controller_tool_metadata=metadata,role_request_builders=TurnRoleRequestBuilders(initial,critic,revision),durable_roles=DurableRoleExecutor(backend),checkpoint_sink=Task011CheckpointEventSink(ledger),committed_tool_outcomes=tuple(current_outcomes)))
                responders[turn_index]=(responder,mapping);return responder
            def load_audit(decision):
                responder,_=responders[decision.identity.agent_turn_index]
                event=ledger.get_checkpoint(responder._final_checkpoint_id(responder._input_fingerprint))
                return OnlineTurnAuditRecord.model_validate_json(json.dumps(event.payload['audit_record']))
            agent=TransactionalAgentRole(identity=ei,trajectory_store=ts,responder_factory=factory,audit_loader=load_audit)
            def binding(messages):
                ordinal=len(environment.records)+1
                if messages[0].sender==RoleType.USER:
                    return ToolActionBinding(action_sha256=canonical_sha256([m.content for m in messages]),call_ids=tuple(m.openai_tool_call_id or ('user-control-'+canonical_sha256([eid,ordinal,index,m.content])[7:]) for index,m in enumerate(messages)),selected_skill_ids=(None,)*len(messages),canonical_tool_ids=('end_conversation',)*len(messages),effect_classes=('conversation_control',)*len(messages),action_ordinal=ordinal)
                decision=agent.decisions[-1];action=decision.final_action.action
                calls=tuple(action.calls) if hasattr(action,'calls') else (action,)
                _,mapping=responders[decision.identity.agent_turn_index]
                effects=tuple(by_name[mapping[c.name]].effect.value for c in calls)
                if 'external_read' in effects:raise RuntimeError('SmokeRequiresUnavailableExternalFixture')
                bound=ToolActionBinding(action_sha256=canonical_sha256(decision.final_action.model_dump(mode='json')),call_ids=tuple(c.call_id for c in calls),selected_skill_ids=tuple(c.selected_skill_id for c in calls),canonical_tool_ids=tuple(mapping[c.name] for c in calls),effect_classes=effects,action_ordinal=ordinal)
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
            result=runner.run(EpisodeRunInput(scenario=lease.scenario,manifest_record=rec,identity=ei,roles=roles,skill_versions={s.skill_id:s.version for s in seed},eligible_for_train_offline_consumption=False))
            visible=extract_visible_messages(PipelineAgent)
            report={'scenario_id':rec.scenario_id,'result':result.model_dump(mode='json'),'visible_messages':[m.model_dump(mode='json') for m in visible if m.sender.value!='SYSTEM'],'decisions':[d.model_dump(mode='json') for d in agent.decisions]}
            if result.evaluator_record_reference:report['evaluation']=ts.load_evaluator(result.evaluator_record_reference).model_dump(mode='json')
            episode_reports.append(report);save(run/f'episode-{pos}.json',report)
            print(json.dumps({'phase':'episode_end','scenario_id':rec.scenario_id,'status':result.status.value,'error':result.sanitized_failure_class,'agent_turns':len(agent.decisions),'similarity':report.get('evaluation',{}).get('similarity')}),flush=True)
            if result.sanitized_failure_class:break
    # Native provider usage stays separate by role, with actual values only.
    attempt_reports.extend(a.model_dump(mode='json') for g in qgs.values() for a in g.recorder.attempts)
    save(run/'provider-attempts.json',attempt_reports)
    total_tokens=sum(a['metrics']['usage']['total_tokens'] or 0 for a in attempt_reports)
    usage_complete=all(a['metrics']['usage']['usage_complete'] for a in attempt_reports)
    cost,complete=ledger.effective_output_cost()
    save(run/'summary.json',{'development_only':True,'episodes':len(episode_reports),'total_tokens':total_tokens,'usage_complete':usage_complete,'statuses':[r['result']['status'] for r in episode_reports],'total_cost':cost,'cost_complete':complete,'cost_unit':'qwen_effective_output_tokens'})
    save(ROOT/'latest-run.json',{'run_id':run_id,'run_dir':str(run),'status':'finished'})
    store.close();cache.close();return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        # Never print exception text or local variables from credential-holding process.
        import traceback
        frames=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(error.__traceback__)]
        root=ROOT/'latest-run.json'
        info=json.loads(root.read_text()) if root.exists() else {}
        error_report={'status':'failed','error_class':type(error).__name__,'frames':frames}
        if info.get('run_dir'):save(Path(info['run_dir'])/'failure.json',error_report)
        print(json.dumps(error_report),flush=True)
        raise SystemExit(2)
