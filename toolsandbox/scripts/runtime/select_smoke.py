import os,locale,time,json,random
from pathlib import Path
os.environ['TZ']='UTC';time.tzset();locale.setlocale(locale.LC_ALL,'C.UTF-8')
from toolsandbox_pipeline.reproducibility.dataset_manifest import load_build_config
from toolsandbox_pipeline.reproducibility.clock import FixedWorldClock
from toolsandbox_pipeline.schemas.dataset import SplitManifest,REGISTRIES
from toolsandbox_pipeline.reproducibility.splits import build_family_registry,build_augmented_registry
from toolsandbox_pipeline.reproducibility.dataset_access import DatasetAccessGate
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.online.tool_metadata import load_tool_metadata
from toolsandbox_pipeline.toolsandbox_adapter.tools import build_adapter_turn
from toolsandbox_pipeline.toolsandbox_adapter.pipeline_agent import PipelineAgent
from tool_sandbox.common.execution_context import new_context
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios import named_scenarios,named_single_tool_call_scenarios,named_multiple_tool_call_scenarios,named_multiple_user_turn_scenarios,named_insufficient_information_scenarios
root=Path('/root/toolsandbox-runtime');p=root/'dataset/train_manifest.json'
m=SplitManifest.model_validate_json(p.read_bytes());digest=file_hash(p.read_bytes())
cfg,_=load_build_config(Path('/root/toolsandbox/configs/reproducibility/dataset_build_v1.json'))
metadata={x.canonical_tool_name:x for x in load_tool_metadata()};audit=[]
with FixedWorldClock(cfg):
 families=build_family_registry(dict(zip(REGISTRIES,(named_single_tool_call_scenarios,named_multiple_tool_call_scenarios,named_multiple_user_turn_scenarios,named_insufficient_information_scenarios))),ToolBackend.DEFAULT)
 registry=build_augmented_registry(families,named_scenarios,ToolBackend.DEFAULT)
 gate=DatasetAccessGate(manifest_path=p,expected_manifest_sha256=digest,reconstruct=lambda r:registry[r.scenario_id],audit=audit.append)
 for i in range(0,len(m.scenarios),8):
  try:
   lease=gate.load(requested_ids=[m.scenarios[i].scenario_id],run_id='smoke-selection',phase='development_smoke',manifest_sha256=digest)[0]
  except Exception as e:
   print('Access rejection:',type(e.__context__).__name__,str(e.__context__))
   from toolsandbox_pipeline.reproducibility.scenario_hashes import scenario_hashes
   actual=scenario_hashes(registry[m.scenarios[i].scenario_id])
   print('Hash equality:',{k:v==getattr(m.scenarios[i],k) for k,v in actual.items()})
   import copy
   print('Deepcopy stable:',actual==scenario_hashes(copy.deepcopy(registry[m.scenarios[i].scenario_id])))
   raise
  with new_context(lease.scenario.starting_context):
   turn=build_adapter_turn(PipelineAgent,())
   names=tuple(turn.controller_context.agent_to_execution_name.values())
  if all(metadata[n].effect.value!='external_read' for n in names):
   selection={'development_only':True,'selection_rule':'first_train_family_with_local_only_no_distraction_tools','dataset_manifest_sha256':digest,'scenario_ids':[r.scenario_id for r in m.scenarios[i:i+2]],'manifest_positions':[i,i+1]}
   (root/'smoke-selection.json').write_text(json.dumps(selection,indent=2))
   print(json.dumps(selection));break
 else:raise RuntimeError('NoLocalSmokeFamily')
(root/'selection-access-audit.json').write_text(json.dumps(audit))
