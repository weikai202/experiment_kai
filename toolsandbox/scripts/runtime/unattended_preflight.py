import json,os
from pathlib import Path
from toolsandbox_pipeline.providers.preflight import run_preflight
from toolsandbox_pipeline.schemas.runtime import QwenConfig,EmbeddingConfig,UserSimulatorConfig
os.environ['QWEN_BASE_URL']='http://127.0.0.1:18080/v1'
os.environ['QWEN_API_KEY']='EMPTY'
r=Path('/root/toolsandbox-runtime')
results=[]
for mode,config in [('qwen',QwenConfig.model_validate_json((r/'qwen-config.json').read_bytes())),('embedding',EmbeddingConfig()),('user-simulator',UserSimulatorConfig())]:
    result=run_preflight(mode,config)
    results.append(result)
    (r/'unattended-preflight.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'mode':mode,'status':result['status'],'error':result.get('error')}),flush=True)
raise SystemExit(0 if all(x['status']=='pass' for x in results) else 1)
