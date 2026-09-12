"""Fixed, explicitly authorized five-tool preflight; sanitized output only."""
import json
import os
from pathlib import Path
import time
from tool_sandbox.common.execution_context import ExecutionContext, new_context
import tool_sandbox.tools.rapid_api_search_tools as native
from toolsandbox_pipeline.reproducibility.fixture_store import load_backend_manifest, FRANKFURTER_BACKEND_CONFIG_SHA256
from toolsandbox_pipeline.reproducibility.fixture_cli import RequestsTransport
from toolsandbox_pipeline.reproducibility.rapidapi_boundary import RapidAPIBoundary
from toolsandbox_pipeline.schemas.fixtures import ExternalReadContext


def main():
    start=time.monotonic()
    manifest,digest=load_backend_manifest(Path('/root/toolsandbox/configs/reproducibility/rapidapi_backends_frankfurter_v2.json'),expected_sha256=FRANKFURTER_BACKEND_CONFIG_SHA256)
    attempts=[];rows=[]
    def context():
        return ExternalReadContext(schema_version=1,run_id='external-preflight',profile='official_live',phase='setup',scenario_family_id='not_applicable',scenario_id='not_applicable',state_id='sha256:'+'0'*64,logical_tool_call_id=f'preflight_{len(rows)}',backend_manifest_sha256=digest,fixture_manifest_sha256=None)
    calls=(('convert_currency',lambda:native.convert_currency(1,'USD','EUR')),
           ('search_lat_lon',lambda:native.search_lat_lon(37.7749,-122.4194)),
           ('search_location_around_lat_lon',lambda:native.search_location_around_lat_lon('coffee',37.7749,-122.4194)),
           ('search_weather_around_lat_lon',lambda:native.search_weather_around_lat_lon(0,37.7749,-122.4194)),
           ('search_stock',lambda:native.search_stock('AAPL')))
    with RapidAPIBoundary(mode='official_live',profile='official_live',backend_manifest=manifest,backend_manifest_sha256=digest,context_provider=context,attempt_sink=attempts.append,transport=RequestsTransport(),credential_provider=lambda:os.environ['RAPID_API_KEY']):
        with new_context(ExecutionContext()):
            for name,call in calls:
                before=len(attempts)
                try:
                    result=call()
                    row={'tool':name,'status':'pass' if result is not None and result != [] and result != {} else 'empty_result','result_type':type(result).__name__}
                except Exception as exc:
                    row={'tool':name,'status':'fail','error_class':type(exc).__name__}
                row['attempts']=[{'status':a.status,'http_status':a.status_code,'dispatched':a.dispatched,'exception_class':a.sanitized_exception_class} for a in attempts[before:]]
                rows.append(row)
                print(json.dumps(row),flush=True)
    report={'status':'pass' if all(r['status']=='pass' for r in rows) else 'fail','results':rows,'backend_manifest_sha256':digest,'total_running_time_seconds':time.monotonic()-start,'total_tokens':0,'dataset_scenarios_run':0}
    Path('/root/toolsandbox-runtime/external-preflight-result.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='pass' else 2

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status':'fail','error_class':type(exc).__name__}),flush=True)
        raise SystemExit(2)
