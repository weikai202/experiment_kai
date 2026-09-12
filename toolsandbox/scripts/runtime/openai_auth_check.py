"""One fixed authentication probe; prints whitelisted status only, never error text."""
import json
import os
from openai import OpenAI, APIStatusError
from tool_sandbox.common.utils import all_logging_disabled

report={'endpoint':'api.openai.com','operation':'models.list','total_tokens':0}
try:
    with all_logging_disabled(), OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url='https://api.openai.com/v1',organization='',max_retries=0,timeout=20) as client:
        client.models.list()
    report.update(status='pass',http_status=200)
except APIStatusError as error:
    code=error.code
    allowed={'invalid_api_key','invalid_organization','organization_deactivated','project_not_found','account_deactivated','insufficient_permissions','ip_not_authorized','missing_api_key'}
    report.update(status='fail',http_status=error.status_code,error_code=code if code in allowed else 'other_api_error')
except Exception as error:
    report.update(status='fail',error_class=type(error).__name__)
print(json.dumps(report),flush=True)
raise SystemExit(0 if report['status']=='pass' else 2)
