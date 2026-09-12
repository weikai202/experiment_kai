"""Fixed runner: booleans and allowlisted HTTP diagnostics, never credential text."""
import hmac
import json
import os
import httpx
from secret_launcher import load_saved_credentials

ALLOWED_CODES={'invalid_api_key','invalid_organization','organization_deactivated','project_not_found','account_deactivated','insufficient_permissions','ip_not_authorized','missing_api_key'}

def sanitized(response):
    out={'http_status':response.status_code}
    try:
        code=response.json().get('error',{}).get('code')
    except Exception:
        code=None
    if response.status_code != 200:
        out['error_code']=code if code in ALLOWED_CODES else 'other_api_error'
    return out

def main():
    saved=load_saved_credentials()
    key=os.environ.get('OPENAI_API_KEY','')
    saved_key=saved.get('OPENAI_API_KEY','')
    report={
        'saved_openai_present':bool(saved_key),
        'runtime_key_matches_saved':hmac.compare_digest(key.encode(),saved_key.encode()),
        'openai_equals_rapidapi':bool(saved.get('RAPID_API_KEY')) and hmac.compare_digest(key.encode(),saved['RAPID_API_KEY'].encode()),
        'has_whitespace':any(c.isspace() for c in key),
        'has_outer_quotes':len(key)>1 and key[0]==key[-1] and key[0] in ('"',"'"),
        'has_non_ascii':not key.isascii(),
        'proxy_environment_present':any(os.environ.get(n) for n in ('HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','https_proxy','http_proxy','all_proxy')),
        'inherited_openai_present':os.environ.get('CREDENTIAL_AUDIT_INHERITED_PRESENT')=='1',
        'inherited_openai_matches_saved':os.environ.get('CREDENTIAL_AUDIT_INHERITED_MATCH')=='1',
    }
    request=httpx.Request('GET','https://api.openai.com/v1/models',headers={'Authorization':'Bearer '+key})
    report['authorization_header_matches_saved']=hmac.compare_digest(request.headers['Authorization'].encode(),('Bearer '+saved_key).encode())
    report['probes']={}
    for name,trust_env in [('direct_without_proxy',False),('standard_environment',True)]:
        try:
            with httpx.Client(trust_env=trust_env,timeout=20,follow_redirects=False) as client:
                response=client.send(request)
                report['probes'][name]=sanitized(response)
        except Exception as exc:
            report['probes'][name]={'error_class':type(exc).__name__}
    print(json.dumps(report,sort_keys=True),flush=True)
    from pathlib import Path
    Path('/root/toolsandbox-runtime/credential-chain-audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'audit_status':'failed','error_class':type(exc).__name__}),flush=True)
        raise SystemExit(2)
