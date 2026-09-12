import json
from pathlib import Path
import pytest
import tool_sandbox.tools.rapid_api_search_tools as upstream
from tool_sandbox.common.execution_context import ExecutionContext,new_context

from toolsandbox_pipeline.reproducibility.fixture_store import (
    FRANKFURTER_BACKEND_CONFIG_SHA256, PINNED_BACKEND_CONFIG_SHA256,
    load_backend_manifest,build_fixture_request,FixtureStore,canonical_sha256,
)
from toolsandbox_pipeline.reproducibility.rapidapi_boundary import (
    RapidAPIBoundary,RapidAPIBoundaryError,ExternalReadFailed,ExternalReadUnknownOutcome,
)
from toolsandbox_pipeline.reproducibility.fixture_capture import capture_fixtures
from toolsandbox_pipeline.schemas.fixtures import FixtureCaptureRequestManifest
from tests.reproducibility.test_rapidapi_boundary import context,FakeResponse,FakeTransport
from tests.reproducibility.test_fixture_capture import capture_manifest

ROOT=Path(__file__).parents[2]
NEW=ROOT/'configs/reproducibility/rapidapi_backends_frankfurter_v2.json'
RATE={'date':'2026-09-13','base':'USD','quote':'EUR','rate':0.86}


def backend():
    return load_backend_manifest(NEW,expected_sha256=FRANKFURTER_BACKEND_CONFIG_SHA256)


def run_currency(result,amount=2):
    manifest,digest=backend(); attempts=[];transport=FakeTransport(result)
    boundary=RapidAPIBoundary(mode='official_live',profile='official_live',backend_manifest=manifest,
        backend_manifest_sha256=digest,context_provider=lambda:context(digest,profile='official_live'),
        attempt_sink=attempts.append,transport=transport,
        credential_provider=lambda:pytest.fail('currency must not retrieve any API key'))
    return boundary,transport,attempts


def test_native_currency_free_backend_preserves_float_and_records_rate_date():
    boundary,transport,attempts=run_currency(FakeResponse(200,json.dumps(RATE).encode()))
    original=upstream.convert_currency
    with boundary,new_context(ExecutionContext()):
        assert upstream.convert_currency(2,'usd','eur') == 1.72
    assert upstream.convert_currency is original
    url,request=transport.calls[0]
    assert url=='https://api.frankfurter.dev/v2/rate/USD/EUR'
    assert request==dict(params={},headers={},timeout=30.0,allow_redirects=False)
    assert attempts[0].backend_version=='frankfurter-v2-rate-v1'
    assert attempts[0].status=='completed'


@pytest.mark.parametrize('bad',[{}, {**RATE,'base':'GBP'}, {**RATE,'quote':'GBP'},
    {**RATE,'date':'bad'}, {**RATE,'rate':0}, {**RATE,'rate':True}, {**RATE,'rate':-1},
    {**RATE,'rate':'0.86'}, {**RATE,'rate':float('nan')}])
def test_invalid_rate_one_failed_attempt_no_fallback(bad):
    boundary,transport,attempts=run_currency(FakeResponse(200,json.dumps(bad).encode()))
    with boundary,new_context(ExecutionContext()),pytest.raises(ExternalReadFailed):
        upstream.convert_currency(2,'usd','eur')
    assert len(transport.calls)==len(attempts)==1
    assert attempts[0].status=='failed'


@pytest.mark.parametrize('result,error,status',[(FakeResponse(404,b'{"message":"not found"}'),ExternalReadFailed,'failed'),
                                              (TimeoutError('private'),ExternalReadUnknownOutcome,'unknown_outcome')])
def test_provider_failures_are_not_retried(result,error,status):
    boundary,transport,attempts=run_currency(result)
    with boundary,new_context(ExecutionContext()),pytest.raises(error):
        upstream.convert_currency(2,'usd','eur')
    assert len(transport.calls)==1 and attempts[0].status==status


def test_old_currency_provider_is_replay_only_and_fixture_identity_differs():
    old,oldhash=load_backend_manifest(ROOT/'configs/reproducibility/rapidapi_backends_v1.json',expected_sha256=PINNED_BACKEND_CONFIG_SHA256)
    new,newhash=backend()
    args={'from':'USD','to':'EUR','amount':2};a=old.backends[0]
    requests=[build_fixture_request(m,url=a.url,params=args,headers={'X-RapidAPI-Host':a.host}) for m in (old,new)]
    assert requests[0].fixture_key!=requests[1].fixture_key
    assert requests[0].request_url_identity!=requests[1].request_url_identity
    transport=FakeTransport(None);attempts=[]
    with RapidAPIBoundary(mode='official_live',profile='official_live',backend_manifest=old,
        backend_manifest_sha256=oldhash,context_provider=lambda:context(oldhash,profile='official_live'),
        attempt_sink=attempts.append,transport=transport,
        credential_provider=lambda:pytest.fail('retired backend must fail before key')) as boundary:
        with pytest.raises(RapidAPIBoundaryError):boundary.dispatch(a.url,args,{'X-RapidAPI-Host':a.host})
    assert not transport.calls and attempts[0].status=='rejected_before_dispatch'


def test_currency_capture_no_key_provenance_and_replay(tmp_path):
    manifest,digest=backend();a=manifest.backends[0]
    request=build_fixture_request(manifest,url=a.url,params={'from':'USD','to':'EUR','amount':2},headers={'X-RapidAPI-Host':a.host})
    _,_,oldcapture=capture_manifest()
    payload=oldcapture.model_dump(mode='json');payload.update(backend_manifest_sha256=digest,
        requests=[request.model_dump(mode='json')],ordered_fixture_keys=[request.fixture_key])
    capture=FixtureCaptureRequestManifest.model_validate_json(json.dumps(payload))
    transport=FakeTransport(FakeResponse(200,json.dumps(RATE).encode()))
    report=capture_fixtures(capture,output_root=tmp_path/'capture',backend_manifest=manifest,
        backend_manifest_sha256=digest,transport=transport,credential_provider=lambda:pytest.fail('no key'),
        utc_now=lambda:'2026-09-12T00:00:00Z')
    path=next((tmp_path/'capture'/'fixtures').glob('*/manifest.json'))
    store=FixtureStore.open(path,expected_manifest_sha256=report.fixture_manifest_sha256,
        backend_manifest=manifest,backend_manifest_sha256=digest)
    entry=store.lookup_entry(request.fixture_key)
    assert entry.source=='frankfurter_capture'
    assert entry.normalized_response_body['provider_response']==RATE
    assert entry.response_body_sha256==canonical_sha256(entry.normalized_response_body)
    attempts=[]
    with RapidAPIBoundary(mode='replay',profile='strict_replay',backend_manifest=manifest,
        backend_manifest_sha256=digest,fixture_store=store,fixture_miss_sink=lambda _:pytest.fail('fixture miss'),
        context_provider=lambda:context(digest,store.manifest_sha256),attempt_sink=attempts.append),new_context(ExecutionContext()):
        assert upstream.convert_currency(2,'USD','EUR')==1.72
    assert attempts[0].status=='fixture_hit' and len(transport.calls)==1


def test_cli_can_preflight_new_manifest_without_network(tmp_path,capsys):
    from toolsandbox_pipeline.reproducibility.fixture_store import publish_fixture_bundle
    from toolsandbox_pipeline.reproducibility.fixture_cli import main
    from toolsandbox_pipeline.schemas.fixtures import FixtureEntry
    from toolsandbox_pipeline.toolsandbox_adapter.currency_backend import adapt_response
    manifest,digest=backend();a=manifest.backends[0]
    request=build_fixture_request(manifest,url=a.url,params={'from':'USD','to':'EUR','amount':2},headers={'X-RapidAPI-Host':a.host})
    body=adapt_response(RATE,request.effective_arguments)
    entry=FixtureEntry(**request.model_dump(mode='python'),status_code=200,normalized_response_body=body,
        response_body_sha256=canonical_sha256(body),captured_at_utc='2026-09-12T00:00:00Z',source='frankfurter_capture')
    bundle,_=publish_fixture_bundle(tmp_path/'bundle',[entry],backend_manifest=manifest,
        backend_manifest_sha256=digest,created_at_utc='2026-09-12T00:00:00Z')
    assert main(['preflight-fixtures','--backend-config',str(NEW),
        '--expected-backend-config-sha256',digest,'--fixture-manifest',str(bundle/'manifest.json')])==0
    output=json.loads(capsys.readouterr().out)
    assert output['backend_manifest_sha256']==digest
