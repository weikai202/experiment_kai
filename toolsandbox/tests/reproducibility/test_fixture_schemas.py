from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.reproducibility.fixture_cli import PINNED_BACKEND_CONFIG_SHA256
from toolsandbox_pipeline.reproducibility.fixture_store import (
    FixtureValidationError,
    fixture_key,
    load_backend_manifest,
    parse_strict_json,
    request_url_identity,
)
from toolsandbox_pipeline.schemas.fixtures import (
    ExternalReadContext,
    FixtureCaptureRequestManifest,
    FixtureRequest,
    TOOL_NAMES,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reproducibility/rapidapi_backends_v1.json"
DIGEST = "sha256:" + "0" * 64


def backend_manifest():
    return load_backend_manifest(CONFIG, expected_sha256=PINNED_BACKEND_CONFIG_SHA256)


def test_exact_backend_manifest_and_hash():
    manifest, digest = backend_manifest()
    assert digest == PINNED_BACKEND_CONFIG_SHA256
    assert tuple(item.canonical_tool_name for item in manifest.backends) == TOOL_NAMES
    assert manifest.request_timeout_seconds == 30.0
    assert manifest.allow_redirects is False
    assert all(item.method == "GET" and item.url.startswith("https://") for item in manifest.backends)
    with pytest.raises(FixtureValidationError):
        load_backend_manifest(CONFIG, expected_sha256=DIGEST)


def test_fixture_key_golden_and_exact_json_semantics():
    payload = {
        "tool_name": "search_stock",
        "arguments": {"query": "Café \u6771\u4eac", "filters": [1, 1.0, True]},
        "backend_version": "v1",
    }
    expected = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert fixture_key("search_stock", payload["arguments"], "v1") == expected
    assert fixture_key("search_stock", {"n": 1}, "v1") != fixture_key("search_stock", {"n": 1.0}, "v1")
    assert fixture_key("search_stock", {"n": 1}, "v1") != fixture_key("search_stock", {"n": 1}, "v2")
    assert fixture_key("search_stock", {"n": 1}, "v1") != fixture_key("convert_currency", {"n": 1}, "v1")


@pytest.mark.parametrize(
    "arguments",
    [
        {"api_key": "secret"},
        {"nested": {"Authorization": "secret"}},
        {"n": float("nan")},
        {"n": float("inf")},
        {"n": (1, 2)},
        {"n": {1, 2}},
        {1: "not a string key"},
    ],
)
def test_fixture_key_rejects_secrets_nonfinite_and_unsupported(arguments):
    with pytest.raises(FixtureValidationError):
        fixture_key("search_stock", arguments, "v1")


def test_duplicate_and_nonfinite_json_rejected():
    with pytest.raises(FixtureValidationError):
        parse_strict_json(b'{"a":1,"a":2}')
    with pytest.raises(FixtureValidationError):
        parse_strict_json(b'{"a":NaN}')
    with pytest.raises(FixtureValidationError):
        parse_strict_json(b"\xff")


def test_context_and_capture_manifest_are_frozen_strict():
    manifest, backend_hash = backend_manifest()
    context = ExternalReadContext(
        schema_version=1,
        run_id="run",
        profile="strict_replay",
        phase="online",
        scenario_family_id="family",
        scenario_id="scenario",
        state_id=DIGEST,
        logical_tool_call_id="call",
        backend_manifest_sha256=backend_hash,
        fixture_manifest_sha256=DIGEST,
    )
    with pytest.raises(ValidationError):
        context.run_id = "changed"
    with pytest.raises(ValidationError):
        ExternalReadContext.model_validate({**context.model_dump(), "schema_version": "1"})
    with pytest.raises(ValidationError):
        ExternalReadContext.model_validate(
            {**context.model_dump(), "profile": "official_live", "fixture_manifest_sha256": DIGEST}
        )
    backend = manifest.backends[3]
    request = FixtureRequest(
        schema_version=1,
        fixture_key=fixture_key("search_stock", {"query": "AAPL"}, backend.backend_version),
        canonical_tool_name="search_stock",
        effective_arguments={"query": "AAPL"},
        backend_version=backend.backend_version,
        request_url_identity=request_url_identity(backend),
    )
    capture = FixtureCaptureRequestManifest(
        schema_version=1,
        adapter_version="toolsandbox-rapidapi-boundary-v1",
        upstream_commit="165848b9a78cead7ca7fe7c89c688b58e6501219",
        backend_manifest_sha256=backend_hash,
        purpose="synthetic_contract_probe",
        train_manifest_sha256=None,
        fixture_miss_audit_sha256=None,
        requests=(request,),
        ordered_fixture_keys=(request.fixture_key,),
        maximum_request_count=1,
        requests_per_second=1.0,
        concurrency_limit=1,
        created_at_utc="2026-09-09T00:00:00Z",
    )
    assert capture.requests == (request,)
    with pytest.raises(ValidationError):
        FixtureCaptureRequestManifest.model_validate(
            {**capture.model_dump(), "purpose": "train_fixture_preparation"}
        )
