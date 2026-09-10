from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.reproducibility.fixture_cli import (
    PINNED_BACKEND_CONFIG_SHA256,
    main,
)
from toolsandbox_pipeline.reproducibility.fixture_store import (
    build_fixture_request,
    load_backend_manifest,
    publish_fixture_bundle,
    sha256_bytes,
)
from toolsandbox_pipeline.schemas.fixtures import (
    FixtureCaptureRequestManifest,
    FixtureEntry,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reproducibility/rapidapi_backends_v1.json"


@dataclass
class FakeResponse:
    status_code: int
    content: bytes


class FakeTransport:
    def __init__(self):
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return FakeResponse(200, b'{"synthetic":"response-body"}')


def prepared(tmp_path):
    backend, backend_hash = load_backend_manifest(
        CONFIG, expected_sha256=PINNED_BACKEND_CONFIG_SHA256
    )
    record = backend.backends[3]
    request = build_fixture_request(
        backend,
        url=record.url,
        params={"query": "PRIVATE_QUERY"},
        headers={"X-RapidAPI-Host": record.host},
    )
    capture_manifest = FixtureCaptureRequestManifest(
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
        requests_per_second=100.0,
        concurrency_limit=1,
        created_at_utc="2026-09-09T00:00:00Z",
    )
    request_path = tmp_path / "capture-request.json"
    request_path.write_bytes(canonical_json_bytes(capture_manifest.model_dump(mode="json")))
    return backend, backend_hash, request, request_path


def test_capture_cli_isolated_and_sanitized(tmp_path, capsys):
    _, _, _, request_path = prepared(tmp_path)
    transport = FakeTransport()
    result = main(
        [
            "capture-fixtures",
            "--backend-config",
            str(CONFIG),
            "--request-manifest",
            str(request_path),
            "--expected-request-manifest-sha256",
            sha256_bytes(request_path.read_bytes()),
            "--output-dir",
            str(tmp_path / "captured"),
        ],
        transport=transport,
        credential_provider=lambda: "PRIVATE_KEY",
        monotonic=iter([0.0, 0.0, 0.1, 0.2]).__next__,
    )
    output = capsys.readouterr().out
    assert result == 0 and transport.calls == 1
    assert '"status":"completed"' in output
    for forbidden in (
        "PRIVATE_KEY",
        "PRIVATE_QUERY",
        "response-body",
        "rapidapi.com",
        "X-RapidAPI",
    ):
        assert forbidden not in output
    assert '"total_tokens":0' in output and '"usage_complete":true' in output


def test_preflight_cli_is_local_and_sanitized(tmp_path, capsys):
    backend, backend_hash, request, _ = prepared(tmp_path)
    body = {"synthetic": "response-body"}
    entry = FixtureEntry(
        schema_version=1,
        fixture_key=request.fixture_key,
        canonical_tool_name=request.canonical_tool_name,
        effective_arguments=request.effective_arguments,
        backend_version=request.backend_version,
        request_url_identity=request.request_url_identity,
        status_code=200,
        normalized_response_body=body,
        response_body_sha256=canonical_sha256(body),
        captured_at_utc="2026-09-09T00:00:00Z",
        source="rapidapi_capture",
    )
    bundle, _ = publish_fixture_bundle(
        tmp_path / "private",
        [entry],
        backend_manifest=backend,
        backend_manifest_sha256=backend_hash,
        created_at_utc="2026-09-09T00:00:01Z",
    )
    assert main(
        [
            "preflight-fixtures",
            "--backend-config",
            str(CONFIG),
            "--fixture-manifest",
            str(bundle / "manifest.json"),
        ],
        credential_provider=lambda: pytest.fail("credential must not be read"),
    ) == 0
    output = capsys.readouterr().out
    assert '"status":"pass"' in output and '"entry_count":1' in output
    assert "PRIVATE_QUERY" not in output and "response-body" not in output


def test_relative_or_unapproved_capture_rejected_before_secret_or_transport(tmp_path, capsys):
    _, _, _, request_path = prepared(tmp_path)
    transport = FakeTransport()
    credentials = []
    result = main(
        [
            "capture-fixtures",
            "--backend-config",
            str(CONFIG),
            "--request-manifest",
            str(request_path),
            "--expected-request-manifest-sha256",
            "sha256:" + "0" * 64,
            "--output-dir",
            str(tmp_path / "captured"),
        ],
        transport=transport,
        credential_provider=lambda: credentials.append(True) or "PRIVATE_KEY",
    )
    assert result == 2 and transport.calls == 0 and credentials == []
    output = capsys.readouterr().out
    assert "PRIVATE" not in output
    result = main(
        [
            "preflight-fixtures",
            "--backend-config",
            "relative.json",
            "--fixture-manifest",
            "relative.json",
        ]
    )
    assert result == 2
    assert "relative.json" not in capsys.readouterr().out
