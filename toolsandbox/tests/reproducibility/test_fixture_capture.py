from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes
from toolsandbox_pipeline.reproducibility.fixture_capture import (
    FixtureCaptureError,
    capture_fixtures,
    load_capture_request,
)
from toolsandbox_pipeline.reproducibility.fixture_cli import PINNED_BACKEND_CONFIG_SHA256
from toolsandbox_pipeline.reproducibility.fixture_store import (
    FixtureStore,
    FixtureValidationError,
    build_fixture_request,
    load_backend_manifest,
    sha256_bytes,
)
from toolsandbox_pipeline.schemas.fixtures import FixtureCaptureRequestManifest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reproducibility/rapidapi_backends_v1.json"


def capture_manifest(count=1):
    backend, backend_hash = load_backend_manifest(
        CONFIG, expected_sha256=PINNED_BACKEND_CONFIG_SHA256
    )
    requests = []
    for query in ("AAPL", "MSFT")[:count]:
        record = backend.backends[3]
        requests.append(
            build_fixture_request(
                backend,
                url=record.url,
                params={"query": query},
                headers={"X-RapidAPI-Host": record.host},
            )
        )
    requests.sort(key=lambda item: item.fixture_key.encode())
    manifest = FixtureCaptureRequestManifest(
        schema_version=1,
        adapter_version="toolsandbox-rapidapi-boundary-v1",
        upstream_commit="165848b9a78cead7ca7fe7c89c688b58e6501219",
        backend_manifest_sha256=backend_hash,
        purpose="synthetic_contract_probe",
        train_manifest_sha256=None,
        fixture_miss_audit_sha256=None,
        requests=tuple(requests),
        ordered_fixture_keys=tuple(item.fixture_key for item in requests),
        maximum_request_count=count,
        requests_per_second=100.0,
        concurrency_limit=1,
        created_at_utc="2026-09-09T00:00:00Z",
    )
    return backend, backend_hash, manifest


@dataclass
class FakeResponse:
    status_code: int
    content: bytes


class SequenceTransport:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


def ticking():
    values = iter([0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4])
    return values.__next__


def test_capture_request_hash_and_backend_validation(tmp_path):
    backend, backend_hash, manifest = capture_manifest()
    path = tmp_path / "request.json"
    path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    digest = sha256_bytes(path.read_bytes())
    assert load_capture_request(
        path,
        expected_sha256=digest,
        backend_manifest=backend,
        backend_manifest_sha256=backend_hash,
    ) == manifest
    with pytest.raises(FixtureValidationError):
        load_capture_request(
            path,
            expected_sha256="sha256:" + "0" * 64,
            backend_manifest=backend,
            backend_manifest_sha256=backend_hash,
        )


def test_successful_capture_one_request_each_and_publish(tmp_path):
    backend, backend_hash, manifest = capture_manifest(count=2)
    transport = SequenceTransport(
        [
            FakeResponse(200, b'{"data":{"stock":[{"symbol":"ONE:X"}]}}'),
            FakeResponse(200, b'{"data":{"stock":[{"symbol":"TWO:X"}]}}'),
        ]
    )
    sleeps = []
    report = capture_fixtures(
        manifest,
        output_root=tmp_path / "capture",
        backend_manifest=backend,
        backend_manifest_sha256=backend_hash,
        transport=transport,
        credential_provider=lambda: "synthetic-key",
        monotonic=ticking(),
        utc_now=lambda: "2026-09-09T00:00:00Z",
        sleep=sleeps.append,
        attempt_id_factory=iter(("a1", "a2")).__next__,
    )
    assert report.status == "completed"
    assert report.request_count == report.success_count == 2
    assert report.failure_count == 0
    assert len(transport.calls) == 2
    assert all(call[1]["timeout"] == 30.0 and call[1]["allow_redirects"] is False for call in transport.calls)
    assert all(call[1]["headers"]["X-RapidAPI-Key"] == "synthetic-key" for call in transport.calls)
    bundle = tmp_path / "capture" / "fixtures" / report.fixture_generation_id
    store = FixtureStore.open(
        bundle / "manifest.json",
        expected_manifest_sha256=report.fixture_manifest_sha256,
        backend_manifest=backend,
        backend_manifest_sha256=backend_hash,
    )
    assert store.entry_count == 2
    audit = (tmp_path / "capture" / "capture_audit.jsonl").read_text()
    assert "synthetic-key" not in audit
    assert "ONE" not in audit and "AAPL" not in audit


@pytest.mark.parametrize(
    "failure,expected_status",
    [
        (TimeoutError("private detail"), "unknown_outcome"),
        (ConnectionError("private detail"), "unknown_outcome"),
        (RuntimeError("private detail"), "failed"),
        (FakeResponse(503, b'{"private":"body"}'), "failed"),
        (FakeResponse(200, b'{"a":1,"a":2}'), "failed"),
    ],
)
def test_capture_failure_has_no_retry_or_partial_bundle(tmp_path, failure, expected_status):
    backend, backend_hash, manifest = capture_manifest(count=2)
    transport = SequenceTransport([failure])
    with pytest.raises(FixtureCaptureError) as captured:
        capture_fixtures(
            manifest,
            output_root=tmp_path / "capture",
            backend_manifest=backend,
            backend_manifest_sha256=backend_hash,
            transport=transport,
            credential_provider=lambda: "synthetic-key",
            monotonic=ticking(),
            utc_now=lambda: "2026-09-09T00:00:00Z",
            sleep=lambda _: None,
        )
    report = captured.value.report
    assert report.status == "failed"
    assert len(transport.calls) == 1
    assert not (tmp_path / "capture" / "fixtures").exists()
    audit = (tmp_path / "capture" / "capture_audit.jsonl").read_text()
    assert expected_status in audit
    assert "private detail" not in audit
    assert '"private":"body"' not in audit


def test_rejection_precedes_credential_retrieval_and_dispatch(tmp_path):
    backend, backend_hash, manifest = capture_manifest()
    output = tmp_path / "existing"
    output.mkdir()
    credential_calls = []
    transport = SequenceTransport([])
    with pytest.raises(FixtureValidationError):
        capture_fixtures(
            manifest,
            output_root=output,
            backend_manifest=backend,
            backend_manifest_sha256=backend_hash,
            transport=transport,
            credential_provider=lambda: credential_calls.append(True) or "synthetic-key",
        )
    assert credential_calls == []
    assert transport.calls == []
