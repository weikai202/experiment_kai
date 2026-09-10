from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import pytest
import requests
import tool_sandbox.tools.rapid_api_search_tools as upstream
from tool_sandbox.common.execution_context import ExecutionContext, new_context

from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.reproducibility.fixture_cli import PINNED_BACKEND_CONFIG_SHA256
from toolsandbox_pipeline.reproducibility.fixture_store import (
    FixtureStore,
    build_fixture_request,
    load_backend_manifest,
    publish_fixture_bundle,
)
from toolsandbox_pipeline.reproducibility.rapidapi_boundary import (
    ExternalFixtureMiss,
    ExternalReadUnknownOutcome,
    PINNED_RAPID_API_GET_REQUEST,
    RapidAPIBoundary,
    RapidAPIBoundaryError,
    audit_pinned_upstream,
)
from toolsandbox_pipeline.schemas.fixtures import ExternalReadContext, FixtureEntry


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reproducibility/rapidapi_backends_v1.json"


REQUESTS = {
    "convert_currency": (
        {"from": "USD", "to": "EUR", "amount": 2},
        {"result": {"convertedAmount": 1.5}},
    ),
    "search_lat_lon": (
        {"location": "1.0,2.0", "language": "en"},
        {"results": [{"address": "Synthetic address"}]},
    ),
    "search_location_around_lat_lon": (
        {"query": "cafe", "lat": 1.0, "lng": 2.0, "limit": 4, "country": "us", "lang": "en"},
        {"data": [{"business_id": "private", "place_id": "x", "name": "Cafe"}]},
    ),
    "search_stock": (
        {"query": "AAPL"},
        {"data": {"stock": [{"symbol": "AAPL:NASDAQ", "price": 10.0}]}},
    ),
    "search_weather_around_lat_lon": (
        {"q": "1.0, 2.0", "days": 1},
        {
            "forecast": {
                "forecastday": [
                    {
                        "day": {"temp_c": 20.0, "temp_f": 68.0},
                        "astro": {"sunrise": "06:00"},
                    }
                ]
            },
            "location": {"name": "Synthetic"},
            "current": {"feelslike_c": 19.0, "feelslike_f": 66.0},
        },
    ),
}


def make_store(tmp_path):
    manifest, backend_hash = load_backend_manifest(
        CONFIG, expected_sha256=PINNED_BACKEND_CONFIG_SHA256
    )
    entries = []
    requests_by_name = {}
    for backend in manifest.backends:
        arguments, body = REQUESTS[backend.canonical_tool_name]
        request = build_fixture_request(
            manifest,
            url=backend.url,
            params=arguments,
            headers={"X-RapidAPI-Host": backend.host},
        )
        requests_by_name[backend.canonical_tool_name] = request
        entries.append(
            FixtureEntry(
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
        )
    bundle, fixture_hash = publish_fixture_bundle(
        tmp_path / "private",
        entries,
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        created_at_utc="2026-09-09T00:00:01Z",
    )
    store = FixtureStore.open(
        bundle / "manifest.json",
        expected_manifest_sha256=fixture_hash,
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
    )
    return manifest, backend_hash, store, requests_by_name


def context(backend_hash, fixture_hash=None, profile="strict_replay", call="call"):
    return ExternalReadContext(
        schema_version=1,
        run_id="run",
        profile=profile,
        phase="online",
        scenario_family_id="family",
        scenario_id="scenario",
        state_id="sha256:" + "1" * 64,
        logical_tool_call_id=call,
        backend_manifest_sha256=backend_hash,
        fixture_manifest_sha256=fixture_hash,
    )


def replay_boundary(manifest, backend_hash, store, attempts, misses):
    return RapidAPIBoundary(
        mode="replay",
        profile="strict_replay",
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        fixture_store=store,
        context_provider=lambda: context(backend_hash, store.manifest_sha256),
        attempt_sink=attempts.append,
        fixture_miss_sink=misses.append,
        monotonic=iter([1.0, 1.25] * 20).__next__,
        utc_now=lambda: "2026-09-09T00:00:00Z",
        attempt_id_factory=lambda: f"attempt_{len(attempts)}",
    )


def test_source_audit_and_construction_have_no_patch(tmp_path):
    manifest, backend_hash, store, _ = make_store(tmp_path)
    original_requests = upstream.requests
    boundary = replay_boundary(manifest, backend_hash, store, [], [])
    assert audit_pinned_upstream().startswith("sha256:")
    assert upstream.rapid_api_get_request is PINNED_RAPID_API_GET_REQUEST
    assert upstream.requests is original_requests is requests
    del boundary


def test_all_five_native_postprocessors_receive_replay_json(tmp_path, monkeypatch):
    manifest, backend_hash, store, _ = make_store(tmp_path)
    attempts = []
    monkeypatch.setattr(upstream, "get_location_service_status", lambda: True)
    monkeypatch.setattr(upstream, "get_current_location", lambda: {"latitude": 1.0, "longitude": 2.0})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: pytest.fail("HTTP is forbidden in replay"))
    class RejectEnvironment(dict):
        def __contains__(self, key):
            raise AssertionError(f"environment access is forbidden: {key}")

        def __getitem__(self, key):
            raise AssertionError(f"environment access is forbidden: {key}")

    with replay_boundary(manifest, backend_hash, store, attempts, []):
        monkeypatch.setattr(upstream.os, "environ", RejectEnvironment())
        with new_context(ExecutionContext()):
            assert upstream.convert_currency(2, "usd", "eur") == 1.5
            assert upstream.search_lat_lon(1.0, 2.0) == "Synthetic address"
            assert upstream.search_location_around_lat_lon("cafe") == [{"name": "Cafe"}]
            assert upstream.search_stock("AAPL") == {"symbol": "AAPL", "price": 10.0}
            weather = upstream.search_weather_around_lat_lon(0, 1.0, 2.0)
            assert weather["current_temperature"] == 20.0
            assert weather["perceived_temperature"] == 19.0
            assert "temp_f" not in weather and "feelslike_f" not in weather
    assert [item.canonical_tool_name for item in attempts] == list(REQUESTS)
    assert all(item.status == "fixture_hit" and not item.dispatched for item in attempts)
    # Location default resolution is represented in the effective fixture identity.
    location_attempt = attempts[2]
    assert location_attempt.fixture_key == next(
        entry.fixture_key
        for entry in store._entries.values()
        if entry.canonical_tool_name == "search_location_around_lat_lon"
    )


def test_replay_miss_is_fixed_public_failure_with_no_fallback(tmp_path):
    manifest, backend_hash, store, _ = make_store(tmp_path)
    attempts, misses = [], []
    boundary = replay_boundary(manifest, backend_hash, store, attempts, misses)
    stock = manifest.backends[3]
    with boundary:
        with pytest.raises(ExternalFixtureMiss, match="^EXTERNAL_FIXTURE_MISS$"):
            boundary.dispatch(
                stock.url,
                {"query": "MISSING"},
                {"X-RapidAPI-Host": stock.host},
            )
    assert len(attempts) == len(misses) == 1
    assert attempts[0].status == "fixture_miss"
    assert misses[0].effective_arguments == {"query": "MISSING"}
    assert store.entry_count == 5


def test_scope_restoration_nested_and_cross_thread_rejection(tmp_path):
    manifest, backend_hash, store, _ = make_store(tmp_path)
    boundary = replay_boundary(manifest, backend_hash, store, [], [])
    other = replay_boundary(manifest, backend_hash, store, [], [])
    failures = []
    original_requests = requests
    with pytest.raises(RuntimeError):
        with boundary:
            assert upstream.requests is not original_requests
            assert requests is original_requests
            with pytest.raises(RapidAPIBoundaryError, match="ALREADY_ACTIVE"):
                with other:
                    pass
            thread = threading.Thread(
                target=lambda: failures.append(
                    pytest.raises(RapidAPIBoundaryError, boundary.dispatch, "", {}, {})
                )
            )
            thread.start()
            thread.join()
            raise RuntimeError("synthetic")
    assert upstream.rapid_api_get_request is PINNED_RAPID_API_GET_REQUEST
    assert upstream.requests is original_requests
    assert len(failures) == 1


@dataclass
class FakeResponse:
    status_code: int
    content: bytes


class FakeTransport:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_official_live_exact_one_dispatch_and_audit(tmp_path):
    manifest, backend_hash, _, _ = make_store(tmp_path)
    transport = FakeTransport(FakeResponse(200, b'{"data":{"stock":[{"symbol":"AAPL:NASDAQ"}]}}'))
    attempts = []
    boundary = RapidAPIBoundary(
        mode="official_live",
        profile="official_live",
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        context_provider=lambda: context(backend_hash, profile="official_live"),
        attempt_sink=attempts.append,
        transport=transport,
        credential_provider=lambda: "synthetic-key",
        monotonic=iter([1.0, 1.5]).__next__,
        utc_now=lambda: "2026-09-09T00:00:00Z",
        attempt_id_factory=lambda: "attempt_live",
    )
    stock = manifest.backends[3]
    with boundary:
        body = boundary.dispatch(stock.url, {"query": "AAPL"}, {"X-RapidAPI-Host": stock.host})
    assert body["data"]["stock"][0]["symbol"] == "AAPL:NASDAQ"
    assert len(transport.calls) == 1
    url, kwargs = transport.calls[0]
    assert url == stock.url
    assert kwargs == {
        "params": {"query": "AAPL"},
        "headers": {
            "X-RapidAPI-Host": stock.host,
            "X-RapidAPI-Key": "synthetic-key",
        },
        "timeout": 30.0,
        "allow_redirects": False,
    }
    assert attempts[0].status == "completed"
    assert attempts[0].latency_seconds == 0.5
    assert "synthetic-key" not in attempts[0].model_dump_json()


@pytest.mark.parametrize("error", [TimeoutError("private"), ConnectionError("private")])
def test_live_unknown_outcome_is_never_retried(tmp_path, error):
    manifest, backend_hash, _, _ = make_store(tmp_path)
    transport = FakeTransport(error)
    attempts = []
    boundary = RapidAPIBoundary(
        mode="official_live",
        profile="official_live",
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        context_provider=lambda: context(backend_hash, profile="official_live"),
        attempt_sink=attempts.append,
        transport=transport,
        credential_provider=lambda: "synthetic-key",
        monotonic=iter([1.0, 1.2]).__next__,
        utc_now=lambda: "2026-09-09T00:00:00Z",
    )
    stock = manifest.backends[3]
    with boundary:
        with pytest.raises(ExternalReadUnknownOutcome, match="EXTERNAL_READ_UNKNOWN_OUTCOME"):
            boundary.dispatch(stock.url, {"query": "AAPL"}, {"X-RapidAPI-Host": stock.host})
    assert len(transport.calls) == len(attempts) == 1
    assert attempts[0].status == "unknown_outcome"


def test_stale_context_and_unknown_request_rejected_before_dispatch(tmp_path):
    manifest, backend_hash, _, _ = make_store(tmp_path)
    transport = FakeTransport(FakeResponse(200, b"{}"))
    boundary = RapidAPIBoundary(
        mode="official_live",
        profile="official_live",
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        context_provider=lambda: context(
            "sha256:" + "9" * 64, profile="official_live"
        ),
        attempt_sink=lambda _: None,
        transport=transport,
        credential_provider=lambda: "synthetic-key",
    )
    with boundary:
        with pytest.raises(RapidAPIBoundaryError, match="STALE"):
            boundary.dispatch("https://invalid.example/x", {}, {})
    assert transport.calls == []
