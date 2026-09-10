from __future__ import annotations

from pathlib import Path

import pytest

from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.reproducibility.fixture_cli import PINNED_BACKEND_CONFIG_SHA256
from toolsandbox_pipeline.reproducibility.fixture_store import (
    FixtureStore,
    FixtureValidationError,
    build_fixture_request,
    load_backend_manifest,
    publish_fixture_bundle,
)
from toolsandbox_pipeline.schemas.fixtures import FixtureEntry


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/reproducibility/rapidapi_backends_v1.json"


def setup_entry(body=None, captured_at="2026-09-09T00:00:00Z"):
    manifest, backend_hash = load_backend_manifest(
        CONFIG, expected_sha256=PINNED_BACKEND_CONFIG_SHA256
    )
    backend = manifest.backends[3]
    request = build_fixture_request(
        manifest,
        url=backend.url,
        params={"query": "AAPL"},
        headers={"X-RapidAPI-Host": backend.host},
    )
    body = body or {"data": {"stock": [{"symbol": "AAPL:NASDAQ", "price": 1.0}]}}
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
        captured_at_utc=captured_at,
        source="rapidapi_capture",
    )
    return manifest, backend_hash, request, entry


def publish(tmp_path, entry=None):
    manifest, backend_hash, request, default_entry = setup_entry()
    bundle, fixture_hash = publish_fixture_bundle(
        tmp_path / "private",
        [default_entry if entry is None else entry],
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        created_at_utc="2026-09-09T00:00:01Z",
    )
    return manifest, backend_hash, request, bundle, fixture_hash


def test_publish_open_permissions_and_deep_copy(tmp_path):
    manifest, backend_hash, request, bundle, fixture_hash = publish(tmp_path)
    assert bundle.parent.name == "fixtures"
    assert bundle.stat().st_mode & 0o777 == 0o700
    assert (bundle / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (bundle / "entries").iterdir())
    store = FixtureStore.open(
        bundle / "manifest.json",
        expected_manifest_sha256=fixture_hash,
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
    )
    first = store.lookup(request.fixture_key)
    first["data"]["stock"][0]["symbol"] = "MUTATED"
    assert store.lookup(request.fixture_key)["data"]["stock"][0]["symbol"] == "AAPL:NASDAQ"
    assert store.lookup("sha256:" + "f" * 64) is None


def test_identical_publication_reuses_verified_generation(tmp_path):
    manifest, backend_hash, _, entry = setup_entry()
    first, first_hash = publish_fixture_bundle(
        tmp_path / "private",
        [entry],
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        created_at_utc="2026-09-09T00:00:01Z",
    )
    second, second_hash = publish_fixture_bundle(
        tmp_path / "private",
        [entry],
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        created_at_utc="2026-09-09T00:00:01Z",
    )
    assert (first, first_hash) == (second, second_hash)


def test_capture_timestamp_excluded_from_generation_but_conflicting_bytes_fail(tmp_path):
    manifest, backend_hash, _, first_entry = setup_entry()
    _, _, _, second_entry = setup_entry(captured_at="2026-09-09T01:00:00Z")
    first, _ = publish_fixture_bundle(
        tmp_path / "private",
        [first_entry],
        backend_manifest=manifest,
        backend_manifest_sha256=backend_hash,
        created_at_utc="2026-09-09T00:00:01Z",
    )
    with pytest.raises(FixtureValidationError):
        publish_fixture_bundle(
            tmp_path / "private",
            [second_entry],
            backend_manifest=manifest,
            backend_manifest_sha256=backend_hash,
            created_at_utc="2026-09-09T00:00:01Z",
        )
    assert first.exists()


@pytest.mark.parametrize("mutation", ["body", "extra", "mode"])
def test_tamper_unexpected_file_and_permission_fail_closed(tmp_path, mutation):
    manifest, backend_hash, request, bundle, fixture_hash = publish(tmp_path)
    entry_path = bundle / "entries" / f"{request.fixture_key.removeprefix('sha256:')}.json"
    if mutation == "body":
        entry_path.write_bytes(entry_path.read_bytes().replace(b"AAPL", b"MSFT"))
    elif mutation == "extra":
        extra = bundle / "unexpected"
        extra.write_text("x")
        extra.chmod(0o600)
    else:
        entry_path.chmod(0o644)
    with pytest.raises(FixtureValidationError):
        FixtureStore.open(
            bundle / "manifest.json",
            expected_manifest_sha256=fixture_hash,
            backend_manifest=manifest,
            backend_manifest_sha256=backend_hash,
        )


def test_wrong_backend_hash_and_symlink_manifest_rejected(tmp_path):
    manifest, backend_hash, _, bundle, fixture_hash = publish(tmp_path)
    with pytest.raises(FixtureValidationError):
        FixtureStore.open(
            bundle / "manifest.json",
            expected_manifest_sha256=fixture_hash,
            backend_manifest=manifest,
            backend_manifest_sha256="sha256:" + "0" * 64,
        )
    link = tmp_path / "manifest-link.json"
    link.symlink_to(bundle / "manifest.json")
    with pytest.raises(FixtureValidationError):
        FixtureStore.open(
            link,
            expected_manifest_sha256=fixture_hash,
            backend_manifest=manifest,
            backend_manifest_sha256=backend_hash,
        )
