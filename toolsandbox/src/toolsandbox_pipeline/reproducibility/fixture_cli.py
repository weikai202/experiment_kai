"""Narrow local fixture preflight and externally authorized capture CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Sequence

import requests

from toolsandbox_pipeline.reproducibility.fixture_capture import (
    FixtureCaptureError,
    capture_fixtures,
    load_capture_request,
)
from toolsandbox_pipeline.reproducibility.fixture_store import (
    FixtureStore,
    PINNED_BACKEND_CONFIG_SHA256,
    load_backend_manifest,
    sha256_bytes,
)


class RequestsTransport:
    """One-shot transport adapter; retrying sessions are deliberately absent."""

    def get(self, url, *, params, headers, timeout, allow_redirects):
        return requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolsandbox-fixtures")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight-fixtures")
    preflight.add_argument("--backend-config", required=True)
    preflight.add_argument("--fixture-manifest", required=True)
    preflight.add_argument("--expected-backend-config-sha256")
    capture = subparsers.add_parser("capture-fixtures")
    capture.add_argument("--backend-config", required=True)
    capture.add_argument("--request-manifest", required=True)
    capture.add_argument("--expected-request-manifest-sha256", required=True)
    capture.add_argument("--output-dir", required=True)
    capture.add_argument("--expected-backend-config-sha256")
    return parser


def _absolute(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError("absolute path required")
    return path


def _safe_print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _preflight(args: argparse.Namespace, *, monotonic=time.monotonic) -> int:
    started = monotonic()
    backend, backend_sha256 = load_backend_manifest(
        _absolute(args.backend_config),
        expected_sha256=(getattr(args, "expected_backend_config_sha256", None)
                         or sha256_bytes(_absolute(args.backend_config).read_bytes())),
    )
    fixture_path = _absolute(args.fixture_manifest)
    fixture_sha256 = sha256_bytes(fixture_path.read_bytes())
    store = FixtureStore.open(
        fixture_path,
        expected_manifest_sha256=fixture_sha256,
        backend_manifest=backend,
        backend_manifest_sha256=backend_sha256,
    )
    _safe_print(
        {
            "backend_manifest_sha256": backend_sha256,
            "entry_count": store.entry_count,
            "fixture_generation_id": store.manifest.fixture_generation_id,
            "fixture_manifest_sha256": fixture_sha256,
            "status": "pass",
            "total_running_time_seconds": float(max(0.0, monotonic() - started)),
            "total_tokens": 0,
            "usage_complete": True,
        }
    )
    return 0


def _capture(
    args: argparse.Namespace,
    *,
    transport=None,
    credential_provider=None,
    monotonic=time.monotonic,
) -> int:
    backend, backend_sha256 = load_backend_manifest(
        _absolute(args.backend_config),
        expected_sha256=(getattr(args, "expected_backend_config_sha256", None)
                         or sha256_bytes(_absolute(args.backend_config).read_bytes())),
    )
    request_manifest = load_capture_request(
        _absolute(args.request_manifest),
        expected_sha256=args.expected_request_manifest_sha256,
        backend_manifest=backend,
        backend_manifest_sha256=backend_sha256,
    )
    output_dir = _absolute(args.output_dir)
    selected_transport = RequestsTransport() if transport is None else transport
    selected_credential_provider = (
        (lambda: os.environ["RAPID_API_KEY"])
        if credential_provider is None
        else credential_provider
    )
    try:
        report = capture_fixtures(
            request_manifest,
            output_root=output_dir,
            backend_manifest=backend,
            backend_manifest_sha256=backend_sha256,
            transport=selected_transport,
            credential_provider=selected_credential_provider,
            monotonic=monotonic,
        )
    except FixtureCaptureError as exc:
        report = exc.report
        result = report.model_dump(mode="json")
        result.update({"total_tokens": 0, "usage_complete": True})
        _safe_print(result)
        return 2
    result = report.model_dump(mode="json")
    result.update({"total_tokens": 0, "usage_complete": True})
    _safe_print(result)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    transport=None,
    credential_provider=None,
    monotonic=time.monotonic,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight-fixtures":
            return _preflight(args, monotonic=monotonic)
        return _capture(
            args,
            transport=transport,
            credential_provider=credential_provider,
            monotonic=monotonic,
        )
    except Exception as exc:
        _safe_print({"error_class": type(exc).__name__, "status": "fail"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
