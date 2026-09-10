"""Explicit, approval-bound RapidAPI fixture capture with no retry path."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Callable
from uuid import uuid4

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.reproducibility.fixture_store import (
    FixtureValidationError,
    backend_by_name,
    parse_strict_json,
    publish_fixture_bundle,
    sha256_bytes,
    validate_request_against_backend,
)
from toolsandbox_pipeline.reproducibility.rapidapi_boundary import (
    ExternalReadFailed,
    HTTPTransport,
    classify_transport_exception,
    normalize_response,
)
from toolsandbox_pipeline.schemas.fixtures import (
    CaptureReport,
    ExternalReadAttempt,
    ExternalReadContext,
    FixtureCaptureRequestManifest,
    FixtureEntry,
    RapidAPIBackendManifest,
)


class FixtureCaptureError(RuntimeError):
    def __init__(self, report: CaptureReport) -> None:
        super().__init__("FIXTURE_CAPTURE_FAILED")
        self.report = report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_capture_request(
    path: str | Path,
    *,
    expected_sha256: str,
    backend_manifest: RapidAPIBackendManifest,
    backend_manifest_sha256: str,
) -> FixtureCaptureRequestManifest:
    request_path = Path(path)
    if not request_path.is_absolute() or request_path.is_symlink() or not request_path.is_file():
        raise FixtureValidationError("absolute regular capture request manifest required")
    for parent in request_path.parents:
        if parent.is_symlink():
            raise FixtureValidationError("symlink capture request path is forbidden")
    raw = request_path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise FixtureValidationError("capture request manifest hash mismatch")
    try:
        parse_strict_json(raw)
        manifest = FixtureCaptureRequestManifest.model_validate_json(raw)
    except Exception as exc:
        raise FixtureValidationError("invalid capture request manifest") from exc
    if raw != canonical_json_bytes(manifest.model_dump(mode="json")):
        raise FixtureValidationError("capture request manifest must use canonical JSON")
    if manifest.backend_manifest_sha256 != backend_manifest_sha256:
        raise FixtureValidationError("capture request backend identity mismatch")
    for request in manifest.requests:
        validate_request_against_backend(request, backend_manifest)
    return manifest


def _capture_context(request, backend_manifest_sha256: str) -> ExternalReadContext:
    return ExternalReadContext(
        schema_version=1,
        run_id="fixture_capture",
        profile="official_live",
        phase="setup",
        scenario_family_id="not_applicable",
        scenario_id="not_applicable",
        state_id=canonical_sha256({"capture_request": request.fixture_key}),
        logical_tool_call_id="capture_" + request.fixture_key.removeprefix("sha256:"),
        backend_manifest_sha256=backend_manifest_sha256,
        fixture_manifest_sha256=None,
    )


def _attempt(
    *,
    attempt_id: str,
    context: ExternalReadContext,
    request,
    status: str,
    dispatched: bool,
    status_code: int | None,
    response_body_sha256: str | None,
    sanitized_exception_class: str | None,
    started_at_utc: str,
    ended_at_utc: str,
    latency_seconds: float,
    response_byte_count: int | None = None,
) -> ExternalReadAttempt:
    return ExternalReadAttempt(
        schema_version=1,
        attempt_id=attempt_id,
        context=context,
        mode="capture",
        canonical_tool_name=request.canonical_tool_name,
        effect="external_read",
        fixture_key=request.fixture_key,
        backend_version=request.backend_version,
        status=status,
        dispatched=dispatched,
        status_code=status_code,
        response_byte_count=response_byte_count,
        response_body_sha256=response_body_sha256,
        sanitized_exception_class=sanitized_exception_class,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        latency_seconds=float(max(0.0, latency_seconds)),
    )


def _write_audit(output_root: Path, attempts: tuple[ExternalReadAttempt, ...]) -> None:
    raw = b"".join(canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in attempts)
    path = output_root / "capture_audit.jsonl"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def capture_fixtures(
    request_manifest: FixtureCaptureRequestManifest,
    *,
    output_root: str | Path,
    backend_manifest: RapidAPIBackendManifest,
    backend_manifest_sha256: str,
    transport: HTTPTransport,
    credential_provider: Callable[[], str],
    monotonic: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], str] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
    attempt_id_factory: Callable[[], str] = lambda: "era_" + uuid4().hex,
) -> CaptureReport:
    """Capture an already validated immutable request manifest exactly once."""

    start_all = monotonic()
    if type(request_manifest) is not FixtureCaptureRequestManifest:
        raise TypeError("validated capture request manifest required")
    if request_manifest.backend_manifest_sha256 != backend_manifest_sha256:
        raise FixtureValidationError("capture/backend identity mismatch")
    root = Path(output_root)
    if not root.is_absolute() or root.exists() or root.is_symlink():
        raise FixtureValidationError("capture output root must be a new absolute path")
    for parent in root.parents:
        if parent.is_symlink():
            raise FixtureValidationError("symlink output path is forbidden")
    for request in request_manifest.requests:
        validate_request_against_backend(request, backend_manifest)
    # All input/output authorization validation precedes credential retrieval.
    credential = credential_provider()
    if type(credential) is not str or not credential:
        raise FixtureValidationError("capture credential unavailable")
    root.mkdir(mode=0o700)
    attempts: list[ExternalReadAttempt] = []
    entries: list[FixtureEntry] = []
    previous_dispatch: float | None = None
    failure_class: str | None = None
    for request in request_manifest.requests:
        if previous_dispatch is not None:
            delay = (1.0 / request_manifest.requests_per_second) - (monotonic() - previous_dispatch)
            if delay > 0:
                sleep(delay)
        record = backend_by_name(backend_manifest, request.canonical_tool_name)
        attempt_id = attempt_id_factory()
        started_at = utc_now()
        started = monotonic()
        previous_dispatch = started
        context = _capture_context(request, backend_manifest_sha256)
        try:
            response = transport.get(
                record.url,
                params=dict(request.effective_arguments),
                headers={"X-RapidAPI-Host": record.host, "X-RapidAPI-Key": credential},
                timeout=backend_manifest.request_timeout_seconds,
                allow_redirects=backend_manifest.allow_redirects,
            )
        except Exception as exc:
            failure_class, unknown = classify_transport_exception(exc)
            attempts.append(
                _attempt(
                    attempt_id=attempt_id,
                    context=context,
                    request=request,
                    status="unknown_outcome" if unknown else "failed",
                    dispatched=True,
                    status_code=None,
                    response_body_sha256=None,
                    sanitized_exception_class=failure_class,
                    started_at_utc=started_at,
                    ended_at_utc=utc_now(),
                    latency_seconds=monotonic() - started,
                )
            )
            break
        try:
            status_code, body, body_sha256, byte_count = normalize_response(response)
        except Exception as exc:
            failure_class = type(exc).__name__
            attempts.append(
                _attempt(
                    attempt_id=attempt_id,
                    context=context,
                    request=request,
                    status="failed",
                    dispatched=True,
                    status_code=getattr(response, "status_code", None),
                    response_body_sha256=None,
                    sanitized_exception_class=failure_class,
                    started_at_utc=started_at,
                    ended_at_utc=utc_now(),
                    latency_seconds=monotonic() - started,
                    response_byte_count=(
                        len(response.content)
                        if type(getattr(response, "content", None)) is bytes
                        else None
                    ),
                )
            )
            break
        if status_code != 200:
            attempts.append(
                _attempt(
                    attempt_id=attempt_id,
                    context=context,
                    request=request,
                    status="failed",
                    dispatched=True,
                    status_code=status_code,
                    response_body_sha256=body_sha256,
                    sanitized_exception_class="HTTPStatusError",
                    started_at_utc=started_at,
                    ended_at_utc=utc_now(),
                    latency_seconds=monotonic() - started,
                    response_byte_count=byte_count,
                )
            )
            failure_class = "HTTPStatusError"
            break
        attempts.append(
            _attempt(
                attempt_id=attempt_id,
                context=context,
                request=request,
                status="completed",
                dispatched=True,
                status_code=200,
                response_body_sha256=body_sha256,
                sanitized_exception_class=None,
                started_at_utc=started_at,
                ended_at_utc=utc_now(),
                latency_seconds=monotonic() - started,
                response_byte_count=byte_count,
            )
        )
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
                response_body_sha256=body_sha256,
                captured_at_utc=utc_now(),
                source="rapidapi_capture",
            )
        )
    elapsed = float(max(0.0, monotonic() - start_all))
    generation_id: str | None = None
    fixture_manifest_sha256: str | None = None
    if len(entries) == len(request_manifest.requests):
        bundle, fixture_manifest_sha256 = publish_fixture_bundle(
            root,
            entries,
            backend_manifest=backend_manifest,
            backend_manifest_sha256=backend_manifest_sha256,
            created_at_utc=utc_now(),
        )
        generation_id = bundle.name
    _write_audit(root, tuple(attempts))
    latencies = [item.latency_seconds for item in attempts]
    failure_count = len(request_manifest.requests) - len(entries)
    status_counts = Counter(item.status for item in attempts)
    if failure_count > len(attempts) - len(entries):
        status_counts["not_dispatched_after_failure"] += failure_count - (len(attempts) - len(entries))
    errors = tuple(
        sorted(
            {
                item.sanitized_exception_class
                for item in attempts
                if item.sanitized_exception_class is not None
            }
            | ({"AbortedAfterFailure"} if failure_count > len(attempts) - len(entries) else set())
        )
    )
    report = CaptureReport(
        schema_version=1,
        status="completed" if failure_count == 0 else "failed",
        request_count=len(request_manifest.requests),
        success_count=len(entries),
        failure_count=failure_count,
        fixture_generation_id=generation_id,
        fixture_manifest_sha256=fixture_manifest_sha256,
        total_running_time_seconds=elapsed,
        status_counts=dict(status_counts),
        latency_min_seconds=min(latencies) if latencies else None,
        latency_max_seconds=max(latencies) if latencies else None,
        latency_mean_seconds=sum(latencies) / len(latencies) if latencies else None,
        sanitized_error_classes=errors,
    )
    if report.status == "failed":
        raise FixtureCaptureError(report)
    return report
