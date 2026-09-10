"""Scoped interception of the pinned ToolSandbox RapidAPI raw-JSON boundary."""

from __future__ import annotations

import ast
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution
import inspect
import json
import os
from pathlib import Path
import threading
import time
from types import ModuleType
from typing import Any, Callable, Protocol
from uuid import uuid4

import requests as process_requests
import tool_sandbox.tools.rapid_api_search_tools as upstream_rapidapi

from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.reproducibility.fixture_store import (
    EXPECTED_BACKENDS,
    FixtureStore,
    FixtureValidationError,
    PINNED_BACKEND_CONFIG_SHA256,
    build_fixture_request,
    parse_strict_json,
)
from toolsandbox_pipeline.schemas.fixtures import (
    ExternalReadAttempt,
    ExternalReadContext,
    FixtureMissEvidence,
    FixtureRequest,
    RapidAPIBackendManifest,
    TOOL_NAMES,
    UPSTREAM_COMMIT,
)


PINNED_RAPIDAPI_SOURCE_SHA256 = "sha256:2e369320fb6ce3a203e324fe55979ee1dfdb14f5ff9afc3eefeb3aa2a1a86fb1"
PINNED_RAPID_API_GET_REQUEST = upstream_rapidapi.rapid_api_get_request


class ExternalFixtureMiss(RuntimeError):
    def __init__(self) -> None:
        super().__init__("EXTERNAL_FIXTURE_MISS")


class ExternalReadUnknownOutcome(RuntimeError):
    def __init__(self) -> None:
        super().__init__("EXTERNAL_READ_UNKNOWN_OUTCOME")


class ExternalReadFailed(RuntimeError):
    def __init__(self) -> None:
        super().__init__("EXTERNAL_READ_FAILED")


class RapidAPIBoundaryError(RuntimeError):
    pass


class HTTPResponse(Protocol):
    status_code: int
    content: bytes


class HTTPTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
        allow_redirects: bool,
    ) -> HTTPResponse: ...


class _RejectingRequestsProxy:
    def __getattr__(self, name: str) -> Any:
        del name
        raise RapidAPIBoundaryError("RAPIDAPI_NETWORK_BYPASS_DENIED")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _installed_upstream_commit() -> str:
    try:
        metadata = distribution("tool-sandbox").read_text("direct_url.json")
        if metadata is None:
            raise ValueError
        return str(json.loads(metadata)["vcs_info"]["commit_id"])
    except Exception as exc:
        raise RapidAPIBoundaryError("UPSTREAM_IDENTITY_UNAVAILABLE") from exc


def audit_pinned_upstream(module: ModuleType = upstream_rapidapi) -> str:
    """Verify exact source and that only the five registered tools cross the seam."""

    if module is not upstream_rapidapi or module.__name__ != "tool_sandbox.tools.rapid_api_search_tools":
        raise RapidAPIBoundaryError("UPSTREAM_MODULE_IDENTITY_DRIFT")
    source_path_text = inspect.getsourcefile(module)
    if source_path_text is None:
        raise RapidAPIBoundaryError("UPSTREAM_SOURCE_UNAVAILABLE")
    raw = Path(source_path_text).read_bytes()
    actual_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    if actual_hash != PINNED_RAPIDAPI_SOURCE_SHA256 or _installed_upstream_commit() != UPSTREAM_COMMIT:
        raise RapidAPIBoundaryError("UPSTREAM_SOURCE_IDENTITY_DRIFT")
    tree = ast.parse(raw.decode("utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    registered = {
        name
        for name, value in vars(module).items()
        if callable(value)
        and getattr(value, "is_tool", False)
        and getattr(value, "__module__", None) == module.__name__
    }
    if registered != set(TOOL_NAMES):
        raise RapidAPIBoundaryError("UPSTREAM_TOOL_REGISTRATION_DRIFT")
    callers: set[str] = set()
    for name, node in functions.items():
        seam_calls = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "rapid_api_get_request"
        ]
        if seam_calls:
            callers.add(name)
            if len(seam_calls) != 1:
                raise RapidAPIBoundaryError("UPSTREAM_BOUNDARY_CALL_COUNT_DRIFT")
        direct_requests = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "requests"
        ]
        if direct_requests and name != "rapid_api_get_request":
            raise RapidAPIBoundaryError("UPSTREAM_DIRECT_REQUEST_BYPASS")
    if callers != set(TOOL_NAMES):
        raise RapidAPIBoundaryError("UPSTREAM_BOUNDARY_COVERAGE_DRIFT")
    return actual_hash


def normalize_response(response: HTTPResponse) -> tuple[int, dict[str, Any], str, int]:
    if type(response.status_code) is not int or type(response.content) is not bytes:
        raise FixtureValidationError("invalid transport response shape")
    parsed = parse_strict_json(response.content)
    if type(parsed) is not dict:
        raise FixtureValidationError("RapidAPI response must be a JSON object")
    body_hash = canonical_sha256(parsed)
    return response.status_code, parsed, body_hash, len(response.content)


def classify_transport_exception(exc: BaseException) -> tuple[str, bool]:
    if isinstance(
        exc,
        (TimeoutError, ConnectionError, process_requests.Timeout, process_requests.ConnectionError),
    ):
        return type(exc).__name__, True
    return type(exc).__name__, False


class RapidAPIBoundary(AbstractContextManager["RapidAPIBoundary"]):
    """Process-scoped, non-reentrant native-tool boundary."""

    _entry_lock = threading.Lock()
    _active = False

    def __init__(
        self,
        *,
        mode: str,
        profile: str,
        backend_manifest: RapidAPIBackendManifest,
        backend_manifest_sha256: str,
        context_provider: Callable[[], ExternalReadContext],
        attempt_sink: Callable[[ExternalReadAttempt], None],
        fixture_store: FixtureStore | None = None,
        fixture_miss_sink: Callable[[FixtureMissEvidence], None] | None = None,
        transport: HTTPTransport | None = None,
        credential_provider: Callable[[], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] = _utc_now,
        attempt_id_factory: Callable[[], str] = lambda: "era_" + uuid4().hex,
    ) -> None:
        if mode not in ("replay", "official_live"):
            raise RapidAPIBoundaryError("UNKNOWN_RAPIDAPI_MODE")
        if (mode, profile) not in (("replay", "strict_replay"), ("official_live", "official_live")):
            raise RapidAPIBoundaryError("RAPIDAPI_MODE_PROFILE_MISMATCH")
        if type(backend_manifest) is not RapidAPIBackendManifest:
            raise TypeError("validated backend manifest required")
        actual_backends = tuple(
            (
                item.canonical_tool_name,
                item.backend_version,
                item.url,
                item.host,
                item.permitted_parameter_names,
            )
            for item in backend_manifest.backends
        )
        if (
            backend_manifest_sha256 != PINNED_BACKEND_CONFIG_SHA256
            or actual_backends != EXPECTED_BACKENDS
            or backend_manifest.request_timeout_seconds != 30.0
        ):
            raise RapidAPIBoundaryError("RAPIDAPI_BACKEND_MANIFEST_DRIFT")
        if mode == "replay" and (fixture_store is None or fixture_miss_sink is None):
            raise RapidAPIBoundaryError("REPLAY_DEPENDENCY_MISSING")
        if mode == "official_live" and (transport is None or credential_provider is None):
            raise RapidAPIBoundaryError("LIVE_DEPENDENCY_MISSING")
        self.mode = mode
        self.profile = profile
        self.backend_manifest = backend_manifest
        self.backend_manifest_sha256 = backend_manifest_sha256
        self.context_provider = context_provider
        self.attempt_sink = attempt_sink
        self.fixture_store = fixture_store
        self.fixture_miss_sink = fixture_miss_sink
        self.transport = transport
        self.credential_provider = credential_provider
        self.monotonic = monotonic
        self.utc_now = utc_now
        self.attempt_id_factory = attempt_id_factory
        self._process_id = os.getpid()
        self._thread_id: int | None = None
        self._entered = False
        self._original_requests: Any = None

    def __enter__(self) -> "RapidAPIBoundary":
        if os.getpid() != self._process_id:
            raise RapidAPIBoundaryError("RAPIDAPI_WRONG_PROCESS")
        if not self._entry_lock.acquire(blocking=False):
            raise RapidAPIBoundaryError("RAPIDAPI_BOUNDARY_ALREADY_ACTIVE")
        try:
            if RapidAPIBoundary._active or self._entered:
                raise RapidAPIBoundaryError("RAPIDAPI_BOUNDARY_ALREADY_ACTIVE")
            audit_pinned_upstream()
            if upstream_rapidapi.rapid_api_get_request is not PINNED_RAPID_API_GET_REQUEST:
                raise RapidAPIBoundaryError("UPSTREAM_CALLABLE_IDENTITY_DRIFT")
            if upstream_rapidapi.requests is not process_requests:
                raise RapidAPIBoundaryError("UPSTREAM_REQUESTS_IDENTITY_DRIFT")
            self._original_requests = upstream_rapidapi.requests
            upstream_rapidapi.rapid_api_get_request = self.dispatch
            upstream_rapidapi.requests = _RejectingRequestsProxy()
            self._thread_id = threading.get_ident()
            self._entered = True
            RapidAPIBoundary._active = True
            return self
        except Exception:
            self._entry_lock.release()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if not self._entered:
            return None
        try:
            upstream_rapidapi.rapid_api_get_request = PINNED_RAPID_API_GET_REQUEST
            upstream_rapidapi.requests = self._original_requests
        finally:
            RapidAPIBoundary._active = False
            self._entered = False
            self._thread_id = None
            self._entry_lock.release()
        return None

    def _context(self) -> ExternalReadContext:
        if not self._entered or threading.get_ident() != self._thread_id or os.getpid() != self._process_id:
            raise RapidAPIBoundaryError("RAPIDAPI_BOUNDARY_SCOPE_VIOLATION")
        context = self.context_provider()
        if type(context) is not ExternalReadContext:
            raise RapidAPIBoundaryError("EXTERNAL_READ_CONTEXT_MISSING")
        if (
            context.profile != self.profile
            or context.backend_manifest_sha256 != self.backend_manifest_sha256
            or (
                self.mode == "replay"
                and (
                    self.fixture_store is None
                    or context.fixture_manifest_sha256 != self.fixture_store.manifest_sha256
                )
            )
        ):
            raise RapidAPIBoundaryError("EXTERNAL_READ_CONTEXT_STALE")
        return context

    def dispatch(self, url: str, params: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
        context = self._context()
        started = self.monotonic()
        started_at = self.utc_now()
        attempt_id = self.attempt_id_factory()
        request: FixtureRequest | None = None
        try:
            request = build_fixture_request(self.backend_manifest, url=url, params=params, headers=headers)
            if self.mode == "replay":
                assert self.fixture_store is not None and self.fixture_miss_sink is not None
                entry = self.fixture_store.lookup_entry(request.fixture_key)
                if entry is None:
                    self.fixture_miss_sink(
                        FixtureMissEvidence(
                            schema_version=1,
                            context=context,
                            fixture_key=request.fixture_key,
                            canonical_tool_name=request.canonical_tool_name,
                            backend_version=request.backend_version,
                            effective_arguments=request.effective_arguments,
                        )
                    )
                    self._record(
                        attempt_id, context, request, started, started_at,
                        status="fixture_miss", dispatched=False, status_code=None,
                        response_body_sha256=None, sanitized_exception_class="FixtureMiss",
                    )
                    raise ExternalFixtureMiss()
                self._record(
                    attempt_id, context, request, started, started_at,
                    status="fixture_hit", dispatched=False, status_code=entry.status_code,
                    response_body_sha256=entry.response_body_sha256, sanitized_exception_class=None,
                )
                return entry.model_copy(deep=True).normalized_response_body
            return self._dispatch_live(context, request, started, started_at, attempt_id)
        except (
            ExternalFixtureMiss,
            ExternalReadUnknownOutcome,
            ExternalReadFailed,
            RapidAPIBoundaryError,
        ):
            raise
        except Exception as exc:
            if request is not None:
                self._record(
                    attempt_id, context, request, started, started_at,
                    status="rejected_before_dispatch", dispatched=False, status_code=None,
                    response_body_sha256=None, sanitized_exception_class=type(exc).__name__,
                )
            raise RapidAPIBoundaryError("EXTERNAL_READ_REJECTED") from exc

    def _dispatch_live(
        self,
        context: ExternalReadContext,
        request: FixtureRequest,
        started: float,
        started_at: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        assert self.transport is not None and self.credential_provider is not None
        record = next(
            item for item in self.backend_manifest.backends
            if item.canonical_tool_name == request.canonical_tool_name
        )
        try:
            credential = self.credential_provider()
            if type(credential) is not str or not credential:
                raise PermissionError("credential unavailable")
        except Exception as exc:
            self._record(
                attempt_id, context, request, started, started_at,
                status="rejected_before_dispatch", dispatched=False, status_code=None,
                response_body_sha256=None, sanitized_exception_class=type(exc).__name__,
            )
            raise RapidAPIBoundaryError("EXTERNAL_READ_REJECTED") from exc
        try:
            response = self.transport.get(
                record.url,
                params=dict(request.effective_arguments),
                headers={"X-RapidAPI-Host": record.host, "X-RapidAPI-Key": credential},
                timeout=self.backend_manifest.request_timeout_seconds,
                allow_redirects=self.backend_manifest.allow_redirects,
            )
        except Exception as exc:
            sanitized_class, unknown = classify_transport_exception(exc)
            self._record(
                attempt_id, context, request, started, started_at,
                status="unknown_outcome" if unknown else "failed", dispatched=True,
                status_code=None, response_body_sha256=None,
                sanitized_exception_class=sanitized_class,
            )
            if unknown:
                raise ExternalReadUnknownOutcome() from exc
            raise ExternalReadFailed() from exc
        try:
            status_code, body, body_hash, byte_count = normalize_response(response)
        except Exception as exc:
            self._record(
                attempt_id, context, request, started, started_at,
                status="failed", dispatched=True,
                status_code=getattr(response, "status_code", None), response_body_sha256=None,
                sanitized_exception_class=type(exc).__name__,
            )
            raise ExternalReadFailed() from exc
        if status_code != 200:
            self._record(
                attempt_id, context, request, started, started_at,
                status="failed", dispatched=True, status_code=status_code,
                response_body_sha256=body_hash, sanitized_exception_class="HTTPStatusError",
                response_byte_count=byte_count,
            )
            raise ExternalReadFailed()
        self._record(
            attempt_id, context, request, started, started_at,
            status="completed", dispatched=True, status_code=200,
            response_body_sha256=body_hash, sanitized_exception_class=None,
            response_byte_count=byte_count,
        )
        return body

    def _record(
        self,
        attempt_id: str,
        context: ExternalReadContext,
        request: FixtureRequest,
        started: float,
        started_at: str,
        *,
        status: str,
        dispatched: bool,
        status_code: int | None,
        response_body_sha256: str | None,
        sanitized_exception_class: str | None,
        response_byte_count: int | None = None,
    ) -> None:
        ended = self.monotonic()
        self.attempt_sink(
            ExternalReadAttempt(
                schema_version=1,
                attempt_id=attempt_id,
                context=context,
                mode=self.mode,
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
                started_at_utc=started_at,
                ended_at_utc=self.utc_now(),
                latency_seconds=float(max(0.0, ended - started)),
            )
        )
