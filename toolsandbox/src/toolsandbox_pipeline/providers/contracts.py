"""One-dispatch transport boundary. Identity and retry authority stay with callers."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from time import monotonic
from typing import Annotated, Any, Callable, Literal, Protocol

from pydantic import ConfigDict, Field
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.usage import PhysicalAttemptMetrics, TokenUsage
from toolsandbox_pipeline.metrics.usage import normalize_usage


class ProviderRole(str, Enum):
    VANILLA = "vanilla"
    POLICY = "policy"
    CRITIC = "critic"
    REVISION = "revision"
    MEMORY_CANDIDATE = "memory_candidate"
    MEMORY_REVIEW = "memory_review"
    FAILURE_MODE_UPDATE = "failure_mode_update"
    SKILL_CANDIDATE = "skill_candidate"
    EMBEDDING = "embedding"
    USER_SIMULATOR = "user_simulator"


ROLE_BOOTSTRAP_MAX_TOKENS = {
    ProviderRole.VANILLA: 256, ProviderRole.POLICY: 256, ProviderRole.CRITIC: 384,
    ProviderRole.REVISION: 256, ProviderRole.MEMORY_CANDIDATE: 512,
    ProviderRole.MEMORY_REVIEW: 256, ProviderRole.FAILURE_MODE_UPDATE: 512,
    ProviderRole.SKILL_CANDIDATE: 2048,
}
QWEN_GENERATION_ROLES = frozenset(ROLE_BOOTSTRAP_MAX_TOKENS)
Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Fingerprint = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class RequestContext(StrictModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    logical_request_id: Identifier
    attempt_id: Identifier
    role: ProviderRole
    phase: Identifier
    unit_reference: Identifier
    input_fingerprint: Fingerprint
    replayed_after_unknown_outcome: bool
    manifest_identity: Fingerprint


class PhysicalAttemptStatus(str, Enum):
    COMPLETED = "completed"
    REJECTED = "rejected_before_dispatch"
    FAILED = "failed"
    UNKNOWN_OUTCOME = "unknown_outcome"


class PhysicalAttemptResult(StrictModel):
    model_config = ConfigDict(frozen=True)
    context: RequestContext
    status: PhysicalAttemptStatus
    model: str
    returned_model: str | None = None
    finish_reason: Literal["stop", "length"] | None = None
    response_hash: Fingerprint | None = None
    exception_class: str | None = None
    metrics: PhysicalAttemptMetrics


@dataclass(frozen=True)
class TransportResponse:
    """Raw HTTP body for hashing plus the SDK response for upstream compatibility."""
    raw_body: bytes = field(repr=False)
    data: Any = field(repr=False)


@dataclass(frozen=True)
class GatewayResponse:
    value: Any = field(repr=False)
    attempt: PhysicalAttemptResult
    raw_response_body: bytes = field(repr=False)


@dataclass(frozen=True)
class ValidatedProviderResponse:
    """Validated value plus sanitized provider metadata retained for auditing."""
    value: Any = field(repr=False)
    finish_reason: Literal["stop", "length"] | None = None


class ChatTransport(Protocol):
    def create(self, **request: Any) -> TransportResponse: ...


class EmbeddingTransport(Protocol):
    def create(self, **request: Any) -> TransportResponse: ...


class RejectedBeforeDispatch(Exception):
    """A transport may raise this only when it proves no request was sent."""


class ProviderRequestError(Exception):
    def __init__(self, attempt: PhysicalAttemptResult, raw_response_body: bytes | None = None):
        if raw_response_body is not None and type(raw_response_body) is not bytes:
            raise TypeError("raw response body must be bytes")
        self.attempt = attempt
        self._raw_response_body = raw_response_body
        super().__init__(attempt.exception_class or "ProviderRequestError")

    @property
    def raw_response_body(self) -> bytes | None:
        """Restricted response bytes; deliberately absent from exception text/repr."""
        return self._raw_response_body


class OutputTruncated(Exception):
    """A provider explicitly stopped because the selected output ceiling was hit."""


def response_mapping(data):
    value = data if isinstance(data, dict) else data.model_dump(mode="json")
    if not isinstance(value, dict):
        raise ValueError("invalid provider response")
    return value


def physical_request(context: RequestContext, model: str, *, prepare: Callable,
                     validate: Callable, embedding=False, recorder=None) -> GatewayResponse:
    """Prepare locally, call once, validate, and retain metrics even on failure.

    Exception bodies are deliberately discarded. Only fixed public categories
    cross this boundary, including when client construction fails.
    """
    if not isinstance(context, RequestContext):
        raise TypeError("validated RequestContext required")
    if recorder is not None and recorder.role != context.role:
        raise ValueError("recorder role mismatch")
    usage = TokenUsage()
    response_hash = returned_model = finish_reason = exception_class = None
    raw_response_body = None
    value = None
    start_utc = datetime.now(timezone.utc)
    end_utc = start_utc
    elapsed = 0.0
    status = PhysicalAttemptStatus.REJECTED
    try:
        transport, request = prepare()
    except Exception as error:
        exception_class = "RequestPreparationError"
        from toolsandbox_pipeline.providers.openai_clients import MissingCredentialError
        if isinstance(error, MissingCredentialError) and str(error) in ("OPENAI_API_KEY", "QWEN_API_KEY", "QWEN_BASE_URL"):
            exception_class = str(error)
    else:
        start_utc = datetime.now(timezone.utc)
        start = monotonic()
        try:
            response = transport.create(**request)
        except RejectedBeforeDispatch:
            exception_class = "RejectedBeforeDispatch"
        except Exception as error:
            status = PhysicalAttemptStatus.UNKNOWN_OUTCOME
            name = type(error).__name__
            exception_class = name if name in ("TimeoutError", "ConnectionError", "APITimeoutError", "APIConnectionError",
                                               "BadRequestError", "AuthenticationError", "NotFoundError",
                                               "RateLimitError", "InternalServerError", "PermissionDeniedError") else "TransportError"
        else:
            status = PhysicalAttemptStatus.FAILED
        finally:
            elapsed = monotonic() - start
            end_utc = datetime.now(timezone.utc)
        if exception_class is None:
            try:
                if not isinstance(response, TransportResponse) or type(response.raw_body) is not bytes:
                    raise ValueError("raw transport response required")
                raw_response_body = response.raw_body
                response_hash = "sha256:" + sha256(raw_response_body).hexdigest()
                mapping = response_mapping(response.data)
                # Untrusted model strings are not propagated to logs/preflights.
                if mapping.get("model") == model:
                    returned_model = model
                choices = mapping.get("choices")
                if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
                    if choices[0].get("finish_reason") in ("stop", "length"):
                        finish_reason = choices[0]["finish_reason"]
                usage = normalize_usage(mapping.get("usage"), embedding=embedding)
                validated = validate(mapping, response.data)
                if isinstance(validated, ValidatedProviderResponse):
                    value = validated.value
                    finish_reason = validated.finish_reason
                else:
                    value = validated
                status = PhysicalAttemptStatus.COMPLETED
            except OutputTruncated:
                finish_reason = "length"
                exception_class = "OutputTruncated"
            except Exception:
                exception_class = "ProviderOutputError"
    attempt = PhysicalAttemptResult(
        context=context, status=status, model=model, returned_model=returned_model,
        finish_reason=finish_reason, response_hash=response_hash,
        exception_class=exception_class,
        metrics=PhysicalAttemptMetrics(started_at=start_utc, completed_at=end_utc,
                                       latency_seconds=elapsed, usage=usage),
    )
    if recorder is not None:
        recorder.record(attempt)
    if exception_class is not None:
        raise ProviderRequestError(attempt, raw_response_body) from None
    assert raw_response_body is not None
    return GatewayResponse(
        value=value, attempt=attempt, raw_response_body=raw_response_body
    )
