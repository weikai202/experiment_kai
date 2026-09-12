"""One-dispatch frozen Qwen roles for offline memory candidate and review."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import ConfigDict, Field

from toolsandbox_pipeline.offline.memory_prompts import (
    MemoryPromptSet,
    candidate_envelope,
    prompt_input_fingerprint,
    review_envelope,
)
from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    PhysicalAttemptResult,
    PhysicalAttemptStatus,
    ProviderRole,
    RequestContext,
)
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryReviewOutput,
    PolicyMemoryCandidate,
    PolicyMemoryCandidateDecision,
    PolicyTrajectoryProjection,
    WorldMemoryCandidate,
    WorldMemoryCandidateDecision,
    WorldTrajectoryProjection,
)
from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig


class OfflineMemoryRoleError(RuntimeError):
    pass


class OfflineMemoryTokenLimits(StrictModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1]
    status: Literal["provisional", "calibrated"]
    memory_candidate: Annotated[int, Field(gt=0)]
    memory_review: Annotated[int, Field(gt=0)]


class OfflinePromptMessage(StrictModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["system", "user"]
    content: Annotated[str, Field(min_length=1)]


class PreparedMemoryRequest(StrictModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["memory_candidate", "memory_review"]
    memory_role: Literal["policy", "world"]
    messages: tuple[OfflinePromptMessage, OfflinePromptMessage]
    output_model_name: Literal[
        "PolicyMemoryCandidateDecision",
        "WorldMemoryCandidateDecision",
        "MemoryReviewOutput",
    ]
    output_schema_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    structured_output_wire_mode: Literal["guided_json", "structured_outputs_json"]
    max_tokens: Annotated[int, Field(gt=0)]
    token_limit_config_status: Literal["provisional", "calibrated"]
    token_limit_config_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    unit_reference: Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
    prompt_version: Literal["v1", "v2"]
    prompt_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    user_envelope_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    canonical_input_fingerprint: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


@dataclass(frozen=True)
class MemoryRoleCallResult:
    prepared_request: PreparedMemoryRequest
    attempt: PhysicalAttemptResult
    output: object = field(repr=False)


class OfflineQwenResponseDurabilitySeam(Protocol):
    def load_completed_response(self, context: RequestContext) -> GatewayResponse | None: ...

    def persist_completed_response(self, response: GatewayResponse) -> None: ...


def load_token_limits(path: Path) -> tuple[OfflineMemoryTokenLimits, str]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OfflineMemoryRoleError("absolute token-limit config required")
    raw = path.read_bytes()
    try:
        limits = OfflineMemoryTokenLimits.model_validate_json(raw)
    except Exception as error:
        raise OfflineMemoryRoleError("invalid offline memory token limits") from error
    if limits.status == "provisional" and (
        limits.memory_candidate != 512 or limits.memory_review != 256
    ):
        raise OfflineMemoryRoleError("provisional memory limits are frozen")
    if limits.status == "calibrated" and (
        limits.memory_candidate % 64 or limits.memory_review % 64
    ):
        raise OfflineMemoryRoleError("calibrated memory limits must be multiples of 64")
    return limits, "sha256:" + sha256(raw).hexdigest()


def _prepared(
    *,
    role: Literal["memory_candidate", "memory_review"],
    memory_role: Literal["policy", "world"],
    prompt: str,
    prompt_sha256: str,
    envelope: str,
    output_model: type[StrictModel],
    unit_reference: str,
    limits: OfflineMemoryTokenLimits,
    limits_sha256: str,
    structured_output_wire_mode: str,
    prompt_version: Literal["v1", "v2"] = "v1",
) -> PreparedMemoryRequest:
    maximum = limits.memory_candidate if role == "memory_candidate" else limits.memory_review
    schema_sha = canonical_sha256(output_model.model_json_schema())
    fingerprint = prompt_input_fingerprint(
        role=role,
        prompt_sha256=prompt_sha256,
        user_envelope=envelope,
        output_schema_sha256=schema_sha,
        max_tokens=maximum,
        structured_output_wire_mode=structured_output_wire_mode,
    )
    return PreparedMemoryRequest(
        role=role,
        memory_role=memory_role,
        messages=(
            OfflinePromptMessage(role="system", content=prompt),
            OfflinePromptMessage(role="user", content=envelope),
        ),
        output_model_name=output_model.__name__,
        output_schema_sha256=schema_sha,
        structured_output_wire_mode=structured_output_wire_mode,
        max_tokens=maximum,
        token_limit_config_status=limits.status,
        token_limit_config_sha256=limits_sha256,
        unit_reference=unit_reference,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        user_envelope_sha256=canonical_sha256(json.loads(envelope)),
        canonical_input_fingerprint=fingerprint,
    )


def prepare_candidate_request(
    projection: PolicyTrajectoryProjection | WorldTrajectoryProjection,
    *,
    unit_reference: str,
    prompts: MemoryPromptSet,
    limits: OfflineMemoryTokenLimits,
    limits_sha256: str,
    structured_output_wire_mode: str,
    input_representation: Literal["v1", "packed-v2"] = "v1",
) -> PreparedMemoryRequest:
    if type(projection) is PolicyTrajectoryProjection:
        memory_role = "policy"
        output_model = PolicyMemoryCandidateDecision
    elif type(projection) is WorldTrajectoryProjection:
        memory_role = "world"
        output_model = WorldMemoryCandidateDecision
    else:
        raise TypeError("strict trajectory projection required")
    prompt = prompts.candidate
    prompt_sha = prompts.candidate_sha256
    envelope = candidate_envelope(projection)
    version = "v1"
    if input_representation == "packed-v2":
        from .reflection_packing import NOTICE, candidate_input
        envelope = candidate_input(projection)
        prompt = NOTICE + prompt
        prompt_sha = "sha256:" + sha256(prompt.encode("utf-8")).hexdigest()
        version = "v2"
    elif input_representation != "v1":
        raise ValueError("unknown reflection input representation")
    return _prepared(
        prompt_version=version,
        role="memory_candidate",
        memory_role=memory_role,
        prompt=prompt,
        prompt_sha256=prompt_sha,
        envelope=envelope,
        output_model=output_model,
        unit_reference=unit_reference,
        limits=limits,
        limits_sha256=limits_sha256,
        structured_output_wire_mode=structured_output_wire_mode,
    )


def prepare_review_request(
    candidate: PolicyMemoryCandidate | WorldMemoryCandidate,
    matches: tuple[PolicyMemory | WorldMemory, ...],
    *,
    unit_reference: str,
    prompts: MemoryPromptSet,
    limits: OfflineMemoryTokenLimits,
    limits_sha256: str,
    structured_output_wire_mode: str,
) -> PreparedMemoryRequest:
    return _prepared(
        role="memory_review",
        memory_role=candidate.role,
        prompt=prompts.review,
        prompt_sha256=prompts.review_sha256,
        envelope=review_envelope(candidate, matches),
        output_model=MemoryReviewOutput,
        unit_reference=unit_reference,
        limits=limits,
        limits_sha256=limits_sha256,
        structured_output_wire_mode=structured_output_wire_mode,
    )


class MemoryRoleRunner:
    """Validate one prepared request, dispatch once, persist, then expose output."""

    def __init__(self, gateway, *, manifest_identity: str, durability_seam=None):
        self.gateway = gateway
        self.manifest_identity = manifest_identity
        self.durability_seam: OfflineQwenResponseDurabilitySeam | None = durability_seam

    def run(
        self, prepared: PreparedMemoryRequest, context: RequestContext
    ) -> MemoryRoleCallResult:
        if type(prepared) is not PreparedMemoryRequest or type(context) is not RequestContext:
            raise TypeError("prepared request and RequestContext required")
        expected_role = ProviderRole(prepared.role)
        if (
            context.role is not expected_role
            or context.unit_reference != prepared.unit_reference
            or context.input_fingerprint != prepared.canonical_input_fingerprint
            or context.manifest_identity != self.manifest_identity
        ):
            raise OfflineMemoryRoleError("offline request context mismatch")
        output_model = {
            "PolicyMemoryCandidateDecision": PolicyMemoryCandidateDecision,
            "WorldMemoryCandidateDecision": WorldMemoryCandidateDecision,
            "MemoryReviewOutput": MemoryReviewOutput,
        }[prepared.output_model_name]
        expected_output_name = (
            "MemoryReviewOutput"
            if prepared.role == "memory_review"
            else (
                "PolicyMemoryCandidateDecision"
                if prepared.memory_role == "policy"
                else "WorldMemoryCandidateDecision"
            )
        )
        expected_fingerprint = prompt_input_fingerprint(
            role=prepared.role,
            prompt_sha256=prepared.prompt_sha256,
            user_envelope=prepared.messages[1].content,
            output_schema_sha256=prepared.output_schema_sha256,
            max_tokens=prepared.max_tokens,
            structured_output_wire_mode=prepared.structured_output_wire_mode,
        )
        if (
            prepared.output_model_name != expected_output_name
            or prepared.output_schema_sha256
            != canonical_sha256(output_model.model_json_schema())
            or prepared.prompt_sha256
            != "sha256:" + sha256(prepared.messages[0].content.encode("utf-8")).hexdigest()
            or prepared.user_envelope_sha256
            != canonical_sha256(json.loads(prepared.messages[1].content))
            or prepared.canonical_input_fingerprint != expected_fingerprint
            or prepared.structured_output_wire_mode
            != self.gateway.config.structured_output_wire_mode
        ):
            raise OfflineMemoryRoleError("offline prepared request mismatch")
        response = None
        if self.durability_seam is not None:
            response = self.durability_seam.load_completed_response(context)
        if response is None:
            selected_limit = RoleTokenLimitConfig(
                version="offline-memory-v1",
                role=prepared.role,
                stage="bootstrap" if prepared.token_limit_config_status == "provisional" else "calibrated",
                max_tokens=prepared.max_tokens,
                evidence_manifest_identity=(
                    None if prepared.token_limit_config_status == "provisional" else prepared.token_limit_config_sha256
                ),
            )
            response = self.gateway.generate(
                context,
                [message.model_dump() for message in prepared.messages],
                output_model,
                max_tokens=prepared.max_tokens,
                token_limit_config=selected_limit,
            )
            self._validate_response(response, context, output_model)
            if self.durability_seam is not None:
                self.durability_seam.persist_completed_response(response)
        else:
            self._validate_response(response, context, output_model)
        return MemoryRoleCallResult(
            prepared_request=prepared,
            attempt=response.attempt,
            output=response.value,
        )

    def _validate_response(self, response, context, output_model) -> None:
        if not isinstance(response, GatewayResponse):
            raise OfflineMemoryRoleError("invalid completed offline response")
        attempt = response.attempt
        saved = attempt.context
        stable_current = (
            context.logical_request_id,
            context.role,
            context.phase,
            context.unit_reference,
            context.input_fingerprint,
            context.manifest_identity,
        )
        stable_saved = (
            saved.logical_request_id,
            saved.role,
            saved.phase,
            saved.unit_reference,
            saved.input_fingerprint,
            saved.manifest_identity,
        )
        raw = response.raw_response_body
        expected_model = self.gateway.config.served_model_id or self.gateway.config.model
        if (
            attempt.status is not PhysicalAttemptStatus.COMPLETED
            or attempt.finish_reason != "stop"
            or attempt.model != expected_model
            or attempt.returned_model != expected_model
            or stable_current != stable_saved
            or type(raw) is not bytes
            or attempt.response_hash != "sha256:" + sha256(raw).hexdigest()
            or type(response.value) is not output_model
        ):
            raise OfflineMemoryRoleError("invalid completed offline response")


__all__ = [
    "MemoryRoleCallResult",
    "MemoryRoleRunner",
    "OfflineMemoryRoleError",
    "OfflineMemoryTokenLimits",
    "OfflineQwenResponseDurabilitySeam",
    "PreparedMemoryRequest",
    "load_token_limits",
    "prepare_candidate_request",
    "prepare_review_request",
]
