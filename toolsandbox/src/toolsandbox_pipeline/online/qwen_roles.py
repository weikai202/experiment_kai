"""Exactly one gateway call per prepared online request."""
from hashlib import sha256
from typing import Protocol

from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    PhysicalAttemptStatus,
    ProviderRequestError,
    ProviderRole,
    RequestContext,
)
from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.critic import CriticOutput
from .prompt_builder import fingerprint
from .prompt_contracts import PreparedRoleRequest, RoleCallResult


class QwenResponseDurabilitySeam(Protocol):
    """Caller-owned completed-response storage used by online recovery."""

    def load_completed_response(
        self, context: RequestContext
    ) -> GatewayResponse | None: ...

    def persist_completed_response(self, response: GatewayResponse) -> None: ...


class QwenResponseDurabilityError(RuntimeError):
    """Sanitized failure at the injected durability boundary."""


class _Runner:
    role = None

    def __init__(self, gateway, *, manifest_identity, mode="offline",
                 durability_seam=None):
        if mode not in ("offline", "formal", "calibration"):
            raise ValueError("invalid runner mode")
        self.gateway = gateway
        self.manifest_identity = manifest_identity
        self.mode = mode
        self._durability_seam: QwenResponseDurabilitySeam | None = durability_seam
        self._revised_states = {}

    def run(self, prepared, context):
        if type(prepared) is not PreparedRoleRequest or type(context) is not RequestContext:
            raise TypeError("prepared request and RequestContext required")
        prepared = PreparedRoleRequest.model_validate(prepared)
        if (prepared.role, context.role.value, context.unit_reference, context.input_fingerprint, context.manifest_identity) != (
            self.role, self.role, prepared.state_id, prepared.canonical_input_fingerprint, self.manifest_identity
        ):
            raise ValueError("request role/state/fingerprint/manifest mismatch")
        if self.mode == "formal" and prepared.token_limit_config_status != "calibrated":
            raise ValueError("formal request requires promoted calibration")
        output_model = CriticOutput if self.role == "critic" else ActionEnvelope
        schema_hash = canonical_sha256(output_model.model_json_schema())
        if prepared.output_model_name != output_model.__name__ or prepared.output_schema_sha256 != schema_hash:
            raise ValueError("output schema identity mismatch")
        if prepared.canonical_input_fingerprint != fingerprint(self.role, prepared.messages, schema_hash, prepared.max_tokens, self.gateway.config):
            raise ValueError("prepared request fingerprint mismatch")
        identity = (prepared.generation_id, prepared.state_id)
        if self.role == "revision" and self.mode != "calibration":
            previous_logical_request_id = self._revised_states.get(identity)
            if (
                previous_logical_request_id is not None
                and previous_logical_request_id != context.logical_request_id
            ):
                raise ValueError("second Revision forbidden")
            self._revised_states[identity] = context.logical_request_id

        response = self._load_completed_response(context)
        if response is None:
            try:
                response = self.gateway.generate(
                    context,
                    [m.model_dump() for m in prepared.messages],
                    output_model,
                    max_tokens=prepared.max_tokens,
                    token_limit_config=prepared.selected_limit,
                )
            except ProviderRequestError as error:
                if (
                    self.mode == "calibration"
                    and error.attempt.exception_class == "OutputTruncated"
                    and error.attempt.metrics.usage.usage_complete
                ):
                    return RoleCallResult(
                        prepared_request=prepared,
                        attempt=error.attempt,
                        output_json=None,
                        truncated=True,
                    )
                raise
            self._validate_completed_response(context, response, output_model)
            self._persist_completed_response(response)
        else:
            self._validate_completed_response(context, response, output_model)

        if not response.attempt.metrics.usage.usage_complete:
            raise ProviderRequestError(response.attempt)
        return RoleCallResult(
            prepared_request=prepared,
            attempt=response.attempt,
            output_json=response.value.model_dump_json(),
        )

    def _load_completed_response(
        self, context: RequestContext
    ) -> GatewayResponse | None:
        if self._durability_seam is None:
            return None
        try:
            return self._durability_seam.load_completed_response(context)
        except Exception:
            raise QwenResponseDurabilityError(
                "completed Qwen response lookup failed"
            ) from None

    def _persist_completed_response(self, response: GatewayResponse) -> None:
        if self._durability_seam is None:
            return
        try:
            self._durability_seam.persist_completed_response(response)
        except Exception:
            raise QwenResponseDurabilityError(
                "completed Qwen response persistence failed"
            ) from None

    def _validate_completed_response(
        self, context: RequestContext, response: GatewayResponse, output_model
    ) -> None:
        if not isinstance(response, GatewayResponse):
            raise QwenResponseDurabilityError("invalid completed Qwen response")
        attempt = response.attempt
        saved_context = attempt.context
        expected_model = self.gateway.config.served_model_id or self.gateway.config.model
        current_identity = (
            context.logical_request_id,
            context.role,
            context.phase,
            context.unit_reference,
            context.input_fingerprint,
            context.manifest_identity,
        )
        saved_identity = (
            saved_context.logical_request_id,
            saved_context.role,
            saved_context.phase,
            saved_context.unit_reference,
            saved_context.input_fingerprint,
            saved_context.manifest_identity,
        )
        raw_body = response.raw_response_body
        if (
            attempt.status is not PhysicalAttemptStatus.COMPLETED
            or attempt.model != expected_model
            or attempt.returned_model != expected_model
            or attempt.finish_reason != "stop"
            or current_identity != saved_identity
            or type(raw_body) is not bytes
            or attempt.response_hash
            != "sha256:" + sha256(raw_body).hexdigest()
            or type(response.value) is not output_model
        ):
            raise QwenResponseDurabilityError("invalid completed Qwen response")
        # Re-run the exact strict local contract over the stored validated value.
        try:
            output_model.model_validate_json(response.value.model_dump_json())
            if output_model is ActionEnvelope:
                from toolsandbox_pipeline.toolsandbox_adapter.action_decode import audit_provider_action
                audit_provider_action(response, allow_legacy=True)
        except Exception:
            raise QwenResponseDurabilityError(
                "invalid completed Qwen response"
            ) from None


class InitialPolicyRunner(_Runner):
    role = "policy"


class CriticRunner(_Runner):
    role = "critic"


class RevisionRunner(_Runner):
    role = "revision"
