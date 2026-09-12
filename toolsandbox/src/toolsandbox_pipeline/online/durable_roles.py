"""Validated durable-role boundary used by online orchestration.

The low-level Task 011 transaction adapter is injected so this module never owns
retry policy or a default checkpoint location.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from toolsandbox_pipeline.checkpointing import (
    LLMLedger,
    LLMRecoveryAction,
    LogicalLLMRequestIdentity,
    LogicalRequestStatus,
    QwenEffectKind,
)
from toolsandbox_pipeline.online.prompt_contracts import PreparedRoleRequest
from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    ProviderRequestError,
    ProviderRole,
)
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.critic import CriticOutput
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnStage
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope


@dataclass(frozen=True)
class DurableRoleExecution:
    prepared_request: PreparedRoleRequest
    output: ActionEnvelope | CriticOutput
    logical_request_id: str
    source_attempt_id: str
    reused_completed_response: bool = False

    def __post_init__(self) -> None:
        expected = CriticOutput if self.prepared_request.role == "critic" else ActionEnvelope
        if type(self.output) is not expected:
            raise TypeError("durable role output type mismatch")
        if not self.logical_request_id or not self.source_attempt_id:
            raise ValueError("durable role identities must be non-empty")


@dataclass(frozen=True)
class DurableRoleApplication:
    execution: DurableRoleExecution
    application_id: str
    committed_checkpoint_id: str

    def __post_init__(self) -> None:
        if not self.application_id or not self.committed_checkpoint_id:
            raise ValueError("application identities must be non-empty")


@dataclass(frozen=True)
class DurableOnlineActionCommit:
    effect_id: str
    committed_checkpoint_id: str

    def __post_init__(self) -> None:
        if not self.effect_id or not self.committed_checkpoint_id:
            raise ValueError("online action commit identities must be non-empty")


class DurableRoleBackend(Protocol):
    """Task 011 adapter: one recovery decision and at most one dispatch."""

    def execute(self, prepared: PreparedRoleRequest) -> DurableRoleExecution: ...

    def apply(
        self,
        execution: DurableRoleExecution,
        *,
        stage: OnlineTurnStage,
        state_id: str,
        application_payload: object,
    ) -> DurableRoleApplication: ...

    def commit_online_action(
        self,
        *,
        checkpoint_id: str,
        action: ActionEnvelope,
        application_ids: tuple[str, ...],
        checkpoint_payload: object,
    ) -> DurableOnlineActionCommit: ...


class OnlineRoleRunner(Protocol):
    def run(self, prepared: PreparedRoleRequest, context): ...


def _json(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)  # type: ignore[union-attr]
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    return value


def _payload_dict(value: object) -> dict:
    converted = _json(value)
    if not isinstance(converted, dict):
        raise TypeError("durable checkpoint payload must be a JSON object")
    canonical_sha256(converted)
    return converted


class LedgerQwenResponseSeam:
    """Exact Task 008 durability seam backed by Task 011 verified blobs."""

    def __init__(self, ledger: LLMLedger):
        if type(ledger) is not LLMLedger:
            raise TypeError("LLMLedger required")
        self.ledger = ledger

    def load_completed_response(self, context) -> GatewayResponse | None:
        if context.role not in (
            ProviderRole.POLICY,
            ProviderRole.CRITIC,
            ProviderRole.REVISION,
        ):
            raise ValueError("online Qwen role required")
        status = self.ledger.request_status(context.logical_request_id)
        if status is LogicalRequestStatus.PREPARED:
            return None
        if status is LogicalRequestStatus.TERMINAL_FAILURE:
            raise RuntimeError("terminal logical request cannot be reused")
        material = self.ledger.load_completed_response_material(
            context.logical_request_id
        )
        stored_context = material.attempt.context
        stable_current = (
            context.logical_request_id,
            context.role,
            context.phase,
            context.unit_reference,
            context.input_fingerprint,
            context.manifest_identity,
        )
        stable_stored = (
            stored_context.logical_request_id,
            stored_context.role,
            stored_context.phase,
            stored_context.unit_reference,
            stored_context.input_fingerprint,
            stored_context.manifest_identity,
        )
        if stable_stored != stable_current:
            raise RuntimeError("stored response context mismatch")
        model = CriticOutput if context.role is ProviderRole.CRITIC else ActionEnvelope
        value = model.model_validate(material.validated_output, strict=True)
        response = GatewayResponse(value=value, attempt=material.attempt,
                                   raw_response_body=material.raw_response_body)
        self._persist_call_identity(response, allow_legacy=True)
        return response

    def persist_completed_response(self, response: GatewayResponse) -> None:
        if type(response) is not GatewayResponse:
            raise TypeError("GatewayResponse required")
        value = response.value
        if type(value) not in (ActionEnvelope, CriticOutput):
            raise TypeError("online role output required")
        self._persist_call_identity(response, allow_legacy=False)
        self.ledger.complete_response(
            response,
            validated_output=value.model_dump(mode="json"),
            output_schema_name=type(value).__name__,
            output_schema_version=1,
        )


    def _persist_call_identity(self, response, *, allow_legacy):
        from toolsandbox_pipeline.toolsandbox_adapter.action_decode import audit_provider_action
        audit = audit_provider_action(response, allow_legacy=allow_legacy)
        if audit is not None:
            checkpoint_id = 'action-call-identity-' + canonical_sha256([
                audit['version'], audit['logical_request_id'], audit['source_attempt_id']])[7:]
            self.ledger.commit_checkpoint(checkpoint_id, 'action_call_identity', audit)


class Task011CheckpointEventSink:
    """Store non-application online stages through the Task 011 ledger."""

    def __init__(self, ledger: LLMLedger):
        if type(ledger) is not LLMLedger:
            raise TypeError("LLMLedger required")
        self.ledger = ledger

    def load_checkpoint(self, checkpoint_id: str) -> tuple[str, object] | None:
        event = self.ledger.get_checkpoint(checkpoint_id)
        if event is None:
            return None
        return event.event_kind, event.payload

    def commit_stage(
        self,
        *,
        identity,
        state_id: str,
        stage: OnlineTurnStage,
        component_payload: object,
    ) -> str:
        run_identity = self.ledger.store.identity
        if (
            identity.run_id != run_identity.run_id
            or identity.environment_identity != run_identity.environment_identity
            or identity.dataset_manifest_sha256
            != run_identity.dataset_manifest_sha256
            or identity.prompt_manifest_sha256 != run_identity.prompt_manifest_sha256
            or identity.fixture_manifest_sha256 != run_identity.fixture_manifest_sha256
        ):
            raise ValueError("turn/checkpoint run identity mismatch")
        payload = {
            "identity": identity.model_dump(mode="json"),
            "state_id": state_id,
            "stage": stage.value,
            "component": _json(component_payload),
        }
        checkpoint_id = "checkpoint-" + canonical_sha256(payload)[7:]
        event = self.ledger.commit_checkpoint(checkpoint_id, stage.value, payload)
        return event.checkpoint_id


class Task011DurableRoleBackend:
    """One Task 011 recovery decision plus one Task 008 runner invocation."""

    def __init__(
        self,
        *,
        ledger: LLMLedger,
        runners: Mapping[str, OnlineRoleRunner],
        run_id: str,
        phase: str,
        qwen_model: str,
        decoding_configuration_sha256: str,
        manifest_identity: str,
        accounting_scope: AccountingScope,
    ) -> None:
        if type(ledger) is not LLMLedger:
            raise TypeError("LLMLedger required")
        if set(runners) != {"policy", "critic", "revision"}:
            raise ValueError("exact Policy/Critic/Revision runners required")
        if any(getattr(runner, "role", role) != role for role, runner in runners.items()):
            raise ValueError("role runner mapping mismatch")
        if ledger.store.identity.run_id != run_id:
            raise ValueError("durable backend run identity mismatch")
        if (
            type(accounting_scope) is not AccountingScope
            or accounting_scope.run_id != run_id
        ):
            raise ValueError("durable backend accounting scope mismatch")
        if qwen_model != "Qwen/Qwen3-32B":
            raise ValueError("frozen Qwen model required")
        for value in (run_id, phase, qwen_model):
            if type(value) is not str or not value or any(c.isspace() for c in value):
                raise ValueError("invalid durable role identity")
        for value in (decoding_configuration_sha256, manifest_identity):
            if (
                type(value) is not str
                or not value.startswith("sha256:")
                or len(value) != 71
                or any(c not in "0123456789abcdef" for c in value[7:])
            ):
                raise ValueError("invalid durable role digest")
        self.ledger = ledger
        self.runners = dict(runners)
        self.run_id = run_id
        self.phase = phase
        self.qwen_model = qwen_model
        self.decoding_configuration_sha256 = decoding_configuration_sha256
        self.manifest_identity = manifest_identity
        self.accounting_scope = accounting_scope

    def _identity(self, prepared: PreparedRoleRequest) -> LogicalLLMRequestIdentity:
        return LogicalLLMRequestIdentity(
            run_id=self.run_id,
            role=ProviderRole(prepared.role),
            phase=self.phase,
            unit_reference=prepared.state_id,
            input_fingerprint=prepared.canonical_input_fingerprint,
            model=self.qwen_model,
            decoding_configuration_sha256=self.decoding_configuration_sha256,
            output_schema_sha256=prepared.output_schema_sha256,
        )

    def execute(self, prepared: PreparedRoleRequest) -> DurableRoleExecution:
        record = self.ledger.prepare_request(self._identity(prepared))
        self.ledger.bind_accounting_scope(
            record.logical_request_id, self.accounting_scope
        )
        plan = self.ledger.plan_recovery(record.logical_request_id)
        reused = plan.action in (
            LLMRecoveryAction.APPLY_STORED_RESPONSE,
            LLMRecoveryAction.RESTORE_APPLIED_CHECKPOINT,
        )
        if reused:
            material = self.ledger.load_completed_response_material(
                record.logical_request_id
            )
            request_context = material.attempt.context
        elif plan.action in (
            LLMRecoveryAction.DISPATCH_FIRST_ATTEMPT,
            LLMRecoveryAction.DISPATCH_RECOVERY_ATTEMPT,
        ):
            pre_payload = {
                "logical_request_id": record.logical_request_id,
                "role": prepared.role,
                "state_id": prepared.state_id,
                "input_fingerprint": prepared.canonical_input_fingerprint,
                "recovery_action": plan.action.value,
                "prior_attempt_id": plan.prior_attempt_id,
            }
            pre_checkpoint = "checkpoint-" + canonical_sha256(pre_payload)[7:]
            self.ledger.commit_checkpoint(
                pre_checkpoint,
                f"before_{prepared.role}_request",
                pre_payload,
            )
            if plan.prior_attempt_id is None:
                request_context = self.ledger.allocate_attempt(
                    record.logical_request_id,
                    manifest_identity=self.manifest_identity,
                    replayed_after_unknown_outcome=False,
                )
            else:
                request_context = self.ledger.reconcile_and_allocate_attempt(
                    record.logical_request_id,
                    manifest_identity=self.manifest_identity,
                )
            if (
                request_context.replayed_after_unknown_outcome
                != plan.replayed_after_unknown_outcome
            ):
                raise RuntimeError("recovery replay authorization mismatch")
            self.ledger.mark_in_flight(request_context)
        else:
            raise RuntimeError("terminal logical request cannot execute")

        runner = self.runners[prepared.role]
        try:
            result = runner.run(prepared, request_context)
        except ProviderRequestError as error:
            self.ledger.record_failure(error)
            raise
        if result.prepared_request != prepared or result.truncated or result.output is None:
            raise RuntimeError("invalid durable online role result")
        if result.attempt.context != request_context:
            raise RuntimeError("online role attempt identity mismatch")
        status = self.ledger.request_status(record.logical_request_id)
        if status not in (
            LogicalRequestStatus.RESPONSE_COMPLETED,
            LogicalRequestStatus.APPLIED,
        ):
            raise RuntimeError("runner returned before durable response persistence")
        return DurableRoleExecution(
            prepared_request=prepared,
            output=result.output,
            logical_request_id=record.logical_request_id,
            source_attempt_id=result.attempt.context.attempt_id,
            reused_completed_response=reused,
        )

    def apply(
        self,
        execution: DurableRoleExecution,
        *,
        stage: OnlineTurnStage,
        state_id: str,
        application_payload: object,
    ) -> DurableRoleApplication:
        payload = {
            "stage": stage.value,
            "state_id": state_id,
            "logical_request_id": execution.logical_request_id,
            "source_attempt_id": execution.source_attempt_id,
            "application": _json(application_payload),
        }
        digest = canonical_sha256(payload)
        checkpoint_id = "checkpoint-" + digest[7:]
        application = self.ledger.commit_checkpoint_and_apply(
            checkpoint_id=checkpoint_id,
            event_kind=stage.value,
            checkpoint_payload=payload,
            logical_request_id=execution.logical_request_id,
            source_attempt_id=execution.source_attempt_id,
            application_artifact_id=f"online-{stage.value}-{state_id}",
            application_artifact_sha256=digest,
        )
        return DurableRoleApplication(
            execution=execution,
            application_id=application.application_id,
            committed_checkpoint_id=application.committed_checkpoint_id,
        )

    def commit_online_action(
        self,
        *,
        checkpoint_id: str,
        action: ActionEnvelope,
        application_ids: tuple[str, ...],
        checkpoint_payload: object,
    ) -> DurableOnlineActionCommit:
        payload = _payload_dict(checkpoint_payload)
        action_sha256 = canonical_sha256(action.model_dump(mode="json"))
        effect = self.ledger.commit_checkpoint_and_effect(
            checkpoint_id=checkpoint_id,
            event_kind=OnlineTurnStage.FINAL_ACTION_COMMITTED.value,
            checkpoint_payload=payload,
            effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
            effect_artifact_id=f"online-action-{action_sha256[7:]}",
            effect_artifact_sha256=action_sha256,
            ordered_application_ids=application_ids,
        )
        return DurableOnlineActionCommit(
            effect_id=effect.effect_id,
            committed_checkpoint_id=effect.committed_checkpoint_id,
        )


class DurableRoleExecutor:
    """Fail-closed facade over a Task 011-backed durable implementation."""

    def __init__(self, backend: DurableRoleBackend):
        self._backend = backend

    def execute(self, prepared: PreparedRoleRequest) -> DurableRoleExecution:
        if type(prepared) is not PreparedRoleRequest:
            raise TypeError("PreparedRoleRequest required")
        result = self._backend.execute(prepared)
        if type(result) is not DurableRoleExecution or result.prepared_request != prepared:
            raise ValueError("durable backend returned a conflicting request")
        return result

    def apply(
        self,
        execution: DurableRoleExecution,
        *,
        stage: OnlineTurnStage,
        state_id: str,
        application_payload: object,
    ) -> DurableRoleApplication:
        if type(execution) is not DurableRoleExecution:
            raise TypeError("DurableRoleExecution required")
        if stage not in (
            OnlineTurnStage.INITIAL_POLICY_APPLIED,
            OnlineTurnStage.CRITIC_APPLIED,
            OnlineTurnStage.REVISION_APPLIED,
        ):
            raise ValueError("invalid online role application stage")
        result = self._backend.apply(
            execution,
            stage=stage,
            state_id=state_id,
            application_payload=application_payload,
        )
        if type(result) is not DurableRoleApplication or result.execution != execution:
            raise ValueError("durable backend returned a conflicting application")
        return result

    def commit_online_action(
        self,
        *,
        checkpoint_id: str,
        action: ActionEnvelope,
        application_ids: tuple[str, ...],
        checkpoint_payload: object,
    ) -> DurableOnlineActionCommit:
        if type(action) is not ActionEnvelope:
            raise TypeError("ActionEnvelope required")
        if type(checkpoint_id) is not str or not checkpoint_id:
            raise ValueError("final checkpoint ID must be non-empty")
        if not application_ids or len(set(application_ids)) != len(application_ids):
            raise ValueError("one or more unique causal application IDs required")
        result = self._backend.commit_online_action(
            checkpoint_id=checkpoint_id,
            action=action,
            application_ids=application_ids,
            checkpoint_payload=checkpoint_payload,
        )
        if type(result) is not DurableOnlineActionCommit:
            raise ValueError("invalid committed online-action result")
        return result


__all__ = [
    "DurableRoleApplication",
    "DurableRoleBackend",
    "DurableRoleExecution",
    "DurableRoleExecutor",
    "DurableOnlineActionCommit",
    "LedgerQwenResponseSeam",
    "Task011CheckpointEventSink",
    "Task011DurableRoleBackend",
]
