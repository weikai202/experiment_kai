"""Direct durable Qwen responder for the frozen Vanilla comparison system."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator
from tool_sandbox.common.execution_context import RoleType, get_current_context

from toolsandbox_pipeline.checkpointing import (
    LLMLedger,
    LLMRecoveryAction,
    LogicalLLMRequestIdentity,
    QwenEffectKind,
)
from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    PhysicalAttemptStatus,
    ProviderRequestError,
    ProviderRole,
)
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.runtime import QwenConfig, RoleTokenLimitConfig
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.trajectory import EpisodeIdentity, OnlineTurnRecord
from toolsandbox_pipeline.toolsandbox_adapter.contracts import AdapterTurn
from toolsandbox_pipeline.toolsandbox_adapter.pipeline_agent import PipelineAgent
from toolsandbox_pipeline.toolsandbox_adapter.trajectory_store import TrajectoryStore
from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import EpisodeAgentRole


class VanillaResponderError(RuntimeError):
    """Sanitized direct-responder failure."""


class _FrozenStrict(StrictModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class VanillaPromptManifest(_FrozenStrict):
    schema_version: Literal[1]
    role: Literal["vanilla"]
    path: Literal["prompts/vanilla_agent_v1.txt"]
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_version: Literal["v1"]
    output_model_name: Literal["ActionEnvelope"]


class VanillaTokenLimitManifest(_FrozenStrict):
    schema_version: Literal[1]
    status: Literal["provisional", "calibrated"]
    role: Literal["vanilla"]
    output_model_name: Literal["ActionEnvelope"]
    max_tokens: int = Field(gt=0)
    version: str = "vanilla-bootstrap-v1"
    evidence_manifest_identity: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def calibrated_shape(self) -> "VanillaTokenLimitManifest":
        if self.status == "provisional":
            if self.max_tokens != 256 or self.evidence_manifest_identity is not None:
                raise ValueError("Vanilla provisional ceiling must be 256")
        elif self.evidence_manifest_identity is None or self.max_tokens % 64:
            raise ValueError("calibrated Vanilla limit requires evidence and 64 alignment")
        return self

    def selected_limit(self) -> RoleTokenLimitConfig:
        return RoleTokenLimitConfig(
            version=self.version,
            role="vanilla",
            stage="bootstrap" if self.status == "provisional" else "calibrated",
            max_tokens=self.max_tokens,
            evidence_manifest_identity=self.evidence_manifest_identity,
        )


@dataclass(frozen=True)
class LoadedVanillaAssets:
    prompt: str
    prompt_manifest: VanillaPromptManifest
    prompt_manifest_sha256: str
    token_limit: VanillaTokenLimitManifest
    token_limit_sha256: str


@dataclass(frozen=True)
class VanillaResponseEvidence:
    action: ActionEnvelope
    state_id: str
    logical_request_id: str
    source_attempt_id: str
    application_id: str
    effect_id: str
    turn_index: int


class VanillaDecisionRecord(_FrozenStrict):
    system_id: Literal["vanilla"] = "vanilla"
    state_id: str = Field(min_length=1)
    action: ActionEnvelope
    logical_request_id: str = Field(min_length=1)
    source_attempt_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    effect_id: str = Field(min_length=1)


class VanillaRequestNamespace(_FrozenStrict):
    """Immutable cross-scenario namespace for a sequence of Vanilla turns."""

    run_id: str = Field(min_length=1, pattern=r"^\S+$")
    episode_id: str = Field(min_length=1, pattern=r"^\S+$")
    scenario_family_id: str = Field(min_length=1, pattern=r"^\S+$")
    scenario_id: str = Field(min_length=1, pattern=r"^\S+$")
    starting_turn_index: int = Field(ge=0)
    system_variant: Literal["vanilla"] = "vanilla"

    def accounting_scope(self) -> AccountingScope:
        return AccountingScope(
            run_id=self.run_id,
            task_id=self.episode_id,
            scenario_family_id=self.scenario_family_id,
            scenario_id=self.scenario_id,
            system_variant=self.system_variant,
        )


def load_vanilla_assets(project_root: Path) -> LoadedVanillaAssets:
    if not project_root.is_absolute() or project_root.is_symlink():
        raise ValueError("absolute non-symlink project root required")
    prompt_path = project_root / "prompts/vanilla_agent_v1.txt"
    manifest_path = project_root / "prompts/vanilla_manifest.json"
    limit_path = project_root / "configs/vanilla_token_limit.provisional.json"
    for path in (prompt_path, manifest_path, limit_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or unsafe Vanilla asset")
    prompt_bytes = prompt_path.read_bytes()
    if (
        not prompt_bytes.endswith(b"\n")
        or prompt_bytes.endswith(b"\n\n")
        or b"\r" in prompt_bytes
        or prompt_bytes.startswith(b"\xef\xbb\xbf")
    ):
        raise ValueError("invalid Vanilla prompt encoding")
    prompt_manifest = VanillaPromptManifest.model_validate_json(
        manifest_path.read_bytes(), strict=True
    )
    if prompt_manifest.sha256 != file_hash(prompt_bytes):
        raise ValueError("Vanilla prompt hash mismatch")
    token_limit = VanillaTokenLimitManifest.model_validate_json(
        limit_path.read_bytes(), strict=True
    )
    return LoadedVanillaAssets(
        prompt=prompt_bytes.decode("utf-8"),
        prompt_manifest=prompt_manifest,
        prompt_manifest_sha256=file_hash(manifest_path.read_bytes()),
        token_limit=token_limit,
        token_limit_sha256=file_hash(limit_path.read_bytes()),
    )


class VanillaResponder:
    """Use only the Adapter Agent view, then durably apply one direct action."""

    def __init__(
        self,
        *,
        ledger: LLMLedger,
        gateway: object,
        assets: LoadedVanillaAssets,
        namespace: VanillaRequestNamespace,
        phase: str,
        manifest_identity: str,
        mode: Literal["offline", "formal", "calibration"] = "offline",
    ) -> None:
        if type(ledger) is not LLMLedger:
            raise TypeError("LLMLedger required")
        config = getattr(gateway, "config", None)
        if type(config) is not QwenConfig or config.enable_thinking is not False:
            raise TypeError("frozen non-thinking Qwen gateway required")
        if mode == "formal" and assets.token_limit.status != "calibrated":
            raise ValueError("formal Vanilla requires a calibrated token limit")
        if type(namespace) is not VanillaRequestNamespace:
            raise TypeError("VanillaRequestNamespace required")
        if ledger.store.identity.run_id != namespace.run_id:
            raise ValueError("Vanilla ledger run mismatch")
        self.ledger = ledger
        self.gateway = gateway
        self.assets = assets
        self.namespace = namespace
        self.run_id = namespace.run_id
        self.episode_id = namespace.episode_id
        self._turn_index = namespace.starting_turn_index
        self.phase = phase
        self.manifest_identity = manifest_identity
        self.mode = mode
        self.last_evidence: VanillaResponseEvidence | None = None

    def respond(self, turn: AdapterTurn) -> ActionEnvelope:
        if type(turn) is not AdapterTurn:
            raise TypeError("AdapterTurn required")
        envelope = self._agent_view_envelope(turn)
        state_digest = canonical_sha256(
            {
                "episode_id": self.episode_id,
                "scenario_id": self.namespace.scenario_id,
                "turn_index": self._turn_index,
                "agent_view": envelope,
            }
        )
        state_id = "vanilla-state-" + state_digest[7:]
        schema_sha256 = canonical_sha256(ActionEnvelope.model_json_schema())
        messages = [
            {"role": "system", "content": self.assets.prompt},
            {"role": "user", "content": canonical_json_bytes(envelope).decode("utf-8")},
        ]
        qwen_config = self.gateway.config
        input_fingerprint = canonical_sha256(
            {
                "role": "vanilla",
                "provider": qwen_config.provider,
                "model": qwen_config.model,
                "messages": messages,
                "output_schema_sha256": schema_sha256,
                "max_tokens": self.assets.token_limit.max_tokens,
                "temperature": 0.0,
                "seed": 0,
                "top_p": "omitted",
                "enable_thinking": False,
                "structured_output_wire_mode": qwen_config.structured_output_wire_mode,
            }
        )
        identity = LogicalLLMRequestIdentity(
            run_id=self.run_id,
            role=ProviderRole.VANILLA,
            phase=self.phase,
            unit_reference=state_id,
            input_fingerprint=input_fingerprint,
            model=qwen_config.served_model_id or qwen_config.model,
            decoding_configuration_sha256=canonical_sha256(
                qwen_config.model_dump(mode="json")
            ),
            output_schema_sha256=schema_sha256,
        )
        request = self.ledger.prepare_request(identity)
        self.ledger.bind_accounting_scope(
            request.logical_request_id, self.namespace.accounting_scope()
        )
        plan = self.ledger.plan_recovery(request.logical_request_id)
        if plan.action is LLMRecoveryAction.TERMINAL_FAILURE:
            raise VanillaResponderError("terminal Vanilla request")
        if plan.action in {
            LLMRecoveryAction.APPLY_STORED_RESPONSE,
            LLMRecoveryAction.RESTORE_APPLIED_CHECKPOINT,
        }:
            material = self.ledger.load_completed_response_material(
                request.logical_request_id
            )
            action = ActionEnvelope.model_validate(material.validated_output, strict=True)
            source_attempt_id = material.attempt.context.attempt_id
        else:
            pre_payload = {
                "system_id": "vanilla",
                "state_id": state_id,
                "logical_request_id": request.logical_request_id,
                "input_fingerprint": input_fingerprint,
            }
            self.ledger.commit_checkpoint(
                "vanilla-before-" + canonical_sha256(pre_payload)[7:],
                "before_vanilla_request",
                pre_payload,
            )
            if plan.prior_attempt_id is None:
                context = self.ledger.allocate_attempt(
                    request.logical_request_id,
                    manifest_identity=self.manifest_identity,
                    replayed_after_unknown_outcome=False,
                )
            else:
                context = self.ledger.reconcile_and_allocate_attempt(
                    request.logical_request_id,
                    manifest_identity=self.manifest_identity,
                )
            self.ledger.mark_in_flight(context)
            try:
                response = self.gateway.generate(
                    context,
                    messages,
                    ActionEnvelope,
                    max_tokens=self.assets.token_limit.max_tokens,
                    token_limit_config=self.assets.token_limit.selected_limit(),
                )
            except ProviderRequestError as error:
                self.ledger.record_failure(error)
                raise VanillaResponderError("Vanilla provider request failed") from None
            self._validate_response(context, response)
            action = response.value
            source_attempt_id = response.attempt.context.attempt_id
            self.ledger.complete_response(
                response,
                validated_output=action.model_dump(mode="json"),
                output_schema_name="ActionEnvelope",
                output_schema_version=1,
            )
        action_sha256 = canonical_sha256(action.model_dump(mode="json"))
        applications = tuple(
            item
            for item in self.ledger.applications()
            if item.logical_request_id == request.logical_request_id
        )
        if applications:
            if (
                len(applications) != 1
                or applications[0].source_attempt_id != source_attempt_id
                or applications[0].application_artifact_sha256 != action_sha256
            ):
                raise VanillaResponderError("Vanilla application identity conflict")
            application = applications[0]
        else:
            application_payload = {
                "system_id": "vanilla",
                "state_id": state_id,
                "logical_request_id": request.logical_request_id,
                "source_attempt_id": source_attempt_id,
                "action_sha256": action_sha256,
            }
            application_digest = canonical_sha256(application_payload)
            application = self.ledger.commit_checkpoint_and_apply(
                checkpoint_id="vanilla-apply-" + application_digest[7:],
                event_kind="vanilla_action_applied",
                checkpoint_payload=application_payload,
                logical_request_id=request.logical_request_id,
                source_attempt_id=source_attempt_id,
                application_artifact_id="vanilla-action-" + action_sha256[7:],
                application_artifact_sha256=action_sha256,
            )
        effects = tuple(
            effect
            for effect in self.ledger.effects()
            if application.application_id in effect.ordered_application_ids
        )
        if effects:
            if (
                len(effects) != 1
                or effects[0].effect_artifact_sha256 != action_sha256
                or effects[0].ordered_application_ids != (application.application_id,)
            ):
                raise VanillaResponderError("Vanilla effect identity conflict")
            effect = effects[0]
        else:
            effect_payload = {
                "system_id": "vanilla",
                "state_id": state_id,
                "action_sha256": action_sha256,
                "application_id": application.application_id,
            }
            effect_digest = canonical_sha256(effect_payload)
            effect = self.ledger.commit_checkpoint_and_effect(
                checkpoint_id="vanilla-effect-" + effect_digest[7:],
                event_kind="vanilla_action_committed",
                checkpoint_payload=effect_payload,
                effect_kind=QwenEffectKind.COMMITTED_ONLINE_ACTION,
                effect_artifact_id="vanilla-action-" + action_sha256[7:],
                effect_artifact_sha256=action_sha256,
                ordered_application_ids=(application.application_id,),
            )
        self.last_evidence = VanillaResponseEvidence(
            action=action,
            state_id=state_id,
            logical_request_id=request.logical_request_id,
            source_attempt_id=source_attempt_id,
            application_id=application.application_id,
            effect_id=effect.effect_id,
            turn_index=self._turn_index,
        )
        self._turn_index += 1
        return action

    @property
    def next_turn_index(self) -> int:
        return self._turn_index

    @staticmethod
    def _agent_view_envelope(turn: AdapterTurn) -> dict[str, object]:
        view = turn.agent_view
        names = tuple(tool.name for tool in view.available_tools)
        if len(set(names)) != len(names):
            raise ValueError("duplicate agent-facing tool")
        for tool in view.available_tools:
            function = tool.schema_.get("function")
            if not isinstance(function, dict) or function.get("name") != tool.name:
                raise ValueError("agent-facing schema identity mismatch")
        return {
            "visible_messages": [item.model_dump(mode="json") for item in view.visible_messages],
            "available_tools": [
                item.model_dump(mode="json", by_alias=True) for item in view.available_tools
            ],
        }

    def _validate_response(self, context: object, response: GatewayResponse) -> None:
        role = getattr(context, "role", None)
        served_model = self.gateway.config.served_model_id or self.gateway.config.model
        if (
            type(response) is not GatewayResponse
            or type(response.value) is not ActionEnvelope
            or response.attempt.context != context
            or role is not ProviderRole.VANILLA
            or response.attempt.status is not PhysicalAttemptStatus.COMPLETED
            or type(response.raw_response_body) is not bytes
            or response.attempt.finish_reason != "stop"
            or response.attempt.model != served_model
            or response.attempt.returned_model != served_model
            or response.attempt.response_hash
            != "sha256:" + sha256(response.raw_response_body).hexdigest()
        ):
            raise VanillaResponderError("invalid Vanilla response")


class TransactionalVanillaAgentRole(EpisodeAgentRole):
    """Task014 marker implementation for the direct Vanilla responder."""

    role_type = RoleType.AGENT

    def __init__(
        self,
        responder: VanillaResponder,
        *,
        identity: EpisodeIdentity,
        trajectory_store: TrajectoryStore,
        starting_turn_index: int = 0,
    ) -> None:
        if type(responder) is not VanillaResponder:
            raise TypeError("VanillaResponder required")
        if (
            responder.episode_id != identity.episode_id
            or responder.namespace.scenario_id != identity.scenario_id
            or responder.namespace.scenario_family_id != identity.family_id
            or responder.namespace.system_variant != identity.system_variant
            or responder.next_turn_index != starting_turn_index
        ):
            raise ValueError("Vanilla responder episode mismatch")
        self.responder = responder
        self.identity = identity
        self.trajectory_store = trajectory_store
        self._turn_index = starting_turn_index
        self._records: list[OnlineTurnRecord] = []
        self.last_checkpoint_ordinal = 0

    @property
    def records(self) -> tuple[OnlineTurnRecord, ...]:
        return tuple(self._records)

    def respond(self, ending_index: int | None = None) -> None:
        if ending_index is not None:
            raise VanillaResponderError("Vanilla Agent cannot process setup")
        before = get_current_context().max_sandbox_message_index
        PipelineAgent(self.responder).respond()
        evidence = self.responder.last_evidence
        context = get_current_context()
        if (
            evidence is None
            or evidence.turn_index != self._turn_index
            or context.max_sandbox_message_index <= before
        ):
            raise VanillaResponderError("Vanilla action was not appended")
        decision = VanillaDecisionRecord(
            state_id=evidence.state_id,
            action=evidence.action,
            logical_request_id=evidence.logical_request_id,
            source_attempt_id=evidence.source_attempt_id,
            application_id=evidence.application_id,
            effect_id=evidence.effect_id,
        )
        reference = self.trajectory_store.persist_online_decision(decision)
        self._records.append(
            OnlineTurnRecord(
                agent_turn_index=self._turn_index,
                state_id=evidence.state_id,
                decision_reference=reference,
                decision_sha256=reference.sha256,
                final_action_sha256=canonical_sha256(
                    evidence.action.model_dump(mode="json")
                ),
                logical_request_ids=(evidence.logical_request_id,),
                source_attempt_ids=(evidence.source_attempt_id,),
                application_ids=(evidence.application_id,),
            )
        )
        stored = self.trajectory_store.persist_context(context)
        event = self.trajectory_store.commit_context_checkpoint(
            self.identity,
            event_kind="vanilla_agent_message_committed",
            stored=stored,
            recipient=self.role_type.value,
            boundary_ordinal=self._turn_index + 1,
        )
        self.last_checkpoint_ordinal = event.event_ordinal
        self._turn_index += 1


__all__ = [
    "LoadedVanillaAssets",
    "TransactionalVanillaAgentRole",
    "VanillaDecisionRecord",
    "VanillaPromptManifest",
    "VanillaResponder",
    "VanillaResponderError",
    "VanillaRequestNamespace",
    "VanillaResponseEvidence",
    "VanillaTokenLimitManifest",
    "load_vanilla_assets",
]
