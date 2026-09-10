"""Strict durable contracts for checkpoint and request-ledger records."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.providers.contracts import ProviderRole
from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes
from toolsandbox_pipeline.schemas.base import JsonObject, JsonValue, StrictModel


Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Fingerprint = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class CheckpointConfig(StrictModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1]
    checkpoint_protocol_version: Literal["checkpoint-v1"]
    hash_algorithm: Literal["sha256"]
    database: Literal["sqlite3"]
    journal_mode: Literal["DELETE"]
    synchronous: Literal["FULL"]
    foreign_keys: Literal[True]
    busy_timeout_ms: Literal[0]
    single_writer: Literal[True]
    file_mode: Literal["0600"]
    directory_mode: Literal["0700"]
    max_blob_bytes: Literal[268435456]
    max_checkpoint_bytes: Literal[1073741824]


class RunIdentity(StrictModel):
    model_config = ConfigDict(frozen=True)
    run_id: Identifier
    profile: Literal["offline", "strict_replay", "official_live"]
    environment_identity: Fingerprint
    dataset_manifest_sha256: Fingerprint
    config_manifest_sha256: Fingerprint
    prompt_manifest_sha256: Fingerprint
    generation_manifest_sha256: Fingerprint
    fixture_manifest_sha256: Fingerprint


class BlobReference(StrictModel):
    model_config = ConfigDict(frozen=True)
    sha256: Fingerprint
    byte_count: Annotated[int, Field(ge=0)]
    media_type: Literal[
        "application/vnd.toolsandbox.canonical+json",
        "application/vnd.toolsandbox.raw-provider+json",
        "application/vnd.toolsandbox.execution-context+json",
        "application/vnd.toolsandbox.jsonl",
    ]
    schema_name: Identifier
    schema_version: Annotated[int, Field(ge=1)]
    content_visibility: Literal["restricted"]


class LogicalRequestStatus(str, Enum):
    PREPARED = "prepared"
    RESPONSE_COMPLETED = "response_completed"
    APPLIED = "applied"
    TERMINAL_FAILURE = "terminal_failure"


class AttemptLedgerStatus(str, Enum):
    ALLOCATED = "allocated"
    ABANDONED_BEFORE_DISPATCH = "abandoned_before_dispatch"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    REJECTED_BEFORE_DISPATCH = "rejected_before_dispatch"
    FAILED = "failed"
    UNKNOWN_OUTCOME = "unknown_outcome"


class LLMRecoveryAction(str, Enum):
    DISPATCH_FIRST_ATTEMPT = "dispatch_first_llm_attempt"
    DISPATCH_RECOVERY_ATTEMPT = "dispatch_recovery_llm_attempt"
    APPLY_STORED_RESPONSE = "apply_stored_llm_response"
    RESTORE_APPLIED_CHECKPOINT = "restore_applied_llm_checkpoint"
    TERMINAL_FAILURE = "terminal_failure"


class LLMRecoveryPlan(StrictModel):
    model_config = ConfigDict(frozen=True)
    logical_request_id: Annotated[str, Field(pattern=r"^llm-[0-9a-f]{64}$")]
    action: LLMRecoveryAction
    prior_attempt_id: Annotated[
        str, Field(pattern=r"^attempt-[0-9a-f]{64}-[0-9]{8}$")
    ] | None = None
    replayed_after_unknown_outcome: bool

    @model_validator(mode="after")
    def coherent_replay_flag(self) -> "LLMRecoveryPlan":
        if self.replayed_after_unknown_outcome != (
            self.action is LLMRecoveryAction.DISPATCH_RECOVERY_ATTEMPT
        ):
            raise ValueError("replay flag must match recovery dispatch")
        return self


class LogicalLLMRequestIdentity(StrictModel):
    model_config = ConfigDict(frozen=True)
    run_id: Identifier
    role: ProviderRole
    phase: Identifier
    unit_reference: Identifier
    input_fingerprint: Fingerprint
    model: Identifier
    decoding_configuration_sha256: Fingerprint
    output_schema_sha256: Fingerprint | None


class LogicalLLMRequestRecord(StrictModel):
    model_config = ConfigDict(frozen=True)
    logical_request_id: Annotated[str, Field(pattern=r"^llm-[0-9a-f]{64}$")]
    identity: LogicalLLMRequestIdentity
    status: LogicalRequestStatus


class LLMResponseApplication(StrictModel):
    model_config = ConfigDict(frozen=True)
    application_id: Annotated[str, Field(pattern=r"^application-[0-9a-f]{64}$")]
    logical_request_id: Annotated[str, Field(pattern=r"^llm-[0-9a-f]{64}$")]
    source_attempt_id: Annotated[
        str, Field(pattern=r"^attempt-[0-9a-f]{64}-[0-9]{8}$")
    ]
    application_artifact_id: Identifier
    application_artifact_sha256: Fingerprint
    committed_checkpoint_id: Identifier
    created_at_utc: datetime

    @model_validator(mode="after")
    def require_utc(self) -> "LLMResponseApplication":
        if (
            self.created_at_utc.utcoffset() is None
            or self.created_at_utc.utcoffset().total_seconds() != 0
        ):
            raise ValueError("UTC timestamp required")
        return self


class StoredValidatedOutput(StrictModel):
    """Verified completed output loaded for reuse without provider dispatch."""

    model_config = ConfigDict(frozen=True)
    logical_request_id: Annotated[str, Field(pattern=r"^llm-[0-9a-f]{64}$")]
    source_attempt_id: Annotated[
        str, Field(pattern=r"^attempt-[0-9a-f]{64}-[0-9]{8}$")
    ]
    output: JsonValue
    output_reference: BlobReference


class QwenEffectKind(str, Enum):
    COMMITTED_ONLINE_ACTION = "committed_online_action"
    POLICY_MEMORY_MUTATION = "policy_memory_mutation"
    WORLD_MEMORY_MUTATION = "world_memory_mutation"
    FAILURE_MODE_MUTATION = "failure_mode_mutation"
    ACCEPTED_SKILL_MUTATION = "accepted_skill_mutation"


class NonSubstantiveOutcome(str, Enum):
    NONE = "none"
    SKIP = "skip"
    DUPLICATE_NOOP = "duplicate_noop"
    REJECTED_CANDIDATE = "rejected_candidate"


class QwenEffectiveEffect(StrictModel):
    """Append-only link from a committed effect to its causal Qwen applications."""

    model_config = ConfigDict(frozen=True)
    effect_id: Annotated[str, Field(pattern=r"^effect-[0-9a-f]{64}$")]
    effect_kind: QwenEffectKind
    effect_artifact_id: Identifier
    effect_artifact_sha256: Fingerprint
    ordered_application_ids: tuple[
        Annotated[str, Field(pattern=r"^application-[0-9a-f]{64}$")], ...
    ]
    committed_checkpoint_id: Identifier

    @model_validator(mode="after")
    def causal_chain_is_nonempty_and_unique(self) -> "QwenEffectiveEffect":
        if not self.ordered_application_ids:
            raise ValueError("effective effect requires a causal application chain")
        if len(set(self.ordered_application_ids)) != len(
            self.ordered_application_ids
        ):
            raise ValueError("duplicate application in effective effect")
        identity_payload = {
            "protocol": "qwen-effective-effect-v1",
            "effect_kind": self.effect_kind.value,
            "effect_artifact_id": self.effect_artifact_id,
            "effect_artifact_sha256": self.effect_artifact_sha256,
            "ordered_application_ids": list(self.ordered_application_ids),
            "committed_checkpoint_id": self.committed_checkpoint_id,
        }
        expected_effect_id = "effect-" + sha256(
            canonical_json_bytes(identity_payload)
        ).hexdigest()
        if self.effect_id != expected_effect_id:
            raise ValueError("effective effect identity mismatch")
        return self


class CheckpointEvent(StrictModel):
    model_config = ConfigDict(frozen=True)
    checkpoint_id: Identifier
    event_ordinal: Annotated[int, Field(ge=1)]
    event_kind: Identifier
    payload: JsonObject
    created_at_utc: datetime

    @model_validator(mode="after")
    def require_utc(self) -> "CheckpointEvent":
        if (
            self.created_at_utc.utcoffset() is None
            or self.created_at_utc.utcoffset().total_seconds() != 0
        ):
            raise ValueError("UTC timestamp required")
        return self


class ExecutionContextEnvelope(StrictModel):
    model_config = ConfigDict(frozen=True)
    codec_version: Literal["execution-context-v1"]
    upstream_commit: Literal["165848b9a78cead7ca7fe7c89c688b58e6501219"]
    python_patch_version: Identifier
    environment_identity: Fingerprint
    non_console_context_sha256: Fingerprint
    serialized_context_with_base64_console: JsonObject
    serialized_context_sha256: Fingerprint


class ToolTransactionStatus(str, Enum):
    PREPARED = "prepared"
    COMMITTED = "committed"
    FAILED_COMMITTED = "failed_committed"
    RECONCILIATION_REQUIRED = "reconciliation_required"


class ToolAttemptLedgerStatus(str, Enum):
    ALLOCATED = "allocated"
    ABANDONED_BEFORE_EXECUTION = "abandoned_before_execution"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    UNKNOWN_OUTCOME = "unknown_outcome"


class ToolActionIdentity(StrictModel):
    model_config = ConfigDict(frozen=True)
    run_id: Identifier
    scenario_id: Identifier
    pre_action_context_sha256: Fingerprint
    action_fingerprint: Fingerprint
    ordered_call_ids: tuple[Identifier, ...]
    action_ordinal: Annotated[int, Field(ge=1)]
    profile: Literal["offline", "strict_replay", "official_live"]
    effect_classes: Annotated[
        tuple[
            Literal[
                "sandbox_read",
                "sandbox_write",
                "external_read",
                "conversation_control",
            ],
            ...,
        ],
        Field(min_length=1),
    ]
    fixture_manifest_sha256: Fingerprint
    backend_manifest_sha256: Fingerprint

    @model_validator(mode="after")
    def unique_calls(self) -> "ToolActionIdentity":
        if not self.ordered_call_ids or len(set(self.ordered_call_ids)) != len(
            self.ordered_call_ids
        ):
            raise ValueError("one or more unique ordered call IDs required")
        return self


class ToolTransactionRecord(StrictModel):
    model_config = ConfigDict(frozen=True)
    transaction_id: Annotated[str, Field(pattern=r"^tool-[0-9a-f]{64}$")]
    identity: ToolActionIdentity
    pre_context_reference: BlobReference
    status: ToolTransactionStatus
    post_context_reference: BlobReference | None = None
    post_context_sha256: Fingerprint | None = None


class ToolRecoveryAction(str, Enum):
    EXECUTE_FROM_PRE_CONTEXT = "execute_tool_from_pre_context"
    REPLAY_LOCAL_FROM_PRE_CONTEXT = "replay_local_tool_from_pre_context"
    REPLAY_FIXTURE_FROM_PRE_CONTEXT = "replay_fixture_tool_from_pre_context"
    RESTORE_COMMITTED_CONTEXT = "restore_committed_tool_context"
    RECONCILIATION_REQUIRED = "reconciliation_required"


class ToolRecoveryPlan(StrictModel):
    model_config = ConfigDict(frozen=True)
    transaction_id: Annotated[str, Field(pattern=r"^tool-[0-9a-f]{64}$")]
    action: ToolRecoveryAction
    prior_attempt_id: Identifier | None = None


class OfflineUnitStatus(str, Enum):
    PREPARED = "prepared"
    OUTPUTS_STAGED = "outputs_staged"
    COMMITTED = "committed"
    TERMINAL_FAILURE = "terminal_failure"


class OfflineUnitIdentity(StrictModel):
    model_config = ConfigDict(frozen=True)
    run_id: Identifier
    round_index: Annotated[int, Field(ge=0)]
    input_generation_id: Identifier
    unit_kind: Identifier
    unit_key: Identifier
    input_fingerprint: Fingerprint
    expected_output_artifact_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def unique_outputs(self) -> "OfflineUnitIdentity":
        if len(set(self.expected_output_artifact_ids)) != len(
            self.expected_output_artifact_ids
        ):
            raise ValueError("duplicate expected output artifact")
        return self


class OfflineUnitRecord(StrictModel):
    model_config = ConfigDict(frozen=True)
    unit_id: Annotated[str, Field(pattern=r"^offline-[0-9a-f]{64}$")]
    identity: OfflineUnitIdentity
    status: OfflineUnitStatus
    request_ids: tuple[Identifier, ...]
    staged_outputs: dict[str, Fingerprint] | None = None
    committed_outputs: dict[str, Fingerprint] | None = None
    failure_class: Identifier | None = None


class PublicationStatus(str, Enum):
    PREPARED = "prepared"
    COMMITTED = "committed"
    TERMINAL_FAILURE = "terminal_failure"


class UnitRecoveryAction(str, Enum):
    RESUME_OFFLINE_UNIT = "resume_offline_unit"
    REUSE_COMMITTED_OUTPUTS = "reuse_committed_offline_outputs"
    TERMINAL_FAILURE = "terminal_failure"


class PublicationRecoveryAction(str, Enum):
    FINALIZE_PUBLICATION = "finalize_generation_publication"
    RUN_COMPLETE = "run_complete"
    TERMINAL_FAILURE = "terminal_failure"


class GenerationPublicationIdentity(StrictModel):
    model_config = ConfigDict(frozen=True)
    run_id: Identifier
    round_index: Annotated[int, Field(ge=0)]
    input_generation_id: Identifier
    destination_generation_id: Identifier
    staging_manifest_sha256: Fingerprint
    staging_content_sha256: Fingerprint
    mini_bench_decision_sha256: Fingerprint | None = None


class GenerationPublicationRecord(StrictModel):
    model_config = ConfigDict(frozen=True)
    publication_id: Annotated[str, Field(pattern=r"^publication-[0-9a-f]{64}$")]
    identity: GenerationPublicationIdentity
    status: PublicationStatus
    pre_checkpoint_id: Identifier | None = None
    post_checkpoint_id: Identifier | None = None
    failure_class: Identifier | None = None
