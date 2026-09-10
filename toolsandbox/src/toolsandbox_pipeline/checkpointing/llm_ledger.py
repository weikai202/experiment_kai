"""Durable LLM request, application, and substantive-effect ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import json
import sqlite3

from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    PhysicalAttemptResult,
    PhysicalAttemptStatus,
    ProviderRequestError,
    QWEN_GENERATION_ROLES,
    RequestContext,
)
from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.accounting import (
    LogicalRequestAccountingInput,
    PhysicalAttemptAccountingInput,
)
from toolsandbox_pipeline.schemas.base import JsonValue
from toolsandbox_pipeline.schemas.checkpoint import (
    AttemptLedgerStatus,
    CheckpointEvent,
    LLMResponseApplication,
    LogicalLLMRequestIdentity,
    LogicalLLMRequestRecord,
    LLMRecoveryAction,
    LLMRecoveryPlan,
    LogicalRequestStatus,
    NonSubstantiveOutcome,
    QwenEffectiveEffect,
    QwenEffectKind,
    StoredValidatedOutput,
)
from toolsandbox_pipeline.checkpointing.identities import (
    application_id as derive_application_id,
    attempt_id as derive_attempt_id,
    effective_effect_id,
    logical_request_id,
)
from toolsandbox_pipeline.checkpointing.store import CheckpointStore
from toolsandbox_pipeline.schemas.ledger_accounting import (
    AccountingScope,
    Task011AccountingSnapshot,
    Task011LedgerPopulation,
)


class LedgerConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompletedResponseMaterial:
    """Restricted material for a caller-owned provider-specific decoder."""

    attempt: PhysicalAttemptResult
    validated_output: JsonValue = field(repr=False)
    raw_response_body: bytes = field(repr=False)


class LLMLedger:
    def __init__(self, store: CheckpointStore):
        if type(store) is not CheckpointStore:
            raise TypeError("CheckpointStore required")
        self.store = store

    def prepare_request(
        self, identity: LogicalLLMRequestIdentity
    ) -> LogicalLLMRequestRecord:
        if identity.run_id != self.store.identity.run_id:
            raise LedgerConflictError("request run identity mismatch")
        request_id = logical_request_id(identity)
        encoded = canonical_json_bytes(identity.model_dump(mode="json"))
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT identity,status FROM logical_llm_requests WHERE logical_request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO logical_llm_requests VALUES(?,?,?,?,?)",
                    (request_id, encoded, identity.role.value, identity.model, "prepared"),
                )
                status = LogicalRequestStatus.PREPARED
            else:
                if bytes(row["identity"]) != encoded:
                    raise LedgerConflictError("logical request identity conflict")
                status = LogicalRequestStatus(row["status"])
        return LogicalLLMRequestRecord(
            logical_request_id=request_id, identity=identity, status=status
        )

    def bind_accounting_scope(
        self, logical_request_id: str, scope: AccountingScope
    ) -> None:
        """Durably bind one logical request to exactly one reporting scope."""

        if type(scope) is not AccountingScope:
            raise TypeError("AccountingScope required")
        if scope.run_id != self.store.identity.run_id:
            raise LedgerConflictError("accounting scope run identity mismatch")
        encoded = canonical_json_bytes(scope.model_dump(mode="json"))
        with self.store.transaction() as connection:
            request = connection.execute(
                "SELECT 1 FROM logical_llm_requests WHERE logical_request_id=?",
                (logical_request_id,),
            ).fetchone()
            if request is None:
                raise LedgerConflictError("unknown logical request")
            existing = connection.execute(
                "SELECT scope FROM llm_accounting_scopes WHERE logical_request_id=?",
                (logical_request_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO llm_accounting_scopes VALUES(?,?)",
                    (logical_request_id, encoded),
                )
            elif bytes(existing["scope"]) != encoded:
                raise LedgerConflictError("logical request accounting scope conflict")

    def snapshot_accounting(
        self, scope: AccountingScope
    ) -> Task011AccountingSnapshot:
        """Atomically project every Task011 row attributed to one exact scope."""

        if type(scope) is not AccountingScope:
            raise TypeError("AccountingScope required")
        if scope.run_id != self.store.identity.run_id:
            raise LedgerConflictError("accounting scope run identity mismatch")
        encoded_scope = canonical_json_bytes(scope.model_dump(mode="json"))
        with self.store.transaction() as connection:
            unbound = connection.execute(
                """SELECT r.logical_request_id
                   FROM logical_llm_requests r
                   LEFT JOIN llm_accounting_scopes s USING(logical_request_id)
                   WHERE s.logical_request_id IS NULL LIMIT 1"""
            ).fetchone()
            if unbound is not None:
                raise LedgerConflictError(
                    "unbound logical request prevents complete accounting snapshot"
                )
            request_rows = connection.execute(
                """SELECT r.rowid AS request_rowid,r.*
                   FROM logical_llm_requests r
                   JOIN llm_accounting_scopes s USING(logical_request_id)
                   WHERE s.scope=? ORDER BY r.rowid""",
                (encoded_scope,),
            ).fetchall()
            logical_ids = tuple(row["logical_request_id"] for row in request_rows)
            attempts = self._accounting_attempts(connection, request_rows, scope)
            applications = self._accounting_applications(
                connection, logical_ids, request_rows
            )
            logical = self._accounting_logical_requests(
                request_rows, scope, applications
            )
            effects = self._accounting_effects(
                connection, tuple(item.application_id for item in applications)
            )
            high_water = {
                table: {
                    "count": int(
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    ),
                    "max_rowid": int(
                        connection.execute(
                            f"SELECT COALESCE(MAX(rowid),0) FROM {table}"
                        ).fetchone()[0]
                    ),
                }
                for table in (
                    "logical_llm_requests",
                    "physical_llm_attempts",
                    "llm_response_applications",
                    "qwen_effective_effects",
                    "qwen_effect_applications",
                    "llm_accounting_scopes",
                )
            }
            population = Task011LedgerPopulation.build(
                scope=scope,
                high_water_identity=canonical_sha256(
                    ["task011-ledger-high-water-v1", high_water]
                ),
                logical_request_ids=logical_ids,
                physical_attempt_ids=tuple(item.attempt_id for item in attempts),
                application_ids=tuple(item.application_id for item in applications),
                effect_ids=tuple(item.effect_id for item in effects),
            )
            population_reference = self.store.blobs.put(
                canonical_json_bytes(population.model_dump(mode="json")),
                media_type="application/vnd.toolsandbox.canonical+json",
                schema_name="Task011LedgerPopulation",
                schema_version=1,
            )
        return Task011AccountingSnapshot(
            population=population,
            population_reference=population_reference,
            physical_attempts=attempts,
            logical_requests=logical,
            applications=applications,
            substantive_effects=effects,
        )

    def allocate_attempt(
        self,
        logical_request_id_value: str,
        *,
        manifest_identity: str,
        replayed_after_unknown_outcome: bool,
    ) -> RequestContext:
        with self.store.transaction() as connection:
            request = connection.execute(
                "SELECT identity,status FROM logical_llm_requests WHERE logical_request_id=?",
                (logical_request_id_value,),
            ).fetchone()
            if request is None or request["status"] != "prepared":
                raise LedgerConflictError("request is not prepared")
            identity = LogicalLLMRequestIdentity.model_validate_json(request["identity"])
            prior_count = connection.execute(
                "SELECT COUNT(*) FROM physical_llm_attempts WHERE logical_request_id=?",
                (logical_request_id_value,),
            ).fetchone()[0]
            if prior_count or replayed_after_unknown_outcome:
                raise LedgerConflictError(
                    "initial allocation requires no prior attempt and replay=false"
                )
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(attempt_ordinal),0)+1 FROM physical_llm_attempts WHERE logical_request_id=?",
                (logical_request_id_value,),
            ).fetchone()[0]
            new_attempt_id = derive_attempt_id(logical_request_id_value, ordinal)
            context = RequestContext(
                logical_request_id=logical_request_id_value,
                attempt_id=new_attempt_id,
                role=identity.role,
                phase=identity.phase,
                unit_reference=identity.unit_reference,
                input_fingerprint=identity.input_fingerprint,
                replayed_after_unknown_outcome=replayed_after_unknown_outcome,
                manifest_identity=manifest_identity,
            )
            connection.execute(
                "INSERT INTO physical_llm_attempts(attempt_id,logical_request_id,attempt_ordinal,status,request_context) VALUES(?,?,?,?,?)",
                (
                    new_attempt_id,
                    logical_request_id_value,
                    ordinal,
                    AttemptLedgerStatus.ALLOCATED.value,
                    canonical_json_bytes(context.model_dump(mode="json")),
                ),
            )
        return context

    def plan_recovery(self, logical_request_id: str) -> LLMRecoveryPlan:
        """Pure decision over current durable LLM state; performs no mutation."""
        request = self.store._connection.execute(
            "SELECT status FROM logical_llm_requests WHERE logical_request_id=?",
            (logical_request_id,),
        ).fetchone()
        if request is None:
            raise LedgerConflictError("unknown logical request")
        latest = self.store._connection.execute(
            """SELECT attempt_id,status FROM physical_llm_attempts
               WHERE logical_request_id=? ORDER BY attempt_ordinal DESC LIMIT 1""",
            (logical_request_id,),
        ).fetchone()
        logical_status = LogicalRequestStatus(request["status"])
        if logical_status is LogicalRequestStatus.RESPONSE_COMPLETED:
            action = LLMRecoveryAction.APPLY_STORED_RESPONSE
            replay = False
        elif logical_status is LogicalRequestStatus.APPLIED:
            action = LLMRecoveryAction.RESTORE_APPLIED_CHECKPOINT
            replay = False
        elif logical_status is LogicalRequestStatus.TERMINAL_FAILURE:
            action = LLMRecoveryAction.TERMINAL_FAILURE
            replay = False
        elif latest is None or latest["status"] in {
            AttemptLedgerStatus.ALLOCATED.value,
            AttemptLedgerStatus.ABANDONED_BEFORE_DISPATCH.value,
            AttemptLedgerStatus.REJECTED_BEFORE_DISPATCH.value,
        }:
            action = LLMRecoveryAction.DISPATCH_FIRST_ATTEMPT
            replay = False
        elif latest["status"] in {
            AttemptLedgerStatus.IN_FLIGHT.value,
            AttemptLedgerStatus.UNKNOWN_OUTCOME.value,
        }:
            action = LLMRecoveryAction.DISPATCH_RECOVERY_ATTEMPT
            replay = True
        else:
            raise LedgerConflictError("inconsistent prepared request attempt state")
        return LLMRecoveryPlan(
            logical_request_id=logical_request_id,
            action=action,
            prior_attempt_id=None if latest is None else latest["attempt_id"],
            replayed_after_unknown_outcome=replay,
        )

    def reconcile_and_allocate_attempt(
        self, logical_request_id: str, *, manifest_identity: str
    ) -> RequestContext:
        """Atomically classify a stale attempt and allocate one never-reused ID."""
        with self.store.transaction() as connection:
            request = connection.execute(
                "SELECT identity,status FROM logical_llm_requests WHERE logical_request_id=?",
                (logical_request_id,),
            ).fetchone()
            if request is None or request["status"] != "prepared":
                raise LedgerConflictError("request is not recoverable prepared state")
            latest = connection.execute(
                """SELECT attempt_id,attempt_ordinal,status FROM physical_llm_attempts
                   WHERE logical_request_id=? ORDER BY attempt_ordinal DESC LIMIT 1""",
                (logical_request_id,),
            ).fetchone()
            replay = False
            if latest is not None:
                latest_status = AttemptLedgerStatus(latest["status"])
                if latest_status is AttemptLedgerStatus.ALLOCATED:
                    connection.execute(
                        "UPDATE physical_llm_attempts SET status='abandoned_before_dispatch' WHERE attempt_id=? AND status='allocated'",
                        (latest["attempt_id"],),
                    )
                elif latest_status is AttemptLedgerStatus.IN_FLIGHT:
                    connection.execute(
                        "UPDATE physical_llm_attempts SET status='unknown_outcome' WHERE attempt_id=? AND status='in_flight'",
                        (latest["attempt_id"],),
                    )
                    replay = True
                elif latest_status is AttemptLedgerStatus.UNKNOWN_OUTCOME:
                    replay = True
                elif latest_status not in {
                    AttemptLedgerStatus.ABANDONED_BEFORE_DISPATCH,
                    AttemptLedgerStatus.REJECTED_BEFORE_DISPATCH,
                }:
                    raise LedgerConflictError("latest attempt is not recoverable")
            ordinal = 1 if latest is None else latest["attempt_ordinal"] + 1
            identity = LogicalLLMRequestIdentity.model_validate_json(request["identity"])
            new_attempt_id = derive_attempt_id(logical_request_id, ordinal)
            context = RequestContext(
                logical_request_id=logical_request_id,
                attempt_id=new_attempt_id,
                role=identity.role,
                phase=identity.phase,
                unit_reference=identity.unit_reference,
                input_fingerprint=identity.input_fingerprint,
                replayed_after_unknown_outcome=replay,
                manifest_identity=manifest_identity,
            )
            connection.execute(
                "INSERT INTO physical_llm_attempts(attempt_id,logical_request_id,attempt_ordinal,status,request_context) VALUES(?,?,?,?,?)",
                (
                    new_attempt_id,
                    logical_request_id,
                    ordinal,
                    AttemptLedgerStatus.ALLOCATED.value,
                    canonical_json_bytes(context.model_dump(mode="json")),
                ),
            )
        return context

    def attempt_status(self, attempt_id: str) -> AttemptLedgerStatus:
        row = self.store._connection.execute(
            "SELECT status FROM physical_llm_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise LedgerConflictError("unknown physical attempt")
        return AttemptLedgerStatus(row["status"])

    def mark_in_flight(self, context: RequestContext) -> None:
        encoded = canonical_json_bytes(context.model_dump(mode="json"))
        with self.store.transaction() as connection:
            changed = connection.execute(
                "UPDATE physical_llm_attempts SET status='in_flight' WHERE attempt_id=? AND logical_request_id=? AND request_context=? AND status='allocated'",
                (context.attempt_id, context.logical_request_id, encoded),
            ).rowcount
            if changed != 1:
                raise LedgerConflictError("attempt cannot enter in-flight state")

    def complete_response(
        self,
        response: GatewayResponse,
        *,
        validated_output: JsonValue,
        output_schema_name: str,
        output_schema_version: int,
    ) -> None:
        attempt = response.attempt
        if attempt.status is not PhysicalAttemptStatus.COMPLETED:
            raise LedgerConflictError("only completed provider response is reusable")
        raw_reference = self.store.blobs.put(
            response.raw_response_body,
            media_type="application/vnd.toolsandbox.raw-provider+json",
            schema_name="provider_raw_response",
            schema_version=1,
        )
        if raw_reference.sha256 != attempt.response_hash:
            raise LedgerConflictError("raw response hash mismatch")
        validated_bytes = canonical_json_bytes(validated_output)
        validated_reference = self.store.blobs.put(
            validated_bytes,
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name=output_schema_name,
            schema_version=output_schema_version,
        )
        result_bytes = canonical_json_bytes(attempt.model_dump(mode="json"))
        raw_bytes = canonical_json_bytes(raw_reference.model_dump(mode="json"))
        validated_reference_bytes = canonical_json_bytes(
            validated_reference.model_dump(mode="json")
        )
        usage = attempt.metrics.usage
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT status,request_context,result,raw_blob_reference,validated_output_reference FROM physical_llm_attempts WHERE attempt_id=?",
                (attempt.context.attempt_id,),
            ).fetchone()
            if row is None or row["status"] != "in_flight":
                raise LedgerConflictError("attempt is not in flight")
            if bytes(row["request_context"]) != canonical_json_bytes(
                attempt.context.model_dump(mode="json")
            ):
                raise LedgerConflictError("attempt context mismatch")
            connection.execute(
                "UPDATE physical_llm_attempts SET status='completed',result=?,raw_blob_reference=?,validated_output_reference=?,output_tokens=?,usage_complete=? WHERE attempt_id=?",
                (
                    result_bytes,
                    raw_bytes,
                    validated_reference_bytes,
                    usage.output_tokens,
                    int(usage.usage_complete),
                    attempt.context.attempt_id,
                ),
            )
            changed = connection.execute(
                "UPDATE logical_llm_requests SET status='response_completed' WHERE logical_request_id=? AND status='prepared'",
                (attempt.context.logical_request_id,),
            ).rowcount
            if changed != 1:
                raise LedgerConflictError("logical request cannot complete")

    def record_failure(self, error: ProviderRequestError) -> None:
        attempt = error.attempt
        mapping = {
            PhysicalAttemptStatus.REJECTED: AttemptLedgerStatus.REJECTED_BEFORE_DISPATCH,
            PhysicalAttemptStatus.FAILED: AttemptLedgerStatus.FAILED,
            PhysicalAttemptStatus.UNKNOWN_OUTCOME: AttemptLedgerStatus.UNKNOWN_OUTCOME,
        }
        status = mapping.get(attempt.status)
        if status is None:
            raise LedgerConflictError("invalid failure status")
        raw_reference = None
        if error.raw_response_body is not None:
            raw_reference = self.store.blobs.put(
                error.raw_response_body,
                media_type="application/vnd.toolsandbox.raw-provider+json",
                schema_name="provider_failure_raw_response",
                schema_version=1,
            )
            if raw_reference.sha256 != attempt.response_hash:
                raise LedgerConflictError("failure raw response hash mismatch")
        with self.store.transaction() as connection:
            changed = connection.execute(
                "UPDATE physical_llm_attempts SET status=?,result=?,raw_blob_reference=?,output_tokens=?,usage_complete=? WHERE attempt_id=? AND status IN ('allocated','in_flight')",
                (
                    status.value,
                    canonical_json_bytes(attempt.model_dump(mode="json")),
                    None
                    if raw_reference is None
                    else canonical_json_bytes(raw_reference.model_dump(mode="json")),
                    attempt.metrics.usage.output_tokens,
                    int(attempt.metrics.usage.usage_complete),
                    attempt.context.attempt_id,
                ),
            ).rowcount
            if changed != 1:
                raise LedgerConflictError("attempt failure transition rejected")
            if status is AttemptLedgerStatus.FAILED:
                connection.execute(
                    "UPDATE logical_llm_requests SET status='terminal_failure' WHERE logical_request_id=? AND status='prepared'",
                    (attempt.context.logical_request_id,),
                )

    def commit_checkpoint(
        self, checkpoint_id: str, event_kind: str, payload: dict
    ) -> CheckpointEvent:
        now = datetime.now(timezone.utc)
        encoded = canonical_json_bytes(payload)
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM checkpoint_events WHERE checkpoint_id=?", (checkpoint_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["event_kind"] != event_kind
                    or bytes(existing["payload"]) != encoded
                ):
                    raise LedgerConflictError("checkpoint identity conflict")
                return CheckpointEvent(
                    checkpoint_id=checkpoint_id,
                    event_ordinal=existing["event_ordinal"],
                    event_kind=event_kind,
                    payload=json.loads(existing["payload"]),
                    created_at_utc=datetime.fromisoformat(existing["created_at_utc"]),
                )
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(event_ordinal),0)+1 FROM checkpoint_events"
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO checkpoint_events VALUES(?,?,?,?,?)",
                (checkpoint_id, ordinal, event_kind, encoded, now.isoformat()),
            )
        return CheckpointEvent(
            checkpoint_id=checkpoint_id,
            event_ordinal=ordinal,
            event_kind=event_kind,
            payload=payload,
            created_at_utc=now,
        )

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointEvent | None:
        """Read one authoritative committed checkpoint by deterministic ID."""
        row = self.store._connection.execute(
            "SELECT * FROM checkpoint_events WHERE checkpoint_id=?", (checkpoint_id,)
        ).fetchone()
        if row is None:
            return None
        return CheckpointEvent(
            checkpoint_id=row["checkpoint_id"],
            event_ordinal=row["event_ordinal"],
            event_kind=row["event_kind"],
            payload=json.loads(row["payload"]),
            created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
        )

    def apply_response(
        self,
        *,
        logical_request_id: str,
        source_attempt_id: str,
        application_artifact_id: str,
        application_artifact_sha256: str,
        committed_checkpoint_id: str,
    ) -> LLMResponseApplication:
        application_id = derive_application_id(
            logical_request_id,
            source_attempt_id,
            application_artifact_id,
            application_artifact_sha256,
            committed_checkpoint_id,
        )
        now = datetime.now(timezone.utc)
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM llm_response_applications WHERE application_id=?",
                (application_id,),
            ).fetchone()
            if existing is not None:
                return self._application_from_row(existing)
            attempt = connection.execute(
                "SELECT logical_request_id,status FROM physical_llm_attempts WHERE attempt_id=?",
                (source_attempt_id,),
            ).fetchone()
            if (
                attempt is None
                or attempt["logical_request_id"] != logical_request_id
                or attempt["status"] != "completed"
            ):
                raise LedgerConflictError("source attempt is not the completed response")
            try:
                connection.execute(
                    "INSERT INTO llm_response_applications VALUES(?,?,?,?,?,?,?)",
                    (
                        application_id,
                        logical_request_id,
                        source_attempt_id,
                        application_artifact_id,
                        application_artifact_sha256,
                        committed_checkpoint_id,
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise LedgerConflictError("response already applied or checkpoint missing") from error
            changed = connection.execute(
                "UPDATE logical_llm_requests SET status='applied' WHERE logical_request_id=? AND status='response_completed'",
                (logical_request_id,),
            ).rowcount
            if changed != 1:
                raise LedgerConflictError("logical response cannot be applied")
        return LLMResponseApplication(
            application_id=application_id,
            logical_request_id=logical_request_id,
            source_attempt_id=source_attempt_id,
            application_artifact_id=application_artifact_id,
            application_artifact_sha256=application_artifact_sha256,
            committed_checkpoint_id=committed_checkpoint_id,
            created_at_utc=now,
        )

    def commit_checkpoint_and_apply(
        self,
        *,
        checkpoint_id: str,
        event_kind: str,
        checkpoint_payload: dict,
        logical_request_id: str,
        source_attempt_id: str,
        application_artifact_id: str,
        application_artifact_sha256: str,
    ) -> LLMResponseApplication:
        """Atomically commit post-application state and its at-most-once marker."""
        application_id = derive_application_id(
            logical_request_id,
            source_attempt_id,
            application_artifact_id,
            application_artifact_sha256,
            checkpoint_id,
        )
        now = datetime.now(timezone.utc)
        encoded_payload = canonical_json_bytes(checkpoint_payload)
        with self.store.transaction() as connection:
            existing_application = connection.execute(
                "SELECT * FROM llm_response_applications WHERE application_id=?",
                (application_id,),
            ).fetchone()
            if existing_application is not None:
                checkpoint = connection.execute(
                    "SELECT event_kind,payload FROM checkpoint_events WHERE checkpoint_id=?",
                    (checkpoint_id,),
                ).fetchone()
                if (
                    checkpoint is None
                    or checkpoint["event_kind"] != event_kind
                    or bytes(checkpoint["payload"]) != encoded_payload
                ):
                    raise LedgerConflictError("application checkpoint conflict")
                return self._application_from_row(existing_application)
            attempt = connection.execute(
                "SELECT logical_request_id,status FROM physical_llm_attempts WHERE attempt_id=?",
                (source_attempt_id,),
            ).fetchone()
            if (
                attempt is None
                or attempt["logical_request_id"] != logical_request_id
                or attempt["status"] != "completed"
            ):
                raise LedgerConflictError("source attempt is not the completed response")
            existing_checkpoint = connection.execute(
                "SELECT event_kind,payload FROM checkpoint_events WHERE checkpoint_id=?",
                (checkpoint_id,),
            ).fetchone()
            if existing_checkpoint is None:
                ordinal = connection.execute(
                    "SELECT COALESCE(MAX(event_ordinal),0)+1 FROM checkpoint_events"
                ).fetchone()[0]
                connection.execute(
                    "INSERT INTO checkpoint_events VALUES(?,?,?,?,?)",
                    (
                        checkpoint_id,
                        ordinal,
                        event_kind,
                        encoded_payload,
                        now.isoformat(),
                    ),
                )
            elif (
                existing_checkpoint["event_kind"] != event_kind
                or bytes(existing_checkpoint["payload"]) != encoded_payload
            ):
                raise LedgerConflictError("checkpoint identity conflict")
            try:
                connection.execute(
                    "INSERT INTO llm_response_applications VALUES(?,?,?,?,?,?,?)",
                    (
                        application_id,
                        logical_request_id,
                        source_attempt_id,
                        application_artifact_id,
                        application_artifact_sha256,
                        checkpoint_id,
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise LedgerConflictError("response already applied") from error
            changed = connection.execute(
                "UPDATE logical_llm_requests SET status='applied' WHERE logical_request_id=? AND status='response_completed'",
                (logical_request_id,),
            ).rowcount
            if changed != 1:
                raise LedgerConflictError("logical response cannot be applied")
        return LLMResponseApplication(
            application_id=application_id,
            logical_request_id=logical_request_id,
            source_attempt_id=source_attempt_id,
            application_artifact_id=application_artifact_id,
            application_artifact_sha256=application_artifact_sha256,
            committed_checkpoint_id=checkpoint_id,
            created_at_utc=now,
        )

    def load_completed_output(self, logical_request_id: str) -> StoredValidatedOutput:
        """Load a hash-verified output for response reuse; never expose raw bytes."""
        row = self.store._connection.execute(
            """SELECT a.attempt_id,a.validated_output_reference
               FROM logical_llm_requests r
               JOIN physical_llm_attempts a USING(logical_request_id)
               WHERE r.logical_request_id=? AND r.status IN ('response_completed','applied')
                 AND a.status='completed'
               ORDER BY a.attempt_ordinal DESC""",
            (logical_request_id,),
        ).fetchone()
        if row is None or row["validated_output_reference"] is None:
            raise LedgerConflictError("no completed validated output")
        from toolsandbox_pipeline.schemas.checkpoint import BlobReference

        reference = BlobReference.model_validate_json(
            row["validated_output_reference"], strict=True
        )
        output = json.loads(self.store.blobs.read(reference))
        return StoredValidatedOutput(
            logical_request_id=logical_request_id,
            source_attempt_id=row["attempt_id"],
            output=output,
            output_reference=reference,
        )

    def load_completed_response_material(
        self, logical_request_id: str
    ) -> CompletedResponseMaterial:
        """Return exact verified bytes without provider-specific deserialization."""
        row = self.store._connection.execute(
            """SELECT a.result,a.raw_blob_reference,a.validated_output_reference
               FROM logical_llm_requests r
               JOIN physical_llm_attempts a USING(logical_request_id)
               WHERE r.logical_request_id=? AND r.status IN ('response_completed','applied')
                 AND a.status='completed'
               ORDER BY a.attempt_ordinal DESC""",
            (logical_request_id,),
        ).fetchone()
        if (
            row is None
            or row["result"] is None
            or row["raw_blob_reference"] is None
            or row["validated_output_reference"] is None
        ):
            raise LedgerConflictError("no complete reusable response material")
        from toolsandbox_pipeline.providers.contracts import PhysicalAttemptResult
        from toolsandbox_pipeline.schemas.checkpoint import BlobReference

        attempt = PhysicalAttemptResult.model_validate_json(row["result"])
        raw_reference = BlobReference.model_validate_json(row["raw_blob_reference"])
        validated_reference = BlobReference.model_validate_json(
            row["validated_output_reference"]
        )
        raw = self.store.blobs.read(raw_reference)
        if attempt.response_hash != raw_reference.sha256:
            raise LedgerConflictError("stored response hash mismatch")
        validated = json.loads(self.store.blobs.read(validated_reference))
        return CompletedResponseMaterial(
            attempt=attempt,
            validated_output=validated,
            raw_response_body=raw,
        )

    def commit_checkpoint_and_effect(
        self,
        *,
        checkpoint_id: str,
        event_kind: str,
        checkpoint_payload: dict,
        effect_kind: QwenEffectKind,
        effect_artifact_id: str,
        effect_artifact_sha256: str,
        ordered_application_ids: tuple[str, ...],
    ) -> QwenEffectiveEffect:
        """Commit an externally meaningful effect and its checkpoint atomically."""
        effect_id = effective_effect_id(
            effect_kind=effect_kind,
            effect_artifact_id=effect_artifact_id,
            effect_artifact_sha256=effect_artifact_sha256,
            ordered_application_ids=ordered_application_ids,
            committed_checkpoint_id=checkpoint_id,
        )
        effect = QwenEffectiveEffect(
            effect_id=effect_id,
            effect_kind=effect_kind,
            effect_artifact_id=effect_artifact_id,
            effect_artifact_sha256=effect_artifact_sha256,
            ordered_application_ids=ordered_application_ids,
            committed_checkpoint_id=checkpoint_id,
        )
        encoded_payload = canonical_json_bytes(checkpoint_payload)
        now = datetime.now(timezone.utc)
        with self.store.transaction() as connection:
            existing_effect = connection.execute(
                "SELECT 1 FROM qwen_effective_effects WHERE effect_id=?", (effect_id,)
            ).fetchone()
            if existing_effect is not None:
                existing_checkpoint = connection.execute(
                    "SELECT event_kind,payload FROM checkpoint_events WHERE checkpoint_id=?",
                    (checkpoint_id,),
                ).fetchone()
                if (
                    existing_checkpoint is None
                    or existing_checkpoint["event_kind"] != event_kind
                    or bytes(existing_checkpoint["payload"]) != encoded_payload
                    or self.get_effect(effect_id, connection=connection) != effect
                ):
                    raise LedgerConflictError("effective-effect checkpoint conflict")
                return effect
            existing_checkpoint = connection.execute(
                "SELECT event_kind,payload FROM checkpoint_events WHERE checkpoint_id=?",
                (checkpoint_id,),
            ).fetchone()
            if existing_checkpoint is None:
                ordinal = connection.execute(
                    "SELECT COALESCE(MAX(event_ordinal),0)+1 FROM checkpoint_events"
                ).fetchone()[0]
                connection.execute(
                    "INSERT INTO checkpoint_events VALUES(?,?,?,?,?)",
                    (
                        checkpoint_id,
                        ordinal,
                        event_kind,
                        encoded_payload,
                        now.isoformat(),
                    ),
                )
            elif (
                existing_checkpoint["event_kind"] != event_kind
                or bytes(existing_checkpoint["payload"]) != encoded_payload
            ):
                raise LedgerConflictError("checkpoint identity conflict")
            self._insert_effect(connection, effect)
        return effect

    def request_status(self, logical_request_id: str) -> LogicalRequestStatus:
        row = self.store._connection.execute(
            "SELECT status FROM logical_llm_requests WHERE logical_request_id=?",
            (logical_request_id,),
        ).fetchone()
        if row is None:
            raise LedgerConflictError("unknown logical request")
        return LogicalRequestStatus(row["status"])

    def record_effect(
        self,
        *,
        effect_kind: QwenEffectKind,
        effect_artifact_id: str,
        effect_artifact_sha256: str,
        ordered_application_ids: tuple[str, ...],
        committed_checkpoint_id: str,
    ) -> QwenEffectiveEffect:
        effect_id = effective_effect_id(
            effect_kind=effect_kind,
            effect_artifact_id=effect_artifact_id,
            effect_artifact_sha256=effect_artifact_sha256,
            ordered_application_ids=ordered_application_ids,
            committed_checkpoint_id=committed_checkpoint_id,
        )
        effect = QwenEffectiveEffect(
            effect_id=effect_id,
            effect_kind=effect_kind,
            effect_artifact_id=effect_artifact_id,
            effect_artifact_sha256=effect_artifact_sha256,
            ordered_application_ids=ordered_application_ids,
            committed_checkpoint_id=committed_checkpoint_id,
        )
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM qwen_effective_effects WHERE effect_id=?", (effect_id,)
            ).fetchone()
            if existing is not None:
                stored = self.get_effect(effect_id, connection=connection)
                if stored != effect:
                    raise LedgerConflictError("effective-effect identity conflict")
                return stored
            checkpoint = connection.execute(
                "SELECT 1 FROM checkpoint_events WHERE checkpoint_id=?",
                (committed_checkpoint_id,),
            ).fetchone()
            if checkpoint is None:
                raise LedgerConflictError("committed checkpoint does not exist")
            self._insert_effect(connection, effect)
        return effect

    def record_non_substantive_outcome(
        self,
        *,
        outcome: NonSubstantiveOutcome,
        ordered_application_ids: tuple[str, ...],
    ) -> None:
        """Explicitly return no effect for NONE/SKIP/no-op/rejected outcomes."""
        if not isinstance(outcome, NonSubstantiveOutcome):
            raise TypeError("NonSubstantiveOutcome required")
        # Validate references for auditability without creating cost eligibility.
        with self.store.transaction() as connection:
            for application_id in ordered_application_ids:
                row = connection.execute(
                    "SELECT 1 FROM llm_response_applications WHERE application_id=?",
                    (application_id,),
                ).fetchone()
                if row is None:
                    raise LedgerConflictError("unknown response application")

    def get_effect(
        self, effect_id: str, *, connection=None
    ) -> QwenEffectiveEffect | None:
        database = connection or self.store._connection
        row = database.execute(
            "SELECT * FROM qwen_effective_effects WHERE effect_id=?", (effect_id,)
        ).fetchone()
        if row is None:
            return None
        applications = database.execute(
            "SELECT application_id FROM qwen_effect_applications WHERE effect_id=? ORDER BY application_ordinal",
            (effect_id,),
        ).fetchall()
        return QwenEffectiveEffect(
            effect_id=row["effect_id"],
            effect_kind=QwenEffectKind(row["effect_kind"]),
            effect_artifact_id=row["effect_artifact_id"],
            effect_artifact_sha256=row["effect_artifact_sha256"],
            ordered_application_ids=tuple(item["application_id"] for item in applications),
            committed_checkpoint_id=row["committed_checkpoint_id"],
        )

    def applications(self) -> tuple[LLMResponseApplication, ...]:
        """Return the authoritative immutable application projection."""
        rows = self.store._connection.execute(
            "SELECT * FROM llm_response_applications ORDER BY rowid"
        ).fetchall()
        return tuple(self._application_from_row(row) for row in rows)

    def effects(self) -> tuple[QwenEffectiveEffect, ...]:
        """Return committed effects in append order for pure accounting."""
        rows = self.store._connection.execute(
            "SELECT effect_id FROM qwen_effective_effects ORDER BY rowid"
        ).fetchall()
        return tuple(self.get_effect(row["effect_id"]) for row in rows)

    def effective_output_cost(self) -> tuple[int | None, bool]:
        row = self.store._connection.execute(
            """SELECT COUNT(*) AS count_all,
                      COUNT(a.output_tokens) AS count_known,
                      COALESCE(SUM(a.output_tokens),0) AS total
               FROM qwen_effect_applications e
               JOIN llm_response_applications x USING(application_id)
               JOIN physical_llm_attempts a ON a.attempt_id=x.source_attempt_id"""
        ).fetchone()
        complete = row["count_all"] == row["count_known"]
        return (row["total"] if complete else None, complete)

    @staticmethod
    def _provider_for_role(role) -> tuple[str, str]:
        from toolsandbox_pipeline.providers.contracts import ProviderRole

        if role is ProviderRole.EMBEDDING:
            return "openai", "embedding"
        if role is ProviderRole.USER_SIMULATOR:
            return "openai", "chat"
        if role in QWEN_GENERATION_ROLES:
            return "vllm_openai_compatible", "chat"
        raise LedgerConflictError("unsupported accounting provider role")

    @classmethod
    def _accounting_attempts(cls, connection, request_rows, scope):
        records = []
        for request in request_rows:
            identity = LogicalLLMRequestIdentity.model_validate_json(
                request["identity"], strict=True
            )
            if (
                identity.role.value != request["role"]
                or identity.model != request["model"]
            ):
                raise LedgerConflictError("logical request projection tampered")
            provider, endpoint = cls._provider_for_role(identity.role)
            rows = connection.execute(
                """SELECT * FROM physical_llm_attempts
                   WHERE logical_request_id=? ORDER BY attempt_ordinal""",
                (request["logical_request_id"],),
            ).fetchall()
            for row in rows:
                status = AttemptLedgerStatus(row["status"])
                if status is AttemptLedgerStatus.IN_FLIGHT:
                    raise LedgerConflictError(
                        "in-flight attempt cannot enter accounting snapshot"
                    )
                context = RequestContext.model_validate_json(
                    row["request_context"], strict=True
                )
                if (
                    context.logical_request_id != request["logical_request_id"]
                    or context.role is not identity.role
                    or context.phase != identity.phase
                    or context.unit_reference != identity.unit_reference
                    or context.input_fingerprint != identity.input_fingerprint
                ):
                    raise LedgerConflictError("physical attempt context tampered")
                result = None
                if row["result"] is not None:
                    result = PhysicalAttemptResult.model_validate_json(
                        row["result"], strict=True
                    )
                    expected = {
                        AttemptLedgerStatus.COMPLETED: PhysicalAttemptStatus.COMPLETED,
                        AttemptLedgerStatus.REJECTED_BEFORE_DISPATCH: PhysicalAttemptStatus.REJECTED,
                        AttemptLedgerStatus.FAILED: PhysicalAttemptStatus.FAILED,
                        AttemptLedgerStatus.UNKNOWN_OUTCOME: PhysicalAttemptStatus.UNKNOWN_OUTCOME,
                    }.get(status)
                    if result.context != context or result.status is not expected:
                        raise LedgerConflictError("physical attempt result tampered")
                elif status not in {
                    AttemptLedgerStatus.ALLOCATED,
                    AttemptLedgerStatus.ABANDONED_BEFORE_DISPATCH,
                    AttemptLedgerStatus.UNKNOWN_OUTCOME,
                }:
                    raise LedgerConflictError("physical attempt result missing")

                dispatched = status not in {
                    AttemptLedgerStatus.ALLOCATED,
                    AttemptLedgerStatus.ABANDONED_BEFORE_DISPATCH,
                    AttemptLedgerStatus.REJECTED_BEFORE_DISPATCH,
                }
                unknown_kind = None
                if status is AttemptLedgerStatus.UNKNOWN_OUTCOME:
                    exception = result.exception_class if result is not None else ""
                    if exception in {"TimeoutError", "APITimeoutError"}:
                        unknown_kind = "timeout"
                    elif exception in {"ConnectionError", "APIConnectionError"}:
                        unknown_kind = "connection"
                    else:
                        unknown_kind = "other"
                include_metrics = dispatched and result is not None
                records.append(
                    PhysicalAttemptAccountingInput(
                        run_id=scope.run_id,
                        round_index=scope.round_index,
                        task_id=scope.task_id,
                        scenario_family_id=scope.scenario_family_id,
                        scenario_id=scope.scenario_id,
                        system_variant=scope.system_variant,
                        logical_request_id=request["logical_request_id"],
                        attempt_id=row["attempt_id"],
                        attempt_ordinal=row["attempt_ordinal"],
                        role=identity.role.value,
                        phase=identity.phase,
                        provider=provider,
                        model=identity.model,
                        endpoint_kind=endpoint,
                        dispatched=dispatched,
                        replayed_after_unknown_outcome=
                            context.replayed_after_unknown_outcome,
                        status=status.value,
                        unknown_outcome_kind=unknown_kind,
                        started_at_utc=(
                            result.metrics.started_at if include_metrics else None
                        ),
                        completed_at_utc=(
                            result.metrics.completed_at if include_metrics else None
                        ),
                        latency_seconds=(
                            Decimal(str(result.metrics.latency_seconds))
                            if include_metrics
                            else None
                        ),
                        usage=result.metrics.usage if include_metrics else None,
                        response_sha256=(
                            result.response_hash
                            if status is AttemptLedgerStatus.COMPLETED
                            else None
                        ),
                    )
                )
        return tuple(records)

    @classmethod
    def _accounting_applications(cls, connection, logical_ids, request_rows):
        del request_rows
        if not logical_ids:
            return ()
        placeholders = ",".join("?" for _ in logical_ids)
        rows = connection.execute(
            f"""SELECT * FROM llm_response_applications
                WHERE logical_request_id IN ({placeholders}) ORDER BY rowid""",
            logical_ids,
        ).fetchall()
        applications = tuple(cls._application_from_row(row) for row in rows)
        if len({item.logical_request_id for item in applications}.difference(logical_ids)):
            raise LedgerConflictError("application scope mismatch")
        return applications

    @classmethod
    def _accounting_logical_requests(cls, request_rows, scope, applications):
        by_logical = {}
        for application in applications:
            if application.logical_request_id in by_logical:
                raise LedgerConflictError("logical response applied more than once")
            by_logical[application.logical_request_id] = application
        records = []
        for row in request_rows:
            identity = LogicalLLMRequestIdentity.model_validate_json(
                row["identity"], strict=True
            )
            provider, _ = cls._provider_for_role(identity.role)
            status = LogicalRequestStatus(row["status"])
            application = by_logical.get(row["logical_request_id"])
            if (status is LogicalRequestStatus.APPLIED) != (application is not None):
                raise LedgerConflictError("logical application projection mismatch")
            records.append(
                LogicalRequestAccountingInput(
                    run_id=scope.run_id,
                    round_index=scope.round_index,
                    task_id=scope.task_id,
                    scenario_family_id=scope.scenario_family_id,
                    scenario_id=scope.scenario_id,
                    system_variant=scope.system_variant,
                    logical_request_id=row["logical_request_id"],
                    role=identity.role.value,
                    phase=identity.phase,
                    provider=provider,
                    model=identity.model,
                    status=status.value,
                    source_attempt_id=(
                        application.source_attempt_id if application else None
                    ),
                    application_id=(
                        application.application_id if application else None
                    ),
                )
            )
        return tuple(records)

    def _accounting_effects(self, connection, application_ids):
        if not application_ids:
            return ()
        placeholders = ",".join("?" for _ in application_ids)
        rows = connection.execute(
            f"""SELECT DISTINCT e.effect_id,e.rowid
                FROM qwen_effective_effects e
                JOIN qwen_effect_applications x USING(effect_id)
                WHERE x.application_id IN ({placeholders}) ORDER BY e.rowid""",
            application_ids,
        ).fetchall()
        effects = tuple(
            self.get_effect(row["effect_id"], connection=connection) for row in rows
        )
        allowed = set(application_ids)
        if any(
            effect is None
            or not set(effect.ordered_application_ids).issubset(allowed)
            for effect in effects
        ):
            raise LedgerConflictError("effective effect crosses accounting scope")
        return effects

    @staticmethod
    def _insert_effect(connection, effect: QwenEffectiveEffect) -> None:
        for application_id in effect.ordered_application_ids:
            row = connection.execute(
                """SELECT r.role,r.status,a.status AS attempt_status
                   FROM llm_response_applications x
                   JOIN logical_llm_requests r USING(logical_request_id)
                   JOIN physical_llm_attempts a ON a.attempt_id=x.source_attempt_id
                   WHERE x.application_id=?""",
                (application_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "applied"
                or row["attempt_status"] != "completed"
            ):
                raise LedgerConflictError("effect application is not durably applied")
            from toolsandbox_pipeline.providers.contracts import ProviderRole

            if ProviderRole(row["role"]) not in QWEN_GENERATION_ROLES:
                raise LedgerConflictError("non-Qwen application cannot contribute cost")
        try:
            connection.execute(
                "INSERT INTO qwen_effective_effects VALUES(?,?,?,?,?)",
                (
                    effect.effect_id,
                    effect.effect_kind.value,
                    effect.effect_artifact_id,
                    effect.effect_artifact_sha256,
                    effect.committed_checkpoint_id,
                ),
            )
            connection.executemany(
                "INSERT INTO qwen_effect_applications VALUES(?,?,?)",
                [
                    (effect.effect_id, application_id, ordinal)
                    for ordinal, application_id in enumerate(
                        effect.ordered_application_ids
                    )
                ],
            )
        except sqlite3.IntegrityError as error:
            raise LedgerConflictError(
                "application already belongs to an effective effect"
            ) from error

    @staticmethod
    def _application_from_row(row) -> LLMResponseApplication:
        return LLMResponseApplication(
            application_id=row["application_id"],
            logical_request_id=row["logical_request_id"],
            source_attempt_id=row["source_attempt_id"],
            application_artifact_id=row["application_artifact_id"],
            application_artifact_sha256=row["application_artifact_sha256"],
            committed_checkpoint_id=row["committed_checkpoint_id"],
            created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
        )
