"""At-most-once native tool transaction ledger."""

from __future__ import annotations

import sqlite3

from toolsandbox_pipeline.checkpointing.identities import (
    tool_attempt_id,
    tool_transaction_id,
)
from toolsandbox_pipeline.checkpointing.store import CheckpointStore
from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.checkpoint import (
    BlobReference,
    ToolActionIdentity,
    ToolAttemptLedgerStatus,
    ToolRecoveryAction,
    ToolRecoveryPlan,
    ToolTransactionRecord,
    ToolTransactionStatus,
)


class ToolLedgerConflictError(RuntimeError):
    pass


class ToolLedger:
    def __init__(self, store: CheckpointStore):
        if type(store) is not CheckpointStore:
            raise TypeError("CheckpointStore required")
        self.store = store

    def prepare_transaction(
        self,
        identity: ToolActionIdentity,
        *,
        pre_context_reference: BlobReference,
    ) -> ToolTransactionRecord:
        if identity.run_id != self.store.identity.run_id:
            raise ToolLedgerConflictError("tool transaction run mismatch")
        transaction_id = tool_transaction_id(identity)
        encoded_identity = canonical_json_bytes(identity.model_dump(mode="json"))
        encoded_reference = canonical_json_bytes(
            pre_context_reference.model_dump(mode="json")
        )
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM tool_action_transactions WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO tool_action_transactions(transaction_id,identity,pre_context_reference,status) VALUES(?,?,?,'prepared')",
                    (transaction_id, encoded_identity, encoded_reference),
                )
                return ToolTransactionRecord(
                    transaction_id=transaction_id,
                    identity=identity,
                    pre_context_reference=pre_context_reference,
                    status=ToolTransactionStatus.PREPARED,
                )
            if (
                bytes(row["identity"]) != encoded_identity
                or bytes(row["pre_context_reference"]) != encoded_reference
            ):
                raise ToolLedgerConflictError("tool transaction identity conflict")
            return self._record(row)

    def allocate_attempt(self, transaction_id: str) -> str:
        with self.store.transaction() as connection:
            transaction = connection.execute(
                "SELECT status FROM tool_action_transactions WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()
            if transaction is None or transaction["status"] != "prepared":
                raise ToolLedgerConflictError("tool transaction is not prepared")
            prior = connection.execute(
                "SELECT COUNT(*) FROM tool_execution_attempts WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()[0]
            if prior:
                raise ToolLedgerConflictError("use recovery allocation after prior attempt")
            attempt_id = tool_attempt_id(transaction_id, 1)
            connection.execute(
                "INSERT INTO tool_execution_attempts VALUES(?,?,1,'allocated')",
                (attempt_id, transaction_id),
            )
        return attempt_id

    def mark_in_flight(self, attempt_id: str) -> None:
        with self.store.transaction() as connection:
            changed = connection.execute(
                "UPDATE tool_execution_attempts SET status='in_flight' WHERE attempt_id=? AND status='allocated'",
                (attempt_id,),
            ).rowcount
            if changed != 1:
                raise ToolLedgerConflictError("tool attempt cannot enter in-flight")

    def commit_attempt(
        self,
        attempt_id: str,
        *,
        post_context_reference: BlobReference,
        post_context_sha256: str,
        visible_result_identity: str,
        external_read_attempt_ids: tuple[str, ...] = (),
        failed: bool = False,
    ) -> ToolTransactionRecord:
        external_payload = canonical_json_bytes(list(external_read_attempt_ids))
        reference_payload = canonical_json_bytes(
            post_context_reference.model_dump(mode="json")
        )
        target = (
            ToolTransactionStatus.FAILED_COMMITTED
            if failed
            else ToolTransactionStatus.COMMITTED
        )
        with self.store.transaction() as connection:
            attempt = connection.execute(
                "SELECT transaction_id,status FROM tool_execution_attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if attempt is None or attempt["status"] != "in_flight":
                raise ToolLedgerConflictError("tool attempt is not in flight")
            transaction = connection.execute(
                "SELECT * FROM tool_action_transactions WHERE transaction_id=?",
                (attempt["transaction_id"],),
            ).fetchone()
            if transaction is None or transaction["status"] != "prepared":
                raise ToolLedgerConflictError("tool transaction cannot commit")
            connection.execute(
                "UPDATE tool_execution_attempts SET status='completed' WHERE attempt_id=?",
                (attempt_id,),
            )
            connection.execute(
                """UPDATE tool_action_transactions
                   SET status=?,post_context_reference=?,post_context_sha256=?,
                       visible_result_identity=?,external_read_attempt_ids=?
                   WHERE transaction_id=? AND status='prepared'""",
                (
                    target.value,
                    reference_payload,
                    post_context_sha256,
                    visible_result_identity,
                    external_payload,
                    attempt["transaction_id"],
                ),
            )
            committed = connection.execute(
                "SELECT * FROM tool_action_transactions WHERE transaction_id=?",
                (attempt["transaction_id"],),
            ).fetchone()
        return self._record(committed)

    def plan_recovery(self, transaction_id: str) -> ToolRecoveryPlan:
        transaction = self.store._connection.execute(
            "SELECT identity,status FROM tool_action_transactions WHERE transaction_id=?",
            (transaction_id,),
        ).fetchone()
        if transaction is None:
            raise ToolLedgerConflictError("unknown tool transaction")
        latest = self.store._connection.execute(
            """SELECT attempt_id,status FROM tool_execution_attempts
               WHERE transaction_id=? ORDER BY attempt_ordinal DESC LIMIT 1""",
            (transaction_id,),
        ).fetchone()
        status = ToolTransactionStatus(transaction["status"])
        if status in {
            ToolTransactionStatus.COMMITTED,
            ToolTransactionStatus.FAILED_COMMITTED,
        }:
            action = ToolRecoveryAction.RESTORE_COMMITTED_CONTEXT
        elif status is ToolTransactionStatus.RECONCILIATION_REQUIRED:
            action = ToolRecoveryAction.RECONCILIATION_REQUIRED
        elif latest is None or latest["status"] in {
            "allocated",
            "abandoned_before_execution",
        }:
            action = ToolRecoveryAction.EXECUTE_FROM_PRE_CONTEXT
        elif latest["status"] in {"in_flight", "unknown_outcome"}:
            identity = ToolActionIdentity.model_validate_json(transaction["identity"])
            if "external_read" not in identity.effect_classes:
                action = ToolRecoveryAction.REPLAY_LOCAL_FROM_PRE_CONTEXT
            elif identity.profile == "strict_replay":
                action = ToolRecoveryAction.REPLAY_FIXTURE_FROM_PRE_CONTEXT
            else:
                action = ToolRecoveryAction.RECONCILIATION_REQUIRED
        else:
            raise ToolLedgerConflictError("inconsistent tool attempt state")
        return ToolRecoveryPlan(
            transaction_id=transaction_id,
            action=action,
            prior_attempt_id=None if latest is None else latest["attempt_id"],
        )

    def reconcile_and_allocate_attempt(self, transaction_id: str) -> str:
        plan = self.plan_recovery(transaction_id)
        if plan.action in {
            ToolRecoveryAction.RESTORE_COMMITTED_CONTEXT,
            ToolRecoveryAction.RECONCILIATION_REQUIRED,
        }:
            if plan.action is ToolRecoveryAction.RECONCILIATION_REQUIRED:
                with self.store.transaction() as connection:
                    connection.execute(
                        "UPDATE tool_action_transactions SET status='reconciliation_required' WHERE transaction_id=? AND status='prepared'",
                        (transaction_id,),
                    )
            raise ToolLedgerConflictError(plan.action.value)
        with self.store.transaction() as connection:
            latest = connection.execute(
                """SELECT attempt_id,attempt_ordinal,status FROM tool_execution_attempts
                   WHERE transaction_id=? ORDER BY attempt_ordinal DESC LIMIT 1""",
                (transaction_id,),
            ).fetchone()
            if latest is not None:
                if latest["status"] == "allocated":
                    connection.execute(
                        "UPDATE tool_execution_attempts SET status='abandoned_before_execution' WHERE attempt_id=?",
                        (latest["attempt_id"],),
                    )
                elif latest["status"] == "in_flight":
                    connection.execute(
                        "UPDATE tool_execution_attempts SET status='unknown_outcome' WHERE attempt_id=?",
                        (latest["attempt_id"],),
                    )
                elif latest["status"] not in {
                    "abandoned_before_execution",
                    "unknown_outcome",
                }:
                    raise ToolLedgerConflictError("tool attempt is not recoverable")
            ordinal = 1 if latest is None else latest["attempt_ordinal"] + 1
            attempt_id = tool_attempt_id(transaction_id, ordinal)
            try:
                connection.execute(
                    "INSERT INTO tool_execution_attempts VALUES(?,?,?,'allocated')",
                    (attempt_id, transaction_id, ordinal),
                )
            except sqlite3.IntegrityError as error:
                raise ToolLedgerConflictError("tool attempt identity conflict") from error
        return attempt_id

    def restore_committed_context_reference(
        self, transaction_id: str
    ) -> BlobReference:
        row = self.store._connection.execute(
            "SELECT status,post_context_reference FROM tool_action_transactions WHERE transaction_id=?",
            (transaction_id,),
        ).fetchone()
        if (
            row is None
            or row["status"] not in {"committed", "failed_committed"}
            or row["post_context_reference"] is None
        ):
            raise ToolLedgerConflictError("no committed tool context")
        reference = BlobReference.model_validate_json(row["post_context_reference"])
        self.store.blobs.read(reference)
        return reference

    @staticmethod
    def _record(row) -> ToolTransactionRecord:
        return ToolTransactionRecord(
            transaction_id=row["transaction_id"],
            identity=ToolActionIdentity.model_validate_json(row["identity"]),
            pre_context_reference=BlobReference.model_validate_json(
                row["pre_context_reference"]
            ),
            status=ToolTransactionStatus(row["status"]),
            post_context_reference=(
                None
                if row["post_context_reference"] is None
                else BlobReference.model_validate_json(row["post_context_reference"])
            ),
            post_context_sha256=row["post_context_sha256"],
        )


__all__ = ["ToolLedger", "ToolLedgerConflictError"]
