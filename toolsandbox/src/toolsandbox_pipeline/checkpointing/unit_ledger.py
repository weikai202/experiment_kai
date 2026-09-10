"""Idempotent offline-unit and generation-publication markers."""

from __future__ import annotations

import sqlite3

from toolsandbox_pipeline.checkpointing.identities import (
    offline_unit_id,
    publication_id,
)
from toolsandbox_pipeline.checkpointing.store import CheckpointStore
from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.checkpoint import (
    GenerationPublicationIdentity,
    GenerationPublicationRecord,
    OfflineUnitIdentity,
    OfflineUnitRecord,
    OfflineUnitStatus,
    PublicationStatus,
)


class UnitLedgerConflictError(RuntimeError):
    pass


class UnitLedger:
    def __init__(self, store: CheckpointStore):
        if type(store) is not CheckpointStore:
            raise TypeError("CheckpointStore required")
        self.store = store

    def prepare_unit(
        self, identity: OfflineUnitIdentity, *, request_ids: tuple[str, ...] = ()
    ) -> OfflineUnitRecord:
        if identity.run_id != self.store.identity.run_id:
            raise UnitLedgerConflictError("offline unit run mismatch")
        if len(set(request_ids)) != len(request_ids):
            raise UnitLedgerConflictError("duplicate request ID")
        payload = identity.model_dump(mode="json")
        unit_id = offline_unit_id(payload)
        encoded_identity = canonical_json_bytes(payload)
        encoded_requests = canonical_json_bytes(list(request_ids))
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM offline_unit_transactions WHERE unit_id=?", (unit_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO offline_unit_transactions(unit_id,identity,status,request_ids) VALUES(?,?,'prepared',?)",
                    (unit_id, encoded_identity, encoded_requests),
                )
                return OfflineUnitRecord(
                    unit_id=unit_id,
                    identity=identity,
                    status=OfflineUnitStatus.PREPARED,
                    request_ids=request_ids,
                )
            if (
                bytes(row["identity"]) != encoded_identity
                or bytes(row["request_ids"]) != encoded_requests
            ):
                raise UnitLedgerConflictError("offline unit identity conflict")
            return self._unit_record(row)

    def stage_outputs(
        self, unit_id: str, *, outputs: dict[str, str]
    ) -> OfflineUnitRecord:
        self._validate_output_hashes(outputs)
        encoded = canonical_json_bytes(outputs)
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM offline_unit_transactions WHERE unit_id=?", (unit_id,)
            ).fetchone()
            if row is None:
                raise UnitLedgerConflictError("unknown offline unit")
            if row["status"] == "outputs_staged":
                if bytes(row["staged_outputs"]) != encoded:
                    raise UnitLedgerConflictError("staged output conflict")
                return self._unit_record(row)
            if row["status"] != "prepared":
                raise UnitLedgerConflictError("offline unit cannot stage outputs")
            identity = OfflineUnitIdentity.model_validate_json(row["identity"])
            if set(outputs) != set(identity.expected_output_artifact_ids):
                raise UnitLedgerConflictError("unexpected staged artifact identities")
            connection.execute(
                "UPDATE offline_unit_transactions SET status='outputs_staged',staged_outputs=? WHERE unit_id=? AND status='prepared'",
                (encoded, unit_id),
            )
            updated = connection.execute(
                "SELECT * FROM offline_unit_transactions WHERE unit_id=?", (unit_id,)
            ).fetchone()
        return self._unit_record(updated)

    def commit_unit(
        self, unit_id: str, *, outputs: dict[str, str]
    ) -> OfflineUnitRecord:
        self._validate_output_hashes(outputs)
        encoded = canonical_json_bytes(outputs)
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM offline_unit_transactions WHERE unit_id=?", (unit_id,)
            ).fetchone()
            if row is None:
                raise UnitLedgerConflictError("unknown offline unit")
            if row["status"] == "committed":
                if bytes(row["committed_outputs"]) != encoded:
                    raise UnitLedgerConflictError("committed output conflict")
                return self._unit_record(row)
            if row["status"] != "outputs_staged" or bytes(row["staged_outputs"]) != encoded:
                raise UnitLedgerConflictError("only identical staged outputs may commit")
            connection.execute(
                "UPDATE offline_unit_transactions SET status='committed',committed_outputs=? WHERE unit_id=? AND status='outputs_staged'",
                (encoded, unit_id),
            )
            updated = connection.execute(
                "SELECT * FROM offline_unit_transactions WHERE unit_id=?", (unit_id,)
            ).fetchone()
        return self._unit_record(updated)

    def fail_unit(self, unit_id: str, *, failure_class: str) -> OfflineUnitRecord:
        self._validate_identifier(failure_class, "failure class")
        with self.store.transaction() as connection:
            changed = connection.execute(
                "UPDATE offline_unit_transactions SET status='terminal_failure',failure_class=? WHERE unit_id=? AND status IN ('prepared','outputs_staged')",
                (failure_class, unit_id),
            ).rowcount
            if changed != 1:
                raise UnitLedgerConflictError("offline unit cannot fail")
            row = connection.execute(
                "SELECT * FROM offline_unit_transactions WHERE unit_id=?", (unit_id,)
            ).fetchone()
        return self._unit_record(row)

    def prepare_publication(
        self, identity: GenerationPublicationIdentity
    ) -> GenerationPublicationRecord:
        if identity.run_id != self.store.identity.run_id:
            raise UnitLedgerConflictError("publication run mismatch")
        payload = identity.model_dump(mode="json")
        transaction_id = publication_id(payload)
        encoded = canonical_json_bytes(payload)
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM generation_publication_transactions WHERE publication_id=?",
                (transaction_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO generation_publication_transactions(publication_id,identity,status) VALUES(?,?,'prepared')",
                    (transaction_id, encoded),
                )
                return GenerationPublicationRecord(
                    publication_id=transaction_id,
                    identity=identity,
                    status=PublicationStatus.PREPARED,
                )
            if bytes(row["identity"]) != encoded:
                raise UnitLedgerConflictError("publication identity conflict")
            return self._publication_record(row)

    def commit_publication(
        self,
        publication_id: str,
        *,
        pre_checkpoint_id: str,
        post_checkpoint_id: str,
    ) -> GenerationPublicationRecord:
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM generation_publication_transactions WHERE publication_id=?",
                (publication_id,),
            ).fetchone()
            if row is None:
                raise UnitLedgerConflictError("unknown publication")
            if row["status"] == "committed":
                if (
                    row["pre_checkpoint_id"] != pre_checkpoint_id
                    or row["post_checkpoint_id"] != post_checkpoint_id
                ):
                    raise UnitLedgerConflictError("publication commit conflict")
                return self._publication_record(row)
            if row["status"] != "prepared":
                raise UnitLedgerConflictError("publication cannot commit")
            try:
                connection.execute(
                    "UPDATE generation_publication_transactions SET status='committed',pre_checkpoint_id=?,post_checkpoint_id=? WHERE publication_id=? AND status='prepared'",
                    (pre_checkpoint_id, post_checkpoint_id, publication_id),
                )
            except sqlite3.IntegrityError as error:
                raise UnitLedgerConflictError("publication checkpoint missing") from error
            updated = connection.execute(
                "SELECT * FROM generation_publication_transactions WHERE publication_id=?",
                (publication_id,),
            ).fetchone()
        return self._publication_record(updated)

    def fail_publication(
        self, publication_id: str, *, failure_class: str
    ) -> GenerationPublicationRecord:
        self._validate_identifier(failure_class, "failure class")
        with self.store.transaction() as connection:
            changed = connection.execute(
                "UPDATE generation_publication_transactions SET status='terminal_failure',failure_class=? WHERE publication_id=? AND status='prepared'",
                (failure_class, publication_id),
            ).rowcount
            if changed != 1:
                raise UnitLedgerConflictError("publication cannot fail")
            row = connection.execute(
                "SELECT * FROM generation_publication_transactions WHERE publication_id=?",
                (publication_id,),
            ).fetchone()
        return self._publication_record(row)

    @staticmethod
    def _unit_record(row) -> OfflineUnitRecord:
        import json

        return OfflineUnitRecord(
            unit_id=row["unit_id"],
            identity=OfflineUnitIdentity.model_validate_json(row["identity"]),
            status=OfflineUnitStatus(row["status"]),
            request_ids=tuple(json.loads(row["request_ids"])),
            staged_outputs=(
                None if row["staged_outputs"] is None else json.loads(row["staged_outputs"])
            ),
            committed_outputs=(
                None
                if row["committed_outputs"] is None
                else json.loads(row["committed_outputs"])
            ),
            failure_class=row["failure_class"],
        )

    @staticmethod
    def _publication_record(row) -> GenerationPublicationRecord:
        return GenerationPublicationRecord(
            publication_id=row["publication_id"],
            identity=GenerationPublicationIdentity.model_validate_json(row["identity"]),
            status=PublicationStatus(row["status"]),
            pre_checkpoint_id=row["pre_checkpoint_id"],
            post_checkpoint_id=row["post_checkpoint_id"],
            failure_class=row["failure_class"],
        )

    @staticmethod
    def _validate_output_hashes(outputs: dict[str, str]) -> None:
        if type(outputs) is not dict:
            raise UnitLedgerConflictError("outputs must be a mapping")
        for key, value in outputs.items():
            UnitLedger._validate_identifier(key, "artifact ID")
            if (
                type(value) is not str
                or len(value) != 71
                or not value.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in value[7:])
            ):
                raise UnitLedgerConflictError("invalid artifact hash")

    @staticmethod
    def _validate_identifier(value: str, label: str) -> None:
        if type(value) is not str or not value or any(c.isspace() for c in value):
            raise UnitLedgerConflictError(f"invalid {label}")


__all__ = ["UnitLedger", "UnitLedgerConflictError"]
