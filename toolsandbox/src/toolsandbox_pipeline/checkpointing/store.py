"""Single-writer SQLite store for checkpoint and request ledgers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
import os
from pathlib import Path
import sqlite3
from typing import Iterator
import secrets

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes
from toolsandbox_pipeline.schemas.checkpoint import CheckpointConfig, RunIdentity
from toolsandbox_pipeline.checkpointing.blob_store import RestrictedBlobStore


APPLICATION_ID = 0x54534258
USER_VERSION = 2
V1_TABLES = {
    "run_identity",
    "checkpoint_events",
    "logical_llm_requests",
    "physical_llm_attempts",
    "llm_response_applications",
    "qwen_effective_effects",
    "qwen_effect_applications",
    "tool_action_transactions",
    "tool_execution_attempts",
    "offline_unit_transactions",
    "generation_publication_transactions",
}
EXPECTED_TABLES = V1_TABLES | {"llm_accounting_scopes"}


class CheckpointStoreError(RuntimeError):
    pass


def load_checkpoint_config(path: Path) -> CheckpointConfig:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise CheckpointStoreError("checkpoint config must be an absolute regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointStoreError("invalid checkpoint config") from error
    return CheckpointConfig.model_validate(payload, strict=True)


class CheckpointStore:
    def __init__(
        self,
        *,
        run_root: Path,
        identity: RunIdentity,
        config_path: Path,
        create: bool,
    ) -> None:
        if not run_root.is_absolute() or run_root.is_symlink():
            raise CheckpointStoreError("run root must be absolute and not a symlink")
        self.config = load_checkpoint_config(config_path)
        self.identity = identity
        if create:
            run_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        elif not run_root.is_dir():
            raise CheckpointStoreError("existing run root required")
        if run_root.stat().st_mode & 0o777 != 0o700:
            raise CheckpointStoreError("run root mode must be 0700")
        if run_root.stat().st_uid != os.getuid():
            raise CheckpointStoreError("run root owner mismatch")
        checkpoint_root = run_root / "checkpointing"
        checkpoint_root.mkdir(mode=0o700, exist_ok=True)
        os.chmod(checkpoint_root, 0o700)
        self.blobs = RestrictedBlobStore(run_root, self.config)
        self._database_path = checkpoint_root / "ledger.sqlite3"
        if self._database_path.is_symlink():
            raise CheckpointStoreError("ledger must not be a symlink")
        existed = self._database_path.exists()
        self._connection = sqlite3.connect(
            self._database_path, isolation_level=None, timeout=0
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._configure()
            if create:
                if existed:
                    raise CheckpointStoreError("new run already has a ledger")
                self._create_schema()
                self._insert_identity()
            else:
                self._migrate_schema()
                self._validate_schema_and_identity()
            os.chmod(self._database_path, 0o600)
        except Exception:
            self._connection.close()
            raise

    @classmethod
    def create(
        cls, run_root: Path, identity: RunIdentity, config_path: Path
    ) -> "CheckpointStore":
        return cls(
            run_root=run_root,
            identity=identity,
            config_path=config_path,
            create=True,
        )

    @classmethod
    def open(
        cls, run_root: Path, identity: RunIdentity, config_path: Path
    ) -> "CheckpointStore":
        return cls(
            run_root=run_root,
            identity=identity,
            config_path=config_path,
            create=False,
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def export_checkpoint(self, checkpoint_id: str) -> Path:
        """Materialize a replaceable derived snapshot from authoritative SQLite."""
        row = self._connection.execute(
            "SELECT * FROM checkpoint_events WHERE checkpoint_id=?", (checkpoint_id,)
        ).fetchone()
        if row is None:
            raise CheckpointStoreError("unknown checkpoint")
        event_payload = json.loads(row["payload"])
        snapshot = canonical_json_bytes(
            {
                "checkpoint_id": row["checkpoint_id"],
                "event_ordinal": row["event_ordinal"],
                "event_kind": row["event_kind"],
                "payload": event_payload,
                "created_at_utc": row["created_at_utc"],
            }
        )
        if len(snapshot) > self.config.max_checkpoint_bytes:
            raise CheckpointStoreError("checkpoint export exceeds configured limit")
        root = self._database_path.parent / "snapshots"
        root.mkdir(mode=0o700, exist_ok=True)
        os.chmod(root, 0o700)
        target = root / f"{row['event_ordinal']:012d}-{checkpoint_id}.json"
        self._atomic_replace(target, snapshot, root)
        digest = "sha256:" + hashlib.sha256(snapshot).hexdigest()
        latest = canonical_json_bytes(
            {
                "checkpoint_id": checkpoint_id,
                "event_ordinal": row["event_ordinal"],
                "snapshot_sha256": digest,
            }
        )
        self._atomic_replace(root / "latest.json", latest, root)
        return target

    def high_water_marks(self) -> dict[str, int]:
        queries = {
            "checkpoint_event_ordinal": "SELECT COALESCE(MAX(event_ordinal),0) FROM checkpoint_events",
            "logical_llm_requests": "SELECT COUNT(*) FROM logical_llm_requests",
            "llm_accounting_scopes": "SELECT COUNT(*) FROM llm_accounting_scopes",
            "physical_llm_attempts": "SELECT COUNT(*) FROM physical_llm_attempts",
            "llm_response_applications": "SELECT COUNT(*) FROM llm_response_applications",
            "qwen_effective_effects": "SELECT COUNT(*) FROM qwen_effective_effects",
            "tool_action_transactions": "SELECT COUNT(*) FROM tool_action_transactions",
            "tool_execution_attempts": "SELECT COUNT(*) FROM tool_execution_attempts",
            "offline_unit_transactions": "SELECT COUNT(*) FROM offline_unit_transactions",
            "generation_publication_transactions": "SELECT COUNT(*) FROM generation_publication_transactions",
        }
        return {
            name: int(self._connection.execute(query).fetchone()[0])
            for name, query in queries.items()
        }

    @staticmethod
    def _atomic_replace(target: Path, payload: bytes, parent: Path) -> None:
        if target.is_symlink():
            raise CheckpointStoreError("derived checkpoint path must not be a symlink")
        temporary = parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            os.chmod(target, 0o600)
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _configure(self) -> None:
        connection = self._connection
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=0")

    def _create_schema(self) -> None:
        schema = """
        CREATE TABLE run_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload BLOB NOT NULL);
        CREATE TABLE checkpoint_events (
          checkpoint_id TEXT PRIMARY KEY, event_ordinal INTEGER NOT NULL UNIQUE,
          event_kind TEXT NOT NULL, payload BLOB NOT NULL, created_at_utc TEXT NOT NULL
        );
        CREATE TABLE logical_llm_requests (
          logical_request_id TEXT PRIMARY KEY, identity BLOB NOT NULL,
          role TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL
        );
        CREATE TABLE llm_accounting_scopes (
          logical_request_id TEXT PRIMARY KEY, scope BLOB NOT NULL,
          FOREIGN KEY(logical_request_id) REFERENCES logical_llm_requests(logical_request_id)
        );
        CREATE TABLE physical_llm_attempts (
          attempt_id TEXT PRIMARY KEY, logical_request_id TEXT NOT NULL,
          attempt_ordinal INTEGER NOT NULL, status TEXT NOT NULL,
          request_context BLOB NOT NULL, result BLOB, raw_blob_reference BLOB,
          validated_output_reference BLOB, output_tokens INTEGER, usage_complete INTEGER,
          UNIQUE(logical_request_id, attempt_ordinal),
          FOREIGN KEY(logical_request_id) REFERENCES logical_llm_requests(logical_request_id)
        );
        CREATE TABLE llm_response_applications (
          application_id TEXT PRIMARY KEY, logical_request_id TEXT NOT NULL,
          source_attempt_id TEXT NOT NULL UNIQUE, application_artifact_id TEXT NOT NULL,
          application_artifact_sha256 TEXT NOT NULL, committed_checkpoint_id TEXT NOT NULL,
          created_at_utc TEXT NOT NULL,
          FOREIGN KEY(logical_request_id) REFERENCES logical_llm_requests(logical_request_id),
          FOREIGN KEY(source_attempt_id) REFERENCES physical_llm_attempts(attempt_id),
          FOREIGN KEY(committed_checkpoint_id) REFERENCES checkpoint_events(checkpoint_id)
        );
        CREATE TABLE qwen_effective_effects (
          effect_id TEXT PRIMARY KEY, effect_kind TEXT NOT NULL,
          effect_artifact_id TEXT NOT NULL, effect_artifact_sha256 TEXT NOT NULL,
          committed_checkpoint_id TEXT NOT NULL,
          FOREIGN KEY(committed_checkpoint_id) REFERENCES checkpoint_events(checkpoint_id)
        );
        CREATE TABLE qwen_effect_applications (
          effect_id TEXT NOT NULL, application_id TEXT NOT NULL UNIQUE,
          application_ordinal INTEGER NOT NULL,
          PRIMARY KEY(effect_id, application_ordinal),
          FOREIGN KEY(effect_id) REFERENCES qwen_effective_effects(effect_id),
          FOREIGN KEY(application_id) REFERENCES llm_response_applications(application_id)
        );
        CREATE TABLE tool_action_transactions (
          transaction_id TEXT PRIMARY KEY, identity BLOB NOT NULL,
          pre_context_reference BLOB NOT NULL, status TEXT NOT NULL,
          post_context_reference BLOB, post_context_sha256 TEXT,
          visible_result_identity TEXT, external_read_attempt_ids BLOB
        );
        CREATE TABLE tool_execution_attempts (
          attempt_id TEXT PRIMARY KEY, transaction_id TEXT NOT NULL,
          attempt_ordinal INTEGER NOT NULL, status TEXT NOT NULL,
          UNIQUE(transaction_id,attempt_ordinal),
          FOREIGN KEY(transaction_id) REFERENCES tool_action_transactions(transaction_id)
        );
        CREATE TABLE offline_unit_transactions (
          unit_id TEXT PRIMARY KEY, identity BLOB NOT NULL, status TEXT NOT NULL,
          request_ids BLOB NOT NULL, staged_outputs BLOB, committed_outputs BLOB,
          failure_class TEXT
        );
        CREATE TABLE generation_publication_transactions (
          publication_id TEXT PRIMARY KEY, identity BLOB NOT NULL, status TEXT NOT NULL,
          pre_checkpoint_id TEXT, post_checkpoint_id TEXT, failure_class TEXT,
          FOREIGN KEY(pre_checkpoint_id) REFERENCES checkpoint_events(checkpoint_id),
          FOREIGN KEY(post_checkpoint_id) REFERENCES checkpoint_events(checkpoint_id)
        );
        """
        self._connection.executescript(schema)
        self._connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        self._connection.execute(f"PRAGMA user_version={USER_VERSION}")

    def _insert_identity(self) -> None:
        payload = canonical_json_bytes(self.identity.model_dump(mode="json"))
        self._connection.execute(
            "INSERT INTO run_identity(singleton,payload) VALUES(1,?)", (payload,)
        )

    def _validate_schema_and_identity(self) -> None:
        if self._database_path.stat().st_mode & 0o777 != 0o600:
            raise CheckpointStoreError("ledger mode must be 0600")
        checks = {
            "application_id": APPLICATION_ID,
            "user_version": USER_VERSION,
            "foreign_keys": 1,
            "busy_timeout": 0,
        }
        for pragma, expected in checks.items():
            actual = self._connection.execute(f"PRAGMA {pragma}").fetchone()[0]
            if actual != expected:
                raise CheckpointStoreError(f"invalid SQLite {pragma}")
        if self._connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
            raise CheckpointStoreError("invalid SQLite journal mode")
        actual_tables = {
            row[0]
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if actual_tables != EXPECTED_TABLES:
            raise CheckpointStoreError("partial or unknown ledger schema")
        row = self._connection.execute(
            "SELECT payload FROM run_identity WHERE singleton=1"
        ).fetchone()
        expected = canonical_json_bytes(self.identity.model_dump(mode="json"))
        if row is None or bytes(row[0]) != expected:
            raise CheckpointStoreError("run identity mismatch")

    def _migrate_schema(self) -> None:
        """Apply only the reviewed v1-to-v2 additive attribution migration."""

        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version == USER_VERSION:
            return
        actual_tables = {
            row[0]
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if version != 1 or actual_tables != V1_TABLES:
            raise CheckpointStoreError("unsupported or partial ledger migration")
        with self.transaction() as connection:
            connection.execute(
                """CREATE TABLE llm_accounting_scopes (
                     logical_request_id TEXT PRIMARY KEY, scope BLOB NOT NULL,
                     FOREIGN KEY(logical_request_id)
                       REFERENCES logical_llm_requests(logical_request_id)
                   )"""
            )
            connection.execute(f"PRAGMA user_version={USER_VERSION}")
