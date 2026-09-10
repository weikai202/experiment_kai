"""Restricted immutable content-addressed blob storage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets

from toolsandbox_pipeline.schemas.checkpoint import BlobReference, CheckpointConfig
from toolsandbox_pipeline.reproducibility import canonical_json_bytes


class BlobStoreError(RuntimeError):
    pass


class RestrictedBlobStore:
    def __init__(self, run_root: Path, config: CheckpointConfig):
        if not run_root.is_absolute():
            raise BlobStoreError("run root must be absolute")
        self._config = config
        self._root = run_root / "checkpointing" / "blobs" / "sha256"
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self._root, 0o700)

    def put(
        self,
        data: bytes,
        *,
        media_type: str,
        schema_name: str,
        schema_version: int,
    ) -> BlobReference:
        if type(data) is not bytes:
            raise TypeError("blob data must be bytes")
        if len(data) > self._config.max_blob_bytes:
            raise BlobStoreError("blob exceeds configured limit")
        self._validate_content(data, media_type)
        digest = hashlib.sha256(data).hexdigest()
        parent = self._root / digest[:2]
        parent.mkdir(mode=0o700, exist_ok=True)
        os.chmod(parent, 0o700)
        destination = parent / digest[2:]
        if destination.is_symlink():
            raise BlobStoreError("blob path must not be a symlink")
        if destination.exists():
            self._verify_path(destination, data, digest)
        else:
            temporary = parent / f".{digest}.{secrets.token_hex(8)}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    self._verify_path(destination, data, digest)
                else:
                    os.chmod(destination, 0o600)
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
        return BlobReference(
            sha256=f"sha256:{digest}",
            byte_count=len(data),
            media_type=media_type,
            schema_name=schema_name,
            schema_version=schema_version,
            content_visibility="restricted",
        )

    def read(self, reference: BlobReference) -> bytes:
        digest = reference.sha256.removeprefix("sha256:")
        path = self._root / digest[:2] / digest[2:]
        if path.is_symlink() or not path.is_file():
            raise BlobStoreError("missing or unsafe blob")
        if path.stat().st_mode & 0o777 != 0o600:
            raise BlobStoreError("wrong blob mode")
        if path.stat().st_uid != os.getuid():
            raise BlobStoreError("wrong blob owner")
        data = path.read_bytes()
        if len(data) != reference.byte_count:
            raise BlobStoreError("blob size mismatch")
        if hashlib.sha256(data).hexdigest() != digest:
            raise BlobStoreError("blob hash mismatch")
        return data

    @staticmethod
    def _validate_content(data: bytes, media_type: str) -> None:
        canonical_types = {
            "application/vnd.toolsandbox.canonical+json",
            "application/vnd.toolsandbox.execution-context+json",
        }
        if media_type in canonical_types:
            try:
                payload = json.loads(data)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise BlobStoreError("invalid canonical JSON blob") from error
            if canonical_json_bytes(payload) != data:
                raise BlobStoreError("noncanonical JSON blob")
        elif media_type == "application/vnd.toolsandbox.jsonl":
            if data and not data.endswith(b"\n"):
                raise BlobStoreError("JSONL must end with newline")
            for line in data.splitlines():
                try:
                    payload = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise BlobStoreError("invalid JSONL blob") from error
                if canonical_json_bytes(payload) != line:
                    raise BlobStoreError("noncanonical JSONL blob")
        elif media_type == "application/vnd.toolsandbox.raw-provider+json":
            try:
                json.loads(data)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise BlobStoreError("invalid raw provider JSON") from error

    @staticmethod
    def _verify_path(path: Path, expected: bytes, digest: str) -> None:
        if path.stat().st_mode & 0o777 != 0o600:
            raise BlobStoreError("wrong blob mode")
        if path.stat().st_uid != os.getuid():
            raise BlobStoreError("wrong blob owner")
        existing = path.read_bytes()
        if existing != expected or hashlib.sha256(existing).hexdigest() != digest:
            raise BlobStoreError("blob collision")
