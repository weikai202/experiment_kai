"""Crash-aware same-filesystem atomic publication of complete generations."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Protocol

from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.checkpoint import GenerationPublicationIdentity
from toolsandbox_pipeline.schemas.generation import GenerationManifest
from toolsandbox_pipeline.retrieval.index import file_hash


class PublicationLedger(Protocol):
    def prepare_publication(self, identity: GenerationPublicationIdentity): ...
    def commit_publication(
        self, publication_id: str, *, pre_checkpoint_id: str, post_checkpoint_id: str,
    ): ...


class CheckpointWriter(Protocol):
    def commit_checkpoint(self, checkpoint_id: str, event_kind: str, payload: dict): ...


class GenerationPublicationError(RuntimeError):
    pass


def _tree_identity(root: Path) -> tuple[str, str]:
    files = []
    from hashlib import sha256

    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        if path.is_symlink():
            raise GenerationPublicationError("generation contains a symlink")
        raw = path.read_bytes()
        files.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": "sha256:" + sha256(raw).hexdigest(),
        })
    if not files or files[0]["path"] != "manifest.json":
        raise GenerationPublicationError("complete generation manifest missing")
    manifest_hash = next(item["sha256"] for item in files if item["path"] == "manifest.json")
    return manifest_hash, canonical_sha256(["generation-content-v1", files])


def _validate_complete_layout(root: Path, expected_generation_id: str) -> GenerationManifest:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise GenerationPublicationError("complete generation manifest missing")
    try:
        manifest = GenerationManifest.model_validate_json(manifest_path.read_bytes(), strict=True)
    except Exception as error:
        raise GenerationPublicationError("invalid generation manifest") from error
    if manifest.generation_id != expected_generation_id or root.name != expected_generation_id:
        raise GenerationPublicationError("generation directory identity mismatch")
    expected = {"manifest.json", "retrieval_indexes", *(entry.path for entry in manifest.files)}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    if actual != expected:
        raise GenerationPublicationError("incomplete or unexpected generation layout")
    for directory in (root, root / "retrieval_indexes"):
        if directory.is_symlink() or stat.S_IMODE(directory.stat().st_mode) != 0o700:
            raise GenerationPublicationError("generation directories must be private")
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o600:
        raise GenerationPublicationError("generation files must be private")
    for entry in manifest.files:
        path = root / entry.path
        if (
            not path.is_file() or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise GenerationPublicationError("generation files must be private")
        raw = path.read_bytes()
        if file_hash(raw) != entry.sha256 or len(raw.splitlines()) != entry.record_count:
            raise GenerationPublicationError("generation file hash/count mismatch")
    return manifest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_latest(parent: Path, generation_id: str, content_sha256: str) -> None:
    target = parent / "latest.json"
    temporary = parent / f".latest.{generation_id}.tmp"
    payload = canonical_json_bytes({
        "generation_id": generation_id,
        "content_sha256": content_sha256,
    })
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        _fsync_directory(parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_generation(
    staged_generation: Path | str,
    generation_root: Path | str,
    *,
    run_id: str,
    round_index: int,
    input_generation_id: str,
    ledger: PublicationLedger,
    checkpoints: CheckpointWriter,
    mini_bench_decision_sha256: str | None = None,
):
    staged = Path(staged_generation)
    parent = Path(generation_root)
    if (
        not staged.is_absolute() or staged.is_symlink() or not staged.is_dir()
        or not parent.is_absolute() or parent.is_symlink() or not parent.is_dir()
        or staged.stat().st_dev != parent.stat().st_dev
    ):
        raise GenerationPublicationError("same-filesystem absolute staging and generation roots required")
    generation_id = staged.name
    expected = f"g{round_index + 1:03d}"
    if generation_id != expected or input_generation_id != f"g{round_index:03d}":
        raise GenerationPublicationError("round/generation publication mismatch")
    _validate_complete_layout(staged, generation_id)
    manifest_hash, content_hash = _tree_identity(staged)
    identity = GenerationPublicationIdentity(
        run_id=run_id,
        round_index=round_index,
        input_generation_id=input_generation_id,
        destination_generation_id=generation_id,
        staging_manifest_sha256=manifest_hash,
        staging_content_sha256=content_hash,
        mini_bench_decision_sha256=mini_bench_decision_sha256,
    )
    transaction = ledger.prepare_publication(identity)
    pre_id = "before-" + transaction.publication_id
    post_id = "after-" + transaction.publication_id
    payload = {
        "publication_id": transaction.publication_id,
        "generation_id": generation_id,
        "manifest_sha256": manifest_hash,
        "content_sha256": content_hash,
    }
    checkpoints.commit_checkpoint(pre_id, "before_generation_publication", payload)
    destination = parent / generation_id
    if destination.exists():
        try:
            matching = not destination.is_symlink() and _tree_identity(destination) == (
                manifest_hash, content_hash,
            )
        except GenerationPublicationError:
            matching = False
        if not matching:
            raise GenerationPublicationError("conflicting generation destination")
    else:
        os.replace(staged, destination)
        _fsync_directory(parent)
    checkpoints.commit_checkpoint(post_id, "after_generation_publication", payload)
    committed = ledger.commit_publication(
        transaction.publication_id,
        pre_checkpoint_id=pre_id,
        post_checkpoint_id=post_id,
    )
    _write_latest(parent, generation_id, content_hash)
    return committed


def publish_generation_zero(
    staged_generation: Path | str,
    generation_root: Path | str,
) -> tuple[str, str]:
    """Publish attested G000 before the first round; it has no producing round."""

    staged = Path(staged_generation)
    parent = Path(generation_root)
    if (
        not staged.is_absolute() or staged.name != "g000"
        or staged.is_symlink() or not staged.is_dir()
        or not parent.is_absolute() or parent.is_symlink() or not parent.is_dir()
        or staged.stat().st_dev != parent.stat().st_dev
    ):
        raise GenerationPublicationError("valid same-filesystem G000 staging required")
    _validate_complete_layout(staged, "g000")
    identity = _tree_identity(staged)
    destination = parent / "g000"
    if destination.exists():
        try:
            matching = not destination.is_symlink() and _tree_identity(destination) == identity
        except GenerationPublicationError:
            matching = False
        if not matching:
            raise GenerationPublicationError("conflicting G000 destination")
    else:
        os.replace(staged, destination)
        _fsync_directory(parent)
    _write_latest(parent, "g000", identity[1])
    return identity


__all__ = [
    "CheckpointWriter", "GenerationPublicationError", "PublicationLedger",
    "publish_generation", "publish_generation_zero",
]
