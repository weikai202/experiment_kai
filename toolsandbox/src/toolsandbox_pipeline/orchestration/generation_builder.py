"""Deterministically build a complete generation in a private staging parent."""

from __future__ import annotations

import os
from pathlib import Path

from toolsandbox_pipeline.memory.store import load_generation
from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.retrieval.index import build_indexes, file_hash, jsonl_bytes
from toolsandbox_pipeline.schemas.generation import (
    GenerationFileEntry,
    GenerationManifest,
    GenerationSnapshot,
    INDEX_PATHS,
    STORE_PATHS,
)
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory
from toolsandbox_pipeline.schemas.offline_memory import MemoryRoundResult
from toolsandbox_pipeline.schemas.offline_skill import SkillRoundResult
from toolsandbox_pipeline.schemas.skill import SkillRecord


class GenerationBuildError(RuntimeError):
    pass


def _private_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _policy_key(record: PolicyMemory) -> bytes:
    return record.memory_id.encode("utf-8")


def _world_key(record: WorldMemory) -> bytes:
    return record.memory_id.encode("utf-8")


def _skill_key(record: SkillRecord) -> tuple[bytes, int]:
    return record.skill_id.encode("utf-8"), int(record.version[3:])


def apply_generation_overlays(
    parent: GenerationSnapshot,
    *,
    policy_updates: tuple[PolicyMemory, ...] = (),
    world_updates: tuple[WorldMemory, ...] = (),
    skill_updates: tuple[SkillRecord, ...] = (),
) -> tuple[tuple[PolicyMemory, ...], tuple[WorldMemory, ...], tuple[SkillRecord, ...]]:
    """Apply authoritative staged records in their already deterministic order."""

    def replace(records, updates, key):
        if tuple(updates) != tuple(sorted(updates, key=key)):
            raise GenerationBuildError("overlay records must use deterministic order")
        update_keys = tuple(key(record) for record in updates)
        if len(set(update_keys)) != len(update_keys):
            raise GenerationBuildError("duplicate authoritative overlay key")
        by_key = {key(record): record for record in records}
        for record in updates:
            record_key = key(record)
            by_key[record_key] = record
        return tuple(by_key[item] for item in sorted(by_key))

    return (
        replace(parent.policy_memory, policy_updates, _policy_key),
        replace(parent.world_memory, world_updates, _world_key),
        replace(parent.skills, skill_updates, _skill_key),
    )


def apply_round_results(
    parent: GenerationSnapshot,
    *,
    run_id: str,
    round_index: int,
    train_shard_id: str,
    dataset_manifest_sha256: str,
    config_manifest_sha256: str,
    input_generation_id: str,
    output_generation_id: str,
    memory_result: MemoryRoundResult,
    skill_result: SkillRoundResult,
) -> tuple[tuple[PolicyMemory, ...], tuple[WorldMemory, ...], tuple[SkillRecord, ...]]:
    """Validate and flatten exact Task015/016 staged results for G+1."""

    if type(memory_result) is not MemoryRoundResult or type(skill_result) is not SkillRoundResult:
        raise GenerationBuildError("strict Task015/016 round results required")
    expected_output = f"g{round_index + 1:03d}"
    if (
        round_index not in (0, 1, 2)
        or parent.manifest.generation_id != input_generation_id
        or input_generation_id != f"g{round_index:03d}"
        or output_generation_id != expected_output
    ):
        raise GenerationBuildError("round generation identity mismatch")
    memory_identity = memory_result.identity
    skill_identity = skill_result.identity
    identities = (
        memory_identity.run_id,
        skill_identity.run_id,
        memory_identity.round_index,
        skill_identity.round_index,
        memory_identity.current_generation_id,
        skill_identity.current_generation_id,
        memory_identity.next_generation_id,
        skill_identity.next_generation_id,
    )
    if identities != (
        run_id,
        run_id,
        round_index,
        round_index,
        input_generation_id,
        input_generation_id,
        output_generation_id,
        output_generation_id,
    ):
        raise GenerationBuildError("Task015/016 identity mismatch")
    if (
        memory_identity.shard_id != train_shard_id
        or skill_identity.shard_id != train_shard_id
        or train_shard_id != f"train-shard-{round_index}"
        or memory_identity.dataset_manifest_sha256 != dataset_manifest_sha256
        or skill_identity.dataset_manifest_sha256 != dataset_manifest_sha256
        or memory_identity.config_manifest_sha256 != config_manifest_sha256
        or skill_identity.config_manifest_sha256 != config_manifest_sha256
        or skill_result.completion_status != "completed"
    ):
        raise GenerationBuildError("Task015/016 shard or manifest identity mismatch")

    skill_updates = tuple(
        record
        for mutation in skill_result.staged_mutations
        for record in mutation.staged_records
    )
    skill_updates = tuple(sorted(skill_updates, key=_skill_key))
    return apply_generation_overlays(
        parent,
        policy_updates=memory_result.staged_policy_memory,
        world_updates=memory_result.staged_world_memory,
        skill_updates=skill_updates,
    )


def build_generation(
    staging_parent: Path | str,
    *,
    generation_id: str,
    parent_generation_id: str | None,
    policy_memory: tuple[PolicyMemory, ...],
    world_memory: tuple[WorldMemory, ...],
    skills: tuple[SkillRecord, ...],
    tool_inventory: tuple[str, ...],
    tool_inventory_sha256: str,
    cache,
    gateway,
    context_factory,
    record_durable,
) -> GenerationSnapshot:
    parent = Path(staging_parent)
    if (
        not parent.is_absolute() or parent.is_symlink() or not parent.is_dir()
        or any(child for child in parent.iterdir())
    ):
        raise GenerationBuildError("new empty absolute staging parent required")
    if generation_id not in {"g000", "g001", "g002", "g003"}:
        raise GenerationBuildError("generation outside reviewed protocol")
    expected_parent = None if generation_id == "g000" else f"g{int(generation_id[1:]) - 1:03d}"
    if parent_generation_id != expected_parent:
        raise GenerationBuildError("parent generation mismatch")

    policy = tuple(sorted((PolicyMemory.model_validate(item) for item in policy_memory), key=_policy_key))
    world = tuple(sorted((WorldMemory.model_validate(item) for item in world_memory), key=_world_key))
    skill_records = tuple(sorted((SkillRecord.model_validate(item) for item in skills), key=_skill_key))
    scratch = parent / f".{generation_id}.build"
    completed = parent / generation_id
    scratch.mkdir(mode=0o700)
    try:
        store_blobs = (jsonl_bytes(policy), jsonl_bytes(world), jsonl_bytes(skill_records))
        store_entries = tuple(
            GenerationFileEntry(path=path, sha256=file_hash(raw), record_count=len(raw.splitlines()))
            for path, raw in zip(STORE_PATHS, store_blobs)
        )
        for path, raw in zip(STORE_PATHS, store_blobs):
            _private_write(scratch / path, raw)
        index_manifest = build_indexes(
            scratch,
            generation_id=generation_id,
            policy_memory=policy,
            world_memory=world,
            skills=skill_records,
            store_entries=store_entries,
            tool_inventory=tool_inventory,
            cache=cache,
            gateway=gateway,
            context_factory=context_factory,
            record_durable=record_durable,
        )
        os.chmod(scratch / "retrieval_indexes", 0o700)
        for path in INDEX_PATHS:
            os.chmod(scratch / path, 0o600)
        index_entries = tuple(
            GenerationFileEntry(
                path=path,
                sha256=file_hash((scratch / path).read_bytes()),
                record_count=len((scratch / path).read_bytes().splitlines()),
            )
            for path in INDEX_PATHS
        )
        manifest = GenerationManifest(
            schema_version=1,
            generation_id=generation_id,
            parent_generation_id=parent_generation_id,
            publication_status="complete",
            upstream_commit="165848b9a78cead7ca7fe7c89c688b58e6501219",
            tool_inventory_sha256=tool_inventory_sha256,
            embedding=index_manifest.embedding,
            vector_dimension=index_manifest.vector_dimension,
            files=store_entries + index_entries,
        )
        _private_write(scratch / "manifest.json", canonical_json_bytes(manifest.model_dump(mode="json")))
        os.replace(scratch, completed)
        return load_generation(
            completed,
            tool_inventory=tool_inventory,
            tool_inventory_sha256=tool_inventory_sha256,
            expected_generation_id=generation_id,
            expected_embedding=cache.identity,
        )
    except BaseException:
        # Keep any partial private staging tree for restricted forensic inspection.
        raise


__all__ = [
    "GenerationBuildError", "apply_generation_overlays", "apply_round_results",
    "build_generation",
]
