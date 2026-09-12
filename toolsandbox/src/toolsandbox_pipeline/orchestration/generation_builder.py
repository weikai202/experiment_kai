"""Deterministically build a complete generation in a private staging parent."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from toolsandbox_pipeline.memory.store import load_generation
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
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


@dataclass(frozen=True)
class PublishedRoundGeneration:
    generation_id: str
    snapshot: GenerationSnapshot
    publication_record: object


class RoundGenerationCoordinator:
    """Connect validated offline results to the existing build/publication seams.

    One instance owns one fresh staging directory and one round. A successful
    repeated invocation is read-only; an interrupted invocation is left for the
    explicit publication recovery path, never silently restarted.
    """

    def __init__(
        self, *, run_id: str, round_index: int,
        dataset_manifest_sha256: str, config_manifest_sha256: str,
        parent_manifest_sha256: str, generation_root: Path,
        staging_parent: Path, tool_inventory: tuple[str, ...],
        tool_inventory_sha256: str, cache, gateway, context_factory,
        record_durable, ledger, checkpoints,
    ) -> None:
        if type(round_index) is not int or round_index not in (0, 1, 2):
            raise GenerationBuildError("round outside reviewed protocol")
        self.run_id = run_id
        self.round_index = round_index
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.config_manifest_sha256 = config_manifest_sha256
        self.parent_manifest_sha256 = parent_manifest_sha256
        self.generation_root = generation_root
        self.staging_parent = staging_parent
        self.tool_inventory = tool_inventory
        self.tool_inventory_sha256 = tool_inventory_sha256
        self.cache = cache
        self.build_dependencies = dict(
            gateway=gateway, context_factory=context_factory,
            record_durable=record_durable,
        )
        self.ledger = ledger
        self.checkpoints = checkpoints
        self._started_input = None
        self._completed = None

    def _load(self, generation_id):
        return load_generation(
            self.generation_root / generation_id,
            tool_inventory=self.tool_inventory,
            tool_inventory_sha256=self.tool_inventory_sha256,
            expected_generation_id=generation_id,
            expected_embedding=self.cache.identity,
        )

    def build_and_publish(
        self, *, round_index: int, input_generation_id: str,
        memory_result: MemoryRoundResult, skill_result: SkillRoundResult,
    ) -> PublishedRoundGeneration:
        from toolsandbox_pipeline.orchestration.generation_publisher import publish_generation

        if (round_index != self.round_index
                or input_generation_id != f"g{self.round_index:03d}"):
            raise GenerationBuildError("coordinator round identity mismatch")
        if type(memory_result) is not MemoryRoundResult or type(skill_result) is not SkillRoundResult:
            raise GenerationBuildError("strict Task015/016 round results required")
        # Revalidate persisted hashes and all nested invariants, including objects
        # made with model_copy/model_construct, before any embedding or write.
        memory_result = MemoryRoundResult.model_validate_json(memory_result.model_dump_json())
        skill_result = SkillRoundResult.model_validate_json(skill_result.model_dump_json())
        input_hash = canonical_sha256([
            memory_result.model_dump(mode="json"), skill_result.model_dump(mode="json"),
        ])
        output_id = f"g{self.round_index + 1:03d}"
        if self._started_input is not None:
            if input_hash != self._started_input:
                raise GenerationBuildError("coordinator result identity conflict")
            if self._completed is None:
                raise GenerationBuildError("interrupted publication requires explicit recovery")
            snapshot = self._load(output_id)
            if snapshot != self._completed.snapshot:
                raise GenerationBuildError("published generation changed")
            return self._completed

        for directory in (self.generation_root, self.staging_parent):
            if (not directory.is_absolute() or not directory.is_dir()
                    or directory.resolve() != directory):
                raise GenerationBuildError("absolute symlink-free coordinator directories required")
        if (any(self.staging_parent.iterdir())
                or self.staging_parent.stat().st_dev != self.generation_root.stat().st_dev):
            raise GenerationBuildError("new empty same-filesystem staging parent required")
        destination = self.generation_root / output_id
        if destination.exists() or destination.is_symlink():
            raise GenerationBuildError("existing destination requires explicit recovery")
        parent = self._load(input_generation_id)
        if (canonical_sha256(parent.manifest.model_dump(mode="json")) != self.parent_manifest_sha256
                or memory_result.source_generation_sha256 != self.parent_manifest_sha256):
            raise GenerationBuildError("source generation hash mismatch")
        active_skills = {item.skill_id: item for item in parent.skills if item.status == "active"}
        expected_skills = tuple(sorted(active_skills, key=lambda value: value.encode("utf-8")))
        if (skill_result.ordered_skill_ids != expected_skills
                or tuple(item.skill_id for item in skill_result.staged_mutations) != expected_skills
                or tuple(item.skill_id for item in skill_result.unit_results) != expected_skills):
            raise GenerationBuildError("Skill result coverage/order mismatch")
        for mutation, unit in zip(skill_result.staged_mutations, skill_result.unit_results):
            if (mutation.previous_record != active_skills[mutation.skill_id]
                    or unit.status == "terminal_failure"
                    or unit.staged_mutation_sha256 != canonical_sha256(mutation.model_dump(mode="json"))):
                raise GenerationBuildError("Skill result source/mutation mismatch")
        policy, world, skills = apply_round_results(
            parent, run_id=self.run_id, round_index=self.round_index,
            train_shard_id=f"train-shard-{self.round_index}",
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            config_manifest_sha256=self.config_manifest_sha256,
            input_generation_id=input_generation_id, output_generation_id=output_id,
            memory_result=memory_result, skill_result=skill_result,
        )
        self._started_input = input_hash
        staged = build_generation(
            self.staging_parent, generation_id=output_id,
            parent_generation_id=input_generation_id,
            policy_memory=policy, world_memory=world, skills=skills,
            tool_inventory=self.tool_inventory,
            tool_inventory_sha256=self.tool_inventory_sha256,
            cache=self.cache, **self.build_dependencies,
        )
        decisions = [
            mutation.mini_bench_result_sha256
            for mutation in skill_result.staged_mutations
            if mutation.mini_bench_result_sha256 is not None
        ]
        publication = publish_generation(
            self.staging_parent / output_id, self.generation_root,
            run_id=self.run_id, round_index=self.round_index,
            input_generation_id=input_generation_id, ledger=self.ledger,
            checkpoints=self.checkpoints,
            mini_bench_decision_sha256=(canonical_sha256(decisions) if decisions else None),
        )
        if getattr(publication, "status", None) != "committed":
            raise GenerationBuildError("publication did not commit")
        published = self._load(output_id)
        if published != staged:
            raise GenerationBuildError("published generation differs from validated staging")
        self._completed = PublishedRoundGeneration(output_id, published, publication)
        return self._completed


__all__ = [
    "GenerationBuildError", "PublishedRoundGeneration", "RoundGenerationCoordinator",
    "apply_generation_overlays", "apply_round_results",
    "build_generation",
]
