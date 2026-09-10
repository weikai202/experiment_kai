"""One fail-fast train round with direct timing and explicit component seams."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import Protocol

from toolsandbox_pipeline.metrics.timing import ScopeTimer
from toolsandbox_pipeline.offline.memory_orchestrator import (
    MemoryTrajectoryReference,
    SealedMemoryTrajectoryBuffer,
)
from toolsandbox_pipeline.offline.skill_orchestrator import (
    SealedSkillTrajectoryBuffer,
    SkillTrajectoryEntry,
)
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryUpdateIdentity,
    PolicyTrajectoryProjection,
    WorldTrajectoryProjection,
)
from toolsandbox_pipeline.schemas.offline_skill import (
    FailureModeLineageRecord,
    SkillFailureEvidence,
    SkillRoundIdentity,
)
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeExecutionStatus,
    EpisodeResult,
    TrustedTrajectory,
)


class EpisodeExecutor(Protocol):
    def run_scenario(
        self, *, scenario_id: str, manifest_position: int, generation_id: str,
    ): ...


class RoundBuffer(Protocol):
    def append(self, episode_result) -> None: ...
    def seal(self) -> "SealedRoundInputs": ...
    def archive_and_revoke(
        self,
        *,
        capability: "RawArchiveCapability",
        lineage_records: tuple[object, ...],
    ): ...


class RawArchiveCapability:
    """Opaque one-round authority; only its issuing RoundBuffer may consume it."""

    __slots__ = ("_issuer", "_round_index", "_revoked")

    def __init__(self, issuer: object, round_index: int) -> None:
        self._issuer = issuer
        self._round_index = round_index
        self._revoked = False

    def consume(self, issuer: object) -> None:
        if self._revoked or issuer is not self._issuer:
            raise PermissionError("raw trajectory archive capability denied")
        self._revoked = True


@dataclass(frozen=True)
class SealedRoundInputs:
    memory: SealedMemoryTrajectoryBuffer
    skill: SealedSkillTrajectoryBuffer
    archive_capability: RawArchiveCapability

    def __post_init__(self) -> None:
        if type(self.memory) is not SealedMemoryTrajectoryBuffer:
            raise TypeError("exact Task015 sealed buffer required")
        if type(self.skill) is not SealedSkillTrajectoryBuffer:
            raise TypeError("exact Task016 sealed buffer required")
        if type(self.archive_capability) is not RawArchiveCapability:
            raise TypeError("opaque raw archive capability required")


@dataclass(frozen=True)
class RoundArchiveReference:
    run_id: str
    round_index: int
    archive_manifest_sha256: str
    trajectory_reference_sha256: str
    lineage_records_sha256: str
    trajectory_count: int
    lineage_count: int


class TrajectoryLoader(Protocol):
    def load_trajectory(self, reference) -> TrustedTrajectory: ...


class MemoryProjectionAdapter(Protocol):
    def project(
        self, trajectory: TrustedTrajectory
    ) -> tuple[PolicyTrajectoryProjection, WorldTrajectoryProjection | None]: ...


class FailureEvidenceAdapter(Protocol):
    def project(self, trajectory: TrustedTrajectory) -> tuple[SkillFailureEvidence, ...]: ...


def _private_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class Task014RoundBuffer:
    """One-round Task014 adapter with distinct sealed views and revocable archive."""

    def __init__(
        self,
        *,
        run_id: str,
        round_index: int,
        shard_id: str,
        generation_id: str,
        dataset_manifest_sha256: str,
        config_manifest_sha256: str,
        memory_prompt_manifest_sha256: str,
        skill_prompt_manifest_sha256: str,
        skill_token_limit_config_sha256: str,
        archive_directory: Path,
        trajectory_loader: TrajectoryLoader,
        memory_projection: MemoryProjectionAdapter,
        failure_evidence: FailureEvidenceAdapter,
    ) -> None:
        expected_generation = f"g{round_index:03d}"
        if (
            round_index not in (0, 1, 2)
            or generation_id != expected_generation
            or shard_id != f"train-shard-{round_index}"
        ):
            raise ValueError("round buffer identity mismatch")
        if (
            not archive_directory.is_absolute()
            or archive_directory.exists()
            or archive_directory.is_symlink()
            or not archive_directory.parent.is_dir()
            or archive_directory.parent.is_symlink()
        ):
            raise ValueError("new absolute restricted archive directory required")
        self.run_id = run_id
        self.round_index = round_index
        self.shard_id = shard_id
        self.generation_id = generation_id
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.config_manifest_sha256 = config_manifest_sha256
        self.memory_prompt_manifest_sha256 = memory_prompt_manifest_sha256
        self.skill_prompt_manifest_sha256 = skill_prompt_manifest_sha256
        self.skill_token_limit_config_sha256 = skill_token_limit_config_sha256
        self.archive_directory = archive_directory
        self.trajectory_loader = trajectory_loader
        self.memory_projection = memory_projection
        self.failure_evidence = failure_evidence
        self._results: list[EpisodeResult] = []
        self._trajectories: tuple[TrustedTrajectory, ...] | None = None
        self._sealed: SealedRoundInputs | None = None
        self._revoked = False

    def append(self, episode_result) -> None:
        if self._revoked or self._sealed is not None:
            raise PermissionError("round buffer is closed")
        if (
            type(episode_result) is not EpisodeResult
            or episode_result.status is not EpisodeExecutionStatus.COMPLETED_EVALUATED
            or not episode_result.identity.phase.startswith("train")
            or episode_result.identity.run_id != self.run_id
            or episode_result.identity.round_index != self.round_index
            or episode_result.identity.shard_id != self.shard_id
            or episode_result.identity.generation_id != self.generation_id
            or episode_result.trusted_trajectory_reference is None
        ):
            raise ValueError("only current completed train episodes may enter round buffer")
        position = episode_result.identity.manifest_position
        if position != len(self._results):
            raise ValueError("episode results must follow exact manifest order")
        self._results.append(episode_result)

    def seal(self) -> SealedRoundInputs:
        if self._revoked:
            raise PermissionError("round buffer capability revoked")
        if self._sealed is not None:
            return self._sealed
        if not self._results:
            raise ValueError("cannot seal an empty round")
        trajectories = tuple(
            self.trajectory_loader.load_trajectory(result.trusted_trajectory_reference)
            for result in self._results
        )
        for result, trajectory in zip(self._results, trajectories):
            if (
                type(trajectory) is not TrustedTrajectory
                or trajectory.identity != result.identity
                or not trajectory.eligible_for_train_offline_consumption
                or trajectory.ending_context_sha256 != result.ending_context_sha256
            ):
                raise ValueError("loaded trajectory/result mismatch")

        memory_entries = []
        skill_entries = []
        for result, trajectory in zip(self._results, trajectories):
            policy, world = self.memory_projection.project(trajectory)
            memory_entries.append(
                MemoryTrajectoryReference(
                    trajectory_id=trajectory.trajectory_id,
                    trajectory_sha256=result.trusted_trajectory_reference.sha256,
                    run_id=self.run_id,
                    round_index=self.round_index,
                    shard_id=self.shard_id,
                    generation_id=self.generation_id,
                    dataset_manifest_sha256=self.dataset_manifest_sha256,
                    config_manifest_sha256=self.config_manifest_sha256,
                    split="train",
                    status="completed_evaluated",
                    eligible_for_train_offline_consumption=True,
                    manifest_position=trajectory.identity.manifest_position,
                    policy_projection=policy,
                    world_projection=world,
                )
            )
            skill_entries.append(
                SkillTrajectoryEntry(
                    trajectory=trajectory,
                    failure_evidence=self.failure_evidence.project(trajectory),
                )
            )

        memory_references = [
            {
                "trajectory_id": entry.trajectory_id,
                "trajectory_sha256": entry.trajectory_sha256,
                "manifest_position": entry.manifest_position,
            }
            for entry in memory_entries
        ]
        memory_buffer_sha256 = canonical_sha256(
            {
                "protocol": "sealed-memory-buffer-v1",
                "run_id": self.run_id,
                "round_index": self.round_index,
                "shard_id": self.shard_id,
                "generation_id": self.generation_id,
                "entries": memory_references,
            }
        )
        memory = SealedMemoryTrajectoryBuffer(
            identity=MemoryUpdateIdentity(
                run_id=self.run_id,
                round_index=self.round_index,
                shard_id=self.shard_id,
                current_generation_id=self.generation_id,
                next_generation_id=f"g{self.round_index + 1:03d}",
                dataset_manifest_sha256=self.dataset_manifest_sha256,
                config_manifest_sha256=self.config_manifest_sha256,
                prompt_manifest_sha256=self.memory_prompt_manifest_sha256,
                sealed_input_buffer_sha256=memory_buffer_sha256,
            ),
            entries=tuple(memory_entries),
        )
        skill_buffer_sha256 = canonical_sha256(
            {
                "protocol": "sealed-skill-buffer-v1",
                "run_id": self.run_id,
                "round_index": self.round_index,
                "shard_id": self.shard_id,
                "generation_id": self.generation_id,
                "entries": [
                    [
                        trajectory.trajectory_id,
                        canonical_sha256(trajectory.model_dump(mode="json")),
                        trajectory.identity.manifest_position,
                    ]
                    for trajectory in trajectories
                ],
            }
        )
        skill = SealedSkillTrajectoryBuffer(
            SkillRoundIdentity(
                run_id=self.run_id,
                round_index=self.round_index,
                shard_id=self.shard_id,
                current_generation_id=self.generation_id,
                next_generation_id=f"g{self.round_index + 1:03d}",
                dataset_manifest_sha256=self.dataset_manifest_sha256,
                config_manifest_sha256=self.config_manifest_sha256,
                prompt_manifest_sha256=self.skill_prompt_manifest_sha256,
                token_limit_config_sha256=self.skill_token_limit_config_sha256,
                sealed_input_buffer_sha256=skill_buffer_sha256,
            ),
            tuple(skill_entries),
        )
        self._trajectories = trajectories
        self._sealed = SealedRoundInputs(
            memory=memory,
            skill=skill,
            archive_capability=RawArchiveCapability(self, self.round_index),
        )
        return self._sealed

    def archive_and_revoke(
        self,
        *,
        capability: RawArchiveCapability,
        lineage_records: tuple[object, ...],
    ) -> RoundArchiveReference:
        if self._sealed is None or self._trajectories is None:
            raise ValueError("round must be sealed before archive")
        if any(type(record) is not FailureModeLineageRecord for record in lineage_records):
            raise TypeError("sanitized Task016 lineage records required")
        capability.consume(self)
        references = b"".join(
            canonical_json_bytes(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "trajectory_reference": result.trusted_trajectory_reference.model_dump(
                        mode="json"
                    ),
                }
            )
            + b"\n"
            for result, trajectory in zip(self._results, self._trajectories)
        )
        lineage = b"".join(
            canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
            for record in lineage_records
        )
        reference_hash = "sha256:" + sha256(references).hexdigest()
        lineage_hash = "sha256:" + sha256(lineage).hexdigest()
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "round_index": self.round_index,
            "trajectory_count": len(self._trajectories),
            "trajectory_reference_sha256": reference_hash,
            "lineage_count": len(lineage_records),
            "lineage_records_sha256": lineage_hash,
            "raw_provider_bodies_archived": False,
            "access_phase": "post_run_restricted_audit_only",
        }
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_hash = "sha256:" + sha256(manifest_bytes).hexdigest()
        self.archive_directory.mkdir(mode=0o700)
        _private_write(self.archive_directory / "trajectory_references.jsonl", references)
        _private_write(self.archive_directory / "failure_lineage.jsonl", lineage)
        _private_write(self.archive_directory / "archive_manifest.json", manifest_bytes)
        descriptor = os.open(self.archive_directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._results.clear()
        self._trajectories = None
        self._sealed = None
        self._revoked = True
        return RoundArchiveReference(
            run_id=self.run_id,
            round_index=self.round_index,
            archive_manifest_sha256=manifest_hash,
            trajectory_reference_sha256=reference_hash,
            lineage_records_sha256=lineage_hash,
            trajectory_count=manifest["trajectory_count"],
            lineage_count=manifest["lineage_count"],
        )


class MemoryUpdater(Protocol):
    def run(self, sealed_buffer): ...


class SkillUpdater(Protocol):
    def run(self, sealed_buffer): ...


class GenerationCoordinator(Protocol):
    def build_and_publish(
        self, *, round_index: int, input_generation_id: str,
        memory_result, skill_result,
    ): ...


@dataclass(frozen=True)
class RoundFinalCheckpointIdentity:
    """Exact identity durably committed after publication and before timer close."""

    run_id: str
    round_index: int
    train_shard_id: str
    input_generation_id: str
    published_generation_id: str

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or self.round_index not in (0, 1, 2)
            or self.train_shard_id != f"train-shard-{self.round_index}"
            or self.input_generation_id != f"g{self.round_index:03d}"
            or self.published_generation_id != f"g{self.round_index + 1:03d}"
        ):
            raise ValueError("final round checkpoint identity mismatch")


class RoundFinalizer(Protocol):
    def commit_final_round_checkpoint(
        self, identity: RoundFinalCheckpointIdentity,
    ): ...


class RoundMetricSink(Protocol):
    def finalize_round(
        self, *, round_index: int, input_generation_id: str, train_shard_id: str,
        published_generation_id: str | None, timing, completion_status: str,
        episode_results: tuple[object, ...], memory_result, skill_result,
    ): ...


@dataclass(frozen=True)
class RoundExecutionResult:
    round_index: int
    input_generation_id: str
    train_shard_id: str
    published_generation_id: str | None
    completion_status: str
    timing: object
    final_checkpoint: object | None
    metric_record: object
    archive_reference: object | None
    memory_result: object | None
    skill_result: object | None


class RoundExecutionError(RuntimeError):
    def __init__(self, result: RoundExecutionResult):
        super().__init__("training round failed")
        self.result = result


def _result_failed(value: object) -> bool:
    status = getattr(value, "completion_status", getattr(value, "status", None))
    return status in {"failed", "terminal_failure", "partial"}


class RoundRunner:
    def __init__(
        self, *, run_id: str, boot_id: str, episodes: EpisodeExecutor,
        buffer: RoundBuffer, memory: MemoryUpdater, skills: SkillUpdater,
        generations: GenerationCoordinator, metrics: RoundMetricSink,
        finalizer: RoundFinalizer,
        timer_factory=ScopeTimer,
    ) -> None:
        self.run_id = run_id
        self.boot_id = boot_id
        self.episodes = episodes
        self.buffer = buffer
        self.memory = memory
        self.skills = skills
        self.generations = generations
        self.metrics = metrics
        self.finalizer = finalizer
        self.timer_factory = timer_factory

    def run(
        self, *, round_index: int, train_shard_id: str,
        scenario_ids: tuple[str, ...], input_generation_id: str,
    ) -> RoundExecutionResult:
        expected_generation = f"g{round_index:03d}"
        if (
            round_index not in (0, 1, 2)
            or input_generation_id != expected_generation
            or train_shard_id != f"train-shard-{round_index}"
            or not scenario_ids
            or len(set(scenario_ids)) != len(scenario_ids)
        ):
            raise ValueError("round/generation/shard/scenario identity mismatch")
        timer = self.timer_factory(
            "round_total", f"{self.run_id}:round-{round_index}", self.boot_id,
        )
        results: list[object] = []
        memory_result = None
        skill_result = None
        published = None
        final_checkpoint = None
        timing = None
        metric = None
        metric_attempted = False
        try:
            for position, scenario_id in enumerate(scenario_ids):
                episode = self.episodes.run_scenario(
                    scenario_id=scenario_id,
                    manifest_position=position,
                    generation_id=input_generation_id,
                )
                if _result_failed(episode):
                    raise RuntimeError("terminal episode result")
                self.buffer.append(episode)
                results.append(episode)
            sealed = self.buffer.seal()
            if type(sealed) is not SealedRoundInputs:
                raise TypeError("RoundBuffer must return distinct Task015/016 sealed inputs")
            memory_result = self.memory.run(sealed.memory)
            if _result_failed(memory_result):
                raise RuntimeError("terminal memory update")
            skill_result = self.skills.run(sealed.skill)
            if _result_failed(skill_result):
                raise RuntimeError("terminal Skill update")
            publication = self.generations.build_and_publish(
                round_index=round_index,
                input_generation_id=input_generation_id,
                memory_result=memory_result,
                skill_result=skill_result,
            )
            published = getattr(publication, "generation_id", publication)
            if published != f"g{round_index + 1:03d}":
                raise RuntimeError("wrong published generation")
            final_checkpoint = self.finalizer.commit_final_round_checkpoint(
                RoundFinalCheckpointIdentity(
                    run_id=self.run_id,
                    round_index=round_index,
                    train_shard_id=train_shard_id,
                    input_generation_id=input_generation_id,
                    published_generation_id=published,
                )
            )
            timing = timer.close()
            metric_attempted = True
            metric = self.metrics.finalize_round(
                round_index=round_index,
                input_generation_id=input_generation_id,
                train_shard_id=train_shard_id,
                published_generation_id=published,
                timing=timing,
                completion_status="complete",
                episode_results=tuple(results),
                memory_result=memory_result,
                skill_result=skill_result,
            )
            lineages = tuple(getattr(skill_result, "lineage_records", ()))
            archive = self.buffer.archive_and_revoke(
                capability=sealed.archive_capability,
                lineage_records=lineages,
            )
            return RoundExecutionResult(
                round_index, input_generation_id, train_shard_id, published,
                "complete", timing, final_checkpoint, metric, archive,
                memory_result, skill_result,
            )
        except BaseException as error:
            if timing is None:
                timing = timer.close()
            if not metric_attempted:
                metric_attempted = True
                metric = self.metrics.finalize_round(
                    round_index=round_index,
                    input_generation_id=input_generation_id,
                    train_shard_id=train_shard_id,
                    published_generation_id=published,
                    timing=timing,
                    completion_status="failed",
                    episode_results=tuple(results),
                    memory_result=memory_result,
                    skill_result=skill_result,
                )
            partial = RoundExecutionResult(
                round_index, input_generation_id, train_shard_id, published,
                "failed", timing, final_checkpoint, metric, None,
                memory_result, skill_result,
            )
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise RoundExecutionError(partial) from error


__all__ = [
    "FailureEvidenceAdapter", "MemoryProjectionAdapter", "RawArchiveCapability",
    "RoundArchiveReference", "RoundExecutionError", "RoundExecutionResult",
    "RoundFinalCheckpointIdentity", "RoundFinalizer", "RoundRunner",
    "SealedRoundInputs", "Task014RoundBuffer",
    "TrajectoryLoader",
]
