from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from toolsandbox_pipeline.offline.memory_orchestrator import SealedMemoryTrajectoryBuffer
from toolsandbox_pipeline.offline.skill_orchestrator import SealedSkillTrajectoryBuffer
from toolsandbox_pipeline.orchestration.round_runner import (
    RawArchiveCapability,
    RoundExecutionError,
    RoundFinalCheckpointIdentity,
    RoundRunner,
    SealedRoundInputs,
    Task014RoundBuffer,
)
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.accounting import ScopeTimingInput, TaskAccountingInput
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryUpdateIdentity,
    PolicyTrajectoryProjection,
)
from toolsandbox_pipeline.schemas.offline_skill import SkillRoundIdentity
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeExecutionStatus,
    EpisodeIdentity,
    EpisodeResult,
    TrustedTrajectory,
)


DIGEST = "sha256:" + "d" * 64


class Episodes:
    def __init__(self, events, fail_at=None):
        self.events, self.fail_at = events, fail_at

    def run_scenario(self, *, scenario_id, manifest_position, generation_id):
        self.events.append(("episode", scenario_id, manifest_position, generation_id))
        status = "failed" if manifest_position == self.fail_at else "complete"
        return SimpleNamespace(
            status=status,
            scenario_id=scenario_id,
            generation_id=generation_id,
        )


class Buffer:
    def __init__(self, events):
        self.events, self.rows, self.revoked = events, [], False

    def append(self, result):
        self.events.append(("append", result.scenario_id))
        self.rows.append(result)

    def seal(self):
        self.events.append(("seal",))
        round_index = int(self.rows[0].generation_id[1:])
        shard = f"train-shard-{round_index}"
        memory_hash = canonical_sha256(
            {
                "protocol": "sealed-memory-buffer-v1",
                "run_id": "run",
                "round_index": round_index,
                "shard_id": shard,
                "generation_id": self.rows[0].generation_id,
                "entries": [],
            }
        )
        memory = SealedMemoryTrajectoryBuffer(
            identity=MemoryUpdateIdentity(
                run_id="run",
                round_index=round_index,
                shard_id=shard,
                current_generation_id=self.rows[0].generation_id,
                next_generation_id=f"g{round_index + 1:03d}",
                dataset_manifest_sha256=DIGEST,
                config_manifest_sha256=DIGEST,
                prompt_manifest_sha256=DIGEST,
                sealed_input_buffer_sha256=memory_hash,
            ),
            entries=(),
        )
        skill_hash = canonical_sha256(
            {
                "protocol": "sealed-skill-buffer-v1",
                "run_id": "run",
                "round_index": round_index,
                "shard_id": shard,
                "generation_id": self.rows[0].generation_id,
                "entries": [],
            }
        )
        skill = SealedSkillTrajectoryBuffer(
            SkillRoundIdentity(
                run_id="run",
                round_index=round_index,
                shard_id=shard,
                current_generation_id=self.rows[0].generation_id,
                next_generation_id=f"g{round_index + 1:03d}",
                dataset_manifest_sha256=DIGEST,
                config_manifest_sha256=DIGEST,
                prompt_manifest_sha256=DIGEST,
                token_limit_config_sha256=DIGEST,
                sealed_input_buffer_sha256=skill_hash,
            ),
            (),
        )
        return SealedRoundInputs(
            memory=memory,
            skill=skill,
            archive_capability=RawArchiveCapability(self, round_index),
        )

    def archive_and_revoke(self, *, capability, lineage_records):
        capability.consume(self)
        self.events.append(("archive", lineage_records))
        self.revoked = True
        return "archive"


class Update:
    def __init__(self, events, name):
        self.events, self.name = events, name

    def run(self, sealed):
        expected = (
            SealedMemoryTrajectoryBuffer if self.name == "memory"
            else SealedSkillTrajectoryBuffer
        )
        assert type(sealed) is expected
        self.events.append((self.name,))
        return SimpleNamespace(status="complete", lineage_records=("lineage",) if self.name == "skill" else ())


class Generations:
    def __init__(self, events):
        self.events = events

    def build_and_publish(self, *, round_index, input_generation_id, memory_result, skill_result):
        self.events.append(("publish", round_index, input_generation_id))
        return f"g{round_index + 1:03d}"


class Metrics:
    def __init__(self, events):
        self.events, self.calls = events, []

    def finalize_round(self, **values):
        self.events.append(("metrics", values["completion_status"]))
        self.calls.append(values)
        return values


class Finalizer:
    def __init__(self, events, *, crash_after_first_commit=False):
        self.events = events
        self.crash_after_first_commit = crash_after_first_commit
        self.records = {}

    def commit_final_round_checkpoint(self, identity):
        assert type(identity) is RoundFinalCheckpointIdentity
        key = (
            identity.run_id,
            identity.round_index,
            identity.train_shard_id,
            identity.input_generation_id,
            identity.published_generation_id,
        )
        self.events.append(("checkpoint", key))
        record = self.records.setdefault(key, SimpleNamespace(identity=identity))
        if self.crash_after_first_commit:
            self.crash_after_first_commit = False
            raise RuntimeError("injected crash after durable commit")
        return record


class Timer:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def close(self):
        if self.closed:
            raise RuntimeError("timer closed twice")
        self.closed = True
        self.events.append(("timer_close",))
        return SimpleNamespace(timing_complete=True, total_running_time_seconds=1.0)


def runner(events, fail_at=None, *, finalizer=None):
    metrics = Metrics(events)
    finalizer = finalizer or Finalizer(events)
    value = RoundRunner(
        run_id="run",
        boot_id="boot",
        episodes=Episodes(events, fail_at),
        buffer=Buffer(events),
        memory=Update(events, "memory"),
        skills=Update(events, "skill"),
        generations=Generations(events),
        metrics=metrics,
        finalizer=finalizer,
        timer_factory=lambda *_: Timer(events),
    )
    return value, metrics


def test_round_preserves_manifest_order_memory_before_skill_and_archives_last():
    events = []
    value, metrics = runner(events)
    result = value.run(
        round_index=1,
        train_shard_id="train-shard-1",
        scenario_ids=("a", "b"),
        input_generation_id="g001",
    )
    assert result.published_generation_id == "g002"
    names = [event[0] for event in events]
    assert names == [
        "episode", "append", "episode", "append", "seal",
        "memory", "skill", "publish", "checkpoint", "timer_close",
        "metrics", "archive",
    ]
    assert result.timing.timing_complete
    assert metrics.calls[0]["timing"].total_running_time_seconds is not None
    with pytest.raises(PermissionError, match="capability denied"):
        result.archive_reference  # archive exposes no raw capability
        value.buffer.archive_and_revoke(
            capability=RawArchiveCapability(object(), 1),
            lineage_records=(),
        )


def test_terminal_episode_fails_fast_and_still_emits_direct_partial_round_row():
    events = []
    value, metrics = runner(events, fail_at=1)
    with pytest.raises(RoundExecutionError) as caught:
        value.run(
            round_index=0,
            train_shard_id="train-shard-0",
            scenario_ids=("a", "b", "c"),
            input_generation_id="g000",
        )
    assert caught.value.result.completion_status == "failed"
    assert [event[:2] for event in events if event[0] == "episode"] == [
        ("episode", "a"), ("episode", "b"),
    ]
    assert metrics.calls[0]["completion_status"] == "failed"
    assert metrics.calls[0]["timing"].total_running_time_seconds is not None


def test_crash_after_durable_final_checkpoint_reuses_the_exact_identity_on_retry():
    events = []
    finalizer = Finalizer(events, crash_after_first_commit=True)
    first, first_metrics = runner(events, finalizer=finalizer)
    with pytest.raises(RoundExecutionError) as caught:
        first.run(
            round_index=2,
            train_shard_id="train-shard-2",
            scenario_ids=("a",),
            input_generation_id="g002",
        )
    assert caught.value.result.published_generation_id == "g003"
    assert len(finalizer.records) == 1
    assert first_metrics.calls[0]["completion_status"] == "failed"

    second, _ = runner(events, finalizer=finalizer)
    result = second.run(
        round_index=2,
        train_shard_id="train-shard-2",
        scenario_ids=("a",),
        input_generation_id="g002",
    )
    assert result.final_checkpoint is next(iter(finalizer.records.values()))
    checkpoints = [event for event in events if event[0] == "checkpoint"]
    assert len(checkpoints) == 2
    assert checkpoints[0][1] == checkpoints[1][1]
    assert len(finalizer.records) == 1


def test_task014_reference_bridge_seals_distinct_views_and_revokes_archive(tmp_path):
    blob = BlobReference(
        sha256=DIGEST,
        byte_count=1,
        media_type="application/vnd.toolsandbox.canonical+json",
        schema_name="TrustedTrajectory",
        schema_version=1,
        content_visibility="restricted",
    )
    identity = EpisodeIdentity(
        run_id="run",
        profile="offline",
        phase="train_round",
        round_index=0,
        shard_id="train-shard-0",
        family_id="family",
        scenario_id="scenario",
        episode_id="episode",
        manifest_position=0,
        system_variant="generation_0",
        generation_id="g000",
        starting_context_sha256=DIGEST,
        evaluation_definition_sha256=DIGEST,
        agent_tool_schema_sha256=DIGEST,
        dataset_manifest_sha256=DIGEST,
        runtime_config_sha256=DIGEST,
        prompt_manifest_sha256=DIGEST,
        token_limit_config_sha256=DIGEST,
        fixture_manifest_sha256=DIGEST,
        environment_sha256=DIGEST,
        max_messages=4,
    )
    trajectory = TrustedTrajectory.build(
        identity=identity,
        messages=(),
        online_turns=(),
        tool_actions=(),
        logical_request_ids=(),
        physical_attempt_ids=(),
        ending_context_reference=blob,
        ending_context_sha256=DIGEST,
        evaluator_record_reference=blob,
        evaluator_record_sha256=DIGEST,
        skill_attributions=(),
        eligible_for_train_offline_consumption=True,
    )
    timing = ScopeTimingInput(
        scope_kind="scenario_task",
        scope_id="episode",
        boot_id="boot",
        started_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        start_monotonic_ns=0,
    )
    result = EpisodeResult(
        identity=identity,
        status=EpisodeExecutionStatus.COMPLETED_EVALUATED,
        ending_context_reference=blob,
        ending_context_sha256=DIGEST,
        trusted_trajectory_reference=blob,
        evaluator_record_reference=blob,
        task_accounting_input=TaskAccountingInput(
            run_id="run",
            task_id="episode",
            scenario_family_id="family",
            scenario_id="scenario",
            system_variant="generation_0",
            timing=timing,
            completion_status="complete",
            evaluator_result_sha256=DIGEST,
            final_context_sha256=DIGEST,
        ),
        last_checkpoint_ordinal=1,
    )

    class Loader:
        def load_trajectory(self, reference):
            assert reference == blob
            return trajectory

    class MemoryProjection:
        def project(self, value):
            assert value == trajectory
            return (
                PolicyTrajectoryProjection(
                    trajectory_id=trajectory.trajectory_id,
                    manifest_position=0,
                    visible_states=(),
                    retrieved_policy_memory=(),
                    retrieved_skills=(),
                    proposed_actions=(),
                    final_actions=(),
                    controller_codes=(),
                    visible_tool_outcomes=(),
                    native_similarity=1.0,
                    fully_successful=True,
                    host_attribution="successful",
                ),
                None,
            )

    class Evidence:
        def project(self, value):
            assert value == trajectory
            return ()

    buffer = Task014RoundBuffer(
        run_id="run",
        round_index=0,
        shard_id="train-shard-0",
        generation_id="g000",
        dataset_manifest_sha256=DIGEST,
        config_manifest_sha256=DIGEST,
        memory_prompt_manifest_sha256=DIGEST,
        skill_prompt_manifest_sha256=DIGEST,
        skill_token_limit_config_sha256=DIGEST,
        archive_directory=(tmp_path / "round-0").resolve(),
        trajectory_loader=Loader(),
        memory_projection=MemoryProjection(),
        failure_evidence=Evidence(),
    )
    buffer.append(result)
    sealed = buffer.seal()
    assert type(sealed.memory) is SealedMemoryTrajectoryBuffer
    assert type(sealed.skill) is SealedSkillTrajectoryBuffer
    archive = buffer.archive_and_revoke(
        capability=sealed.archive_capability,
        lineage_records=(),
    )
    assert archive.trajectory_count == 1 and archive.lineage_count == 0
    assert (tmp_path / "round-0/archive_manifest.json").is_file()
    with pytest.raises(PermissionError, match="revoked"):
        buffer.seal()
    with pytest.raises(PermissionError, match="capability denied"):
        sealed.archive_capability.consume(buffer)


@pytest.mark.parametrize(
    "round_index,shard,generation",
    [(1, "train-shard-0", "g001"), (1, "train-shard-1", "g000"), (3, "train-shard-3", "g003")],
)
def test_round_identity_drift_stops_before_timer_and_dispatch(round_index, shard, generation):
    events = []
    value, _ = runner(events)
    with pytest.raises(ValueError):
        value.run(
            round_index=round_index,
            train_shard_id=shard,
            scenario_ids=("a",),
            input_generation_id=generation,
        )
    assert events == []
