"""Checked-in tau3 round composition from native episodes through generation publication."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from . import DOMAINS, QWEN_MODEL
from .access import AccessAudit, TaskRef
from .accounting import UsageAttempt
from .canonical import atomic_write_json, canonical_sha256, read_json
from .evolution import RoundExecution
from .generation_store import GenerationState, GenerationStore, MemoryRecord, SkillRecord
from .ledger import (
    DurableAccountingLedger,
    LogicalRequest,
    OutputApplication,
    PhysicalAttempt,
    SubstantiveEffect,
)
from .minibench import (
    BranchResult,
    DevTaskView,
    SkillCandidate,
    evaluate_paired_branches,
    select_relevant_dev_tasks,
)
from .online_pipeline import TauOnlineTurnPipeline
from .retrieval import EmbeddingProvider, StoredVector


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text(encoding="utf-8").strip() if path.exists() else "process-local"


@dataclass(frozen=True)
class EpisodeEvidence:
    task_key: str
    complete: bool
    reward: float
    failure_lineage_hashes: tuple[str, ...]
    artifact_sha256: str
    ledger_events: tuple[object, ...]


class NativeEpisodeExecutor(Protocol):
    def tool_names_for_task(self, task: TaskRef) -> tuple[str, ...]: ...

    def execute(
        self,
        *,
        task: TaskRef,
        generation: GenerationState,
        retrieved_context: object,
        online_pipeline: TauOnlineTurnPipeline,
        scope_id: str,
        dispatch_id: str,
    ) -> EpisodeEvidence: ...

    def recover(
        self, *, task: TaskRef, scope_id: str, dispatch_id: str
    ) -> EpisodeEvidence | None: ...


@dataclass(frozen=True)
class ProposedUpdate:
    update_kind: str
    record_id: str
    next_version: int
    memory_kind: str | None
    domains: tuple[str, ...]
    tool_dependencies: tuple[str, ...]
    content: str
    logical_request: LogicalRequest
    physical_attempt: PhysicalAttempt
    application_id: str


@dataclass(frozen=True)
class AppliedDecisionOutput:
    application_id: str
    logical_request_id: str
    attempt_id: str
    application_kind: str


class OfflineUpdater(Protocol):
    def propose(
        self,
        *,
        generation: GenerationState,
        failed_episode_hashes: tuple[str, ...],
        scope_id: str,
        operation_id: str,
    ) -> tuple[ProposedUpdate, ...]: ...

    def recover_propose(self, *, operation_id: str) -> tuple[ProposedUpdate, ...] | None: ...

    def review(
        self, proposal: ProposedUpdate, generation: GenerationState, *, operation_id: str
    ) -> tuple[bool, tuple[object, ...]]: ...

    def recover_review(self, *, operation_id: str) -> tuple[bool, tuple[object, ...]] | None: ...


class DevBranchExecutor(Protocol):
    def run_paired(
        self,
        proposal: ProposedUpdate,
        selected_tasks: tuple[tuple[str, str], ...],
        generation: GenerationState,
        scope_id: str,
        host_shared_configuration_sha256: str,
        manifest_sha256: str,
        access_receipt_sha256: str,
        operation_id: str,
    ) -> tuple[tuple[BranchResult, ...], tuple[object, ...]]: ...

    def recover_paired(
        self, *, operation_id: str
    ) -> tuple[tuple[BranchResult, ...], tuple[object, ...]] | None: ...


def _validate_qwen_request_event(event: object, config_sha256: str) -> None:
    if isinstance(event, LogicalRequest) and event.provider_role == "qwen":
        if event.model_id != QWEN_MODEL or event.config_sha256 != config_sha256:
            raise ValueError("Qwen event is not bound to the calibrated model configuration")


def _record_events(
    ledger: DurableAccountingLedger,
    events: tuple[object, ...],
    *,
    qwen_config_sha256: str,
) -> None:
    for event in events:
        _validate_qwen_request_event(event, qwen_config_sha256)
        if isinstance(event, LogicalRequest):
            ledger.record_request(event)
        elif isinstance(event, PhysicalAttempt):
            ledger.record_attempt(event)
        elif isinstance(event, OutputApplication):
            ledger.record_application(event)
        elif isinstance(event, SubstantiveEffect):
            ledger.record_effect(event)
        elif isinstance(event, AppliedDecisionOutput):
            continue
        else:
            raise TypeError("unknown accounting ledger event")


def _applied_qwen_decision_attempts(
    events: tuple[object, ...],
) -> tuple[tuple[AppliedDecisionOutput, LogicalRequest, PhysicalAttempt], ...]:
    requests = {
        row.logical_request_id: row
        for row in events
        if isinstance(row, LogicalRequest) and row.provider_role == "qwen"
    }
    attempts = {row.attempt_id: row for row in events if isinstance(row, PhysicalAttempt)}
    applications = [row for row in events if isinstance(row, AppliedDecisionOutput)]
    if len({row.application_id for row in applications}) != len(applications):
        raise ValueError("duplicate applied decision-output identity")
    result = []
    for application in applications:
        request = requests.get(application.logical_request_id)
        attempt = attempts.get(application.attempt_id)
        if (
            request is None
            or attempt is None
            or attempt.logical_request_id != application.logical_request_id
            or not attempt.completed
        ):
            raise ValueError("applied decision output lacks one complete Qwen attempt")
        result.append((application, request, attempt))
    return tuple(result)


def _event_to_payload(event: object) -> dict:
    supported = (
        LogicalRequest,
        PhysicalAttempt,
        OutputApplication,
        SubstantiveEffect,
        AppliedDecisionOutput,
    )
    if not isinstance(event, supported):
        raise TypeError("unsupported durable operation event")
    return {"event_type": type(event).__name__, "value": asdict(event)}


def _event_from_payload(payload: dict) -> object:
    classes = {
        "LogicalRequest": LogicalRequest,
        "PhysicalAttempt": PhysicalAttempt,
        "OutputApplication": OutputApplication,
        "SubstantiveEffect": SubstantiveEffect,
        "AppliedDecisionOutput": AppliedDecisionOutput,
    }
    try:
        return classes[payload["event_type"]](**payload["value"])
    except KeyError as error:
        raise ValueError("unknown durable operation event") from error


def _proposal_to_payload(row: ProposedUpdate) -> dict:
    payload = asdict(row)
    return payload


def _proposal_from_payload(payload: dict) -> ProposedUpdate:
    value = dict(payload)
    value["domains"] = tuple(value["domains"])
    value["tool_dependencies"] = tuple(value["tool_dependencies"])
    value["logical_request"] = LogicalRequest(**value["logical_request"])
    value["physical_attempt"] = PhysicalAttempt(**value["physical_attempt"])
    return ProposedUpdate(**value)


def _run_operation(
    *,
    progress: dict,
    progress_path: Path,
    operation_id: str,
    operation_kind: str,
    execute,
    recover,
    encode,
    decode,
):
    operations = progress.setdefault("operations", {})
    row = operations.get(operation_id)
    if row is not None:
        if row.get("operation_kind") != operation_kind:
            raise ValueError("durable operation identity collision")
        if row.get("status") == "complete":
            return decode(row["result"])
        if row.get("status") != "claimed":
            raise ValueError("invalid durable operation state")
        result = recover()
        if result is None:
            raise RuntimeError(
                f"claimed {operation_kind} has no durable result; refusing unsafe redispatch"
            )
    else:
        operations[operation_id] = {"operation_kind": operation_kind, "status": "claimed"}
        atomic_write_json(progress_path, progress)
        result = execute()
    operations[operation_id] = {
        "operation_kind": operation_kind,
        "status": "complete",
        "result": encode(result),
    }
    atomic_write_json(progress_path, progress)
    return result


def _recover_publication(store: GenerationStore, state: GenerationState) -> str | None:
    try:
        recovered = store.load(state.generation_id)
    except FileNotFoundError:
        return None
    if recovered.state_sha256 != state.state_sha256:
        raise ValueError("published generation does not match claimed operation")
    return recovered.state_sha256


def _execution_from_payload(payload: dict) -> RoundExecution:
    return RoundExecution(
        completed_task_keys=tuple(payload["completed_task_keys"]),
        committed_substantive_effect_ids=tuple(payload["committed_substantive_effect_ids"]),
        usage_attempts=tuple(UsageAttempt(**row) for row in payload["usage_attempts"]),
        artifact_hashes=tuple(payload["artifact_hashes"]),
        accepted_skill_versions=tuple(tuple(row) for row in payload["accepted_skill_versions"]),
        failure_lineage_hashes=tuple(payload["failure_lineage_hashes"]),
        accounting_ledger_path=payload["accounting_ledger_path"],
        accounting_scope_id=payload["accounting_scope_id"],
        started_monotonic_ns=payload.get("started_monotonic_ns"),
        started_boot_id=payload.get("started_boot_id"),
        started_at_unix_ns=payload["started_at_unix_ns"],
    )


class PipelineRoundEngine:
    """Own episode execution, strict retrieval, updates, Dev acceptance, and publication."""

    def __init__(
        self,
        *,
        run_id: str,
        generation_store: GenerationStore,
        ledger_root: str | Path,
        online_pipeline: TauOnlineTurnPipeline,
        native_executor: NativeEpisodeExecutor,
        updater: OfflineUpdater,
        dev_executor: DevBranchExecutor,
        dev_task_views: tuple[DevTaskView, ...],
        dev_access_audit: AccessAudit,
        embedding_provider: EmbeddingProvider,
        evolution_manifest: dict,
        time_ns=time.time_ns,
        monotonic_ns=time.monotonic_ns,
        boot_id: str | None = None,
    ):
        self.run_id = run_id
        self.generation_store = generation_store
        self.ledger_root = Path(ledger_root)
        self.online_pipeline = online_pipeline
        self.native_executor = native_executor
        self.updater = updater
        self.dev_executor = dev_executor
        self.dev_task_views = dev_task_views
        self.dev_access_audit = dev_access_audit
        self.embedding_provider = embedding_provider
        self.evolution_manifest = evolution_manifest
        calibration = getattr(online_pipeline, "calibration", None)
        if calibration is None:
            raise ValueError("round engine requires a train-smoke calibration receipt")
        calibration.validate_for_manifest(evolution_manifest)
        self.qwen_config_sha256 = calibration.qwen_config_sha256
        self.time_ns = time_ns
        self.monotonic_ns = monotonic_ns
        self.boot_id = boot_id or _boot_id()
        refs_sha = canonical_sha256(
            [{"domain": row.domain, "task_id": row.task_id} for row in dev_task_views]
        )
        if (
            dev_access_audit.split != "dev"
            or dev_access_audit.purpose != "skill_ab_validation"
            or dev_access_audit.task_count != len(dev_task_views)
            or dev_access_audit.task_refs_sha256 != refs_sha
        ):
            raise ValueError("round engine Dev views require an exact ordinary-access audit")
        expected_dev = tuple(
            (domain, task_id)
            for domain in DOMAINS
            for task_id in evolution_manifest["domains"][domain]["dev"]
        )
        if tuple((row.domain, row.task_id) for row in dev_task_views) != expected_dev:
            raise ValueError("round engine Dev views must exactly match the evolution manifest")

    def execute_round(
        self,
        *,
        round_index: int,
        input_generation_id: str,
        output_generation_id: str,
        train_tasks: tuple[TaskRef, ...],
        manifest_sha256: str,
    ) -> RoundExecution:
        if manifest_sha256 != self.evolution_manifest["manifest_sha256"]:
            raise ValueError("round uses a different manifest than its calibration")
        if self.dev_access_audit.manifest_sha256 != manifest_sha256:
            raise ValueError("round and audited Dev access use different manifests")
        expected_tasks = tuple(
            TaskRef(domain, task_id)
            for domain in DOMAINS
            for task_id in self.evolution_manifest["domains"][domain]["rounds"][round_index]
        )
        if train_tasks != expected_tasks:
            raise ValueError("round train tasks must exactly match the manifest shard")
        scope_id = f"{self.run_id}:round:{round_index}"
        ledger_path = self.ledger_root / f"round_{round_index}.accounting.json"
        progress_path = self.ledger_root / f"round_{round_index}.progress.json"
        signature = canonical_sha256(
            {
                "round_index": round_index,
                "input_generation_id": input_generation_id,
                "output_generation_id": output_generation_id,
                "manifest_sha256": manifest_sha256,
                "train_tasks": [asdict(row) for row in train_tasks],
            }
        )
        if progress_path.exists():
            progress = read_json(progress_path)
            if progress.get("round_signature_sha256") != signature:
                raise ValueError("conflicting durable round progress")
            if progress.get("status") == "complete":
                return _execution_from_payload(progress["result"])
        else:
            progress = {
                "started_monotonic_ns": self.monotonic_ns(),
                "started_boot_id": self.boot_id,
                "schema_version": 1,
                "round_signature_sha256": signature,
                "started_at_unix_ns": self.time_ns(),
                "status": "running",
                "active_dispatch": None,
                "completed_episodes": [],
                "operations": {},
            }
            atomic_write_json(progress_path, progress)
        progress.setdefault("operations", {})
        ledger = DurableAccountingLedger(ledger_path)
        generation = self.generation_store.load(input_generation_id)
        failures = [
            value
            for row in progress["completed_episodes"]
            for value in row["failure_lineage_hashes"]
        ]
        artifacts = [
            value
            for row in progress["completed_episodes"]
            for value in (row["artifact_sha256"], row["retrieval_sha256"])
        ]
        completed = [row["task_key"] for row in progress["completed_episodes"]]
        expected_completed = [
            f"{task.domain}:{task.task_id}" for task in train_tasks[: len(completed)]
        ]
        if completed != expected_completed:
            raise ValueError("durable episode progress is not a manifest-ordered prefix")
        for task in train_tasks[len(completed) :]:
            task_key = f"{task.domain}:{task.task_id}"
            dispatch_id = canonical_sha256(
                {"scope_id": scope_id, "task_key": task_key, "round_signature_sha256": signature}
            )
            tool_names = self.native_executor.tool_names_for_task(task)
            retrieval = self.generation_store.retrieve(
                input_generation_id,
                domain=task.domain,
                tool_names=tool_names,
                query=f"domain={task.domain} task={task.task_id} tools={' '.join(tool_names)}",
                provider=self.embedding_provider,
                scope_id=scope_id,
            )
            _record_events(
                ledger, retrieval.ledger_events, qwen_config_sha256=self.qwen_config_sha256
            )
            active = progress["active_dispatch"]
            if active is None:
                progress["active_dispatch"] = {
                    "dispatch_id": dispatch_id,
                    "task_key": task_key,
                    "retrieval_sha256": retrieval.context.retrieval_sha256,
                }
                atomic_write_json(progress_path, progress)
                episode = self.native_executor.execute(
                    task=task,
                    generation=generation,
                    retrieved_context=retrieval.context,
                    online_pipeline=self.online_pipeline,
                    scope_id=scope_id,
                    dispatch_id=dispatch_id,
                )
            else:
                expected_active = {
                    "dispatch_id": dispatch_id,
                    "task_key": task_key,
                    "retrieval_sha256": retrieval.context.retrieval_sha256,
                }
                if active != expected_active:
                    raise ValueError(
                        "durable active dispatch does not match the next manifest task"
                    )
                episode = self.native_executor.recover(
                    task=task, scope_id=scope_id, dispatch_id=dispatch_id
                )
                if episode is None:
                    raise RuntimeError(
                        "claimed native episode has no durable result; refusing unsafe redispatch"
                    )
            if episode.task_key != task_key or not episode.complete:
                raise RuntimeError("native train episode incomplete or identity mismatch")
            _record_events(
                ledger, episode.ledger_events, qwen_config_sha256=self.qwen_config_sha256
            )
            row = {
                "task_key": episode.task_key,
                "reward": episode.reward,
                "failure_lineage_hashes": list(episode.failure_lineage_hashes),
                "artifact_sha256": episode.artifact_sha256,
                "retrieval_sha256": retrieval.context.retrieval_sha256,
            }
            progress["completed_episodes"].append(row)
            progress["active_dispatch"] = None
            atomic_write_json(progress_path, progress)
            completed.append(episode.task_key)
            artifacts.extend((episode.artifact_sha256, retrieval.context.retrieval_sha256))
            if episode.reward != 1.0:
                failures.extend(episode.failure_lineage_hashes)
        progress["status"] = "episodes_complete"
        atomic_write_json(progress_path, progress)

        proposal_operation_id = canonical_sha256(
            {
                "scope_id": scope_id,
                "operation": "offline_propose",
                "failures": failures,
                "generation": generation.state_sha256,
            }
        )
        proposals = _run_operation(
            progress=progress,
            progress_path=progress_path,
            operation_id=proposal_operation_id,
            operation_kind="offline_propose",
            execute=lambda: self.updater.propose(
                generation=generation,
                failed_episode_hashes=tuple(failures),
                scope_id=scope_id,
                operation_id=proposal_operation_id,
            ),
            recover=lambda: self.updater.recover_propose(operation_id=proposal_operation_id),
            encode=lambda rows: [_proposal_to_payload(row) for row in rows],
            decode=lambda rows: tuple(_proposal_from_payload(row) for row in rows),
        )
        memories = list(generation.memories)
        skills = list(generation.skills)
        vectors = list(generation.vectors)
        accepted_effects: list[str] = []
        accepted_skills: list[tuple[str, int]] = []
        for proposal in proposals:
            if proposal.update_kind not in {"memory", "skill"}:
                raise ValueError("offline updater proposed unknown update kind")
            if proposal.update_kind == "memory" and proposal.memory_kind not in {"policy", "world"}:
                raise ValueError("memory update must declare Policy or World scope")
            if proposal.update_kind == "skill" and proposal.memory_kind is not None:
                raise ValueError("Skill update cannot declare a memory scope")
            current = next(
                (
                    row
                    for row in (
                        generation.memories
                        if proposal.update_kind == "memory"
                        else generation.skills
                    )
                    if (row.record_id if proposal.update_kind == "memory" else row.skill_id)
                    == proposal.record_id
                ),
                None,
            )
            current_version = 0 if current is None else current.version
            if proposal.next_version != current_version + 1:
                raise ValueError(
                    "proposal version must be exactly current generation version plus one"
                )
            if (
                proposal.update_kind == "memory"
                and current is not None
                and current.memory_kind != proposal.memory_kind
            ):
                raise ValueError("Memory update cannot change Policy/World scope")
            _validate_qwen_request_event(proposal.logical_request, self.qwen_config_sha256)
            ledger.record_request(proposal.logical_request)
            ledger.record_attempt(proposal.physical_attempt)
            review_operation_id = canonical_sha256(
                {
                    "scope_id": scope_id,
                    "operation": "offline_review",
                    "proposal": _proposal_to_payload(proposal),
                }
            )
            reviewed, review_events = _run_operation(
                progress=progress,
                progress_path=progress_path,
                operation_id=review_operation_id,
                operation_kind="offline_review",
                execute=lambda: self.updater.review(
                    proposal, generation, operation_id=review_operation_id
                ),
                recover=lambda: self.updater.recover_review(operation_id=review_operation_id),
                encode=lambda value: {
                    "reviewed": value[0],
                    "events": [_event_to_payload(row) for row in value[1]],
                },
                decode=lambda value: (
                    value["reviewed"],
                    tuple(_event_from_payload(row) for row in value["events"]),
                ),
            )
            _record_events(ledger, review_events, qwen_config_sha256=self.qwen_config_sha256)
            decision_events: tuple[object, ...] = (
                proposal.logical_request,
                proposal.physical_attempt,
                AppliedDecisionOutput(
                    proposal.application_id,
                    proposal.logical_request.logical_request_id,
                    proposal.physical_attempt.attempt_id,
                    f"{proposal.update_kind}_candidate",
                ),
                *review_events,
            )
            existing_content = None if current is None else current.content
            accepted = (
                reviewed and bool(proposal.content.strip()) and proposal.content != existing_content
            )
            dev_evidence_sha: str | None = None
            if proposal.update_kind == "skill" and accepted:
                selection = select_relevant_dev_tasks(
                    candidate=SkillCandidate(
                        proposal.record_id,
                        current_version,
                        proposal.next_version,
                        proposal.domains,
                        proposal.tool_dependencies,
                    ),
                    manifest_sha256=manifest_sha256,
                    task_views=self.dev_task_views,
                    access_audit=self.dev_access_audit,
                )
                dev_operation_id = canonical_sha256(
                    {
                        "scope_id": scope_id,
                        "operation": "dev_paired",
                        "proposal": _proposal_to_payload(proposal),
                        "selection": asdict(selection),
                    }
                )
                branches, dev_events = _run_operation(
                    progress=progress,
                    progress_path=progress_path,
                    operation_id=dev_operation_id,
                    operation_kind="dev_paired",
                    execute=lambda: self.dev_executor.run_paired(
                        proposal,
                        selection.selected,
                        generation,
                        scope_id,
                        selection.host_shared_configuration_sha256,
                        selection.manifest_sha256,
                        selection.access_receipt_sha256,
                        operation_id=dev_operation_id,
                    ),
                    recover=lambda: self.dev_executor.recover_paired(operation_id=dev_operation_id),
                    encode=lambda value: {
                        "branches": [asdict(row) for row in value[0]],
                        "events": [_event_to_payload(row) for row in value[1]],
                    },
                    decode=lambda value: (
                        tuple(BranchResult(**row) for row in value["branches"]),
                        tuple(_event_from_payload(row) for row in value["events"]),
                    ),
                )
                _record_events(ledger, dev_events, qwen_config_sha256=self.qwen_config_sha256)
                decision_events = (*decision_events, *dev_events)
                result = evaluate_paired_branches(selection, branches)
                accepted = result.accepted
                dev_evidence_sha = result.evidence_sha256
                artifacts.append(result.evidence_sha256)
            effect_id = canonical_sha256(
                {
                    "scope_id": scope_id,
                    "kind": proposal.update_kind,
                    "record_id": proposal.record_id,
                    "version": proposal.next_version,
                    "content": proposal.content,
                    "dev_evidence_sha256": dev_evidence_sha,
                }
            )
            if accepted:
                ledger.record_effect(
                    SubstantiveEffect(
                        effect_id,
                        scope_id,
                        proposal.update_kind,
                        canonical_sha256(proposal.content),
                        True,
                        True,
                    )
                )
                accepted_effects.append(effect_id)
            causal_attempts = _applied_qwen_decision_attempts(decision_events)
            for order, (application, request, attempt) in enumerate(causal_attempts):
                ledger.record_application(
                    OutputApplication(
                        application.application_id,
                        request.logical_request_id,
                        attempt.attempt_id,
                        effect_id if accepted else None,
                        application.application_kind,
                        accepted,
                        order,
                    )
                )
            if not accepted:
                continue
            embedding_operation_id = canonical_sha256(
                {
                    "scope_id": scope_id,
                    "operation": "index_update",
                    "kind": proposal.update_kind,
                    "record_id": proposal.record_id,
                    "version": proposal.next_version,
                    "content": proposal.content,
                }
            )
            vector, embed_events = _run_operation(
                progress=progress,
                progress_path=progress_path,
                operation_id=embedding_operation_id,
                operation_kind="index_update",
                execute=lambda: self.generation_store.embed_record(
                    record_kind=proposal.update_kind,
                    record_id=proposal.record_id,
                    version=proposal.next_version,
                    content=proposal.content,
                    provider=self.embedding_provider,
                    scope_id=scope_id,
                ),
                recover=lambda: self.generation_store.recover_embedded_record(
                    record_kind=proposal.update_kind,
                    record_id=proposal.record_id,
                    version=proposal.next_version,
                    content=proposal.content,
                    provider=self.embedding_provider,
                    scope_id=scope_id,
                ),
                encode=lambda value: {
                    "vector": asdict(value[0]),
                    "events": [_event_to_payload(row) for row in value[1]],
                },
                decode=lambda value: (
                    StoredVector(
                        value["vector"]["record_kind"],
                        value["vector"]["record_id"],
                        value["vector"]["version"],
                        value["vector"]["content_sha256"],
                        value["vector"]["model"],
                        value["vector"]["dimension"],
                        tuple(value["vector"]["vector"]),
                    ),
                    tuple(_event_from_payload(row) for row in value["events"]),
                ),
            )
            _record_events(ledger, embed_events, qwen_config_sha256=self.qwen_config_sha256)
            vectors = [
                row
                for row in vectors
                if not (
                    row.record_kind == proposal.update_kind and row.record_id == proposal.record_id
                )
            ]
            vectors.append(vector)
            if proposal.update_kind == "memory":
                memories = [row for row in memories if row.record_id != proposal.record_id]
                memories.append(
                    MemoryRecord(
                        proposal.record_id,
                        proposal.next_version,
                        proposal.memory_kind or "",
                        proposal.domains,
                        proposal.tool_dependencies,
                        proposal.content,
                    )
                )
            else:
                skills = [row for row in skills if row.skill_id != proposal.record_id]
                skills.append(
                    SkillRecord(
                        proposal.record_id,
                        proposal.next_version,
                        proposal.domains,
                        proposal.tool_dependencies,
                        proposal.content,
                        effect_id,
                        dev_evidence_sha,
                    )
                )
                accepted_skills.append((proposal.record_id, proposal.next_version))
        state = GenerationState(
            output_generation_id,
            input_generation_id,
            tuple(
                sorted(memories, key=lambda row: (row.memory_kind, row.record_id.encode("utf-8")))
            ),
            tuple(sorted(skills, key=lambda row: row.skill_id.encode("utf-8"))),
            tuple(
                sorted(
                    vectors,
                    key=lambda row: (
                        row.record_kind.encode("utf-8"),
                        row.record_id.encode("utf-8"),
                        row.version,
                    ),
                )
            ),
            generation.seed_library_sha256,
        )
        publication_operation_id = canonical_sha256(
            {
                "scope_id": scope_id,
                "operation": "publish_generation",
                "state_sha256": state.state_sha256,
            }
        )
        publication_sha = _run_operation(
            progress=progress,
            progress_path=progress_path,
            operation_id=publication_operation_id,
            operation_kind="publish_generation",
            execute=lambda: self.generation_store.publish(state),
            recover=lambda: _recover_publication(self.generation_store, state),
            encode=lambda value: value,
            decode=lambda value: value,
        )
        artifacts.append(publication_sha)
        requests = {row["logical_request_id"]: row for row in ledger.document["logical_requests"]}
        applications = tuple(ledger.document["applications"])
        attempts = tuple(
            UsageAttempt(
                row["attempt_id"],
                row["logical_request_id"],
                requests[row["logical_request_id"]]["provider_role"],
                row["input_tokens"],
                row["output_tokens"],
                next(
                    (
                        application["effect_id"]
                        for application in applications
                        if application["attempt_id"] == row["attempt_id"]
                        and application["committed"]
                    ),
                    None,
                ),
            )
            for row in ledger.document["physical_attempts"]
        )
        result = RoundExecution(
            completed_task_keys=tuple(completed),
            committed_substantive_effect_ids=tuple(accepted_effects),
            usage_attempts=attempts,
            artifact_hashes=tuple(artifacts) + (ledger.ledger_sha256,),
            accepted_skill_versions=tuple(accepted_skills),
            failure_lineage_hashes=tuple(failures),
            accounting_ledger_path=str(ledger_path),
            accounting_scope_id=scope_id,
            started_at_unix_ns=progress["started_at_unix_ns"],
            started_monotonic_ns=progress["started_monotonic_ns"],
            started_boot_id=progress["started_boot_id"],
        )
        progress["status"] = "complete"
        progress["result"] = asdict(result)
        atomic_write_json(progress_path, progress)
        return result
