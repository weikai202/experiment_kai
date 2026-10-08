from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol

from .accounting import AccountingSnapshot, DirectTimer, UsageLedger
from .canonical import sha256_json
from .checkpoint import CheckpointStore
from .resources import GenerationResourceManifest
from .splits import SplitManifest


@dataclass(frozen=True)
class Generation:
    generation_id: str
    parent_generation_id: str | None
    memory_sha256: str
    skill_library_sha256: str
    accepted_skill_versions: tuple[str, ...]
    generation_sha256: str


@dataclass(frozen=True)
class RoundProduct:
    generation: Generation
    accounting: AccountingSnapshot
    completed_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class CaseDispatch:
    dispatch_id: str
    run_id: str
    round_index: int
    case_id: str
    generation_id: str
    completed_prefix_sha256: str
    operation_state_sha256: str

    @classmethod
    def build(
        cls,
        run_id: str,
        round_index: int,
        case_id: str,
        generation_id: str,
        completed_case_ids: tuple[str, ...],
        operation_state: dict,
    ) -> "CaseDispatch":
        core = {
            "run_id": run_id,
            "round_index": round_index,
            "case_id": case_id,
            "generation_id": generation_id,
            "completed_prefix_sha256": sha256_json(completed_case_ids),
            "operation_state_sha256": sha256_json(operation_state),
        }
        return cls(sha256_json(core), **core)

    def validate(self) -> None:
        core = asdict(self)
        supplied = core.pop("dispatch_id")
        if supplied != sha256_json(core):
            raise ValueError("Case dispatch identity is not canonical")


class RoundOperation(Protocol):
    def execute(
        self,
        dispatch: CaseDispatch,
        generation: Generation,
        ledger: UsageLedger,
        operation_state: dict,
    ) -> dict: ...

    def recover(
        self,
        dispatch: CaseDispatch,
        generation: Generation,
        ledger: UsageLedger,
        operation_state: dict,
    ) -> dict:
        """Reconcile a claimed in-flight case without blindly redispatching it."""


def seed_generation(memory_sha256: str, skill_library: GenerationResourceManifest) -> Generation:
    skill_library.validate()
    if skill_library.generation_id != "g000" or not skill_library.skill_records:
        raise ValueError("G000 requires a non-empty compiled public-schema Skill manifest")
    core = {
        "generation_id": "g000",
        "parent_generation_id": None,
        "memory_sha256": memory_sha256,
        "skill_library_sha256": skill_library.manifest_sha256,
        "accepted_skill_versions": tuple(row[0] for row in skill_library.skill_records),
    }
    return Generation(**core, generation_sha256=sha256_json(core))


class ThreeRoundCoordinator:
    _OPERATION_FIELDS = frozenset({
        "memory_sha256",
        "skill_library_sha256",
        "accepted_skill_versions",
        "failure_lineage_sha256",
        "diagnostic_best_checkpoint_id",
    })

    def __init__(
        self,
        manifest: SplitManifest,
        checkpoints: CheckpointStore,
        clock: Callable[[], float] | None = None,
        after_stage: Callable[[str], None] | None = None,
    ):
        if len(manifest.rounds) != 3 or any(len(r.family_ids) != 40 or len(r.case_ids) != 160 for r in manifest.rounds):
            raise ValueError("BFCL evolution requires three 40-family/160-case rounds")
        self.manifest = manifest
        self.checkpoints = checkpoints
        self.clock = clock
        self.after_stage = after_stage

    def _notify(self, stage: str) -> None:
        if self.after_stage is not None:
            self.after_stage(stage)

    @staticmethod
    def _generation(material: dict) -> Generation:
        row = dict(material)
        row["accepted_skill_versions"] = tuple(row["accepted_skill_versions"])
        return Generation(**row)

    @staticmethod
    def _accounting(material: dict) -> AccountingSnapshot:
        row = dict(material)
        for key in ("missing_usage_attempt_ids", "attempt_ids", "application_ids", "committed_effect_ids"):
            row[key] = tuple(row[key])
        return AccountingSnapshot(**row)

    @classmethod
    def _validate_operation_state(cls, state: dict) -> dict:
        if set(state) - cls._OPERATION_FIELDS:
            raise ValueError("Round operation state contains an unapproved field")
        required = {"memory_sha256", "skill_library_sha256", "accepted_skill_versions"}
        if not required <= set(state):
            raise ValueError("Round operation state is missing generation material")
        output = dict(state)
        output["accepted_skill_versions"] = tuple(output["accepted_skill_versions"])
        return output

    @staticmethod
    def _load_dispatch(payload: dict) -> CaseDispatch:
        row = dict(payload["material"]["dispatch"])
        dispatch = CaseDispatch(**row)
        dispatch.validate()
        return dispatch

    def run_round(
        self,
        run_id: str,
        round_index: int,
        generation: Generation,
        operation: RoundOperation,
    ) -> RoundProduct:
        if generation.generation_id != f"g{round_index:03d}":
            raise ValueError("Round input generation is not the required predecessor")
        stage = self.checkpoints.recovery_stage(run_id, round_index)
        if stage == "accounting_finalized":
            payload = self.checkpoints.load(run_id, round_index, "accounting_finalized")
            assert payload is not None
            material = payload["material"]
            complete = self.checkpoints.load(run_id, round_index, "complete")
            if complete is None or material.get("complete_payload_sha256") != complete["payload_sha256"]:
                raise ValueError("Finalized accounting does not bind the complete checkpoint")
            return RoundProduct(
                self._generation(material["generation"]),
                self._accounting(material["accounting"]),
                tuple(material["completed_case_ids"]),
            )

        shard = self.manifest.rounds[round_index]
        if stage == "not_started":
            timer = DirectTimer(self.clock) if self.clock else DirectTimer()
            self.checkpoints.persist(
                run_id,
                round_index,
                "round_started",
                generation.generation_id,
                (),
                {
                    "started_at": timer.started_at,
                    "input_generation_sha256": generation.generation_sha256,
                },
            )
            self._notify("round_started")
        started = self.checkpoints.load(run_id, round_index, "round_started")
        assert started is not None
        if started["material"]["input_generation_sha256"] != generation.generation_sha256:
            raise ValueError("Recovery input generation does not match round start")
        started_at = float(started["material"]["started_at"])
        timer = DirectTimer(self.clock, started_at) if self.clock else DirectTimer(started_at=started_at)

        effects = self.checkpoints.load(run_id, round_index, "effects_committed")
        if effects is None:
            progress = self.checkpoints.load(run_id, round_index, "episodes_progress")
            if progress is None:
                ledger = UsageLedger()
                operation_state = {
                    "memory_sha256": generation.memory_sha256,
                    "skill_library_sha256": generation.skill_library_sha256,
                    "accepted_skill_versions": generation.accepted_skill_versions,
                }
                completed: tuple[str, ...] = ()
            else:
                completed = tuple(progress["completed_case_ids"])
                if completed != shard.case_ids[:len(completed)]:
                    raise ValueError("Durable case progress is not a manifest-order prefix")
                ledger = UsageLedger.from_payload(progress["material"]["ledger_payload"])
                operation_state = self._validate_operation_state(progress["material"]["operation_state"])

            claimed = self.checkpoints.load(run_id, round_index, "case_dispatch")
            for case_id in shard.case_ids[len(completed):]:
                expected = CaseDispatch.build(
                    run_id,
                    round_index,
                    case_id,
                    generation.generation_id,
                    completed,
                    operation_state,
                )
                active = False
                if claimed is not None:
                    dispatch = self._load_dispatch(claimed)
                    claimed_prefix = tuple(claimed["completed_case_ids"])
                    active = (
                        dispatch == expected
                        and claimed_prefix == completed
                        and dispatch.case_id == case_id
                    )
                if active:
                    claim_ledger = UsageLedger.from_payload(claimed["material"]["ledger_payload"])
                    operation_state = self._validate_operation_state(
                        operation.recover(expected, generation, claim_ledger, dict(operation_state))
                    )
                    ledger = claim_ledger
                    self._notify("case_reconciled")
                else:
                    self.checkpoints.persist(
                        run_id,
                        round_index,
                        "case_dispatch",
                        generation.generation_id,
                        completed,
                        {
                            "dispatch": asdict(expected),
                            "operation_state": operation_state,
                            "ledger_payload": ledger.to_payload(),
                        },
                    )
                    self._notify("case_dispatch_prepared")
                    operation_state = self._validate_operation_state(
                        operation.execute(expected, generation, ledger, dict(operation_state))
                    )
                completed = completed + (case_id,)
                self.checkpoints.persist(
                    run_id,
                    round_index,
                    "episodes_progress",
                    generation.generation_id,
                    completed,
                    {
                        "operation_state": operation_state,
                        "ledger_payload": ledger.to_payload(),
                        "completed_dispatch_id": expected.dispatch_id,
                    },
                )
                self._notify("case_completed")
                claimed = None

            if completed != shard.case_ids:
                raise ValueError("Round must durably complete every case exactly once in manifest order")
            state = {**operation_state, "ledger_payload": ledger.to_payload()}
            self.checkpoints.persist(
                run_id,
                round_index,
                "effects_committed",
                generation.generation_id,
                completed,
                state,
            )
            self._notify("effects_committed")
            effects = self.checkpoints.load(run_id, round_index, "effects_committed")
            assert effects is not None

        state = effects["material"]
        completed = tuple(effects["completed_case_ids"])
        if completed != shard.case_ids:
            raise ValueError("Committed round effects do not cover the exact shard")
        ledger = UsageLedger.from_payload(state["ledger_payload"])
        operation_state = self._validate_operation_state({
            key: value for key, value in state.items() if key != "ledger_payload"
        })
        core = {
            "generation_id": f"g{round_index + 1:03d}",
            "parent_generation_id": generation.generation_id,
            "memory_sha256": operation_state["memory_sha256"],
            "skill_library_sha256": operation_state["skill_library_sha256"],
            "accepted_skill_versions": operation_state["accepted_skill_versions"],
        }
        next_generation = Generation(**core, generation_sha256=sha256_json(core))

        if self.checkpoints.load(run_id, round_index, "generation_published") is None:
            self.checkpoints.persist(
                run_id,
                round_index,
                "generation_published",
                next_generation.generation_id,
                completed,
                {"generation": asdict(next_generation), "ledger_sha256": ledger.ledger_sha256},
            )
            self._notify("generation_published")
        if self.checkpoints.load(run_id, round_index, "durable_completion") is None:
            self.checkpoints.persist(
                run_id,
                round_index,
                "durable_completion",
                next_generation.generation_id,
                completed,
                {
                    "generation_sha256": next_generation.generation_sha256,
                    "ledger_sha256": ledger.ledger_sha256,
                    "completed_case_ids": completed,
                },
            )
            self._notify("durable_completion")

        complete = self.checkpoints.load(run_id, round_index, "complete")
        if complete is None:
            self.checkpoints.persist(
                run_id,
                round_index,
                "complete",
                next_generation.generation_id,
                completed,
                {
                    "generation": asdict(next_generation),
                    "ledger_payload": ledger.to_payload(),
                    "completed_case_ids": completed,
                },
            )
            self._notify("complete")
            complete = self.checkpoints.load(run_id, round_index, "complete")
            assert complete is not None
        complete_ledger = UsageLedger.from_payload(complete["material"]["ledger_payload"])
        if complete_ledger.ledger_sha256 != ledger.ledger_sha256:
            raise ValueError("Complete checkpoint ledger does not match committed effects")

        # The measured round ends only after the complete state checkpoint is fsynced.
        # Persisting the immutable accounting report below is telemetry finalization,
        # not evolution work, and cannot trigger case dispatch.
        latency = timer.stop_after_durable_checkpoint()
        accounting = ledger.snapshot(latency)
        self.checkpoints.persist(
            run_id,
            round_index,
            "accounting_finalized",
            next_generation.generation_id,
            completed,
            {
                "generation": asdict(next_generation),
                "accounting": asdict(accounting),
                "completed_case_ids": completed,
                "complete_payload_sha256": complete["payload_sha256"],
            },
        )
        self._notify("accounting_finalized")
        return RoundProduct(next_generation, accounting, completed)
