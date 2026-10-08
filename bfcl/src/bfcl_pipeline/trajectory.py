from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .accounting import UsageLedger
from .canonical import sha256_json
from .evolution import Generation
from .online import OnlineTurnResult
from .resources import GenerationResourceManifest, RetrievalReceipt


@dataclass(frozen=True)
class RetrievalSelection:
    selection_id: str
    case_id: str
    turn_index: int
    generation_id: str
    skill_version: str
    resource_sha256: str
    creation_effect_id: str
    retrieval_receipt_sha256: str

    @classmethod
    def build(cls, case_id: str, turn_index: int, receipt: RetrievalReceipt) -> "RetrievalSelection":
        receipt.validate()
        if receipt.kind != "skill":
            raise ValueError("Only Skill retrievals can establish Skill-use lineage")
        core = {
            "case_id": case_id,
            "turn_index": turn_index,
            "generation_id": receipt.generation_id,
            "skill_version": receipt.resource_id,
            "resource_sha256": receipt.resource_sha256,
            "creation_effect_id": receipt.creation_effect_id,
            "retrieval_receipt_sha256": receipt.receipt_sha256,
        }
        return cls(sha256_json(core), **core)

    def validate(self, receipts: Mapping[str, RetrievalReceipt]) -> None:
        core = asdict(self)
        supplied = core.pop("selection_id")
        if not all((self.case_id, self.generation_id, self.skill_version, self.resource_sha256, self.creation_effect_id)):
            raise ValueError("Retrieval selection contains an empty identity field")
        if self.turn_index < 0 or supplied != sha256_json(core):
            raise ValueError("Retrieval selection identity is not canonical")
        receipt = receipts.get(self.retrieval_receipt_sha256)
        if receipt is None:
            raise ValueError("Retrieval selection cites an unknown host receipt")
        if (
            receipt.resource_id,
            receipt.resource_sha256,
            receipt.generation_id,
            receipt.creation_effect_id,
        ) != (
            self.skill_version,
            self.resource_sha256,
            self.generation_id,
            self.creation_effect_id,
        ):
            raise ValueError("Retrieval selection does not match its host receipt")


@dataclass(frozen=True)
class DecisionAudit:
    decision_id: str
    case_id: str
    turn_index: int
    generation_id: str
    decision: str
    final_text_sha256: str
    call_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    skill_bindings: tuple[tuple[str, str, str], ...]
    selection_ids: tuple[str, ...]
    response_application_id: str

    @classmethod
    def from_turn(
        cls,
        result: OnlineTurnResult,
        selections: tuple[RetrievalSelection, ...],
        response_application_id: str,
    ) -> "DecisionAudit":
        if tuple(x.skill_version for x in selections) != result.selected_skill_versions:
            raise ValueError("Selected Skill versions do not match the online result")
        core = {
            "case_id": result.case_id,
            "turn_index": result.turn_index,
            "generation_id": result.generation_id,
            "decision": result.decision,
            "final_text_sha256": sha256_json(result.final_text),
            "call_ids": result.committed_call_ids,
            "memory_ids": result.selected_memory_ids,
            "skill_bindings": result.selected_skill_bindings,
        }
        if result.decision_lineage_sha256 != sha256_json(core):
            raise ValueError("Online decision lineage is not canonical")
        return cls(
            result.decision_lineage_sha256,
            result.case_id,
            result.turn_index,
            result.generation_id,
            result.decision,
            sha256_json(result.final_text),
            result.committed_call_ids,
            result.selected_memory_ids,
            result.selected_skill_bindings,
            tuple(x.selection_id for x in selections),
            response_application_id,
        )

    def validate(self, selections: Mapping[str, RetrievalSelection]) -> None:
        core = {
            "case_id": self.case_id,
            "turn_index": self.turn_index,
            "generation_id": self.generation_id,
            "decision": self.decision,
            "final_text_sha256": self.final_text_sha256,
            "call_ids": self.call_ids,
            "memory_ids": self.memory_ids,
            "skill_bindings": self.skill_bindings,
        }
        if self.decision not in {"KEEP", "REVISE"} or self.decision_id != sha256_json(core):
            raise ValueError("Decision audit identity is not canonical")
        selected = tuple(selections.get(x) for x in self.selection_ids)
        if any(x is None for x in selected):
            raise ValueError("Decision cites an unknown retrieval selection")
        expected = tuple(
            (x.skill_version, x.resource_sha256, x.creation_effect_id)
            for x in selected
            if x is not None
        )
        if expected != self.skill_bindings:
            raise ValueError("Decision Skill bindings do not match retrieval selections")
        if any(
            (x.case_id, x.turn_index, x.generation_id)
            != (self.case_id, self.turn_index, self.generation_id)
            for x in selected
            if x is not None
        ):
            raise ValueError("Decision and retrieval selection identity mismatch")


@dataclass(frozen=True)
class ExecutedCallAudit:
    execution_id: str
    case_id: str
    turn_index: int
    generation_id: str
    call_id: str
    decision_id: str
    response_application_id: str
    execution_effect_id: str

    @classmethod
    def build(cls, decision: DecisionAudit, call_id: str, execution_effect_id: str) -> "ExecutedCallAudit":
        core = {
            "case_id": decision.case_id,
            "turn_index": decision.turn_index,
            "generation_id": decision.generation_id,
            "call_id": call_id,
            "decision_id": decision.decision_id,
            "response_application_id": decision.response_application_id,
            "execution_effect_id": execution_effect_id,
        }
        return cls(sha256_json(core), **core)

    def validate(self, decisions: Mapping[str, DecisionAudit], ledger: UsageLedger) -> None:
        core = asdict(self)
        supplied = core.pop("execution_id")
        if supplied != sha256_json(core):
            raise ValueError("Executed call identity is not canonical")
        decision = decisions.get(self.decision_id)
        if decision is None or self.call_id not in decision.call_ids:
            raise ValueError("Executed call is not part of the cited final decision")
        if (self.case_id, self.turn_index, self.generation_id, self.response_application_id) != (
            decision.case_id,
            decision.turn_index,
            decision.generation_id,
            decision.response_application_id,
        ):
            raise ValueError("Executed call and decision identity mismatch")
        effect = ledger.effect(self.execution_effect_id)
        if effect.subject_id != self.call_id:
            raise ValueError("Committed effect is not bound to the exact executed call")
        if not effect.substantive or self.response_application_id not in effect.causal_application_ids:
            raise ValueError("Executed call is not bound to a substantive committed effect")


@dataclass(frozen=True)
class TrustedTrajectory:
    case_id: str
    generation_id: str
    generation: Generation
    resource_manifest: GenerationResourceManifest
    retrieval_receipts: tuple[RetrievalReceipt, ...]
    selections: tuple[RetrievalSelection, ...]
    decisions: tuple[DecisionAudit, ...]
    executed_calls: tuple[ExecutedCallAudit, ...]
    ledger_payload: dict
    trajectory_sha256: str

    def validate(self) -> UsageLedger:
        generation_core = asdict(self.generation)
        supplied_generation_hash = generation_core.pop("generation_sha256")
        if supplied_generation_hash != sha256_json(generation_core):
            raise ValueError("Trajectory generation identity is not canonical")
        self.resource_manifest.validate()
        if (
            self.generation.generation_id != self.generation_id
            or self.resource_manifest.generation_id != self.generation_id
            or self.generation.skill_library_sha256 != self.resource_manifest.manifest_sha256
        ):
            raise ValueError("Trajectory generation does not bind the Skill manifest")
        accepted_skill_records = set(self.resource_manifest.skill_records)
        ledger = UsageLedger.from_payload(self.ledger_payload)
        receipts = {x.receipt_sha256: x for x in self.retrieval_receipts}
        selections = {x.selection_id: x for x in self.selections}
        decisions = {x.decision_id: x for x in self.decisions}
        if (
            len(receipts) != len(self.retrieval_receipts)
            or len(selections) != len(self.selections)
            or len(decisions) != len(self.decisions)
        ):
            raise ValueError("Trusted trajectory contains duplicate identities")
        for receipt in self.retrieval_receipts:
            receipt.validate()
            if receipt.generation_id != self.generation_id:
                raise ValueError("Trajectory and retrieval receipt generation mismatch")
        for selection in self.selections:
            selection.validate(receipts)
            if (selection.case_id, selection.generation_id) != (self.case_id, self.generation_id):
                raise ValueError("Trajectory and retrieval identity mismatch")
            if (selection.skill_version, selection.resource_sha256, selection.creation_effect_id) not in accepted_skill_records:
                raise ValueError("Retrieved Skill is not in the trusted generation manifest")
        for decision in self.decisions:
            decision.validate(selections)
            if (decision.case_id, decision.generation_id) != (self.case_id, self.generation_id):
                raise ValueError("Trajectory and decision identity mismatch")
            application = ledger.application(decision.response_application_id)
            request = ledger.request(application.logical_request_id)
            if application.response_sha256 != decision.final_text_sha256:
                raise ValueError("Decision text does not match the accepted response application")
            if not application.accepted or application.application_kind != "online_action":
                raise ValueError("Decision response was not accepted for online action")
            if (request.case_id, request.turn_index) != (decision.case_id, decision.turn_index):
                raise ValueError("Decision does not match the durable logical request")
        for executed in self.executed_calls:
            executed.validate(decisions, ledger)
        core = {
            "case_id": self.case_id,
            "generation_id": self.generation_id,
            "generation": asdict(self.generation),
            "resource_manifest": asdict(self.resource_manifest),
            "retrieval_receipts": [asdict(x) for x in self.retrieval_receipts],
            "selections": [asdict(x) for x in self.selections],
            "decisions": [asdict(x) for x in self.decisions],
            "executed_calls": [asdict(x) for x in self.executed_calls],
            "ledger_sha256": ledger.ledger_sha256,
        }
        if self.trajectory_sha256 != sha256_json(core):
            raise ValueError("Trusted trajectory hash mismatch")
        return ledger

    def proves_skill_execution(self, skill_version: str, creation_effect_id: str) -> bool:
        self.validate()
        decisions = {x.decision_id: x for x in self.decisions}
        selected_ids = {
            selection.selection_id
            for selection in self.selections
            if selection.skill_version == skill_version and selection.creation_effect_id == creation_effect_id
        }
        if not selected_ids:
            return False
        decision_ids = {
            decision.decision_id
            for decision in self.decisions
            if selected_ids.intersection(decision.selection_ids)
        }
        return any(
            executed.decision_id in decision_ids
            and executed.call_id in decisions[executed.decision_id].call_ids
            for executed in self.executed_calls
        )


def build_trusted_trajectory(
    turn: OnlineTurnResult,
    response_application_id: str,
    call_effect_ids: tuple[str, ...],
    generation: Generation,
    resource_manifest: GenerationResourceManifest,
    ledger: UsageLedger,
) -> TrustedTrajectory:
    if len(call_effect_ids) != len(turn.committed_call_ids):
        raise ValueError("Each committed call requires exactly one execution effect")
    application = ledger.application(response_application_id)
    request = ledger.request(application.logical_request_id)
    if not application.accepted or application.application_kind != "online_action":
        raise ValueError("Online decision must cite an accepted online_action application")
    if application.response_sha256 != sha256_json(turn.final_text):
        raise ValueError("Online final text does not match the accepted response application")
    if (request.case_id, request.turn_index) != (turn.case_id, turn.turn_index):
        raise ValueError("Online result and durable request identity mismatch")
    if tuple(x.resource_id for x in turn.selected_skill_receipts) != turn.selected_skill_versions:
        raise ValueError("Online Skill versions and host retrieval receipts disagree")
    selections = tuple(
        RetrievalSelection.build(turn.case_id, turn.turn_index, receipt)
        for receipt in turn.selected_skill_receipts
    )
    decision = DecisionAudit.from_turn(turn, selections, response_application_id)
    executed = tuple(
        ExecutedCallAudit.build(decision, call_id, effect_id)
        for call_id, effect_id in zip(turn.committed_call_ids, call_effect_ids)
    )
    ledger_payload = ledger.to_payload()
    core = {
        "case_id": turn.case_id,
        "generation_id": turn.generation_id,
        "generation": asdict(generation),
        "resource_manifest": asdict(resource_manifest),
        "retrieval_receipts": [asdict(x) for x in turn.selected_skill_receipts],
        "selections": [asdict(x) for x in selections],
        "decisions": [asdict(decision)],
        "executed_calls": [asdict(x) for x in executed],
        "ledger_sha256": ledger.ledger_sha256,
    }
    trajectory = TrustedTrajectory(
        turn.case_id,
        turn.generation_id,
        generation,
        resource_manifest,
        turn.selected_skill_receipts,
        selections,
        (decision,),
        executed,
        ledger_payload,
        sha256_json(core),
    )
    trajectory.validate()
    return trajectory
