"""Exact committed-call failure lineage and repair reporting for tau3 task trials."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from . import SYSTEM_ORDER
from .canonical import canonical_sha256


@dataclass(frozen=True)
class RetrievalReceipt:
    generation_id: str
    domain: str
    tool_names: tuple[str, ...]
    query_sha256: str
    selected_policy_memory_versions: tuple[tuple[str, int], ...]
    selected_world_memory_versions: tuple[tuple[str, int], ...]
    selected_skill_versions: tuple[tuple[str, int], ...]
    retrieval_sha256: str

    def __post_init__(self) -> None:
        expected = canonical_sha256(
            {
                "generation_id": self.generation_id,
                "domain": self.domain,
                "tool_names": list(self.tool_names),
                "query_sha256": self.query_sha256,
                "policy": [list(value) for value in self.selected_policy_memory_versions],
                "world": [list(value) for value in self.selected_world_memory_versions],
                "skills": [list(value) for value in self.selected_skill_versions],
            }
        )
        if self.retrieval_sha256 != expected:
            raise ValueError("retrieval receipt hash mismatch")


@dataclass(frozen=True)
class DecisionReceipt:
    decision_id: str
    logical_request_id: str
    attempt_id: str
    application_id: str
    response_sha256: str
    executed_effect_id: str


@dataclass(frozen=True)
class CommittedAction:
    call_id: str
    decision_id: str
    tool_name: str
    skill_id: str | None
    skill_version: int | None
    accepted_effect_id: str | None
    visible_outcome_class: str | None
    executed: bool
    committed: bool
    retrieval: RetrievalReceipt | None
    decision: DecisionReceipt | None


def _generation_state_hash(document: dict) -> str:
    payload = dict(document)
    claimed = payload.pop("state_sha256", None)
    if claimed != canonical_sha256(payload):
        raise ValueError("trusted generation-state document hash mismatch")
    return claimed


def trusted_trajectory_sha256(
    *,
    episode_id: str,
    domain: str,
    task_id: str,
    simulator_seed: int,
    system_id: str,
    generation_id: str | None,
    evaluator_record_sha256: str,
    reward: float,
    actions: tuple[CommittedAction, ...],
    generation_state_document: dict,
    accounting_ledger_document: dict,
) -> str:
    generation_sha = _generation_state_hash(generation_state_document)
    return canonical_sha256(
        {
            "episode_id": episode_id,
            "domain": domain,
            "task_id": task_id,
            "simulator_seed": simulator_seed,
            "system_id": system_id,
            "generation_id": generation_id,
            "evaluator_record_sha256": evaluator_record_sha256,
            "reward": reward,
            "actions": [asdict(row) for row in actions],
            "generation_state_sha256": generation_sha,
            "accounting_ledger_sha256": canonical_sha256(accounting_ledger_document),
        }
    )


@dataclass(frozen=True)
class TrustedEpisodeTrace:
    episode_id: str
    domain: str
    task_id: str
    simulator_seed: int
    system_id: str
    generation_id: str | None
    trajectory_sha256: str
    evaluator_record_sha256: str
    reward: float
    actions: tuple[CommittedAction, ...]
    generation_state_document: dict
    accounting_ledger_document: dict

    def __post_init__(self) -> None:
        if type(self.simulator_seed) is not int:
            raise ValueError("integer simulator seed required")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise ValueError("native reward must be a finite float")
        computed = trusted_trajectory_sha256(
            episode_id=self.episode_id,
            domain=self.domain,
            task_id=self.task_id,
            simulator_seed=self.simulator_seed,
            system_id=self.system_id,
            generation_id=self.generation_id,
            evaluator_record_sha256=self.evaluator_record_sha256,
            reward=self.reward,
            actions=self.actions,
            generation_state_document=self.generation_state_document,
            accounting_ledger_document=self.accounting_ledger_document,
        )
        if self.trajectory_sha256 != computed:
            raise ValueError("trusted trajectory hash mismatch")

    @property
    def fully_successful(self) -> bool:
        return self.reward == 1.0


def trusted_trace_from_native_receipt(receipt: dict) -> TrustedEpisodeTrace:
    """Build a trusted trace only from persisted native/retrieval/ledger receipt material."""
    required = {
        "episode_id",
        "domain",
        "task_id",
        "simulator_seed",
        "system_id",
        "generation_id",
        "evaluator_record_sha256",
        "reward",
        "actions",
        "generation_state_document",
        "accounting_ledger_document",
    }
    if set(receipt) != required:
        raise ValueError("native trusted-trace receipt has missing or extra fields")
    actions = []
    for payload in receipt["actions"]:
        value = dict(payload)
        retrieval_payload = value.pop("retrieval")
        decision_payload = value.pop("decision")
        value["retrieval"] = (
            RetrievalReceipt(
                retrieval_payload["generation_id"],
                retrieval_payload["domain"],
                tuple(retrieval_payload["tool_names"]),
                retrieval_payload["query_sha256"],
                tuple(tuple(row) for row in retrieval_payload["selected_policy_memory_versions"]),
                tuple(tuple(row) for row in retrieval_payload["selected_world_memory_versions"]),
                tuple(tuple(row) for row in retrieval_payload["selected_skill_versions"]),
                retrieval_payload["retrieval_sha256"],
            )
            if retrieval_payload is not None
            else None
        )
        value["decision"] = (
            DecisionReceipt(**decision_payload) if decision_payload is not None else None
        )
        actions.append(CommittedAction(**value))
    kwargs = {key: receipt[key] for key in required - {"actions"}}
    kwargs["actions"] = tuple(actions)
    trajectory = trusted_trajectory_sha256(**kwargs)
    return TrustedEpisodeTrace(trajectory_sha256=trajectory, **kwargs)


@dataclass(frozen=True)
class FailureSignature:
    skill_id: str
    signature_sha256: str


@dataclass(frozen=True)
class FailureLineage:
    lineage_id: str
    failure_signature_sha256: str
    skill_id: str
    accepted_skill_version: int | None
    accepted_effect_id: str | None


@dataclass(frozen=True)
class EvaluationRecord:
    system_id: str
    domain: str
    task_id: str
    simulator_seed: int
    reward: float
    manifest_position: int
    trajectory_sha256: str
    evaluator_record_sha256: str

    @property
    def fully_successful(self) -> bool:
        return self.reward == 1.0


@dataclass(frozen=True)
class AttributionRow:
    domain: str
    task_id: str
    simulator_seed: int
    classification: str
    matched_lineage_id: str | None
    generation_0_reward: float
    updated_reward: float
    repaired: bool
    committed_call_proof_sha256: str | None


@dataclass(frozen=True)
class RepairSummary:
    generation_0_failure_count: int
    related_case_count: int
    repaired_case_count: int
    repair_rate: float | None
    unmatched_case_count: int
    ambiguous_case_count: int
    incomplete_evidence_count: int
    updated_minus_generation_0_reward_overall: float
    updated_minus_generation_0_reward_related: float | None


def derive_failure_signatures(trace: TrustedEpisodeTrace) -> tuple[FailureSignature, ...]:
    if trace.system_id != "generation_0" or trace.generation_id != "g000":
        raise ValueError("failure signatures require Generation-0 G000 traces")
    if trace.fully_successful:
        return ()
    result: list[FailureSignature] = []
    seen_calls: set[str] = set()
    for action in trace.actions:
        if action.call_id in seen_calls:
            raise ValueError("duplicate committed call identity")
        seen_calls.add(action.call_id)
        if (
            not action.executed
            or not action.committed
            or not action.decision_id
            or action.skill_id is None
            or action.visible_outcome_class is None
        ):
            continue
        result.append(
            FailureSignature(
                action.skill_id,
                canonical_sha256(
                    {
                        "domain": trace.domain,
                        "skill_id": action.skill_id,
                        "tool_name": action.tool_name,
                        "visible_outcome_class": action.visible_outcome_class,
                    }
                ),
            )
        )
    return tuple(result)


def _identity(row: EvaluationRecord) -> tuple[str, str, int]:
    return row.domain, row.task_id, row.simulator_seed


def _matrix(records: tuple[EvaluationRecord, ...]):
    values: dict[tuple[str, str, str, int], EvaluationRecord] = {}
    order: dict[int, tuple[str, str, int]] = {}
    for row in records:
        key = (row.system_id, *_identity(row))
        if key in values or row.system_id not in SYSTEM_ORDER:
            raise ValueError("duplicate or unknown system result")
        values[key] = row
        if row.system_id == "vanilla":
            if row.manifest_position in order:
                raise ValueError("duplicate manifest position")
            order[row.manifest_position] = _identity(row)
    try:
        identities = tuple(order[index] for index in range(len(order)))
    except KeyError as error:
        raise ValueError("non-contiguous manifest positions") from error
    expected = {(system, *identity) for system in SYSTEM_ORDER for identity in identities}
    if set(values) != expected:
        raise ValueError("incomplete three-system task-trial matrix")
    return identities, values


def _committed_call_proof(trace: TrustedEpisodeTrace | None, lineage: FailureLineage):
    if (
        trace is None
        or trace.system_id != "updated"
        or trace.generation_id != "g003"
        or type(lineage.accepted_skill_version) is not int
        or lineage.accepted_skill_version < 1
        or not lineage.accepted_effect_id
    ):
        return None
    matches = [
        action
        for action in trace.actions
        if action.executed
        and action.committed
        and action.decision_id
        and action.skill_id == lineage.skill_id
        and action.skill_version == lineage.accepted_skill_version
        and action.accepted_effect_id == lineage.accepted_effect_id
        and action.retrieval is not None
        and action.decision is not None
    ]
    if len(matches) != 1:
        return None
    action = matches[0]
    retrieval = action.retrieval
    decision = action.decision
    assert retrieval is not None and decision is not None
    if (
        retrieval.generation_id != "g003"
        or (lineage.skill_id, lineage.accepted_skill_version)
        not in retrieval.selected_skill_versions
        or decision.decision_id != action.decision_id
    ):
        return None
    generation_skills = trace.generation_state_document.get("skills", [])
    generation_match = [
        row
        for row in generation_skills
        if row.get("skill_id") == lineage.skill_id
        and row.get("version") == lineage.accepted_skill_version
        and row.get("accepted_effect_id") == lineage.accepted_effect_id
    ]
    if len(generation_match) != 1:
        return None
    ledger = trace.accounting_ledger_document
    requests = {row["logical_request_id"]: row for row in ledger.get("logical_requests", [])}
    attempts = {row["attempt_id"]: row for row in ledger.get("physical_attempts", [])}
    applications = {row["application_id"]: row for row in ledger.get("applications", [])}
    effects = {row["effect_id"]: row for row in ledger.get("effects", [])}
    request = requests.get(decision.logical_request_id)
    attempt = attempts.get(decision.attempt_id)
    application = applications.get(decision.application_id)
    effect = effects.get(decision.executed_effect_id)
    if (
        request is None
        or request.get("provider_role") != "qwen"
        or attempt is None
        or attempt.get("logical_request_id") != decision.logical_request_id
        or not attempt.get("completed")
        or attempt.get("response_sha256") != decision.response_sha256
        or application is None
        or application.get("logical_request_id") != decision.logical_request_id
        or application.get("attempt_id") != decision.attempt_id
        or application.get("effect_id") != decision.executed_effect_id
        or not application.get("committed")
        or effect is None
        or not effect.get("committed")
        or not effect.get("substantive")
    ):
        return None
    return canonical_sha256(
        {
            "episode_id": trace.episode_id,
            "call_id": action.call_id,
            "decision": asdict(decision),
            "retrieval": asdict(retrieval),
            "skill_id": action.skill_id,
            "skill_version": action.skill_version,
            "accepted_effect_id": action.accepted_effect_id,
            "generation_state_sha256": trace.generation_state_document["state_sha256"],
            "accounting_ledger_sha256": canonical_sha256(ledger),
            "trajectory_sha256": trace.trajectory_sha256,
        }
    )


def analyze_repairs(
    *,
    lineage: tuple[FailureLineage, ...],
    records: tuple[EvaluationRecord, ...],
    generation_0_traces: tuple[TrustedEpisodeTrace, ...],
    updated_traces: tuple[TrustedEpisodeTrace, ...],
):
    identities, matrix = _matrix(records)
    g0_traces = {(row.domain, row.task_id, row.simulator_seed): row for row in generation_0_traces}
    updated = {(row.domain, row.task_id, row.simulator_seed): row for row in updated_traces}
    if len(g0_traces) != len(generation_0_traces) or len(updated) != len(updated_traces):
        raise ValueError("duplicate trusted task-trial trace")
    by_signature: dict[str, list[FailureLineage]] = {}
    by_lineage_id: dict[str, FailureLineage] = {}
    for item in lineage:
        previous = by_lineage_id.get(item.lineage_id)
        if previous is not None and previous != item:
            raise ValueError("conflicting lineage identity")
        by_lineage_id[item.lineage_id] = item
    for item in by_lineage_id.values():
        by_signature.setdefault(item.failure_signature_sha256, []).append(item)
    rows = []
    differences = []
    related_differences = []
    counts = dict(failure=0, related=0, repaired=0, unmatched=0, ambiguous=0, incomplete=0)
    for identity in identities:
        domain, task_id, seed = identity
        g0 = matrix[("generation_0", *identity)]
        g3 = matrix[("updated", *identity)]
        differences.append(g3.reward - g0.reward)
        classification = "not_generation_0_failure"
        matched = None
        proof = None
        if not g0.fully_successful:
            counts["failure"] += 1
            trace = g0_traces.get(identity)
            if (
                trace is None
                or trace.trajectory_sha256 != g0.trajectory_sha256
                or trace.evaluator_record_sha256 != g0.evaluator_record_sha256
                or trace.reward != g0.reward
            ):
                classification = "incomplete_evidence"
                counts["incomplete"] += 1
            else:
                signatures = derive_failure_signatures(trace)
                matches = {
                    item.lineage_id: item
                    for sig in signatures
                    for item in by_signature.get(sig.signature_sha256, ())
                }
                if not signatures:
                    classification = "incomplete_evidence"
                    counts["incomplete"] += 1
                elif not matches:
                    classification = "unmatched"
                    counts["unmatched"] += 1
                elif len(matches) > 1:
                    classification = "ambiguous"
                    counts["ambiguous"] += 1
                else:
                    matched = next(iter(matches.values()))
                    classification = "related_unrepaired"
                    counts["related"] += 1
                    related_differences.append(g3.reward - g0.reward)
                    updated_trace = updated.get(identity)
                    if updated_trace is not None and (
                        updated_trace.trajectory_sha256 != g3.trajectory_sha256
                        or updated_trace.evaluator_record_sha256 != g3.evaluator_record_sha256
                        or updated_trace.reward != g3.reward
                    ):
                        updated_trace = None
                    proof = _committed_call_proof(updated_trace, matched)
                    if g3.fully_successful and proof is not None:
                        classification = "related_repaired"
                        counts["repaired"] += 1
        rows.append(
            AttributionRow(
                domain,
                task_id,
                seed,
                classification,
                matched.lineage_id if matched else None,
                g0.reward,
                g3.reward,
                classification == "related_repaired",
                proof,
            )
        )
    related = counts["related"]
    summary = RepairSummary(
        counts["failure"],
        related,
        counts["repaired"],
        counts["repaired"] / related if related else None,
        counts["unmatched"],
        counts["ambiguous"],
        counts["incomplete"],
        math.fsum(differences) / len(identities),
        math.fsum(related_differences) / related if related else None,
    )
    return tuple(rows), summary
