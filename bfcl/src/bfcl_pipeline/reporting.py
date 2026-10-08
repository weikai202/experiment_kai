from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .canonical import sha256_json
from .evaluation import FinalEvaluationPlan, FinalUnitArtifact, FinalUnitDispatch, SystemRunReceipt
from .trajectory import TrustedTrajectory

EVIDENCE_KINDS = frozenset({"literal_parse_error", "tool_exception", "official_checker_failure", "missing_function_irrelevance", "missing_parameter_irrelevance", "turn_count_mismatch"})
OUTCOME_CLASSES = frozenset({"literal_parse_error", "tool_exception", "unknown_function", "argument_error", "turn_count_mismatch", "irrelevance_failure", "state_mismatch", "response_mismatch", "official_checker_failure"})


@dataclass(frozen=True)
class FailureSignature:
    skill_id: str
    evidence_kind: str
    canonical_tool_dependencies: tuple[str, ...]
    sanitized_outcome_class: str
    signature_sha256: str

    @classmethod
    def build(cls, skill_id: str, evidence_kind: str, dependencies: Iterable[str], outcome_class: str) -> "FailureSignature":
        if evidence_kind not in EVIDENCE_KINDS:
            raise ValueError("Unknown host-defined evidence kind")
        deps = tuple(sorted(set(dependencies), key=lambda x: x.encode("utf-8")))
        if not deps or outcome_class not in OUTCOME_CLASSES:
            raise ValueError("Failure signatures accept only sanitized public classes")
        core = {"skill_id": skill_id, "evidence_kind": evidence_kind, "canonical_tool_dependencies": deps, "sanitized_outcome_class": outcome_class}
        return cls(**core, signature_sha256=sha256_json(core))


@dataclass(frozen=True)
class FailureLineage:
    mode_id: str
    round_index: int
    signature: FailureSignature
    accepted_skill_version: str
    source_evidence_sha256: str
    accepted_skill_effect_id: str


@dataclass(frozen=True)
class SystemCaseResult:
    system: str
    family_id: str
    case_id: str
    variant: str
    valid: bool
    failure_signature_sha256: str | None
    trusted_trajectory: TrustedTrajectory | None = None
    source_artifact: FinalUnitArtifact | None = None

def system_results_from_artifacts(
    plan: FinalEvaluationPlan,
    owner_id: str,
    artifacts: Iterable[FinalUnitArtifact],
) -> tuple[SystemCaseResult, ...]:
    rows = []
    seen_unit_indices: set[int] = set()
    case_ids = tuple(
        f"multi_turn_{variant}_{int(family_id.rsplit('_', 1)[1])}"
        for family_id in plan.sealed_test_family_ids
        for variant in ("base", "miss_func", "miss_param", "long_context")
    )
    unit_count = len(case_ids) * len(plan.systems)
    for artifact in artifacts:
        if artifact.unit_index in seen_unit_indices:
            raise ValueError("Final reporting artifacts contain a duplicate unit")
        if not 0 <= artifact.unit_index < unit_count:
            raise ValueError("Final reporting artifact is outside the frozen plan")
        system_index, case_index = divmod(artifact.unit_index, len(case_ids))
        expected = (
            plan.systems[system_index],
            plan.generation_ids[system_index],
            case_ids[case_index],
        )
        if (artifact.system, artifact.generation_id, artifact.case_id) != expected:
            raise ValueError("Final reporting artifact does not match frozen unit order")
        dispatch = FinalUnitDispatch.build(
            plan,
            owner_id,
            artifact.unit_index,
            artifact.system,
            artifact.generation_id,
            artifact.case_id,
        )
        artifact.validate(plan, dispatch)
        seen_unit_indices.add(artifact.unit_index)
        variant = next(
            (name for name in ("base", "miss_func", "miss_param", "long_context") if artifact.case_id.startswith(f"multi_turn_{name}_")),
            None,
        )
        if variant is None:
            raise ValueError("Final artifact has an invalid BFCL case ID")
        index = int(artifact.case_id.rsplit("_", 1)[1])
        rows.append(SystemCaseResult(
            artifact.system,
            f"bfcl_family_{index:03d}",
            artifact.case_id,
            variant,
            artifact.official_valid,
            artifact.failure_signature_sha256,
            source_artifact=artifact,
        ))
    return tuple(rows)


def artifact_accounting_summary(
    artifacts: Iterable[FinalUnitArtifact],
    system_run_receipts: Iterable[SystemRunReceipt],
) -> dict[str, float | int | bool | None | dict[str, float]]:
    rows = tuple(artifacts)
    receipts = tuple(system_run_receipts)
    if not rows:
        raise ValueError("Accounting summary requires final artifacts")
    if tuple(receipt.system for receipt in receipts) != ("vanilla", "g000", "g003"):
        raise ValueError("Accounting summary requires authoritative receipts for all three system runs")
    complete = all(bool(row.accounting["usage_complete"]) for row in rows)
    effective_complete = all(row.accounting["qwen_effective_output_tokens"] is not None for row in rows)
    return {
        "unit_count": len(rows),
        "direct_latency_seconds": math.fsum(receipt.direct_latency_seconds for receipt in receipts),
        "direct_latency_seconds_by_system": {
            receipt.system: receipt.direct_latency_seconds for receipt in receipts
        },
        "unit_latency_seconds_diagnostic": math.fsum(float(row.accounting["direct_latency_seconds"]) for row in rows),
        "usage_complete": complete,
        "input_tokens": sum(int(row.accounting["input_tokens"]) for row in rows) if complete else None,
        "output_tokens": sum(int(row.accounting["output_tokens"]) for row in rows) if complete else None,
        "total_tokens": sum(int(row.accounting["total_tokens"]) for row in rows) if complete else None,
        "qwen_effective_output_tokens": sum(int(row.accounting["qwen_effective_output_tokens"]) for row in rows) if effective_complete else None,
    }



@dataclass(frozen=True)
class AttributionRow:
    family_id: str
    case_id: str
    classification: str
    mode_id: str | None


def attribute_repairs(g000: Iterable[SystemCaseResult], updated: Iterable[SystemCaseResult], lineage: Iterable[FailureLineage]) -> tuple[AttributionRow, ...]:
    before_rows, after_rows = tuple(g000), tuple(updated)
    if len({x.case_id for x in before_rows}) != len(before_rows) or len({x.case_id for x in after_rows}) != len(after_rows):
        raise ValueError("Attribution inputs contain duplicate case IDs")
    if any(x.system != "g000" for x in before_rows) or any(x.system != "g003" for x in after_rows):
        raise ValueError("Attribution requires G000 and G003 system identities")
    updated_by_id = {row.case_id: row for row in after_rows}
    lineage_by_signature: dict[str, list[FailureLineage]] = {}
    for item in lineage:
        lineage_by_signature.setdefault(item.signature.signature_sha256, []).append(item)
    output: list[AttributionRow] = []
    for before in before_rows:
        after = updated_by_id.get(before.case_id)
        if after is not None and after.family_id != before.family_id:
            raise ValueError("Attribution family identity mismatch")
        if before.valid:
            output.append(AttributionRow(before.family_id, before.case_id, "not_g000_failure", None))
            continue
        if after is None or before.failure_signature_sha256 is None:
            output.append(AttributionRow(before.family_id, before.case_id, "incomplete_evidence", None))
            continue
        matches = lineage_by_signature.get(before.failure_signature_sha256, [])
        if not matches:
            output.append(AttributionRow(before.family_id, before.case_id, "unmatched", None))
        elif len({(x.mode_id, x.accepted_skill_version, x.accepted_skill_effect_id) for x in matches}) != 1:
            output.append(AttributionRow(before.family_id, before.case_id, "ambiguous", None))
        else:
            match = matches[0]
            proof = False
            if after.trusted_trajectory is not None:
                if after.trusted_trajectory.case_id != after.case_id or after.trusted_trajectory.generation_id != "g003":
                    raise ValueError("Attribution trajectory identity mismatch")
                proof = after.trusted_trajectory.proves_skill_execution(match.accepted_skill_version, match.accepted_skill_effect_id)
            elif after.source_artifact is not None:
                artifact = after.source_artifact
                if (artifact.system, artifact.case_id, artifact.official_valid) != (after.system, after.case_id, after.valid):
                    raise ValueError("Reporting row does not match its authoritative final artifact")
                proof = (
                    match.accepted_skill_version,
                    match.accepted_skill_effect_id,
                ) in artifact.proven_skill_origins
            repaired = after.valid and proof
            output.append(AttributionRow(before.family_id, before.case_id, "related_repaired" if repaired else "related_unrepaired", match.mode_id))
    return tuple(output)


@dataclass(frozen=True)
class FamilyStability:
    system: str
    family_id: str
    all_four_variants_successful: bool
    valid_variant_count: int
    related_g000_failure_variant_count: int
    repaired_variant_count: int
    all_related_failures_repaired: bool | None


def family_stability(results: Iterable[SystemCaseResult], attribution: Iterable[AttributionRow]) -> tuple[FamilyStability, ...]:
    rows = tuple(results)
    attr = {x.case_id: x.classification for x in attribution}
    grouped: dict[tuple[str, str], list[SystemCaseResult]] = {}
    for row in rows:
        grouped.setdefault((row.system, row.family_id), []).append(row)
    output: list[FamilyStability] = []
    for (system, family_id), family_rows in sorted(grouped.items()):
        variants = {x.variant for x in family_rows}
        if variants != {"base", "miss_func", "miss_param", "long_context"} or len(family_rows) != 4:
            raise ValueError("Stability reporting requires exactly four BFCL variants")
        if len({x.case_id for x in family_rows}) != 4:
            raise ValueError("Stability reporting rejects duplicate cases")
        related = sum(attr.get(x.case_id, "").startswith("related_") for x in family_rows)
        repaired = sum(attr.get(x.case_id) == "related_repaired" for x in family_rows)
        output.append(FamilyStability(system, family_id, all(x.valid for x in family_rows), sum(x.valid for x in family_rows), related, repaired, None if related == 0 else repaired == related))
    return tuple(output)


def attribution_summary(rows: Iterable[AttributionRow]) -> dict[str, float | int | None | bool]:
    rows = tuple(rows)
    related = sum(x.classification in {"related_repaired", "related_unrepaired"} for x in rows)
    repaired = sum(x.classification == "related_repaired" for x in rows)
    return {
        "related_case_count": related,
        "repaired_case_count": repaired,
        "repair_rate": repaired / related if related else None,
        "repair_rate_complete": related > 0,
        "unmatched_case_count": sum(x.classification == "unmatched" for x in rows),
        "ambiguous_case_count": sum(x.classification == "ambiguous" for x in rows),
        "incomplete_evidence_case_count": sum(x.classification == "incomplete_evidence" for x in rows),
    }


def paired_accuracy_uplift(
    vanilla: Iterable[SystemCaseResult],
    g000: Iterable[SystemCaseResult],
    updated: Iterable[SystemCaseResult],
    attribution: Iterable[AttributionRow],
) -> dict[str, float | int | None]:
    vanilla_rows, before_rows, after_rows = tuple(vanilla), tuple(g000), tuple(updated)
    ids = tuple(x.case_id for x in before_rows)
    if not ids or tuple(x.case_id for x in vanilla_rows) != ids or tuple(x.case_id for x in after_rows) != ids:
        raise ValueError("Final system results must share one non-empty manifest order")
    if len(set(ids)) != len(ids):
        raise ValueError("Final system results contain duplicate cases")
    if any(x.system != "vanilla" for x in vanilla_rows) or any(x.system != "g000" for x in before_rows) or any(x.system != "g003" for x in after_rows):
        raise ValueError("Final system identities are not the frozen three-system plan")
    if tuple(x.family_id for x in vanilla_rows) != tuple(x.family_id for x in before_rows) or tuple(x.family_id for x in after_rows) != tuple(x.family_id for x in before_rows):
        raise ValueError("Final family identities do not align")
    related_ids = {x.case_id for x in attribution if x.classification in {"related_repaired", "related_unrepaired"}}
    if not related_ids <= set(ids):
        raise ValueError("Attribution contains cases outside the final manifest")
    overall_g000 = math.fsum(float(x.valid) for x in before_rows) / len(ids)
    overall_updated = math.fsum(float(x.valid) for x in after_rows) / len(ids)
    overall_vanilla = math.fsum(float(x.valid) for x in vanilla_rows) / len(ids)
    paired_by_id = {x.case_id: x for x in after_rows}
    before_by_id = {x.case_id: x for x in before_rows}
    related_uplift = None
    if related_ids:
        related_uplift = math.fsum(float(paired_by_id[x].valid) - float(before_by_id[x].valid) for x in sorted(related_ids)) / len(related_ids)
    return {
        "case_count": len(ids),
        "related_case_count": len(related_ids),
        "g000_accuracy": overall_g000,
        "updated_accuracy": overall_updated,
        "vanilla_accuracy": overall_vanilla,
        "updated_minus_g000": overall_updated - overall_g000,
        "updated_minus_g000_related": related_uplift,
        "updated_minus_vanilla": overall_updated - overall_vanilla,
    }
