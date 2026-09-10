"""Content-free, exact failure-signature lineage records."""

from __future__ import annotations

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.offline_skill import (
    EvidenceKind,
    FailureModeLineageRecord,
)


def failure_signature(
    *,
    skill_id: str,
    evidence_kind: EvidenceKind,
    canonical_tool_dependencies: tuple[str, ...],
    sanitized_outcome_class: str,
) -> str:
    dependencies = tuple(
        sorted(canonical_tool_dependencies, key=lambda item: item.encode("utf-8"))
    )
    if dependencies != canonical_tool_dependencies or len(set(dependencies)) != len(dependencies):
        raise ValueError("signature dependencies must be unique UTF-8 sorted")
    return canonical_sha256(
        {
            "skill_id": skill_id,
            "evidence_kind": evidence_kind,
            "sorted_public_canonical_tool_dependencies": list(dependencies),
            "sanitized_outcome_class": sanitized_outcome_class,
        }
    )


def lineage_id(
    signature: str,
    mode_id: str,
    source_evidence_sha256: str,
    producing_round: int,
    failure_mode_effect_id: str,
) -> str:
    return "lineage_" + canonical_sha256(
        {
            "failure_signature_sha256": signature,
            "mode_id": mode_id,
            "source_evidence_sha256": source_evidence_sha256,
            "producing_round": producing_round,
            "failure_mode_effect_id": failure_mode_effect_id,
        }
    )[7:]


def build_lineage(
    *,
    skill_id: str,
    evidence_kind: EvidenceKind,
    canonical_tool_dependencies: tuple[str, ...],
    sanitized_outcome_class: str,
    mode_id: str,
    source_evidence_sha256: str,
    producing_round: int,
    failure_mode_effect_id: str,
) -> FailureModeLineageRecord:
    signature = failure_signature(
        skill_id=skill_id,
        evidence_kind=evidence_kind,
        canonical_tool_dependencies=canonical_tool_dependencies,
        sanitized_outcome_class=sanitized_outcome_class,
    )
    return FailureModeLineageRecord(
        lineage_id=lineage_id(
            signature,
            mode_id,
            source_evidence_sha256,
            producing_round,
            failure_mode_effect_id,
        ),
        failure_signature_sha256=signature,
        skill_id=skill_id,
        evidence_kind=evidence_kind,
        canonical_tool_dependencies=canonical_tool_dependencies,
        sanitized_outcome_class=sanitized_outcome_class,
        mode_id=mode_id,
        source_evidence_sha256=source_evidence_sha256,
        producing_round=producing_round,
        failure_mode_effect_id=failure_mode_effect_id,
    )


def link_accepted_skill(
    record: FailureModeLineageRecord,
    *,
    accepted_skill_version: str,
    accepted_skill_effect_id: str,
) -> FailureModeLineageRecord:
    if record.accepted_skill_version is not None:
        candidate = record.model_copy(
            update={
                "accepted_skill_version": accepted_skill_version,
                "accepted_skill_effect_id": accepted_skill_effect_id,
            }
        )
        if candidate != record:
            raise ValueError("conflicting accepted Skill lineage")
        return record
    return FailureModeLineageRecord(
        **record.model_dump(mode="python", exclude={"accepted_skill_version", "accepted_skill_effect_id"}),
        accepted_skill_version=accepted_skill_version,
        accepted_skill_effect_id=accepted_skill_effect_id,
    )


class FailureLineageIndex:
    def __init__(self) -> None:
        self._records: dict[str, FailureModeLineageRecord] = {}

    def add(self, record: FailureModeLineageRecord) -> FailureModeLineageRecord:
        current = self._records.get(record.lineage_id)
        if current is not None and current != record:
            raise ValueError("conflicting failure lineage replay")
        self._records[record.lineage_id] = record
        return record

    def records(self) -> tuple[FailureModeLineageRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records, key=lambda item: item.encode("utf-8")))


__all__ = [
    "FailureLineageIndex",
    "FailureModeLineageRecord",
    "build_lineage",
    "failure_signature",
    "lineage_id",
    "link_accepted_skill",
]
