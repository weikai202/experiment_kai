"""Exactly-once wrapper around the pinned native evaluator."""

from __future__ import annotations

from dataclasses import dataclass

from tool_sandbox.common.evaluation import EvaluationResult

from toolsandbox_pipeline.reproducibility.scenario_hashes import context_sha256
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeIdentity,
    NativeEvaluationMatch,
    TrustedEvaluatorRecord,
)

from .trajectory_store import TrajectoryStore


class NativeEvaluatorError(RuntimeError):
    """Sanitized native-evaluator contract failure."""


@dataclass(frozen=True)
class StoredEvaluation:
    record: TrustedEvaluatorRecord
    reference: BlobReference
    checkpoint_ordinal: int
    reused: bool


class NativeEvaluator:
    def __init__(self, store: TrajectoryStore) -> None:
        if type(store) is not TrajectoryStore:
            raise TypeError("TrajectoryStore required")
        self.store = store

    def evaluate(
        self,
        *,
        identity: EpisodeIdentity,
        scenario: object,
        ending_context: object,
    ) -> StoredEvaluation:
        ending_digest = context_sha256(ending_context)
        existing = self.store.get_evaluator_event(identity)
        if existing is not None:
            payload = existing.payload
            if payload.get("ending_context_sha256") != ending_digest:
                raise NativeEvaluatorError("evaluator checkpoint context conflict")
            reference = BlobReference.model_validate(
                payload.get("evaluator_record_reference"), strict=True
            )
            record = self.store.load_evaluator(reference)
            if (
                record.ending_context_sha256 != ending_digest
                or record.evaluation_definition_sha256
                != identity.evaluation_definition_sha256
            ):
                raise NativeEvaluatorError("stored evaluator identity mismatch")
            return StoredEvaluation(record, reference, existing.event_ordinal, True)

        evaluation = getattr(scenario, "evaluation", None)
        max_messages = getattr(scenario, "max_messages", None)
        if evaluation is None or not callable(getattr(evaluation, "evaluate", None)):
            raise TypeError("native scenario evaluation required")
        if max_messages != identity.max_messages:
            raise NativeEvaluatorError("native evaluator turn limit mismatch")
        result = evaluation.evaluate(
            execution_context=ending_context,
            max_turn_count=max_messages,
        )
        if type(result) is not EvaluationResult:
            raise NativeEvaluatorError("unexpected native evaluator result")
        record = TrustedEvaluatorRecord(
            milestone_similarity=float(result.milestone_similarity),
            minefield_similarity=float(result.minefield_similarity),
            similarity=float(result.similarity),
            turn_count=int(result.turn_count),
            milestone_mapping=self._mapping(result.milestone_mapping),
            minefield_mapping=self._mapping(result.minefield_mapping),
            fully_successful=result.similarity == 1.0,
            evaluation_definition_sha256=identity.evaluation_definition_sha256,
            ending_context_sha256=ending_digest,
        )
        reference, checkpoint = self.store.persist_evaluator(identity, record)
        return StoredEvaluation(record, reference, checkpoint.event_ordinal, False)

    @staticmethod
    def _mapping(mapping: object) -> tuple[NativeEvaluationMatch, ...]:
        if not hasattr(mapping, "items"):
            raise NativeEvaluatorError("unexpected evaluator mapping")
        records: list[NativeEvaluationMatch] = []
        for node_index, value in mapping.items():
            if (
                type(node_index) is not int
                or type(value) is not tuple
                or len(value) != 2
            ):
                raise NativeEvaluatorError("unexpected evaluator mapping")
            snapshot_index, similarity = value
            records.append(
                NativeEvaluationMatch(
                    node_index=node_index,
                    snapshot_index=snapshot_index,
                    similarity=float(similarity),
                )
            )
        return tuple(records)


__all__ = ["NativeEvaluator", "NativeEvaluatorError", "StoredEvaluation"]
