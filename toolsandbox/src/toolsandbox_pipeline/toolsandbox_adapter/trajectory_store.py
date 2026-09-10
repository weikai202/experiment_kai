"""Restricted immutable storage primitives for one episode."""

from __future__ import annotations

from dataclasses import dataclass

from toolsandbox_pipeline.checkpointing.execution_context_codec import (
    decode_execution_context,
    encode_execution_context,
)
from toolsandbox_pipeline.checkpointing.llm_ledger import LLMLedger
from toolsandbox_pipeline.checkpointing.store import CheckpointStore
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.reproducibility.scenario_hashes import context_sha256
from toolsandbox_pipeline.schemas.checkpoint import BlobReference, CheckpointEvent
from toolsandbox_pipeline.schemas.fixtures import ExternalReadAttempt
from toolsandbox_pipeline.schemas.trajectory import (
    EpisodeIdentity,
    TrustedEvaluatorRecord,
    TrustedTrajectory,
)


DECISION_MEDIA_TYPE = "application/vnd.toolsandbox.canonical+json"


class TrajectoryStoreError(RuntimeError):
    """Sanitized immutable episode-storage failure."""


@dataclass(frozen=True)
class StoredContext:
    reference: BlobReference
    context_sha256: str


class TrajectoryStore:
    def __init__(self, store: CheckpointStore, *, environment_identity: str) -> None:
        if type(store) is not CheckpointStore:
            raise TypeError("CheckpointStore required")
        if environment_identity != store.identity.environment_identity:
            raise ValueError("environment identity mismatch")
        self.store = store
        self.environment_identity = environment_identity
        self._ledger = LLMLedger(store)

    def persist_context(self, context: object) -> StoredContext:
        encoded = encode_execution_context(
            context, environment_identity=self.environment_identity
        )
        reference = self.store.blobs.put(
            encoded,
            media_type="application/vnd.toolsandbox.execution-context+json",
            schema_name="ExecutionContextEnvelope",
            schema_version=1,
        )
        digest = context_sha256(context)
        return StoredContext(reference=reference, context_sha256=digest)

    def load_context(self, stored: StoredContext):
        encoded = self.store.blobs.read(stored.reference)
        context = decode_execution_context(
            encoded, expected_environment_identity=self.environment_identity
        )
        if context_sha256(context) != stored.context_sha256:
            raise TrajectoryStoreError("restored context identity mismatch")
        return context

    def commit_context_checkpoint(
        self,
        identity: EpisodeIdentity,
        *,
        event_kind: str,
        stored: StoredContext,
        recipient: str | None,
        boundary_ordinal: int,
    ) -> CheckpointEvent:
        checkpoint_id = self._checkpoint_id(
            identity,
            event_kind,
            boundary_ordinal,
            stored.context_sha256,
        )
        return self._ledger.commit_checkpoint(
            checkpoint_id,
            event_kind,
            {
                "episode_id": identity.episode_id,
                "context_reference": stored.reference.model_dump(mode="json"),
                "context_sha256": stored.context_sha256,
                "recipient": recipient,
                "boundary_ordinal": boundary_ordinal,
            },
        )

    def persist_external_attempt(
        self, identity: EpisodeIdentity, attempt: ExternalReadAttempt
    ) -> BlobReference:
        if attempt.context.run_id != identity.run_id:
            raise TrajectoryStoreError("external attempt run identity mismatch")
        if attempt.context.scenario_id != identity.scenario_id:
            raise TrajectoryStoreError("external attempt scenario identity mismatch")
        payload = canonical_json_bytes(attempt.model_dump(mode="json"))
        reference = self.store.blobs.put(
            payload,
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="ExternalReadAttempt",
            schema_version=1,
        )
        checkpoint_id = "external-attempt-" + canonical_sha256(
            [identity.episode_id, attempt.attempt_id]
        )[7:]
        self._ledger.commit_checkpoint(
            checkpoint_id,
            "external_read_attempt_persisted",
            {
                "episode_id": identity.episode_id,
                "attempt_id": attempt.attempt_id,
                "reference": reference.model_dump(mode="json"),
            },
        )
        return reference

    def persist_online_decision(self, decision: object) -> BlobReference:
        if not hasattr(decision, "model_dump"):
            raise TypeError("strict online decision required")
        payload = canonical_json_bytes(decision.model_dump(mode="json"))
        return self.store.blobs.put(
            payload,
            media_type=DECISION_MEDIA_TYPE,
            schema_name=type(decision).__name__,
            schema_version=1,
        )

    def persist_evaluator(
        self, identity: EpisodeIdentity, record: TrustedEvaluatorRecord
    ) -> tuple[BlobReference, CheckpointEvent]:
        payload = canonical_json_bytes(record.model_dump(mode="json"))
        reference = self.store.blobs.put(
            payload,
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="TrustedEvaluatorRecord",
            schema_version=1,
        )
        checkpoint_id = "episode-evaluator-" + canonical_sha256(
            [identity.episode_id, identity.evaluation_definition_sha256]
        )[7:]
        event = self._ledger.commit_checkpoint(
            checkpoint_id,
            "episode_evaluator_committed",
            {
                "episode_id": identity.episode_id,
                "ending_context_sha256": record.ending_context_sha256,
                "evaluator_record_reference": reference.model_dump(mode="json"),
                "evaluator_record_sha256": reference.sha256,
            },
        )
        return reference, event

    def load_evaluator(
        self, reference: BlobReference
    ) -> TrustedEvaluatorRecord:
        return TrustedEvaluatorRecord.model_validate_json(
            self.store.blobs.read(reference), strict=True
        )

    def get_evaluator_event(self, identity: EpisodeIdentity) -> CheckpointEvent | None:
        checkpoint_id = "episode-evaluator-" + canonical_sha256(
            [identity.episode_id, identity.evaluation_definition_sha256]
        )[7:]
        return self._ledger.get_checkpoint(checkpoint_id)

    def persist_trajectory(
        self, identity: EpisodeIdentity, trajectory: TrustedTrajectory
    ) -> tuple[BlobReference, CheckpointEvent]:
        if trajectory.identity != identity:
            raise TrajectoryStoreError("trajectory episode identity mismatch")
        payload = canonical_json_bytes(trajectory.model_dump(mode="json"))
        reference = self.store.blobs.put(
            payload,
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="TrustedTrajectory",
            schema_version=1,
        )
        checkpoint_id = "episode-final-" + canonical_sha256(
            [identity.run_id, identity.episode_id]
        )[7:]
        event = self._ledger.commit_checkpoint(
            checkpoint_id,
            "episode_completed_evaluated",
            {
                "episode_id": identity.episode_id,
                "trajectory_id": trajectory.trajectory_id,
                "trajectory_reference": reference.model_dump(mode="json"),
                "ending_context_reference": trajectory.ending_context_reference.model_dump(
                    mode="json"
                ),
                "ending_context_sha256": trajectory.ending_context_sha256,
                "evaluator_record_reference": trajectory.evaluator_record_reference.model_dump(
                    mode="json"
                ),
                "evaluator_record_sha256": trajectory.evaluator_record_sha256,
            },
        )
        return reference, event

    def load_trajectory(self, reference: BlobReference) -> TrustedTrajectory:
        return TrustedTrajectory.model_validate_json(
            self.store.blobs.read(reference), strict=True
        )

    def get_final_event(self, identity: EpisodeIdentity) -> CheckpointEvent | None:
        checkpoint_id = "episode-final-" + canonical_sha256(
            [identity.run_id, identity.episode_id]
        )[7:]
        return self._ledger.get_checkpoint(checkpoint_id)

    @staticmethod
    def _checkpoint_id(
        identity: EpisodeIdentity,
        event_kind: str,
        boundary_ordinal: int,
        context_digest: str,
    ) -> str:
        return "episode-boundary-" + canonical_sha256(
            [
                identity.run_id,
                identity.episode_id,
                event_kind,
                boundary_ordinal,
                context_digest,
            ]
        )[7:]


__all__ = ["StoredContext", "TrajectoryStore", "TrajectoryStoreError"]
