"""Pure recovery-plan facade over validated durable ledger state."""

from __future__ import annotations

from toolsandbox_pipeline.schemas.checkpoint import (
    GenerationPublicationRecord,
    LLMRecoveryPlan,
    OfflineUnitRecord,
    OfflineUnitStatus,
    PublicationRecoveryAction,
    PublicationStatus,
    ToolRecoveryPlan,
    UnitRecoveryAction,
)


class RecoveryPlanner:
    """Return decisions only; callers perform any authorized transition."""

    @staticmethod
    def plan_llm(ledger, logical_request_id: str) -> LLMRecoveryPlan:
        return ledger.plan_recovery(logical_request_id)

    @staticmethod
    def plan_tool(ledger, transaction_id: str) -> ToolRecoveryPlan:
        return ledger.plan_recovery(transaction_id)

    @staticmethod
    def plan_offline_unit(record: OfflineUnitRecord) -> UnitRecoveryAction:
        if type(record) is not OfflineUnitRecord:
            raise TypeError("OfflineUnitRecord required")
        if record.status is OfflineUnitStatus.COMMITTED:
            return UnitRecoveryAction.REUSE_COMMITTED_OUTPUTS
        if record.status is OfflineUnitStatus.TERMINAL_FAILURE:
            return UnitRecoveryAction.TERMINAL_FAILURE
        return UnitRecoveryAction.RESUME_OFFLINE_UNIT

    @staticmethod
    def plan_publication(
        record: GenerationPublicationRecord,
    ) -> PublicationRecoveryAction:
        if type(record) is not GenerationPublicationRecord:
            raise TypeError("GenerationPublicationRecord required")
        if record.status is PublicationStatus.COMMITTED:
            return PublicationRecoveryAction.RUN_COMPLETE
        if record.status is PublicationStatus.TERMINAL_FAILURE:
            return PublicationRecoveryAction.TERMINAL_FAILURE
        return PublicationRecoveryAction.FINALIZE_PUBLICATION


__all__ = ["RecoveryPlanner"]
