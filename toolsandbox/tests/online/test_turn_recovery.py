from toolsandbox_pipeline.online.durable_roles import DurableRoleExecution
from toolsandbox_pipeline.online.prompt_contracts import PreparedRoleRequest
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.online_turn import (
    OnlineTurnFailure,
    OnlineTurnFailureCode,
    OnlineTurnStage,
)


def test_reused_execution_preserves_completed_source_attempt_identity():
    prepared = PreparedRoleRequest.model_construct(role="policy", state_id="state-1")
    output = ActionEnvelope(
        action={"type": "assistant_message", "content": "recovered"}
    )
    execution = DurableRoleExecution(
        prepared_request=prepared,
        output=output,
        logical_request_id="logical-stable",
        source_attempt_id="attempt-original",
        reused_completed_response=True,
    )
    assert execution.reused_completed_response
    assert execution.source_attempt_id == "attempt-original"


def test_terminal_failure_is_typed_and_does_not_expose_state_identity():
    failure = OnlineTurnFailure(
        OnlineTurnFailureCode.CHECKPOINT_FAILURE,
        stage=OnlineTurnStage.STATE_BUILT,
        state_id="private-state-id",
    )
    assert str(failure) == "checkpoint_failure"
    assert "private-state-id" not in repr(failure)
