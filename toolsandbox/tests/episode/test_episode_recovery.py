from toolsandbox_pipeline.schemas.trajectory import EpisodeExecutionStatus


def test_recovery_statuses_are_closed_and_explicit():
    assert {item.value for item in EpisodeExecutionStatus} == {
        "running",
        "completed_evaluated",
        "terminal_failure_before_evaluation",
        "reconciliation_required",
    }
