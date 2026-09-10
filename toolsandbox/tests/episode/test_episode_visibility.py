def test_sensitive_imports_and_raw_bytes_are_absent_from_public_records():
    from toolsandbox_pipeline.schemas.trajectory import EpisodeResult, TrustedTrajectory

    assert "raw_response" not in repr(TrustedTrajectory.model_fields)
    assert "evaluator" not in EpisodeResult.model_fields["task_accounting_input"].description if EpisodeResult.model_fields["task_accounting_input"].description else True
