from types import SimpleNamespace

from toolsandbox_pipeline.orchestration.resume import plan_resume
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest

from .test_run_manifest import manifest_payload


class Planner:
    def __init__(self, stage, round_index=None):
        self.stage, self.round_index = stage, round_index

    def plan_run_recovery(self, *, manifest_sha256):
        assert manifest_sha256.startswith("sha256:")
        return SimpleNamespace(stage=self.stage, round_index=self.round_index)


def test_reconciliation_requires_operator_and_completed_run_is_read_only(tmp_path):
    payload = manifest_payload(tmp_path, purpose="formal_training")
    root = tmp_path / "runs" / "formal-case"
    root.mkdir(parents=True)
    manifest = ResolvedRunManifest.model_validate(payload, strict=True)
    blocked = plan_resume(root, manifest, Planner("reconciliation_required", 1))
    assert blocked.status == "operator_action_required"
    assert blocked.opened_execution_dependencies is False
    complete = plan_resume(root, manifest, Planner("run_complete", 2))
    assert complete.status == "complete"
    assert complete.opened_execution_dependencies is False
