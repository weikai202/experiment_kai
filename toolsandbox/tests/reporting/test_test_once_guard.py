from concurrent.futures import ThreadPoolExecutor
import os

import pytest

from toolsandbox_pipeline.reporting.evaluation_plan import (
    CoordinatorRegistryAuthority,
    EvaluationPlanError,
    SystemCompletionReceipt,
    TestOnceGuard as OnceGuard,
)
from tests.reporting.test_system_runner import build_plan
from toolsandbox_pipeline.schemas.checkpoint import BlobReference


def _authorize_all(guard, plan, system_id):
    for scenario in plan.scenarios:
        guard.authorize_episode(
            system_id, scenario.scenario_id, f"episode-{system_id}-{scenario.scenario_id}"
        )


def _authority(root):
    return CoordinatorRegistryAuthority.create(root, authority_id="coordinator-1")


def _receipt(character="a"):
    digest = "sha256:" + character * 64
    return SystemCompletionReceipt(
        result_reference=BlobReference(
            sha256=digest,
            byte_count=1,
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="PersistedSystemResult",
            schema_version=1,
            content_visibility="restricted",
        ),
        result_sha256=digest,
        timing_sha256="sha256:" + "b" * 64,
    )


def _material_reference():
    return BlobReference(
        sha256="sha256:" + "c" * 64,
        byte_count=1,
        media_type="application/vnd.toolsandbox.canonical+json",
        schema_name="StagedSystemMaterial",
        schema_version=1,
        content_visibility="restricted",
    )


def test_fixed_order_completion_and_completed_rerun_denial(tmp_path):
    plan = build_plan(tmp_path / "out")
    guard = OnceGuard.create(_authority(tmp_path / "registry"), plan)
    with pytest.raises(EvaluationPlanError, match="fixed order"):
        guard.start_system("generation_0")
    guard.start_system("vanilla")
    _authorize_all(guard, plan, "vanilla")
    guard.stage_system_material("vanilla", _material_reference())
    receipt = _receipt()
    guard.stage_system_completion("vanilla", receipt)
    guard.complete_system("vanilla", receipt)
    assert guard.state("vanilla") == "completed"
    with pytest.raises(EvaluationPlanError, match="already started"):
        guard.start_system("vanilla")
    with pytest.raises(EvaluationPlanError, match="not resumable"):
        guard.assert_identical_resume("vanilla", plan.plan_sha256)


def test_resume_is_bound_to_exact_plan_and_episode_identity(tmp_path):
    plan = build_plan(tmp_path / "out")
    root = tmp_path / "registry"
    authority = _authority(root)
    guard = OnceGuard.create(authority, plan)
    guard.start_system("vanilla")
    first = plan.scenarios[0]
    guard.authorize_episode("vanilla", first.scenario_id, "episode-stable")
    reopened = OnceGuard.open(authority, plan)
    reopened.assert_identical_resume("vanilla", plan.plan_sha256)
    reopened.authorize_episode("vanilla", first.scenario_id, "episode-stable")
    with pytest.raises(EvaluationPlanError, match="distinct episode"):
        reopened.authorize_episode("vanilla", first.scenario_id, "episode-changed")
    changed = build_plan(tmp_path / "different-output")
    with pytest.raises(EvaluationPlanError, match="invalid test-once ledger|plan identity"):
        OnceGuard.open(authority, changed)


def test_reconciliation_state_is_official_live_only(tmp_path):
    strict = build_plan(tmp_path / "strict")
    strict_guard = OnceGuard.create(_authority(tmp_path / "strict-registry"), strict)
    strict_guard.start_system("vanilla")
    with pytest.raises(EvaluationPlanError, match="official-live"):
        strict_guard.mark_reconciliation_required("vanilla")

    live = build_plan(tmp_path / "live", profile="official_live")
    live_guard = OnceGuard.create(_authority(tmp_path / "live-registry"), live)
    live_guard.start_system("vanilla")
    live_guard.mark_reconciliation_required("vanilla")
    assert live_guard.state("vanilla") == "reconciliation_required"
    live_guard.assert_identical_resume("vanilla", live.plan_sha256)


def test_authority_rejects_bypass_wrong_identity_and_broken_symlink(tmp_path):
    plan = build_plan(tmp_path / "out")
    authority = _authority(tmp_path / "registry")
    with pytest.raises(TypeError, match="created or opened"):
        CoordinatorRegistryAuthority(
            object(), authority.registry_root, "coordinator-1", authority.authority_sha256
        )
    with pytest.raises(EvaluationPlanError, match="identity mismatch"):
        CoordinatorRegistryAuthority.open(
            authority.registry_root, authority_id="different"
        )
    broken = tmp_path / "broken"
    os.symlink(tmp_path / "missing-target", broken)
    with pytest.raises(EvaluationPlanError, match="symlink"):
        CoordinatorRegistryAuthority.create(
            broken / "registry", authority_id="coordinator-1"
        )
    with pytest.raises(EvaluationPlanError, match="symlink"):
        from toolsandbox_pipeline.reporting.evaluation_plan import write_frozen_plan

        write_frozen_plan(broken / "plan.json", plan)


def test_flock_serializes_start_and_preserves_concurrent_episode_updates(tmp_path):
    plan = build_plan(tmp_path / "out")
    authority = _authority(tmp_path / "registry")
    first = OnceGuard.create(authority, plan)
    second = OnceGuard.open(authority, plan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(
            pool.map(
                lambda guard: _start_outcome(guard),
                (first, second),
            )
        )
    assert sorted(outcomes) == ["already-started", "started"]
    scenarios = plan.scenarios[:2]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tuple(
            pool.map(
                lambda item: item[0].authorize_episode(
                    "vanilla", item[1].scenario_id, "episode-" + item[1].scenario_id
                ),
                ((first, scenarios[0]), (second, scenarios[1])),
            )
        )
    reopened = OnceGuard.open(authority, plan)
    payload = reopened._read()
    assert set(payload["systems"]["vanilla"]["episodes"]) >= {
        scenarios[0].scenario_id,
        scenarios[1].scenario_id,
    }


def _start_outcome(guard):
    try:
        guard.start_system("vanilla")
    except EvaluationPlanError:
        return "already-started"
    return "started"
