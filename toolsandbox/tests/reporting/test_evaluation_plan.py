from pathlib import Path
import json

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.reporting.evaluation_plan import (
    EvaluationPlanError,
    load_final_plan,
    write_frozen_plan,
)
from toolsandbox_pipeline.schemas.reporting import (
    category_membership_sha256,
    family_membership_sha256,
    FinalEvaluationPlan,
    PIPELINE_COMPONENTS,
    SystemPlan,
    TestScenarioPlan as ScenarioPlan,
    VANILLA_COMPONENTS,
)
from toolsandbox_pipeline.schemas.dataset import VARIANTS

D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64


def build_plan(output_root: Path, **overrides):
    scenarios = tuple(
        ScenarioPlan(
            manifest_position=index,
            scenario_id=f"family-{index // 8:02d}-variant-{index % 8}",
            scenario_family_id=f"family-{index // 8:02d}",
            variant=VARIANTS[index % 8], categories=("category",),
            starting_context_sha256=D1,
            evaluation_definition_sha256=D1,
            agent_tool_schema_sha256=D1,
        )
        for index in range(200)
    )
    values = dict(
        protocol_version="three-system-final-evaluation-v1",
        user_approval_reference="approval-1",
        frozen_at_utc="2026-09-10T00:00:00Z",
        profile="strict_replay", training_run_id="run-1",
        training_run_sha256=D1, g000_sha256=D1, g003_sha256=D2,
        checkpoint_observation_registry_sha256=D1,
        ordered_system_ids=("vanilla", "generation_0", "updated"),
        systems=(
            SystemPlan(system_id="vanilla", generation_id="g000", allowed_components=VANILLA_COMPONENTS, prompt_manifest_sha256=D1, token_limit_config_sha256=D1, token_limit_status="calibrated"),
            SystemPlan(system_id="generation_0", generation_id="g000", allowed_components=PIPELINE_COMPONENTS, prompt_manifest_sha256=D2, token_limit_config_sha256=D2, token_limit_status="calibrated"),
            SystemPlan(system_id="updated", generation_id="g003", allowed_components=PIPELINE_COMPONENTS, prompt_manifest_sha256=D2, token_limit_config_sha256=D2, token_limit_status="calibrated"),
        ),
        test_dataset_manifest_sha256=D1, scenarios=scenarios,
        family_membership_sha256=family_membership_sha256(scenarios),
        category_membership_sha256=category_membership_sha256(scenarios),
        toolsandbox_source_sha256=D1, dependency_lock_sha256=D1,
        container_image_sha256=D1, environment_sha256=D1,
        world_clock_sha256=D1, fixture_or_live_tool_config_sha256=D1,
        qwen_identity_sha256=D1, qwen_decoding_sha256=D1,
        embedding_identity_sha256=D1, user_simulator_identity_sha256=D1,
        user_prompt_sha256=D1, user_few_shot_sha256=D1,
        user_tools_sha256=D1, user_stop_behavior_sha256=D1,
        prompt_registry_sha256=D1, output_schema_registry_sha256=D1,
        calibrated_token_limit_registry_sha256=D1,
        shared_pipeline_config_sha256=D1, checkpoint_schema_sha256=D1,
        metrics_schema_sha256=D1, reporting_schema_sha256=D1,
        process_count=1, process_order="test_manifest_order",
        seed_policy="scenario_id_sha256_v1",
        output_root=str(output_root),
    )
    values.update(overrides)
    return FinalEvaluationPlan.build(**values)


def test_plan_hash_and_canonical_round_trip(tmp_path):
    plan = build_plan(tmp_path / "out")
    path = tmp_path / "plan.json"
    write_frozen_plan(path, plan)
    assert load_final_plan(path) == plan
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_bytes() == canonical_json_bytes(plan.model_dump(mode="json"))


def test_plan_rejects_generation_or_provisional_limit(tmp_path):
    with pytest.raises(ValidationError):
        SystemPlan(
            system_id="updated", generation_id="g002",
            allowed_components=PIPELINE_COMPONENTS,
            prompt_manifest_sha256=D1, token_limit_config_sha256=D1,
            token_limit_status="calibrated",
        )
    with pytest.raises(ValidationError):
        SystemPlan(
            system_id="vanilla", generation_id="g000",
            allowed_components=VANILLA_COMPONENTS,
            prompt_manifest_sha256=D1, token_limit_config_sha256=D1,
            token_limit_status="provisional",
        )


def test_plan_rejects_secret_or_prior_result(tmp_path):
    with pytest.raises(ValidationError):
        build_plan(tmp_path / "out", user_approval_reference="sk-secret")
    with pytest.raises(ValidationError):
        build_plan(tmp_path / "out", prior_test_result=True)


def test_frozen_plan_cannot_be_overwritten(tmp_path):
    path = tmp_path / "plan.json"
    first = build_plan(tmp_path / "out")
    write_frozen_plan(path, first)
    changed = build_plan(tmp_path / "other")
    with pytest.raises(EvaluationPlanError, match="overwrite"):
        write_frozen_plan(path, changed)


def test_checked_json_schema_exactly_matches_runtime_model():
    path = Path(__file__).parents[2] / "configs" / "run" / "final_evaluation_plan.schema.json"
    assert json.loads(path.read_bytes()) == FinalEvaluationPlan.model_json_schema()
