import pytest

from toolsandbox_pipeline.offline.skill_projection import skill_rewrite_projection
from toolsandbox_pipeline.schemas.skill import (
    SkillApplicability,
    SkillCostProfile,
    SkillOnlineStatistics,
    SkillRecord,
    SkillRiskProfile,
)


def skill():
    return SkillRecord(
        skill_id="skill",
        name="HelpfulSkill",
        description="Use visible evidence safely",
        applicability=SkillApplicability(
            required_state=(), forbidden_state=(), best_used_when=()
        ),
        required_inputs=(),
        expected_outputs=("A useful result",),
        tool_dependencies=("search_stock",),
        success_criteria=("Visible confirmation",),
        failure_mode_buffer=(),
        cost_profile=SkillCostProfile(
            expected_tool_calls=1, latency="low", token_cost="low"
        ),
        risk_profile=SkillRiskProfile(risk_if_skipped="low", risk_if_wrong="medium"),
        instruction="Use verified inputs and return visible evidence",
        online_statistics=SkillOnlineStatistics(
            evaluated_uses=0,
            successes=0,
            failures=0,
            success_rate=0.0,
            last_update_attempt_at_use_count=0,
        ),
        validation=None,
        version="v1.0",
        status="active",
    )


def test_rewrite_projection_is_single_skill_train_only():
    current = skill()
    projection = skill_rewrite_projection(
        skill=current,
        statistics=current.online_statistics,
        failure_modes=(),
        public_tool_schemas=({"function": {"name": "search_stock"}},),
        generalized_train_trajectories=({"outcome_class": "MISSING_INPUT"},),
    )
    assert projection["current_skill"]["skill_id"] == "skill"
    with pytest.raises(ValueError):
        skill_rewrite_projection(
            skill=current,
            statistics=current.online_statistics,
            failure_modes=(),
            public_tool_schemas=(),
            generalized_train_trajectories=({"dev": "forbidden"},),
        )
