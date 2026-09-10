import pytest

from toolsandbox_pipeline.offline.skill_rewrite import (
    next_skill_version,
    validate_skill_candidate,
)
from toolsandbox_pipeline.schemas.offline_skill import SkillContent
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
        applicability=SkillApplicability(required_state=(), forbidden_state=(), best_used_when=()),
        required_inputs=(), expected_outputs=("Useful result",),
        tool_dependencies=("search_stock",), success_criteria=("Visible result",),
        failure_mode_buffer=(),
        cost_profile=SkillCostProfile(expected_tool_calls=1, latency="low", token_cost="low"),
        risk_profile=SkillRiskProfile(risk_if_skipped="low", risk_if_wrong="medium"),
        instruction="Use verified inputs",
        online_statistics=SkillOnlineStatistics(evaluated_uses=0, successes=0, failures=0, success_rate=0.0, last_update_attempt_at_use_count=0),
        validation=None, version="v1.0", status="active",
    )


def test_candidate_identity_scope_and_semantic_change_validation():
    current = skill()
    candidate = SkillContent.from_record(current).model_copy(
        update={"instruction": "Ask for missing inputs before acting"}
    )
    validate_skill_candidate(
        current=current,
        candidate=candidate,
        public_tool_inventory=("search_stock",),
    )
    assert next_skill_version("v1.9") == "v1.10"
    with pytest.raises(ValueError):
        validate_skill_candidate(
            current=current,
            candidate=SkillContent.from_record(current),
            public_tool_inventory=("search_stock",),
        )
