from toolsandbox_pipeline.offline.skill_orchestrator import (
    SealedSkillTrajectoryBuffer,
    SkillUpdateOrchestrator,
)
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.offline_skill import SkillRoundIdentity
from toolsandbox_pipeline.schemas.skill import (
    SkillApplicability,
    SkillCostProfile,
    SkillOnlineStatistics,
    SkillRecord,
    SkillRiskProfile,
)


DIGEST = "sha256:" + "d" * 64


class Never:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected dependency access: {name}")


class Sink:
    def checkpoint(self, **kwargs):
        raise AssertionError("empty round must not checkpoint a model step")


def current_skill():
    return SkillRecord(
        skill_id="skill", name="HelpfulSkill", description="Safe guidance",
        applicability=SkillApplicability(required_state=(), forbidden_state=(), best_used_when=()),
        required_inputs=(), expected_outputs=("Result",), tool_dependencies=("search_stock",),
        success_criteria=("Visible result",), failure_mode_buffer=(),
        cost_profile=SkillCostProfile(expected_tool_calls=1, latency="low", token_cost="low"),
        risk_profile=SkillRiskProfile(risk_if_skipped="low", risk_if_wrong="medium"),
        instruction="Use verified inputs",
        online_statistics=SkillOnlineStatistics(evaluated_uses=0, successes=0, failures=0, success_rate=0.0, last_update_attempt_at_use_count=0),
        validation=None, version="v1.0", status="active",
    )


def test_empty_sealed_round_is_deterministic_and_zero_cost():
    sealed_hash = canonical_sha256({
        "protocol": "sealed-skill-buffer-v1", "run_id": "run", "round_index": 0,
        "shard_id": "shard", "generation_id": "g000", "entries": [],
    })
    identity = SkillRoundIdentity(
        run_id="run", round_index=0, shard_id="shard",
        current_generation_id="g000", next_generation_id="g001",
        dataset_manifest_sha256=DIGEST, config_manifest_sha256=DIGEST,
        prompt_manifest_sha256=DIGEST, token_limit_config_sha256=DIGEST,
        sealed_input_buffer_sha256=sealed_hash,
    )
    buffer = SealedSkillTrajectoryBuffer(identity, ())
    orchestrator = SkillUpdateOrchestrator(
        failure_executor=Never(), candidate_executor=Never(),
        mini_bench_executor=Never(), effect_sink=Sink(),
        public_tool_inventory=("search_stock",),
    )
    result = orchestrator.run(buffer=buffer, current_skills=(current_skill(),))
    assert result.completion_status == "completed"
    assert result.lineage_records == ()
    assert result.unit_results[0].total_cost_application_ids == ()
    assert result == orchestrator.run(buffer=buffer, current_skills=(current_skill(),))
