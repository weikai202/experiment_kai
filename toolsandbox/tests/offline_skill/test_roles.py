from toolsandbox_pipeline.offline.skill_roles import OfflineSkillRoleRunner
from toolsandbox_pipeline.providers.contracts import ProviderRole


def test_role_preparation_pins_model_decoding_and_schema_identity():
    runner = OfflineSkillRoleRunner(
        object(), role=ProviderRole.SKILL_CANDIDATE, max_tokens=2048
    )
    prepared = runner.prepare(
        unit_id="unit",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "{}"},
        ],
    )
    assert prepared.role is ProviderRole.SKILL_CANDIDATE
    assert prepared.max_tokens == 2048
    assert prepared.input_fingerprint.startswith("sha256:")
    assert prepared.output_model.__name__ == "SkillContentCandidate"
