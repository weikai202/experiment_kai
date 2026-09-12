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


def test_explicit_calibrated_limit_is_bound_and_dispatched_without_bootstrap():
    from toolsandbox_pipeline.schemas.runtime import QwenConfig, RoleTokenLimitConfig
    from toolsandbox_pipeline.providers.qwen import QwenGateway
    from tests.providers.test_request_identity import FakeTransport, chat_response, context
    selection = RoleTokenLimitConfig(version='synthetic-calibration', role='failure_mode_update',
        stage='calibrated', max_tokens=576, evidence_manifest_identity='sha256:' + 'a' * 64)
    transport = FakeTransport(chat_response('{"decision":"SKIP","reason":"No reusable mode"}'))
    gateway = QwenGateway(QwenConfig(structured_output_wire_mode='guided_json'), transport=transport)
    runner = OfflineSkillRoleRunner(gateway, role=ProviderRole.FAILURE_MODE_UPDATE, max_tokens=576,
                                   token_limit_config=selection)
    prepared = runner.prepare(unit_id='unit', messages=[{'role':'user','content':'synthetic input'}])
    result = runner.run(prepared, context(ProviderRole.FAILURE_MODE_UPDATE, unit_reference='unit', input_fingerprint=prepared.input_fingerprint))
    assert result.value.as_decision().decision == 'SKIP'
    assert transport.calls[0]['max_tokens'] == 576
    different = selection.model_copy(update={'evidence_manifest_identity':'sha256:' + 'b' * 64})
    other = OfflineSkillRoleRunner(gateway, role=ProviderRole.FAILURE_MODE_UPDATE, max_tokens=576, token_limit_config=different)
    assert other.prepare(unit_id='unit', messages=prepared.messages).input_fingerprint != prepared.input_fingerprint
    import pytest
    with pytest.raises(ValueError, match='prepared request'):
        other.run(prepared, context(ProviderRole.FAILURE_MODE_UPDATE, unit_reference='unit', input_fingerprint=prepared.input_fingerprint))
    assert len(transport.calls) == 1


def test_default_skill_fingerprint_retains_historical_payload():
    from toolsandbox_pipeline.reproducibility import canonical_sha256
    runner = OfflineSkillRoleRunner(object(), role=ProviderRole.SKILL_CANDIDATE, max_tokens=2048)
    prepared = runner.prepare(unit_id='unit', messages=[{'role':'user','content':'{}'}])
    expected = canonical_sha256(dict(role='skill_candidate',model='Qwen/Qwen3-32B',temperature=0.0,seed=0,
        top_p='omitted',enable_thinking=False,messages=prepared.messages,
        output_schema=prepared.output_model.model_json_schema(),max_tokens=2048))
    assert prepared.input_fingerprint == expected
    assert prepared.token_limit_config is None
