import pytest
from pydantic import ValidationError
from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig, UserSimulatorConfig
from toolsandbox_pipeline.providers.openai_clients import create_transport, MissingCredentialError


def resolved_qwen(**changes):
    values = dict(structured_output_wire_mode="guided_json", expected_vllm_version="0.17.0",
                  container_digest="sha256:" + "c" * 64, served_model_id="Qwen/Qwen3-32B",
                  structured_output_backend="xgrammar", generation_config_policy="vllm",
                  server_launch_configuration="reviewed-launch-v1", context_limit=32768,
                  output_limit=2048)
    values.update(changes)
    return QwenConfig(**values)


@pytest.mark.parametrize("changes", [
    {"model": "other"}, {"temperature": 1.0}, {"temperature": False}, {"seed": False},
    {"enable_thinking": 0}, {"enable_thinking": True}, {"seed": 1}, {"top_p": 1.0},
    {"timeout_seconds": float("inf")}, {"timeout_seconds": 0.0},
    {"structured_output_wire_mode": "auto"}, {"api_key_env": "OPENAI_API_KEY"},
])
def test_qwen_overrides_rejected(changes):
    with pytest.raises(ValidationError):
        resolved_qwen(**changes)


@pytest.mark.parametrize("config,changes", [
    (EmbeddingConfig, {"model": "text-embedding-3-large"}),
    (EmbeddingConfig, {"dimensions": 1}),
    (EmbeddingConfig, {"encoding_format": "base64"}),
    (EmbeddingConfig, {"base_url": "https://example.org/v1"}),
    (EmbeddingConfig, {"base_url": "https://name:secret-sentinel@example.org/v1", "base_url_manifest_identity": "proxy"}),
    (UserSimulatorConfig, {"model": "gpt-4o-mini"}),
    (UserSimulatorConfig, {"temperature": 0.0}),
    (UserSimulatorConfig, {"top_p": 1.0}),
    (UserSimulatorConfig, {"seed": 0}),
])
def test_openai_configs_reject_overrides(config, changes):
    with pytest.raises(ValidationError) as caught:
        config(**changes)
    assert "secret-sentinel" not in str(caught.value)


def test_runtime_frozen_and_alternate_endpoint_recorded():
    config = EmbeddingConfig(base_url="https://example.org/v1", base_url_manifest_identity="proxy-v1")
    with pytest.raises(ValidationError):
        config.model = "other"
    assert config.base_url_manifest_identity == "proxy-v1"


def test_client_isolation_zero_retries_and_timeout():
    received = []
    def factory(**kwargs):
        received.append(kwargs)
        return object()
    env = {"OPENAI_API_KEY": "openai-sentinel", "QWEN_API_KEY": "qwen-sentinel",
           "QWEN_BASE_URL": "http://localhost:8000/v1", "OPENAI_BASE_URL": "https://wrong.example/v1"}
    q = create_transport(resolved_qwen(), environ=env, client_factory=factory)
    e = create_transport(EmbeddingConfig(), embedding=True, environ=env, client_factory=factory)
    u = create_transport(UserSimulatorConfig(), environ=env, client_factory=factory)
    assert len({id(q._client), id(e._client), id(u._client)}) == 3
    assert received[0]["api_key"] == "qwen-sentinel"
    assert received[0]["base_url"] == "http://localhost:8000/v1"
    for request in received[1:]:
        assert request["api_key"] == "openai-sentinel"
        assert request["base_url"] == "https://api.openai.com/v1"
    assert all(r["max_retries"] == 0 and r["timeout"] == 60.0 for r in received)
    assert "sentinel" not in repr(q) + repr(e) + repr(u)


def test_locked_openai_http_client_constructs_without_network():
    transport = create_transport(
        EmbeddingConfig(),
        embedding=True,
        environ={"OPENAI_API_KEY": "test-placeholder"},
    )
    transport.close()


def test_unresolved_config_before_environment_read():
    class NoEnvironment(dict):
        def get(self, *args):
            pytest.fail("environment read before configuration validation")
    with pytest.raises(ValueError, match="unresolved"):
        create_transport(QwenConfig(structured_output_wire_mode="guided_json"), environ=NoEnvironment())


@pytest.mark.parametrize("config,env,name", [
    (EmbeddingConfig(), {}, "OPENAI_API_KEY"),
    (UserSimulatorConfig(), {}, "OPENAI_API_KEY"),
    (resolved_qwen(), {}, "QWEN_BASE_URL"),
    (resolved_qwen(), {"QWEN_BASE_URL": "http://localhost:8000/v1", "OPENAI_API_KEY": "sentinel"}, "QWEN_API_KEY"),
])
def test_missing_credentials_report_name_only(config, env, name):
    with pytest.raises(MissingCredentialError) as caught:
        create_transport(config, environ=env)
    assert str(caught.value) == name


def test_empty_key_requires_explicit_local_selection():
    env = {"QWEN_BASE_URL": "http://localhost:8000/v1", "QWEN_API_KEY": "EMPTY"}
    with pytest.raises(ValueError):
        create_transport(resolved_qwen(), environ=env)
    create_transport(resolved_qwen(allow_empty_local_key=True), environ=env, client_factory=lambda **kw: object())
    env["QWEN_BASE_URL"] = "https://remote.example/v1"
    with pytest.raises(ValueError):
        create_transport(resolved_qwen(allow_empty_local_key=True), environ=env)


def test_versioned_token_limit_selection():
    from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig
    bootstrap = RoleTokenLimitConfig(version="bootstrap-v1", role="policy", stage="bootstrap", max_tokens=256)
    assert bootstrap.max_tokens == 256
    with pytest.raises(ValidationError):
        RoleTokenLimitConfig(version="bootstrap-v1", role="policy", stage="bootstrap", max_tokens=512)
    with pytest.raises(ValidationError):
        RoleTokenLimitConfig(version="v1", role="policy", stage="calibrated", max_tokens=512)
    with pytest.raises(ValidationError):
        RoleTokenLimitConfig(version="v1", role="policy", stage="calibrated", max_tokens=513,
                             evidence_manifest_identity="sha256:" + "a" * 64)
    config = RoleTokenLimitConfig(version="v1", role="policy", stage="calibrated", max_tokens=512,
                                 evidence_manifest_identity="sha256:" + "a" * 64)
    with pytest.raises(ValidationError):
        config.max_tokens = 1024


def test_default_critic_mode_preserves_historical_configuration_identity():
    import json
    config = resolved_qwen()
    expected = dict(timeout_seconds=60.0, provider='vllm_openai_compatible', model='Qwen/Qwen3-32B',
                    temperature=0.0, seed=0, enable_thinking=False, structured_output_wire_mode='guided_json',
                    base_url_env='QWEN_BASE_URL', api_key_env='QWEN_API_KEY', allow_empty_local_key=False,
                    expected_vllm_version='0.17.0', container_digest='sha256:' + 'c' * 64,
                    served_model_id='Qwen/Qwen3-32B', structured_output_backend='xgrammar',
                    generation_config_policy='vllm', server_launch_configuration='reviewed-launch-v1',
                    context_limit=32768, output_limit=2048)
    assert config.model_dump() == expected
    assert config.model_dump(mode='json') == expected
    assert json.loads(config.model_dump_json()) == expected
    assert QwenConfig.model_validate(expected).model_dump() == expected


def grammar_config(**changes):
    from toolsandbox_pipeline.providers.critic_grammar import critic_grammar_sha256
    values = dict(structured_output_wire_mode='structured_outputs_json',
                  critic_structured_output_mode='ordered_unique_grammar_v1',
                  critic_grammar_sha256=critic_grammar_sha256())
    values.update(changes)
    return resolved_qwen(**values)


def test_explicit_critic_grammar_configuration_roundtrip_and_identity():
    from toolsandbox_pipeline.reproducibility import canonical_sha256
    config = grammar_config()
    payload = config.model_dump(mode='json')
    assert payload['critic_structured_output_mode'] == 'ordered_unique_grammar_v1'
    assert payload['critic_grammar_sha256'].startswith('sha256:')
    assert QwenConfig.model_validate_json(config.model_dump_json()) == config
    config.validate_external()
    assert canonical_sha256(payload) != canonical_sha256(resolved_qwen(structured_output_wire_mode='structured_outputs_json').model_dump(mode='json'))


@pytest.mark.parametrize('changes', [
    {'critic_structured_output_mode': 'auto'},
    {'critic_structured_output_mode': 'json_schema'},
    {'critic_grammar_sha256': None},
    {'critic_grammar_sha256': 'sha256:' + '0' * 64},
    {'critic_grammar_sha256': 'invalid'},
    {'structured_output_wire_mode': 'guided_json'},
    {'structured_output_backend': 'outlines'},
    {'structured_output_backend': None},
])
def test_invalid_critic_grammar_configuration_fails_closed(changes):
    with pytest.raises(ValidationError):
        grammar_config(**changes)
