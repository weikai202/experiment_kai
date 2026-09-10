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
