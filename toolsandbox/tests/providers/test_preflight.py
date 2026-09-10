import importlib
import json
import os
import socket
import sys
from types import SimpleNamespace

import pytest
from toolsandbox_pipeline.providers import preflight
from toolsandbox_pipeline.providers.openai_clients import SDKTransport, MissingCredentialError
from toolsandbox_pipeline.schemas.runtime import EmbeddingConfig, UserSimulatorConfig, QwenConfig
from tests.providers.test_request_identity import FakeTransport, chat_response
from tests.providers.test_embedding import embedding_response
from tests.providers.test_runtime_config import resolved_qwen


@pytest.mark.parametrize("mode,config,payload,embedding", [
    ("qwen", resolved_qwen(), chat_response(), False),
    ("embedding", EmbeddingConfig(), embedding_response(), True),
    ("user-simulator", UserSimulatorConfig(), chat_response("ok", "gpt-4o-mini-2024-07-18"), False),
])
def test_one_selected_service(mode, config, payload, embedding):
    calls = []
    transport = FakeTransport(payload)
    def factory(received, **options):
        calls.append((received, options))
        return transport
    report = preflight.run_preflight(mode, config, transport_factory=factory)
    assert calls == [(config, {"embedding": embedding})]
    assert len(transport.calls) == 1 and transport.closed
    assert report["status"] == "pass" and report["usage_complete"]
    assert report["expected_model"] == report["returned_model"] == payload["model"]
    assert report["total_tokens"] == payload["usage"]["total_tokens"]
    assert report["response_hash"].startswith("sha256:")
    assert report["latency_seconds"] >= 0
    if embedding:
        assert report["vector_dimension"] == 2
    assert not {"messages", "prompt", "response", "content", "base_url"} & report.keys()


@pytest.mark.parametrize("mode,config,payload", [
    ("qwen", resolved_qwen(), chat_response()),
    ("embedding", EmbeddingConfig(), embedding_response()),
    ("user-simulator", UserSimulatorConfig(), chat_response("ok", "gpt-4o-mini-2024-07-18")),
])
def test_missing_usage_fails_preflight(mode, config, payload):
    payload.pop("usage")
    report = preflight.run_preflight(mode, config, transport_factory=lambda *a, **k: FakeTransport(payload))
    assert report["status"] == "fail"
    assert not report["usage_complete"] and report["total_tokens"] is None


def test_unresolved_qwen_does_not_construct_transport():
    def unexpected(*args, **kwargs):
        pytest.fail("transport constructed for unresolved deployment")
    report = preflight.run_preflight("qwen", QwenConfig(structured_output_wire_mode="guided_json"), transport_factory=unexpected)
    assert report["status"] == "blocked"


@pytest.mark.parametrize("error", [ValueError("secret-sentinel"), MissingCredentialError("secret-sentinel")])
def test_factory_errors_sanitized(error):
    def factory(*a, **kw):
        raise error
    report = preflight.run_preflight("embedding", EmbeddingConfig(), transport_factory=factory)
    assert report["status"] == "blocked"
    assert "secret-sentinel" not in json.dumps(report)


def test_model_mismatch_and_endpoint_errors_sanitized():
    for transport in (FakeTransport(chat_response("secret-content", "secret-sentinel")),
                      FakeTransport(error=TimeoutError("secret-sentinel"))):
        report = preflight.run_preflight("qwen", resolved_qwen(), transport_factory=lambda *a, **k: transport)
        assert report["status"] == "fail"
        assert "secret-sentinel" not in json.dumps(report)
        assert "secret-content" not in json.dumps(report)
        assert len(transport.calls) == 1


@pytest.mark.parametrize("argv", [[], ["all"], ["secret-sentinel"], ["embedding"],
                                    ["embedding", "--config", "/nonexistent/secret-sentinel"]])
def test_cli_invalid_args_are_sanitized_nonzero(argv, capsys):
    assert preflight.main(argv) != 0
    out = capsys.readouterr()
    assert "secret-sentinel" not in out.out + out.err
    assert json.loads(out.out)["status"] == "blocked"


def test_cli_selected_mode_and_failure_exit(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(EmbeddingConfig().model_dump_json())
    calls = []
    def run(mode, config):
        calls.append((mode, config))
        return {"status": "fail", "usage_complete": False}
    monkeypatch.setattr(preflight, "run_preflight", run)
    assert preflight.main(["embedding", "--config", str(path)]) == 1
    assert calls == [("embedding", EmbeddingConfig())]
    assert json.loads(capsys.readouterr().out)["status"] == "fail"


def test_sdk_transport_raw_body_and_single_call():
    calls = []
    parsed = chat_response()
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(http_response=SimpleNamespace(content=b'raw bytes'), parse=lambda: parsed)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=create))))
    response = SDKTransport(client).create(model="Qwen/Qwen3-32B")
    assert response.raw_body == b'raw bytes' and response.data is parsed
    assert calls == [{"model": "Qwen/Qwen3-32B"}]


def test_imports_and_construction_have_no_provider_side_effects():
    # Isolate reloaded class identities from the other tests. Third-party plugin
    # discovery is fixed locally; no real environment (including plugins) is read.
    import subprocess
    script = r"""
import importlib
import os
import socket
import sys
import builtins
import openai
import pydantic.plugin._loader
import toolsandbox_pipeline.providers.preflight
import toolsandbox_pipeline.providers.user_simulator
pydantic.plugin._loader.get_plugins = lambda: []
def forbidden(*args, **kwargs):
    raise AssertionError("provider side effect during import")
class NoEnvironment(dict):
    __getitem__ = forbidden
    get = forbidden
os.environ = NoEnvironment()
os.getenv = forbidden
socket.socket.connect = forbidden
openai.OpenAI = forbidden
original_open = builtins.open
def no_write(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in "wax+"):
        forbidden()
    return original_open(file, mode, *args, **kwargs)
builtins.open = no_write
modules = ["schemas.runtime", "schemas.usage", "providers.contracts", "metrics.usage",
           "providers.openai_clients", "providers.qwen", "providers.embedding",
           "providers.user_simulator", "providers.preflight", "providers"]
for name in modules:
    importlib.reload(sys.modules["toolsandbox_pipeline." + name])
from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig
from toolsandbox_pipeline.providers import QwenGateway, EmbeddingGateway
from toolsandbox_pipeline.providers.user_simulator import InstrumentedGPT4oMiniUser
QwenGateway(QwenConfig(structured_output_wire_mode="guided_json"))
EmbeddingGateway(EmbeddingConfig())
InstrumentedGPT4oMiniUser(context_provider=forbidden)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            env={}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
