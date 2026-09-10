"""Explicit single-service setup probes; only the dedicated runner invokes live.

Usage: uv run python -m toolsandbox_pipeline.providers.preflight MODE --config PATH
PATH is a non-secret JSON provider configuration. No dataset is selected.
"""
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig, UserSimulatorConfig
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.providers.contracts import ProviderRole, RequestContext, ProviderRequestError
from toolsandbox_pipeline.providers.openai_clients import create_transport, MissingCredentialError
from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256

CONFIGS = {"qwen": QwenConfig, "embedding": EmbeddingConfig, "user-simulator": UserSimulatorConfig}
ROLES = {"qwen": ProviderRole.POLICY, "embedding": ProviderRole.EMBEDDING,
         "user-simulator": ProviderRole.USER_SIMULATOR}


def run_preflight(mode, config, *, transport_factory=create_transport):
    """Inject a factory in offline tests. Live CLI has no fake transport option."""
    if mode not in CONFIGS or type(config) is not CONFIGS[mode]:
        raise ValueError("preflight mode/config mismatch")
    expected = config.served_model_id or config.model if mode == "qwen" else config.model
    report = dict(mode=mode, status="blocked", expected_model=expected,
                  returned_model=None, wire_mode=getattr(config, "structured_output_wire_mode", None),
                  vector_dimension=None, latency_seconds=None, input_tokens=None,
                  output_tokens=None, total_tokens=None, usage_complete=False, response_hash=None)
    report["finish_reason"] = None
    transport = None
    attempt = None
    try:
        config.validate_external()
        if mode == "qwen":
            messages = [{"role": "user", "content": 'Return {"action":{"type":"assistant_message","content":"ok"}} only.'}]
            fingerprint = canonical_sha256({"messages": messages, "schema": ActionEnvelope.model_json_schema(), "max_tokens": 256})
        elif mode == "embedding":
            fingerprint = canonical_sha256("preflight")
        else:
            messages = [{"role": "user", "content": "Reply with ok."}]
            fingerprint = canonical_sha256(messages)
        context = RequestContext(logical_request_id="preflight-" + uuid4().hex,
                                 attempt_id=uuid4().hex, role=ROLES[mode], phase="preflight",
                                 unit_reference="setup", input_fingerprint=fingerprint,
                                 replayed_after_unknown_outcome=False,
                                 manifest_identity=canonical_sha256(config.model_dump(mode="json")))
        transport = transport_factory(config, embedding=mode == "embedding")
        if mode == "qwen":
            from toolsandbox_pipeline.providers.qwen import QwenGateway
            result = QwenGateway(config, transport=transport).generate(context, messages, ActionEnvelope, max_tokens=256)
            if result.value.model_dump(mode="json") != {"action": {"type": "assistant_message", "content": "ok"}}:
                report["error"] = "ProbeOutputMismatch"
            attempt = result.attempt
        elif mode == "embedding":
            from toolsandbox_pipeline.providers.embedding import EmbeddingGateway
            gateway = EmbeddingGateway(config, transport=transport)
            result = gateway.embed(context, "preflight")
            report["vector_dimension"] = gateway.dimension
            attempt = result.attempt
        else:
            from toolsandbox_pipeline.providers.user_simulator import UserSimulatorGateway
            result = UserSimulatorGateway(config, transport=transport).chat(context, messages)
            attempt = result.attempt
        report["status"] = "pass" if attempt.metrics.usage.usage_complete and "error" not in report else "fail"
        if not attempt.metrics.usage.usage_complete:
            report["error"] = "MissingUsage"
        if attempt.returned_model is None:
            report.update(status="fail", error="MissingModelIdentity")
    except MissingCredentialError as error:
        # The factory's credential exception contains only a fixed variable name.
        name = str(error)
        report["error"] = name if name in ("OPENAI_API_KEY", "QWEN_API_KEY", "QWEN_BASE_URL") else "MissingCredential"
    except ProviderRequestError as error:
        attempt = error.attempt
        report.update(status="fail", error=attempt.exception_class)
    except Exception:
        report["error"] = "PreflightConfigurationOrEndpointError"
    finally:
        if transport is not None and hasattr(transport, "close"):
            try:
                transport.close()
            except Exception:
                report.update(status="fail", error="TransportCloseError")
    if attempt is not None:
        report.update(returned_model=attempt.returned_model, response_hash=attempt.response_hash,
                      finish_reason=attempt.finish_reason,
                      latency_seconds=attempt.metrics.latency_seconds,
                      input_tokens=attempt.metrics.usage.input_tokens,
                      output_tokens=attempt.metrics.usage.output_tokens,
                      total_tokens=attempt.metrics.usage.total_tokens,
                      usage_complete=attempt.metrics.usage.usage_complete)
    return report


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid preflight arguments")


def main(argv=None):
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument("mode", choices=tuple(CONFIGS))
        parser.add_argument("--config", required=True, type=Path)
        args = parser.parse_args(argv)
        config = CONFIGS[args.mode].model_validate_json(args.config.read_text(encoding="utf-8"))
        report = run_preflight(args.mode, config)
    except Exception:
        report = {"status": "blocked", "error": "InvalidPreflightConfiguration", "usage_complete": False}
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
