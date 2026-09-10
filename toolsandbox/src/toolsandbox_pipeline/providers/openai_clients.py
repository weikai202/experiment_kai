"""Lazy SDK clients with explicit routing, timeout, and zero retries."""
from collections.abc import Mapping
from urllib.parse import urlsplit

from toolsandbox_pipeline.schemas.runtime import QwenConfig, validate_endpoint
from toolsandbox_pipeline.providers.contracts import TransportResponse


class MissingCredentialError(Exception):
    """Only a fixed environment-variable name is safe to expose."""


class SDKTransport:
    def __init__(self, client, *, embedding=False):
        self._client = client
        self._embedding = embedding

    def create(self, **request):
        resource = self._client.embeddings if self._embedding else self._client.chat.completions
        from tool_sandbox.common.utils import all_logging_disabled
        with all_logging_disabled():
            response = resource.with_raw_response.create(**request)
        return TransportResponse(raw_body=response.http_response.content, data=response.parse())

    def close(self):
        self._client.close()


def create_transport(config, *, embedding=False, environ: Mapping[str, str] | None = None,
                     client_factory=None):
    config.validate_external()
    if environ is None:
        import os
        environ = os.environ
    if isinstance(config, QwenConfig):
        base_url = environ.get("QWEN_BASE_URL")
        if not base_url:
            raise MissingCredentialError("QWEN_BASE_URL")
        validate_endpoint(base_url)
        key_name = "QWEN_API_KEY"
    else:
        base_url = config.base_url
        key_name = "OPENAI_API_KEY"
    key = environ.get(key_name)
    if not key:
        raise MissingCredentialError(key_name)
    if isinstance(config, QwenConfig) and key == "EMPTY":
        if not config.allow_empty_local_key or urlsplit(base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("local no-auth endpoint was not selected")
    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI
    client = None
    try:
        # Explicit organization prevents SDK inheritance from OPENAI_ORG_ID.
        client = client_factory(api_key=key, base_url=base_url, organization="",
                                max_retries=0, timeout=config.timeout_seconds)
    except Exception:
        pass
    if client is None:
        raise ValueError("ClientConstructionError") from None
    return SDKTransport(client, embedding=embedding)
