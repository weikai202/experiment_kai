"""Pinned local-only ExecutionContext checkpoint codec."""

from __future__ import annotations

import base64
import copy
import platform

from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.reproducibility.scenario_hashes import context_sha256
from toolsandbox_pipeline.schemas.checkpoint import ExecutionContextEnvelope


UPSTREAM_COMMIT = "165848b9a78cead7ca7fe7c89c688b58e6501219"


class ExecutionContextCodecError(RuntimeError):
    pass


def encode_execution_context(context, *, environment_identity: str) -> bytes:
    from tool_sandbox.common.execution_context import DatabaseNamespace, ExecutionContext

    if type(context) is not ExecutionContext:
        raise TypeError("native ExecutionContext required")
    serialized = context.to_dict(serialize_console=True)
    if type(serialized.get("interactive_console")) is not bytes:
        raise ExecutionContextCodecError("upstream console serialization drift")
    expected_keys = {
        "_dbs",
        "interactive_console",
        "tool_allow_list",
        "tool_deny_list",
        "trace_tool",
        "tool_augmentation_list",
        "preferred_tool_backend",
    }
    if set(serialized) != expected_keys or set(serialized["_dbs"]) != set(DatabaseNamespace):
        raise ExecutionContextCodecError("upstream context serialization shape drift")
    payload = {
        "_dbs": {
            namespace.name: serialized["_dbs"][namespace]
            for namespace in DatabaseNamespace
        },
        "interactive_console": base64.b64encode(
            serialized["interactive_console"]
        ).decode("ascii"),
        "tool_allow_list": serialized["tool_allow_list"],
        "tool_deny_list": serialized["tool_deny_list"],
        "trace_tool": serialized["trace_tool"],
        "tool_augmentation_list": serialized["tool_augmentation_list"],
        "preferred_tool_backend": serialized["preferred_tool_backend"].name,
    }
    digest = canonical_sha256(payload)
    envelope = ExecutionContextEnvelope(
        codec_version="execution-context-v1",
        upstream_commit=UPSTREAM_COMMIT,
        python_patch_version=platform.python_version(),
        environment_identity=environment_identity,
        non_console_context_sha256=context_sha256(context),
        serialized_context_with_base64_console=payload,
        serialized_context_sha256=digest,
    )
    return canonical_json_bytes(envelope.model_dump(mode="json"))


def decode_execution_context(
    encoded: bytes,
    *,
    expected_environment_identity: str,
    expected_upstream_commit: str = UPSTREAM_COMMIT,
    expected_python_patch_version: str | None = None,
):
    """Decode only already-authenticated local blob bytes supplied explicitly."""
    from tool_sandbox.common.execution_context import (
        DatabaseNamespace,
        ExecutionContext,
    )
    from tool_sandbox.common.tool_discovery import ToolBackend

    if type(encoded) is not bytes:
        raise TypeError("encoded envelope must be bytes")
    try:
        envelope = ExecutionContextEnvelope.model_validate_json(encoded, strict=True)
    except Exception as error:
        raise ExecutionContextCodecError("invalid execution context envelope") from error
    python_version = expected_python_patch_version or platform.python_version()
    if (
        envelope.environment_identity != expected_environment_identity
        or envelope.upstream_commit != expected_upstream_commit
        or envelope.python_patch_version != python_version
    ):
        raise ExecutionContextCodecError("execution context identity mismatch")
    payload = envelope.serialized_context_with_base64_console
    if canonical_sha256(payload) != envelope.serialized_context_sha256:
        raise ExecutionContextCodecError("serialized context hash mismatch")
    try:
        console_text = payload["interactive_console"]
        if type(console_text) is not str:
            raise ValueError
        console = base64.b64decode(console_text, validate=True)
        database_payload = payload["_dbs"]
        if type(database_payload) is not dict or set(database_payload) != {
            item.name for item in DatabaseNamespace
        }:
            raise ValueError
        wire = {
            "_dbs": {
                namespace.name: copy.deepcopy(database_payload[namespace.name])
                for namespace in DatabaseNamespace
            },
            "interactive_console": console,
            "tool_allow_list": copy.deepcopy(payload["tool_allow_list"]),
            "tool_deny_list": copy.deepcopy(payload["tool_deny_list"]),
            "trace_tool": payload["trace_tool"],
            "tool_augmentation_list": copy.deepcopy(
                payload["tool_augmentation_list"]
            ),
            "preferred_tool_backend": ToolBackend[payload["preferred_tool_backend"]],
        }
    except (KeyError, TypeError, ValueError, base64.binascii.Error) as error:
        raise ExecutionContextCodecError("invalid serialized context payload") from error
    try:
        restored = ExecutionContext.from_dict(wire)
    except Exception as error:
        raise ExecutionContextCodecError("execution context deserialization failed") from error
    if context_sha256(restored) != envelope.non_console_context_sha256:
        raise ExecutionContextCodecError("non-console context hash mismatch")
    return restored


__all__ = [
    "ExecutionContextCodecError",
    "UPSTREAM_COMMIT",
    "decode_execution_context",
    "encode_execution_context",
]
