import json

import pytest
from tool_sandbox.common.execution_context import ExecutionContext

from toolsandbox_pipeline.checkpointing import (
    ExecutionContextCodecError,
    decode_execution_context,
    encode_execution_context,
)
from toolsandbox_pipeline.reproducibility.scenario_hashes import context_sha256


ENVIRONMENT = "sha256:" + "1" * 64


def test_execution_context_console_round_trip_is_deeply_isolated():
    original = ExecutionContext()
    original.interactive_console.locals["checkpoint_sentinel"] = [1, 2]
    encoded = encode_execution_context(original, environment_identity=ENVIRONMENT)
    restored = decode_execution_context(
        encoded, expected_environment_identity=ENVIRONMENT
    )
    assert context_sha256(restored) == context_sha256(original)
    assert restored.interactive_console.locals["checkpoint_sentinel"] == [1, 2]
    restored.interactive_console.locals["checkpoint_sentinel"].append(3)
    assert original.interactive_console.locals["checkpoint_sentinel"] == [1, 2]


def test_context_codec_rejects_identity_and_hash_tampering():
    encoded = encode_execution_context(
        ExecutionContext(), environment_identity=ENVIRONMENT
    )
    with pytest.raises(ExecutionContextCodecError, match="identity mismatch"):
        decode_execution_context(
            encoded, expected_environment_identity="sha256:" + "2" * 64
        )
    payload = json.loads(encoded)
    payload["serialized_context_with_base64_console"]["trace_tool"] = not payload[
        "serialized_context_with_base64_console"
    ]["trace_tool"]
    with pytest.raises(ExecutionContextCodecError, match="hash mismatch"):
        decode_execution_context(
            json.dumps(payload).encode(), expected_environment_identity=ENVIRONMENT
        )


def test_context_decode_is_explicit_and_rejects_non_envelope_bytes():
    with pytest.raises(ExecutionContextCodecError, match="invalid execution"):
        decode_execution_context(
            b"not-json", expected_environment_identity=ENVIRONMENT
        )
