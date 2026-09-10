"""Stable checkpoint-ledger identities."""

from __future__ import annotations

from hashlib import sha256

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes
from toolsandbox_pipeline.schemas.checkpoint import (
    LogicalLLMRequestIdentity,
    QwenEffectKind,
    ToolActionIdentity,
)


def logical_request_id(identity: LogicalLLMRequestIdentity) -> str:
    payload = {
        "protocol": "llm-request-v1",
        "run_id": identity.run_id,
        "role": identity.role.value,
        "phase": identity.phase,
        "unit_reference": identity.unit_reference,
        "input_fingerprint": identity.input_fingerprint,
        "model": identity.model,
        "decoding_configuration_sha256": identity.decoding_configuration_sha256,
        "output_schema_sha256": identity.output_schema_sha256,
    }
    return "llm-" + sha256(canonical_json_bytes(payload)).hexdigest()


def attempt_id(request_id: str, ordinal: int) -> str:
    if not _valid_logical_id(request_id):
        raise ValueError("invalid logical request ID")
    if type(ordinal) is not int or ordinal < 1 or ordinal > 99_999_999:
        raise ValueError("invalid attempt ordinal")
    return f"attempt-{request_id[4:]}-{ordinal:08d}"


def application_id(
    request_id: str,
    source_attempt_id: str,
    artifact_id: str,
    artifact_sha256: str,
    checkpoint_id: str,
) -> str:
    payload = {
        "protocol": "llm-application-v1",
        "logical_request_id": request_id,
        "source_attempt_id": source_attempt_id,
        "application_artifact_id": artifact_id,
        "application_artifact_sha256": artifact_sha256,
        "committed_checkpoint_id": checkpoint_id,
    }
    return "application-" + sha256(canonical_json_bytes(payload)).hexdigest()


def effective_effect_id(
    *,
    effect_kind: QwenEffectKind,
    effect_artifact_id: str,
    effect_artifact_sha256: str,
    ordered_application_ids: tuple[str, ...],
    committed_checkpoint_id: str,
) -> str:
    payload = {
        "protocol": "qwen-effective-effect-v1",
        "effect_kind": effect_kind.value,
        "effect_artifact_id": effect_artifact_id,
        "effect_artifact_sha256": effect_artifact_sha256,
        "ordered_application_ids": list(ordered_application_ids),
        "committed_checkpoint_id": committed_checkpoint_id,
    }
    return "effect-" + sha256(canonical_json_bytes(payload)).hexdigest()


def tool_transaction_id(identity: ToolActionIdentity) -> str:
    payload = {
        "protocol": "tool-transaction-v1",
        **identity.model_dump(mode="json"),
    }
    return "tool-" + sha256(canonical_json_bytes(payload)).hexdigest()


def tool_attempt_id(transaction_id: str, ordinal: int) -> str:
    if (
        type(transaction_id) is not str
        or not transaction_id.startswith("tool-")
        or len(transaction_id) != 69
        or any(character not in "0123456789abcdef" for character in transaction_id[5:])
    ):
        raise ValueError("invalid tool transaction ID")
    if type(ordinal) is not int or ordinal < 1 or ordinal > 99_999_999:
        raise ValueError("invalid tool attempt ordinal")
    return f"tool-attempt-{transaction_id[5:]}-{ordinal:08d}"


def offline_unit_id(payload: dict) -> str:
    return "offline-" + sha256(
        canonical_json_bytes({"protocol": "offline-unit-v1", **payload})
    ).hexdigest()


def publication_id(payload: dict) -> str:
    return "publication-" + sha256(
        canonical_json_bytes({"protocol": "generation-publication-v1", **payload})
    ).hexdigest()


def _valid_logical_id(value: str) -> bool:
    return (
        type(value) is str
        and value.startswith("llm-")
        and len(value) == 68
        and all(character in "0123456789abcdef" for character in value[4:])
    )
