"""Canonical JSON serialization and non-self-referential state identity."""

from __future__ import annotations

import hashlib
import hmac
import json

from pydantic import TypeAdapter

from toolsandbox_pipeline.schemas.base import JsonObject, JsonValue


_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


def canonical_json_bytes(payload: JsonValue) -> bytes:
    """Return the one project-wide canonical representation of a JSON value."""

    validated = _JSON_VALUE_ADAPTER.validate_python(payload, strict=True)
    return json.dumps(
        validated,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: JsonValue) -> str:
    """Hash a JSON value's canonical UTF-8 bytes."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_state_id(payload_without_state_id: JsonObject) -> str:
    """Compute a state ID, rejecting an already self-identifying payload."""

    if not isinstance(payload_without_state_id, dict):
        raise TypeError("state payload must be a JSON object")
    if "state_id" in payload_without_state_id:
        raise ValueError("state payload to hash must not contain state_id")
    return canonical_sha256(payload_without_state_id)


def verify_state_id(full_state_payload: JsonObject) -> bool:
    """Verify a top-level state ID without mutating the supplied payload."""

    if not isinstance(full_state_payload, dict):
        raise TypeError("state payload must be a JSON object")
    stored_state_id = full_state_payload.get("state_id")
    if not isinstance(stored_state_id, str):
        raise ValueError("state_id must be present and must be a string")
    payload_without_state_id = dict(full_state_payload)
    del payload_without_state_id["state_id"]
    expected_state_id = compute_state_id(payload_without_state_id)
    return hmac.compare_digest(stored_state_id, expected_state_id)
