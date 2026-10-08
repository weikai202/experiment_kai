from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_hashed_json(path: Path, payload: dict[str, Any]) -> str:
    digest = sha256_json(payload)
    envelope = {"payload": payload, "payload_sha256": digest}
    path.write_bytes(canonical_bytes(envelope) + b"\n")
    return digest


def read_hashed_json(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError("Invalid hashed JSON envelope")
    if sha256_json(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("Manifest hash mismatch")
    return envelope["payload"]
