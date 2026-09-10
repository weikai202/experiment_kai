"""Recursive, exact argument grounding against Agent-visible evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path

from pydantic import ConfigDict, Field

from toolsandbox_pipeline.reproducibility import canonical_json_bytes
from toolsandbox_pipeline.schemas.base import JsonObject, JsonValue, StrictModel
from toolsandbox_pipeline.schemas.state import CompactVerifiedState, VisibleRole


class _FrozenModel(StrictModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)


class GroundingSourceKind(str, Enum):
    VERIFIED_FACT = "verified_fact"
    USER_MESSAGE_SPAN = "user_message_span"
    SCHEMA = "schema"
    TRANSFORM = "transform"


class GroundingEvidence(_FrozenModel):
    argument_pointer: str
    source_kind: GroundingSourceKind
    source_ref: str = Field(min_length=1)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    fact_pointer: str | None = None
    transform_name: str | None = None
    transform_version: str | None = None
    result_value_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class GroundingResult(_FrozenModel):
    evidence: tuple[GroundingEvidence, ...]
    ungrounded_pointers: tuple[str, ...]


class TransformSpec(_FrozenModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


TransformRegistry = Mapping[tuple[str, str], Callable[[JsonValue], JsonValue]]


def load_grounding_transforms(path: Path | str | None = None) -> tuple[TransformSpec, ...]:
    """Load the reviewed production allowlist; code is never loaded from config."""
    config_path = Path(path) if path else Path(__file__).resolve().parents[3] / "configs/grounding_transforms.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "transforms"} or payload["schema_version"] != "1.0" or not isinstance(payload["transforms"], list):
        raise ValueError("invalid grounding transform registry")
    return tuple(TransformSpec.model_validate(item, strict=True) for item in payload["transforms"])


def strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(strict_json_equal(left[k], right[k]) for k in left)  # type: ignore[index]
    if isinstance(left, list):
        return len(left) == len(right) and all(strict_json_equal(a, b) for a, b in zip(left, right, strict=True))  # type: ignore[arg-type]
    return left == right


def _escape(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")


def iter_scalar_leaves(value: JsonValue, pointer: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_scalar_leaves(child, pointer + "/" + _escape(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_scalar_leaves(child, pointer + f"/{index}")
    else:
        yield pointer, value


def resolve_json_pointer(document: object, pointer: str) -> tuple[bool, object]:
    if pointer == "":
        return True, document
    if not pointer.startswith("/"):
        return False, None
    current = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, (list, tuple)) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return False, None
    return True, current


def _schema_at_pointer(parameters: JsonObject, pointer: str) -> JsonObject | None:
    schema: object = parameters
    if pointer:
        for raw in pointer[1:].split("/"):
            token = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(schema, dict) and schema.get("type") == "array":
                schema = schema.get("items")
            elif isinstance(schema, dict):
                props = schema.get("properties")
                schema = props.get(token) if isinstance(props, dict) else None
            else:
                schema = None
            if schema is None:
                return None
    return schema if isinstance(schema, dict) else None


def _value_hash(value: JsonValue) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def ground_arguments(
    arguments: JsonObject,
    augmented_parameters_schema: JsonObject,
    state: CompactVerifiedState,
    *,
    transforms: tuple[TransformSpec, ...] = (),
    transform_registry: TransformRegistry | None = None,
) -> GroundingResult:
    """Ground each scalar leaf without consulting callable signatures or hidden state."""

    registry = transform_registry or {}
    evidence: list[GroundingEvidence] = []
    missing: list[str] = []
    base_sources: list[tuple[GroundingSourceKind, str, JsonValue, dict[str, object]]] = []
    for fact_pointer, fact in state.verified_facts.items():
        base_sources.append((GroundingSourceKind.VERIFIED_FACT, fact.source_message_id, fact.value, {"fact_pointer": fact_pointer}))
    for message in state.visible_messages:
        if message.sender is VisibleRole.USER and message.recipient is VisibleRole.AGENT:
            base_sources.append((GroundingSourceKind.USER_MESSAGE_SPAN, message.message_id, message.content, {}))

    for pointer, value in iter_scalar_leaves(arguments):
        value_hash = _value_hash(value)
        matched: list[GroundingEvidence] = []
        for kind, ref, source_value, extras in base_sources:
            if strict_json_equal(value, source_value):
                matched.append(GroundingEvidence(argument_pointer=pointer, source_kind=kind, source_ref=ref, result_value_hash=value_hash, **extras))
            if kind is GroundingSourceKind.USER_MESSAGE_SPAN and isinstance(value, str):
                start = 0
                while True:
                    offset = source_value.find(value, start)  # type: ignore[union-attr]
                    if offset < 0:
                        break
                    matched.append(GroundingEvidence(argument_pointer=pointer, source_kind=kind, source_ref=ref, start_offset=offset, end_offset=offset + len(value), result_value_hash=value_hash))
                    start = offset + max(1, len(value))

        leaf_schema = _schema_at_pointer(augmented_parameters_schema, pointer)
        if leaf_schema is not None:
            for keyword in ("const", "default"):
                if keyword in leaf_schema and strict_json_equal(value, leaf_schema[keyword]):
                    matched.append(GroundingEvidence(argument_pointer=pointer, source_kind=GroundingSourceKind.SCHEMA, source_ref=f"{pointer or '/'}#{keyword}", result_value_hash=value_hash))
            enum = leaf_schema.get("enum")
            if isinstance(enum, list) and any(strict_json_equal(value, candidate) for candidate in enum):
                matched.append(GroundingEvidence(argument_pointer=pointer, source_kind=GroundingSourceKind.SCHEMA, source_ref=f"{pointer or '/'}#enum", result_value_hash=value_hash))

        for spec in transforms:
            fn = registry.get((spec.name, spec.version))
            if fn is None:
                raise ValueError(f"unregistered grounding transform: {spec.name}@{spec.version}")
            for source_kind, ref, source_value, extras in base_sources:
                transformed = fn(source_value)
                if strict_json_equal(value, transformed):
                    matched.append(GroundingEvidence(argument_pointer=pointer, source_kind=GroundingSourceKind.TRANSFORM, source_ref=ref, transform_name=spec.name, transform_version=spec.version, result_value_hash=value_hash, fact_pointer=extras.get("fact_pointer")))
        if matched:
            evidence.extend(matched)
        else:
            missing.append(pointer)
    return GroundingResult(evidence=tuple(evidence), ungrounded_pointers=tuple(missing))


__all__ = ["GroundingEvidence", "GroundingResult", "GroundingSourceKind", "TransformSpec", "ground_arguments", "iter_scalar_leaves", "load_grounding_transforms", "resolve_json_pointer", "strict_json_equal"]
