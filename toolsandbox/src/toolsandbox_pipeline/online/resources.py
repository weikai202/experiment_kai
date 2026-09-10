"""Deterministic resource-template resolution and independence checks."""

from __future__ import annotations

import json
import math
import re

from toolsandbox_pipeline.schemas.action import FunctionCall
from toolsandbox_pipeline.schemas.tool_metadata import ControllerToolMetadata, ResourceTemplate, ToolEffect

from .grounding import resolve_json_pointer


_PLACEHOLDER = re.compile(r"\{arg:(/[^{}]*)\}")


def validate_resource_template(template: str) -> None:
    stripped = _PLACEHOLDER.sub("", template)
    if "{" in stripped or "}" in stripped:
        raise ValueError(f"invalid resource template: {template}")


def resolve_resource_template(template: ResourceTemplate | str, arguments: dict) -> str | None:
    raw = template.template if isinstance(template, ResourceTemplate) else template
    validate_resource_template(raw)
    unresolved = False

    def replace(match: re.Match[str]) -> str:
        nonlocal unresolved
        found, value = resolve_json_pointer(arguments, match.group(1))
        if not found or isinstance(value, (dict, list)) or (isinstance(value, float) and not math.isfinite(value)):
            unresolved = True
            return ""
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    resolved = _PLACEHOLDER.sub(replace, raw)
    return None if unresolved else resolved


def calls_are_independent(calls: tuple[FunctionCall, ...], metadata: tuple[ControllerToolMetadata, ...]) -> bool:
    if len(calls) != len(metadata) or any(not item.parallel_safe or item.effect in (ToolEffect.CONVERSATION_CONTROL, ToolEffect.EXTERNAL_WRITE) for item in metadata):
        return False
    resources: list[tuple[set[str], set[str]]] = []
    for call, item in zip(calls, metadata, strict=True):
        reads = [resolve_resource_template(t, call.arguments) for t in item.read_resources]
        writes = [resolve_resource_template(t, call.arguments) for t in item.write_resources]
        if any(value is None for value in reads + writes):
            return False
        resources.append((set(reads), set(writes)))  # type: ignore[arg-type]
    for index, (reads, writes) in enumerate(resources):
        for other_reads, other_writes in resources[index + 1:]:
            if writes & (other_reads | other_writes) or other_writes & reads:
                return False
    return True


__all__ = ["calls_are_independent", "resolve_resource_template", "validate_resource_template"]
