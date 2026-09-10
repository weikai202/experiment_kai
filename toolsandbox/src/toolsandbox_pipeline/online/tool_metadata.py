"""Manifest-verified loading of committed Controller metadata."""

from __future__ import annotations

import hashlib
import importlib
import json
from importlib.metadata import distribution
from pathlib import Path

from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.tool_metadata import ControllerToolMetadata, ToolEffect, ToolMetadataManifest, ToolRisk
from toolsandbox_pipeline.online.resources import validate_resource_template

UPSTREAM_COMMIT = "165848b9a78cead7ca7fe7c89c688b58e6501219"
EXPECTED_EFFECT_COUNTS = {
    ToolEffect.SANDBOX_READ: 17, ToolEffect.SANDBOX_WRITE: 11,
    ToolEffect.EXTERNAL_READ: 5, ToolEffect.EXTERNAL_WRITE: 0,
    ToolEffect.CONVERSATION_CONTROL: 1,
}
TOOL_MODULES = (
    "tool_sandbox.tools.contact", "tool_sandbox.tools.messaging",
    "tool_sandbox.tools.rapid_api_search_tools", "tool_sandbox.tools.reminder",
    "tool_sandbox.tools.setting", "tool_sandbox.tools.user_tools",
    "tool_sandbox.tools.utilities",
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def public_tool_inventory() -> tuple[str, ...]:
    names: set[str] = set()
    for module_name in TOOL_MODULES:
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if getattr(value, "is_tool", False) and getattr(value, "__module__", None) == module_name:
                names.add(name)
    return tuple(sorted(names))


def installed_upstream_commit() -> str:
    payload = distribution("tool-sandbox").read_text("direct_url.json")
    if payload is None:
        raise RuntimeError("tool-sandbox direct_url.json is missing")
    data = json.loads(payload)
    try:
        return data["vcs_info"]["commit_id"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("tool-sandbox installation lacks pinned VCS identity") from exc


def canonical_metadata_bytes(records: tuple[ControllerToolMetadata, ...]) -> bytes:
    return b"".join(canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records)


def load_tool_metadata(metadata_path: Path | str | None = None, manifest_path: Path | str | None = None) -> tuple[ControllerToolMetadata, ...]:
    metadata_file = Path(metadata_path) if metadata_path else PROJECT_ROOT / "configs/tool_metadata/controller_tool_metadata.jsonl"
    manifest_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "configs/tool_metadata/manifest.json"
    manifest = ToolMetadataManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"), strict=True)
    if manifest.upstream_commit != UPSTREAM_COMMIT or installed_upstream_commit() != UPSTREAM_COMMIT:
        raise RuntimeError("installed ToolSandbox commit does not match metadata manifest")
    lines = metadata_file.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("metadata JSONL must contain only non-empty records")
    records = tuple(ControllerToolMetadata.model_validate_json(line, strict=True) for line in lines)
    for record in records:
        for template in record.read_resources + record.write_resources:
            validate_resource_template(template.template)
    names = tuple(record.canonical_tool_name for record in records)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("metadata records must have unique lexicographically sorted names")
    inventory = public_tool_inventory()
    if names != inventory:
        raise ValueError("metadata names do not match pinned public tool inventory")
    if manifest.record_count != len(records):
        raise ValueError("metadata record count mismatch")
    counts = {effect: sum(record.effect is effect for record in records) for effect in ToolEffect}
    if counts != EXPECTED_EFFECT_COUNTS or manifest.effect_counts != EXPECTED_EFFECT_COUNTS:
        raise ValueError("metadata effect counts mismatch")
    metadata_hash = "sha256:" + hashlib.sha256(canonical_metadata_bytes(records)).hexdigest()
    if metadata_hash != manifest.metadata_sha256:
        raise ValueError("metadata content hash mismatch")
    if canonical_sha256(list(inventory)) != manifest.public_inventory_sha256:
        raise ValueError("public inventory hash mismatch")
    baseline = {ToolEffect.SANDBOX_READ: ToolRisk.LOW, ToolEffect.SANDBOX_WRITE: ToolRisk.MEDIUM, ToolEffect.EXTERNAL_READ: ToolRisk.MEDIUM, ToolEffect.EXTERNAL_WRITE: ToolRisk.HIGH, ToolEffect.CONVERSATION_CONTROL: ToolRisk.HIGH}
    if any(record.risk is not baseline[record.effect] for record in records):
        raise ValueError("metadata risk differs from the manifest baseline without a rationale")
    return records


__all__ = ["EXPECTED_EFFECT_COUNTS", "UPSTREAM_COMMIT", "canonical_metadata_bytes", "installed_upstream_commit", "load_tool_metadata", "public_tool_inventory"]
