"""Deterministic seed Skills derived only from pinned public tool schemas."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from tool_sandbox.common.tool_conversion import convert_to_openai_tools

from toolsandbox_pipeline.online.tool_metadata import (
    TOOL_MODULES,
    UPSTREAM_COMMIT,
    installed_upstream_commit,
    load_tool_metadata,
    public_tool_inventory,
)
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.skill import (
    SkillApplicability,
    SkillCostProfile,
    SkillOnlineStatistics,
    SkillRecord,
    SkillRiskProfile,
    validate_skill_inventory,
)
from toolsandbox_pipeline.schemas.tool_metadata import ToolEffect, ToolRisk


GENERATOR_VERSION = "seed-skill-builder-v1"
EXCLUDED_TOOLS = ("end_conversation",)
_ALGORITHM = {
    "semantic_source": "intersection-visible-fixed-template-v1",
    "identity": "sha256(generator-version, canonical-public-schema)",
    "ordering": "skill-id-utf8",
    "canonical_identity_location": "tool_dependencies_only",
}


def _file_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _public_tools() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for module_name in TOOL_MODULES:
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if getattr(value, "is_tool", False) and getattr(value, "__module__", None) == module_name:
                if name in tools and tools[name] is not value:
                    raise ValueError("duplicate public tool identity")
                tools[name] = value
    ordered_names = public_tool_inventory()
    if tuple(sorted(tools)) != ordered_names:
        raise ValueError("public callable inventory mismatch")
    return {name: tools[name] for name in ordered_names}


def extract_public_tool_schemas() -> tuple[dict[str, Any], ...]:
    """Return the canonical pinned public OpenAI-tool schemas in name order."""

    if installed_upstream_commit() != UPSTREAM_COMMIT:
        raise RuntimeError("installed ToolSandbox commit mismatch")
    tools = _public_tools()
    schemas = tuple(convert_to_openai_tools(tools))
    if len(schemas) != len(tools):
        raise ValueError("public schema cardinality mismatch")
    names = tuple(schema.get("function", {}).get("name") for schema in schemas)
    if names != tuple(tools):
        raise ValueError("public schema name/order mismatch")
    for schema in schemas:
        canonical_json_bytes(schema)
    return schemas


def _seed_skill(schema: dict[str, Any], *, effect: ToolEffect, risk: ToolRisk) -> SkillRecord:
    function = schema["function"]
    canonical_name = function["name"]
    schema_hash = canonical_sha256(schema)
    skill_id = "skill_" + canonical_sha256([
        GENERATOR_VERSION, schema_hash,
    ]).removeprefix("sha256:")
    latency = "high" if effect is ToolEffect.EXTERNAL_READ else "low"
    risk_level = risk.value
    record = SkillRecord(
        skill_id=skill_id,
        name="Schema-guided single capability use",
        description="Use one visible capability only when its current public schema fits the requested operation.",
        applicability=SkillApplicability(
            required_state=(), forbidden_state=(),
            best_used_when=("One visible schema directly supports the next required operation.",),
        ),
        required_inputs=(),
        expected_outputs=("A schema-valid observable result for the requested operation.",),
        tool_dependencies=(canonical_name,),
        success_criteria=("The call satisfies the visible schema and its result is checked before continuing.",),
        failure_mode_buffer=(),
        cost_profile=SkillCostProfile(
            expected_tool_calls=1, latency=latency, token_cost="low",
        ),
        risk_profile=SkillRiskProfile(
            risk_if_skipped="medium", risk_if_wrong=risk_level,
        ),
        instruction=(
            "Check the visible required fields, submit one schema-valid call, "
            "and verify its visible result before choosing the next action."
        ),
        online_statistics=SkillOnlineStatistics(
            evaluated_uses=0, successes=0, failures=0, success_rate=0.0,
            last_update_attempt_at_use_count=0,
        ),
        validation=None,
        version="v1.0",
        status="active",
    )
    return record


def build_seed_skills(
    *,
    metadata_path: Path | str | None = None,
    metadata_manifest_path: Path | str | None = None,
) -> tuple[tuple[SkillRecord, ...], dict[str, Any]]:
    metadata = load_tool_metadata(metadata_path, metadata_manifest_path)
    schemas = extract_public_tool_schemas()
    inventory = tuple(record.canonical_tool_name for record in metadata)
    schema_by_name = {schema["function"]["name"]: schema for schema in schemas}
    if tuple(schema_by_name) != inventory:
        raise ValueError("metadata/public schema inventory mismatch")
    metadata_by_name = {record.canonical_tool_name: record for record in metadata}
    skills = tuple(sorted((
        _seed_skill(
            schema_by_name[name],
            effect=metadata_by_name[name].effect,
            risk=metadata_by_name[name].risk,
        )
        for name in inventory if name not in EXCLUDED_TOOLS
    ), key=lambda record: record.skill_id.encode("utf-8")))
    dependencies = tuple(record.tool_dependencies[0] for record in skills)
    expected = tuple(name for name in inventory if name not in EXCLUDED_TOOLS)
    if set(dependencies) != set(expected) or len(dependencies) != len(set(dependencies)):
        raise ValueError("seed Skill coverage mismatch")
    for skill in skills:
        validate_skill_inventory(skill, inventory)
    schema_projection = tuple(
        {"canonical_tool_name": schema["function"]["name"], "schema": schema}
        for schema in schemas
    )
    provenance = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "public_inventory_sha256": canonical_sha256(list(inventory)),
        "public_schema_inventory_sha256": canonical_sha256(list(schema_projection)),
        "algorithm_sha256": canonical_sha256(_ALGORITHM),
        "excluded_tool_names": list(EXCLUDED_TOOLS),
        "skill_count": len(skills),
        "skill_ids_sha256": canonical_sha256([record.skill_id for record in skills]),
        "source_types": ["toolsandbox_public_tool_schema"],
        "dev_test_or_scenario_artifacts_used": False,
    }
    return skills, provenance


def seed_skill_bytes(skills: tuple[SkillRecord, ...]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        for record in skills
    )


def publish_seed_skill_library(
    output_directory: Path | str,
    *,
    metadata_path: Path | str | None = None,
    metadata_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("output directory must be a new absolute path")
    parent = output.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise ValueError("output parent must be an existing non-symlink directory")
    skills, provenance = build_seed_skills(
        metadata_path=metadata_path,
        metadata_manifest_path=metadata_manifest_path,
    )
    library = seed_skill_bytes(skills)
    report = {
        **provenance,
        "seed_skills_sha256": _file_hash(library),
        "seed_skills_record_count": len(skills),
    }
    report_bytes = canonical_json_bytes(report)
    staging = parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(mode=0o700)
    try:
        for name, payload in (
            ("seed_skills.jsonl", library),
            ("seed_skill_provenance.json", report_bytes),
        ):
            fd = os.open(staging / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
    return report


__all__ = [
    "GENERATOR_VERSION", "build_seed_skills", "extract_public_tool_schemas",
    "publish_seed_skill_library", "seed_skill_bytes",
]
