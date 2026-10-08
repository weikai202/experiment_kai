"""Deterministic G000 Skill compilation from pinned public agent tool schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from . import DOMAINS, TAU_COMMIT
from .canonical import canonical_sha256, content_hash, verify_content_hash


@dataclass(frozen=True)
class SeedSkillSpec:
    skill_id: str
    version: int
    domains: tuple[str, ...]
    tool_dependencies: tuple[str, ...]
    content: str
    source_schema_sha256: str


def _validate_source_path(path: str, domain: str) -> None:
    expected = f"src/tau2/domains/{domain}/tools.py"
    if path != expected:
        raise ValueError(f"{domain} schemas must come only from {expected}")
    lowered = path.lower()
    if any(value in lowered for value in ("user_tool", "/tasks/", "evaluator", "reference")):
        raise ValueError("private task, evaluator, reference, and user-tool sources are forbidden")


def _validate_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if type(schema) is not dict or set(schema) != {"type", "function"}:
        raise ValueError("seed input must be an exact OpenAI function schema")
    function = schema["function"]
    if schema["type"] != "function" or type(function) is not dict:
        raise ValueError("invalid public function schema")
    if set(function) != {"name", "description", "parameters"}:
        raise ValueError("function schema has non-public or unsupported fields")
    name = function["name"]
    description = function["description"]
    parameters = function["parameters"]
    if type(name) is not str or not name or type(description) is not str:
        raise ValueError("tool name and description must be strings")
    if type(parameters) is not dict or parameters.get("type") != "object":
        raise ValueError("tool parameters must be a JSON object schema")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def compile_seed_library(
    domain_sources: dict[str, dict[str, Any]], *, source_commit: str
) -> dict[str, Any]:
    """Compile schema-only public tool metadata; no task or private state is accepted."""

    if source_commit != TAU_COMMIT:
        raise ValueError("seed library requires the pinned tau source commit")
    if tuple(domain_sources) != DOMAINS:
        raise ValueError("seed sources must be ordered airline, retail, telecom")
    compiled_domains: dict[str, Any] = {}
    skills: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for domain in DOMAINS:
        source = domain_sources[domain]
        if set(source) != {"source_path", "source_sha256", "schemas"}:
            raise ValueError("domain source requires only path, source hash, and schemas")
        _validate_source_path(source["source_path"], domain)
        if not str(source["source_sha256"]).startswith("sha256:"):
            raise ValueError("domain source requires a raw-file content hash")
        schemas = tuple(_validate_schema(value) for value in source["schemas"])
        schemas = tuple(sorted(schemas, key=lambda row: row["function"]["name"].encode("utf-8")))
        names = [row["function"]["name"] for row in schemas]
        if not names or len(names) != len(set(names)):
            raise ValueError("each domain requires unique public agent tools")
        schema_rows = []
        for schema in schemas:
            function = schema["function"]
            schema_sha = canonical_sha256(schema)
            skill_id = f"{domain}.{function['name']}"
            if skill_id in seen_ids:
                raise ValueError("duplicate seed Skill identity")
            seen_ids.add(skill_id)
            parameters = json.dumps(
                function["parameters"], ensure_ascii=True, sort_keys=True, separators=(",", ":")
            )
            content = (
                f"Public {domain} tool `{function['name']}`. "
                f"{function['description'].strip()} Parameters JSON Schema: {parameters}"
            )
            skill = SeedSkillSpec(
                skill_id,
                0,
                (domain,),
                (function["name"],),
                content,
                schema_sha,
            )
            skill_row = asdict(skill)
            skill_row["domains"] = list(skill_row["domains"])
            skill_row["tool_dependencies"] = list(skill_row["tool_dependencies"])
            skills.append(skill_row)
            schema_rows.append({"schema": schema, "schema_sha256": schema_sha})
        compiled_domains[domain] = {
            "source_path": source["source_path"],
            "source_sha256": source["source_sha256"],
            "schema_bundle_sha256": canonical_sha256(schema_rows),
            "tools": schema_rows,
        }
    document = {
        "schema_version": 1,
        "kind": "tau3_public_agent_tool_seed_library",
        "source_commit": source_commit,
        "provenance_policy": "public-agent-tools-schema-only-no-tasks-private-state-or-evaluator",
        "domains": compiled_domains,
        "skills": sorted(skills, key=lambda row: row["skill_id"].encode("utf-8")),
    }
    document["library_sha256"] = content_hash(document)
    verify_content_hash(document, field="library_sha256")
    return document


def verify_seed_library(document: dict[str, Any]) -> None:
    verify_content_hash(document, field="library_sha256")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "tau3_public_agent_tool_seed_library"
        or document.get("source_commit") != TAU_COMMIT
        or tuple(document.get("domains", {})) != DOMAINS
    ):
        raise ValueError("invalid seed library identity or source pin")
    rebuilt = compile_seed_library(
        {
            domain: {
                "source_path": document["domains"][domain]["source_path"],
                "source_sha256": document["domains"][domain]["source_sha256"],
                "schemas": [row["schema"] for row in document["domains"][domain]["tools"]],
            }
            for domain in DOMAINS
        },
        source_commit=document["source_commit"],
    )
    if rebuilt != document:
        raise ValueError("seed library content is not the deterministic schema compilation")


def publish_g000(store, document: dict[str, Any], embedding_provider, *, scope_id: str):
    """Embed and publish the verified schema-derived G000 state with an injected provider."""

    verify_seed_library(document)
    from .generation_store import GenerationState, SkillRecord

    skills = []
    vectors = []
    ledger_events = []
    for row in document["skills"]:
        skill = SkillRecord(
            row["skill_id"],
            row["version"],
            tuple(row["domains"]),
            tuple(row["tool_dependencies"]),
            row["content"],
            None,
            None,
        )
        vector, events = store.embed_record(
            record_kind="skill",
            record_id=skill.skill_id,
            version=skill.version,
            content=skill.content,
            provider=embedding_provider,
            scope_id=scope_id,
        )
        skills.append(skill)
        vectors.append(vector)
        ledger_events.extend(events)
    state = GenerationState(
        "g000",
        None,
        (),
        tuple(skills),
        tuple(vectors),
        document["library_sha256"],
    )
    return state, store.publish(state), tuple(ledger_events)
