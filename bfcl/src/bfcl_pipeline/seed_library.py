from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from .canonical import canonical_bytes, sha256_json
from .dataset import BFCL_VARIANTS
from .evolution import Generation
from .resources import GenerationResource, GenerationResourceManifest

BFCL_PINNED_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
PUBLIC_SCHEMA_FIELDS = frozenset({"name", "description", "parameters"})


@dataclass(frozen=True)
class SchemaProvenance:
    variant: str
    case_id: str
    exposure: str
    turn_index: int | None
    function_name: str
    schema_sha256: str


@dataclass(frozen=True)
class SeedSkillLibrary:
    protocol: str
    source_revision: str
    schema_provenance: tuple[SchemaProvenance, ...]
    resources: tuple[GenerationResource, ...]
    resource_manifest: GenerationResourceManifest
    library_sha256: str

    def validate(self) -> None:
        if self.protocol != "bfcl_public_schema_seed_library_v1" or self.source_revision != BFCL_PINNED_REVISION:
            raise ValueError("Seed Skill library source identity is invalid")
        for resource in self.resources:
            resource.validate()
            if resource.generation_id != "g000" or resource.kind != "skill":
                raise ValueError("Seed library contains a non-G000 Skill resource")
        self.resource_manifest.validate()
        if self.resource_manifest != GenerationResourceManifest.build("g000", self.resources):
            raise ValueError("Seed library resource manifest is invalid")
        core = {
            "protocol": self.protocol,
            "source_revision": self.source_revision,
            "schema_provenance": [asdict(row) for row in self.schema_provenance],
            "resource_sha256": tuple(resource.resource_sha256 for resource in self.resources),
            "resource_manifest_sha256": self.resource_manifest.manifest_sha256,
        }
        if self.library_sha256 != sha256_json(core):
            raise ValueError("Seed Skill library hash is invalid")


def _public_schema(schema: Mapping) -> dict:
    if not isinstance(schema, Mapping) or not isinstance(schema.get("name"), str):
        raise ValueError("BFCL public function schema requires a name")
    public = {key: schema[key] for key in ("name", "description", "parameters") if key in schema}
    if not isinstance(public.get("parameters", {}), Mapping):
        raise ValueError("BFCL public parameters schema is invalid")
    # Round-trip canonical JSON to reject non-JSON/private runtime objects.
    return json.loads(canonical_bytes(public).decode("utf-8"))


def compile_seed_skill_library(
    records_by_variant: Mapping[str, Iterable[Mapping]],
    source_revision: str,
) -> SeedSkillLibrary:
    if source_revision != BFCL_PINNED_REVISION:
        raise ValueError("Seed compilation requires the pinned BFCL revision")
    if set(records_by_variant) != set(BFCL_VARIANTS):
        raise ValueError("Seed compilation requires all four BFCL variants")
    provenance: list[SchemaProvenance] = []
    unique_schemas: dict[tuple[str, str], dict] = {}
    for variant in BFCL_VARIANTS:
        rows = sorted(tuple(records_by_variant[variant]), key=lambda row: str(row.get("id", "")).encode("utf-8"))
        for record in rows:
            case_id = record.get("id")
            if not isinstance(case_id, str) or not case_id.startswith(f"multi_turn_{variant}_"):
                raise ValueError("Seed compiler received an invalid BFCL case ID")
            for schema in record.get("function", ()):
                public = _public_schema(schema)
                schema_hash = sha256_json(public)
                unique_schemas[(public["name"], schema_hash)] = public
                provenance.append(SchemaProvenance(variant, case_id, "initial", None, public["name"], schema_hash))
            delayed = record.get("missed_function", {})
            if not isinstance(delayed, Mapping):
                raise ValueError("missed_function must be a public turn map")
            for turn in sorted(delayed, key=lambda value: int(value)):
                for schema in delayed[turn]:
                    public = _public_schema(schema)
                    schema_hash = sha256_json(public)
                    unique_schemas[(public["name"], schema_hash)] = public
                    provenance.append(SchemaProvenance(variant, case_id, "delayed", int(turn), public["name"], schema_hash))
    provenance_rows = tuple(sorted(
        provenance,
        key=lambda row: (
            BFCL_VARIANTS.index(row.variant),
            row.case_id.encode("utf-8"),
            row.exposure,
            -1 if row.turn_index is None else row.turn_index,
            row.function_name.encode("utf-8"),
            row.schema_sha256,
        ),
    ))
    resources = tuple(
        GenerationResource.build(
            resource_id=f"bfcl-schema:{name}:{schema_hash[:16]}",
            generation_id="g000",
            kind="skill",
            content=canonical_bytes(schema).decode("utf-8"),
            dependencies=(name,),
            creation_effect_id=sha256_json({
                "source_revision": source_revision,
                "function_name": name,
                "schema_sha256": schema_hash,
            }),
        )
        for (name, schema_hash), schema in sorted(unique_schemas.items(), key=lambda item: (item[0][0].encode("utf-8"), item[0][1]))
    )
    if not resources:
        raise ValueError("Seed compiler found no public BFCL function schemas")
    manifest = GenerationResourceManifest.build("g000", resources)
    core = {
        "protocol": "bfcl_public_schema_seed_library_v1",
        "source_revision": source_revision,
        "schema_provenance": [asdict(row) for row in provenance_rows],
        "resource_sha256": tuple(resource.resource_sha256 for resource in resources),
        "resource_manifest_sha256": manifest.manifest_sha256,
    }
    library = SeedSkillLibrary(
        core["protocol"],
        source_revision,
        provenance_rows,
        resources,
        manifest,
        sha256_json(core),
    )
    library.validate()
    return library


def generation_from_seed_library(memory_sha256: str, library: SeedSkillLibrary) -> Generation:
    library.validate()
    core = {
        "generation_id": "g000",
        "parent_generation_id": None,
        "memory_sha256": memory_sha256,
        "skill_library_sha256": library.resource_manifest.manifest_sha256,
        "accepted_skill_versions": tuple(resource.resource_id for resource in library.resources),
    }
    return Generation(**core, generation_sha256=sha256_json(core))


def seed_library_to_payload(library: SeedSkillLibrary) -> dict:
    library.validate()
    return asdict(library)


def seed_library_from_payload(payload: Mapping) -> SeedSkillLibrary:
    provenance = tuple(SchemaProvenance(**row) for row in payload["schema_provenance"])
    resources = tuple(
        GenerationResource(
            resource_id=row["resource_id"],
            generation_id=row["generation_id"],
            kind=row["kind"],
            content=row["content"],
            canonical_tool_dependencies=tuple(row["canonical_tool_dependencies"]),
            creation_effect_id=row["creation_effect_id"],
            resource_sha256=row["resource_sha256"],
        )
        for row in payload["resources"]
    )
    manifest_row = payload["resource_manifest"]
    manifest = GenerationResourceManifest(
        manifest_row["generation_id"],
        tuple(tuple(row) for row in manifest_row["skill_records"]),
        manifest_row["manifest_sha256"],
    )
    library = SeedSkillLibrary(payload["protocol"], payload["source_revision"], provenance, resources, manifest, payload["library_sha256"])
    library.validate()
    return library
