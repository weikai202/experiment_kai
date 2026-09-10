"""Strict reproducibility identities, private manifests, and sanitized audits."""
from typing import Annotated, Literal
from pydantic import Field, field_validator, model_validator
from .memory import Count, Digest, FrozenRecord, Identifier, unique

REGISTRIES = ("single_tool_call", "multiple_tool_call", "multiple_user_turn", "insufficient_information")
VARIANTS = ("no_distraction", "three_distraction_tools", "ten_distraction_tools", "all_tools",
            "three_distraction_tool_description_scrambled", "three_distraction_argument_type_scrambled",
            "three_distraction_argument_description_scrambled", "three_distraction_tool_name_scrambled")
SUFFIXES = ("", "_3_distraction_tools", "_10_distraction_tools", "_all_tools",
            "_3_distraction_tools_tool_description_scrambled", "_3_distraction_tools_arg_type_scrambled",
            "_3_distraction_tools_arg_description_scrambled", "_3_distraction_tools_tool_name_scrambled")
REQUIRED_CATEGORIES = ("NO_DISTRACTION_TOOLS", "THREE_DISTRACTION_TOOLS", "TEN_DISTRACTION_TOOLS", "ALL_TOOLS_AVAILABLE",
                       "TOOL_DESCRIPTION_SCRAMBLED", "ARG_TYPE_SCRAMBLED", "ARG_DESCRIPTION_SCRAMBLED", "TOOL_NAME_SCRAMBLED")
Split = Literal["train", "dev", "test"]
Registry = Literal["single_tool_call", "multiple_tool_call", "multiple_user_turn", "insufficient_information"]
Purpose = Literal["development", "retrieval_smoke", "online_token_calibration", "train_round", "skill_ab_validation"]


class DatasetBuildConfig(FrozenRecord):
    schema_version: Literal[1]
    upstream_commit: Literal["165848b9a78cead7ca7fe7c89c688b58e6501219"]
    preferred_tool_backend: Literal["DEFAULT"]
    data_seed: Literal[0]
    test_fraction: Literal[0.2]
    dev_fraction: Literal[0.2]
    num_update_rounds: Literal[3]
    world_epoch_iso: Literal["2024-05-01T12:00:00Z"]
    world_epoch_unix: Literal[1714564800.0]
    timezone: Literal["UTC"]
    locale: Literal["C.UTF-8"]
    clock_adapter_version: Literal["toolsandbox-fixed-clock-v1"]
    family_registry_order: tuple[str, ...]
    variant_order: tuple[str, ...]

    @field_validator("schema_version", "data_seed", "num_update_rounds", "test_fraction", "dev_fraction", "world_epoch_unix", mode="before")
    @classmethod
    def exact_types(cls, value, info):
        expected = int if info.field_name in ("schema_version", "data_seed", "num_update_rounds") else float
        if type(value) is not expected:
            raise ValueError("build constants require exact numeric types")
        return value

    @model_validator(mode="after")
    def exact_orders(self):
        if self.family_registry_order != REGISTRIES or self.variant_order != VARIANTS:
            raise ValueError("registry/variant order mismatch")
        return self


class DatasetEnvironment(FrozenRecord):
    upstream_commit: Literal["165848b9a78cead7ca7fe7c89c688b58e6501219"]
    upstream_source_sha256: Digest
    dependency_lock_sha256: Digest
    python_patch_version: Annotated[str, Field(pattern=r"^3\.10\.[0-9]+$")]
    platform: Identifier
    timezone: Literal["UTC"]
    locale: Literal["C.UTF-8"]
    build_config_sha256: Digest
    clock_adapter_version: Literal["toolsandbox-fixed-clock-v1"]
    preferred_tool_backend: Literal["DEFAULT"]
    container_image_digest: Digest | None


class FamilyRecord(FrozenRecord):
    family_id: Identifier
    source_registry: Registry
    split: Split
    train_shard: Annotated[int, Field(ge=0, le=2)] | None
    ordered_variant_scenario_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def invariants(self):
        if (self.split == "train") != (self.train_shard is not None):
            raise ValueError("only train has a shard")
        if self.ordered_variant_scenario_ids != tuple(self.family_id + suffix for suffix in SUFFIXES):
            raise ValueError("variants must be forward-generated from family ID")
        return self


class ScenarioRecord(FrozenRecord):
    scenario_id: Identifier
    scenario_family_id: Identifier
    variant: str
    categories: tuple[Identifier, ...]
    max_messages: Annotated[int, Field(gt=0)]
    starting_context_sha256: Digest
    evaluation_definition_sha256: Digest
    agent_facing_tool_names_sha256: Digest
    agent_facing_tool_schema_sha256: Digest

    @model_validator(mode="after")
    def variant_valid(self):
        if self.variant not in VARIANTS or self.scenario_id != self.scenario_family_id + SUFFIXES[VARIANTS.index(self.variant)]:
            raise ValueError("scenario/variant identity mismatch")
        unique(self.categories)
        if REQUIRED_CATEGORIES[VARIANTS.index(self.variant)] not in self.categories:
            raise ValueError("missing augmentation category")
        index = VARIANTS.index(self.variant)
        expected = {REQUIRED_CATEGORIES[index]}
        if index >= 4:
            expected.add("THREE_DISTRACTION_TOOLS")
        if set(self.categories) & set(REQUIRED_CATEGORIES) != expected:
            raise ValueError("unexpected augmentation categories")
        return self


class SplitManifest(FrozenRecord):
    schema_version: Literal[1]
    split: Split
    build_config_sha256: Digest
    environment_sha256: Digest
    families: tuple[FamilyRecord, ...]
    scenarios: tuple[ScenarioRecord, ...]

    @model_validator(mode="after")
    def invariants(self):
        count = 79 if self.split == "train" else 25
        if len(self.families) != count or len(self.scenarios) != count * 8:
            raise ValueError("split counts mismatch")
        unique(tuple(f.family_id for f in self.families))
        expected = tuple(sid for family in self.families for sid in family.ordered_variant_scenario_ids)
        if any(f.split != self.split for f in self.families) or tuple(s.scenario_id for s in self.scenarios) != expected:
            raise ValueError("split or scenario order mismatch")
        for family, offset in zip(self.families, range(0, len(self.scenarios), 8)):
            if any(s.scenario_family_id != family.family_id for s in self.scenarios[offset:offset + 8]):
                raise ValueError("cross-family scenario")
        if self.split == "train" and tuple(f.train_shard for f in self.families) != (0,) * 27 + (1,) * 26 + (2,) * 26:
            raise ValueError("train shard order mismatch")
        return self


class DatasetIndex(FrozenRecord):
    schema_version: Literal[1]
    config_path: Identifier
    config_sha256: Digest
    environment: DatasetEnvironment
    environment_sha256: Digest
    status: Literal["setup_only", "complete"]
    family_count: Literal[129]
    scenario_count: Literal[1032]
    train_manifest_sha256: Digest
    dev_manifest_sha256: Digest
    test_manifest_sha256: Digest

    @model_validator(mode="after")
    def identity(self):
        from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
        if canonical_sha256(self.environment.model_dump(mode="json")) != self.environment_sha256 or self.config_sha256 != self.environment.build_config_sha256:
            raise ValueError("environment identity mismatch")
        if (self.status == "complete") != (self.environment.container_image_digest is not None):
            raise ValueError("unresolved environment must be setup_only")
        return self


class DatasetAccessRequest(FrozenRecord):
    split: Literal["train", "dev"] = "train"
    purpose: Purpose = "development"
    manifest_sha256: Digest
    requested_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    run_id: Identifier
    phase: Identifier
    train_shard: Annotated[int, Field(ge=0, le=2)] | None = None

    @model_validator(mode="after")
    def permitted(self):
        unique(self.requested_ids)
        if (self.split == "dev") != (self.purpose == "skill_ab_validation"):
            raise ValueError("split/purpose mismatch")
        if (self.purpose == "train_round") != (self.train_shard is not None):
            raise ValueError("train_round requires exact shard")
        return self
