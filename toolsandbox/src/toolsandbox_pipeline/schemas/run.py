"""Strict train-run and query-only checkpoint observation contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator, model_serializer

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.checkpoint import CheckpointConfig


Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
GenerationId = Literal["g000", "g001", "g002", "g003"]
CostUnit = Literal["qwen_effective_output_tokens"]
NONCOMPARABLE = "different_train_shards_not_directly_comparable"
UPSTREAM_COMMIT = "165848b9a78cead7ca7fe7c89c688b58e6501219"


class FrozenRunRecord(StrictModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, allow_inf_nan=False,
        protected_namespaces=(),
    )


class QwenRunIdentity(FrozenRunRecord):
    model: Literal["Qwen/Qwen3-32B"] = "Qwen/Qwen3-32B"
    endpoint_identity_sha256: Digest
    container_digest: Digest | None
    registry_declared_digest: Digest | None = None
    image_verification: Literal["verified", "unverified"] = "verified"

    @model_validator(mode="after")
    def honest_image(self):
        if self.image_verification == "unverified":
            if self.container_digest is not None or self.registry_declared_digest is None:
                raise ValueError("unverified image requires declared digest and no actual digest")
        elif self.container_digest is None:
            raise ValueError("verified image requires actual digest")
        return self

    @model_serializer(mode="wrap")
    def omit_legacy_image_defaults(self, handler):
        value = handler(self)
        if self.image_verification == "verified" and self.registry_declared_digest is None:
            value.pop("registry_declared_digest", None)
            value.pop("image_verification", None)
        return value
    server_configuration_sha256: Digest
    decoding_configuration_sha256: Digest
    structured_output_wire_mode: Literal["guided_json", "structured_outputs_json"]
    enable_thinking: Literal[False] = False


class EmbeddingRunIdentity(FrozenRunRecord):
    model: Literal["text-embedding-3-small"] = "text-embedding-3-small"
    client_configuration_sha256: Digest
    expected_dimension: Annotated[int, Field(gt=0)]


class UserSimulatorRunIdentity(FrozenRunRecord):
    model: Literal["gpt-4o-mini-2024-07-18"] = "gpt-4o-mini-2024-07-18"
    profile_sha256: Digest
    prompt_sha256: Digest
    few_shot_sha256: Digest
    tool_schema_sha256: Digest
    stop_configuration_sha256: Digest


class TrainSmokeSelection(FrozenRunRecord):
    family_ordinal: Literal[0] = 0
    variants: tuple[
        Literal["no_distraction"], Literal["three_distraction_tools"]
    ]
    reselection_from_outcomes_forbidden: Literal[True] = True

    @model_validator(mode="after")
    def exact_variants(self) -> "TrainSmokeSelection":
        if self.variants != ("no_distraction", "three_distraction_tools"):
            raise ValueError("train smoke variants are immutable")
        return self


class TrainSmokeConfig(FrozenRunRecord):
    schema_version: Literal[1] = 1
    config_id: Literal["toolsandbox-train-smoke-v1"]
    formal: Literal[False]
    split: Literal["train"]
    selection: TrainSmokeSelection
    profile: Literal["strict_replay"]
    external_read_mode: Literal["fixture"]
    qwen_model: Literal["Qwen/Qwen3-32B"]
    qwen_enable_thinking: Literal[False]
    embedding_model: Literal["text-embedding-3-small"]
    user_simulator_model: Literal["gpt-4o-mini-2024-07-18"]
    artifact_namespace: Literal["runs/smoke"]
    publish_formal_generation: Literal[False]
    length_finish_reason_policy: Literal["calibration_required"]


class SeedArtifactIdentity(FrozenRunRecord):
    policy_memory_path: Annotated[str, Field(min_length=1)]
    policy_memory_sha256: Digest
    world_memory_path: Annotated[str, Field(min_length=1)]
    world_memory_sha256: Digest
    skills_path: Annotated[str, Field(min_length=1)]
    skills_sha256: Digest
    source_manifest_path: Annotated[str, Field(min_length=1)]
    source_manifest_sha256: Digest
    public_tool_schema_inventory_sha256: Digest
    skill_generator_version: Annotated[str, Field(min_length=1)]
    skill_generator_sha256: Digest
    skill_provenance_sha256: Digest

    @field_validator(
        "policy_memory_path", "world_memory_path", "skills_path",
        "source_manifest_path",
    )
    @classmethod
    def absolute_non_symlink_file(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or path.is_symlink():
            raise ValueError("seed paths must be absolute non-symlinks")
        return value


class FixtureRunIdentity(FrozenRunRecord):
    mode: Literal["fixture", "live", "local"]
    manifest_sha256: Digest
    backend_configuration_sha256: Digest


class ResolvedRunManifest(FrozenRunRecord):
    """One non-secret immutable root identity for a smoke or formal train run."""

    schema_version: Literal[1] = 1
    protocol_version: Literal["toolsandbox-evolution-v1", "toolsandbox-evolution-live-v2"] = "toolsandbox-evolution-v1"
    run_id: Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    purpose: Literal["train_smoke", "formal_training"]
    profile: Literal["strict_replay", "official_live"]
    data_seed: Literal[0] = 0
    dataset_manifest_path: Annotated[str, Field(min_length=1)]
    dataset_manifest_sha256: Digest
    ordered_train_shard_ids: tuple[Literal["train-shard-0", "train-shard-1", "train-shard-2"], ...]
    dev_access_policy: Literal["selector_only"] = "selector_only"
    test_access_policy: Literal["forbidden"] = "forbidden"
    upstream_commit: Literal[UPSTREAM_COMMIT] = UPSTREAM_COMMIT
    upstream_source_sha256: Digest
    dependency_lock_sha256: Digest
    container_image_digest: Digest | None
    registry_declared_digest: Digest | None = None
    image_verification: Literal["verified", "unverified"] = "verified"
    environment_sha256: Digest
    python_patch_version: Annotated[str, Field(pattern=r"^3\.10\.[0-9]+$")]
    timezone: Literal["UTC"] = "UTC"
    locale: Literal["C.UTF-8"] = "C.UTF-8"
    world_clock_iso: Literal["2024-05-01T12:00:00Z"] = "2024-05-01T12:00:00Z"
    scenario_registry_sha256: Digest
    tool_inventory_sha256: Digest
    evaluator_registry_sha256: Digest
    schema_bundle_sha256: Digest
    qwen: QwenRunIdentity
    embedding: EmbeddingRunIdentity
    user_simulator: UserSimulatorRunIdentity
    online_prompt_manifest_sha256: Digest
    offline_prompt_manifest_sha256: Digest
    online_token_limits_sha256: Digest
    offline_token_limits_sha256: Digest
    calibrated_token_limits: Literal[True]
    seeds: SeedArtifactIdentity
    fixture: FixtureRunIdentity
    checkpoint: CheckpointConfig
    checkpoint_registry_schema_sha256: Digest
    metrics_schema_sha256: Digest
    cost_unit: CostUnit = "qwen_effective_output_tokens"
    process_count: Literal[1] = 1
    ordering_policy: Literal["manifest_order_single_process"] = "manifest_order_single_process"
    run_root: Annotated[str, Field(min_length=1)]

    @field_validator("dataset_manifest_path", "run_root")
    @classmethod
    def absolute_non_symlink_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or path.is_symlink():
            raise ValueError("run and dataset paths must be absolute non-symlinks")
        return value

    @model_validator(mode="after")
    def protocol_invariants(self):
        if self.protocol_version == "toolsandbox-evolution-v1":
            if (self.container_image_digest is None or self.image_verification != "verified"
                    or self.registry_declared_digest is not None or self.qwen.image_verification != "verified"
                    or self.qwen.registry_declared_digest is not None):
                raise ValueError("v1 requires verified actual image identities")
        elif self.purpose != "formal_training" or self.profile != "official_live":
            raise ValueError("live v2 is restricted to official-live formal training")
        if self.image_verification == "unverified":
            if self.container_image_digest is not None or self.registry_declared_digest is None:
                raise ValueError("unverified image requires declared digest and no actual digest")
        elif self.container_image_digest is None:
            raise ValueError("verified image requires actual digest")
        if self.ordered_train_shard_ids != (
            "train-shard-0", "train-shard-1", "train-shard-2",
        ):
            raise ValueError("exact ordered three-shard plan required")
        if self.purpose == "train_smoke":
            if self.profile != "strict_replay" or self.fixture.mode != "fixture":
                raise ValueError("train smoke requires strict fixture replay")
            if "/runs/smoke/" not in self.run_root.rstrip("/") + "/":
                raise ValueError("train smoke requires the runs/smoke namespace")
        if self.purpose == "formal_training" and self.run_root.rstrip("/").endswith("/smoke"):
            raise ValueError("formal run cannot use the smoke namespace")
        forbidden_names = ("api_key", "secret", "authorization", "credential")
        for key, value in _walk_strings(self.model_dump(mode="json")):
            lowered_key = key.lower()
            lowered_value = value.lower()
            if any(token in lowered_key for token in forbidden_names):
                raise ValueError("secret-bearing manifest field forbidden")
            if lowered_value.startswith(("http://", "https://")):
                raise ValueError("raw endpoint URL forbidden in run manifest")
            if value.startswith(("sk-", "Bearer ")):
                raise ValueError("secret-like manifest value forbidden")
        return self

    @model_serializer(mode="wrap")
    def omit_legacy_image_defaults(self, handler):
        value = handler(self)
        if self.protocol_version == "toolsandbox-evolution-v1":
            value.pop("registry_declared_digest", None)
            value.pop("image_verification", None)
        return value

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(["resolved-run-manifest-v1", self.model_dump(mode="json")])


def _walk_strings(value, key: str = ""):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_strings(child, child_key)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child, key)
    elif isinstance(value, str):
        yield key, value


class GenerationObservation(FrozenRunRecord):
    generation_id: GenerationId
    generation_manifest_sha256: Digest
    generation_content_sha256: Digest
    producing_round_index: Annotated[int, Field(ge=0, le=2)] | None = None
    train_shard_id: Annotated[str, Field(min_length=1)] | None = None
    mean_native_similarity: Annotated[float, Field(ge=0, le=1)] | None = None
    fully_successful_rate: Annotated[float, Field(ge=0, le=1)] | None = None
    round_metric_record_id: Digest | None = None
    total_running_time_seconds: Annotated[float, Field(ge=0)] | None = None
    total_tokens: Annotated[int, Field(ge=0)] | None = None
    usage_complete: bool | None = None
    total_cost: Annotated[int, Field(ge=0)] | None = None
    cost_unit: CostUnit | None = None
    cost_complete: bool | None = None
    completion_status: Literal["complete", "failed", "partial"] | None = None

    @model_validator(mode="after")
    def generation_round_mapping(self):
        number = int(self.generation_id[1:])
        metrics = (
            self.producing_round_index, self.train_shard_id,
            self.mean_native_similarity, self.fully_successful_rate,
            self.round_metric_record_id, self.total_running_time_seconds,
            self.total_tokens, self.usage_complete, self.total_cost,
            self.cost_unit, self.cost_complete, self.completion_status,
        )
        if number == 0:
            if any(value is not None for value in metrics):
                raise ValueError("G000 has no producing-round metrics")
            return self
        if self.producing_round_index != number - 1:
            raise ValueError("generation/producing-round mismatch")
        required = (
            self.train_shard_id, self.mean_native_similarity,
            self.fully_successful_rate, self.round_metric_record_id,
            self.total_running_time_seconds, self.usage_complete,
            self.cost_unit, self.cost_complete, self.completion_status,
        )
        if any(value is None for value in required):
            raise ValueError("produced generation requires complete observation fields")
        if self.usage_complete != (self.total_tokens is not None):
            raise ValueError("usage completeness mismatch")
        if self.cost_complete != (self.total_cost is not None):
            raise ValueError("cost completeness mismatch")
        return self


class DiagnosticCheckpointPointer(FrozenRunRecord):
    metric: Literal["mean_native_similarity", "fully_successful_rate"]
    generation_id: Literal["g001", "g002", "g003"]
    observed_value: Annotated[float, Field(ge=0, le=1)]
    selection_allowed: Literal[False] = False
    comparability: Literal["different_train_shards_not_directly_comparable"] = NONCOMPARABLE


class CheckpointObservationRegistry(FrozenRunRecord):
    schema_version: Literal[1] = 1
    run_id: Annotated[str, Field(min_length=1)]
    entries: tuple[GenerationObservation, GenerationObservation, GenerationObservation, GenerationObservation]
    updated_generation_id: Literal["g003"] = "g003"
    highest_observed_train_mean_similarity: DiagnosticCheckpointPointer
    highest_observed_train_fully_successful_rate: DiagnosticCheckpointPointer
    selection_allowed: Literal[False] = False
    comparability: Literal["different_train_shards_not_directly_comparable"] = NONCOMPARABLE
    registry_sha256: Digest

    @model_validator(mode="after")
    def identities_and_pointers(self):
        if tuple(entry.generation_id for entry in self.entries) != (
            "g000", "g001", "g002", "g003",
        ):
            raise ValueError("registry requires ordered G000-G003")
        for field, metric in (
            ("highest_observed_train_mean_similarity", "mean_native_similarity"),
            ("highest_observed_train_fully_successful_rate", "fully_successful_rate"),
        ):
            pointer = getattr(self, field)
            expected = max(
                self.entries[1:],
                key=lambda row: (getattr(row, metric), -int(row.generation_id[1:])),
            )
            if (
                pointer.metric != metric
                or pointer.generation_id != expected.generation_id
                or pointer.observed_value != getattr(expected, metric)
            ):
                raise ValueError("diagnostic pointer mismatch")
        payload = self.model_dump(mode="json", exclude={"registry_sha256"})
        if self.registry_sha256 != canonical_sha256([
            "checkpoint-observation-registry-v1", payload,
        ]):
            raise ValueError("registry identity mismatch")
        return self


__all__ = [
    "CheckpointObservationRegistry", "DiagnosticCheckpointPointer",
    "EmbeddingRunIdentity", "FixtureRunIdentity", "GenerationObservation",
    "NONCOMPARABLE", "QwenRunIdentity", "ResolvedRunManifest",
    "SeedArtifactIdentity", "UserSimulatorRunIdentity",
    "TrainSmokeConfig", "TrainSmokeSelection",
]
