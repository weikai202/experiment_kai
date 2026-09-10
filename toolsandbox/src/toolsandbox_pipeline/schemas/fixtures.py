"""Strict identities for RapidAPI fixture replay and audited dispatch."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.schemas.base import JsonObject
from toolsandbox_pipeline.schemas.memory import Digest, FrozenRecord, Identifier, unique


UPSTREAM_COMMIT = "165848b9a78cead7ca7fe7c89c688b58e6501219"
ADAPTER_VERSION = "toolsandbox-rapidapi-boundary-v1"
FIXTURE_SCHEMA_VERSION = 1
TOOL_NAMES = (
    "convert_currency",
    "search_lat_lon",
    "search_location_around_lat_lon",
    "search_stock",
    "search_weather_around_lat_lon",
)
UtcTimestamp = Annotated[
    str,
    Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"),
]


class BackendRecord(FrozenRecord):
    canonical_tool_name: Literal[
        "convert_currency",
        "search_lat_lon",
        "search_location_around_lat_lon",
        "search_stock",
        "search_weather_around_lat_lon",
    ]
    backend_version: Identifier
    method: Literal["GET"]
    url: Annotated[str, Field(pattern=r"^https://[^/?#]+/[^?#]*$")]
    host: Annotated[str, Field(pattern=r"^[a-z0-9.-]+\.rapidapi\.com$")]
    permitted_parameter_names: Annotated[tuple[Identifier, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_parameters(self) -> "BackendRecord":
        unique(self.permitted_parameter_names)
        return self


class RapidAPIBackendManifest(FrozenRecord):
    schema_version: Literal[1]
    adapter_version: Literal["toolsandbox-rapidapi-boundary-v1"]
    upstream_commit: Literal["165848b9a78cead7ca7fe7c89c688b58e6501219"]
    request_timeout_seconds: Annotated[float, Field(gt=0, le=120)]
    allow_redirects: Literal[False]
    backends: tuple[BackendRecord, ...]

    @field_validator("request_timeout_seconds", mode="before")
    @classmethod
    def exact_timeout_type(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("request timeout requires an exact float")
        return value

    @model_validator(mode="after")
    def exact_inventory(self) -> "RapidAPIBackendManifest":
        names = tuple(record.canonical_tool_name for record in self.backends)
        if names != TOOL_NAMES:
            raise ValueError("backend inventory/order mismatch")
        pairs = tuple((record.url, record.host) for record in self.backends)
        unique(pairs)
        return self


class ExternalReadContext(FrozenRecord):
    schema_version: Literal[1]
    run_id: Identifier
    profile: Literal["strict_replay", "official_live"]
    phase: Identifier
    scenario_family_id: Identifier
    scenario_id: Identifier
    state_id: Digest
    logical_tool_call_id: Identifier
    backend_manifest_sha256: Digest
    fixture_manifest_sha256: Digest | None

    @model_validator(mode="after")
    def profile_fixture_pairing(self) -> "ExternalReadContext":
        if (self.profile == "strict_replay") != (self.fixture_manifest_sha256 is not None):
            raise ValueError("strict replay alone requires a fixture manifest")
        return self


class FixtureRequest(FrozenRecord):
    schema_version: Literal[1]
    fixture_key: Digest
    canonical_tool_name: Literal[
        "convert_currency",
        "search_lat_lon",
        "search_location_around_lat_lon",
        "search_stock",
        "search_weather_around_lat_lon",
    ]
    effective_arguments: JsonObject
    backend_version: Identifier
    request_url_identity: Digest

    @model_validator(mode="after")
    def key_matches_request(self) -> "FixtureRequest":
        expected = canonical_sha256(
            {
                "tool_name": self.canonical_tool_name,
                "arguments": self.effective_arguments,
                "backend_version": self.backend_version,
            }
        )
        if expected != self.fixture_key:
            raise ValueError("fixture key mismatch")
        return self


class FixtureEntry(FrozenRecord):
    schema_version: Literal[1]
    fixture_key: Digest
    canonical_tool_name: Literal[
        "convert_currency",
        "search_lat_lon",
        "search_location_around_lat_lon",
        "search_stock",
        "search_weather_around_lat_lon",
    ]
    effective_arguments: JsonObject
    backend_version: Identifier
    request_url_identity: Digest
    status_code: Literal[200]
    normalized_response_body: JsonObject
    response_body_sha256: Digest
    captured_at_utc: UtcTimestamp
    source: Literal["rapidapi_capture"]

    @model_validator(mode="after")
    def identities_match(self) -> "FixtureEntry":
        request = FixtureRequest(
            schema_version=1,
            fixture_key=self.fixture_key,
            canonical_tool_name=self.canonical_tool_name,
            effective_arguments=self.effective_arguments,
            backend_version=self.backend_version,
            request_url_identity=self.request_url_identity,
        )
        del request
        if canonical_sha256(self.normalized_response_body) != self.response_body_sha256:
            raise ValueError("fixture response body hash mismatch")
        return self

    def content_identity_payload(self) -> JsonObject:
        payload = self.model_dump(mode="json")
        del payload["captured_at_utc"]
        return payload


class FixtureFileRecord(FrozenRecord):
    fixture_key: Digest
    entry_file_sha256: Digest


class FixtureManifest(FrozenRecord):
    schema_version: Literal[1]
    adapter_version: Literal["toolsandbox-rapidapi-boundary-v1"]
    upstream_commit: Literal["165848b9a78cead7ca7fe7c89c688b58e6501219"]
    backend_manifest_sha256: Digest
    backend_versions: tuple[Identifier, ...]
    fixture_generation_id: Annotated[str, Field(pattern=r"^fg_[0-9a-f]{64}$")]
    entry_count: Annotated[int, Field(ge=1)]
    ordered_fixture_keys: tuple[Digest, ...]
    entry_files: tuple[FixtureFileRecord, ...]
    canonical_entry_set_sha256: Digest
    created_at_utc: UtcTimestamp
    publication_status: Literal["complete"]

    @model_validator(mode="after")
    def layout_identity(self) -> "FixtureManifest":
        if self.entry_count != len(self.ordered_fixture_keys):
            raise ValueError("fixture entry count mismatch")
        unique(self.ordered_fixture_keys, ordered=True)
        if tuple(record.fixture_key for record in self.entry_files) != self.ordered_fixture_keys:
            raise ValueError("entry-file order/key mismatch")
        if self.fixture_generation_id != "fg_" + self.canonical_entry_set_sha256.removeprefix("sha256:"):
            raise ValueError("fixture generation identity mismatch")
        return self


class FixtureMissEvidence(FrozenRecord):
    schema_version: Literal[1]
    context: ExternalReadContext
    fixture_key: Digest
    canonical_tool_name: Literal[
        "convert_currency",
        "search_lat_lon",
        "search_location_around_lat_lon",
        "search_stock",
        "search_weather_around_lat_lon",
    ]
    backend_version: Identifier
    effective_arguments: JsonObject


class ExternalReadAttempt(FrozenRecord):
    schema_version: Literal[1]
    attempt_id: Identifier
    context: ExternalReadContext
    mode: Literal["replay", "official_live", "capture"]
    canonical_tool_name: Literal[
        "convert_currency",
        "search_lat_lon",
        "search_location_around_lat_lon",
        "search_stock",
        "search_weather_around_lat_lon",
    ]
    effect: Literal["external_read"]
    fixture_key: Digest
    backend_version: Identifier
    status: Literal[
        "fixture_hit",
        "fixture_miss",
        "completed",
        "failed",
        "rejected_before_dispatch",
        "unknown_outcome",
    ]
    dispatched: bool
    status_code: int | None
    response_byte_count: Annotated[int, Field(ge=0)] | None = None
    response_body_sha256: Digest | None
    sanitized_exception_class: Identifier | None
    started_at_utc: UtcTimestamp
    ended_at_utc: UtcTimestamp
    latency_seconds: Annotated[float, Field(ge=0)]

    @field_validator("latency_seconds", mode="before")
    @classmethod
    def exact_latency_type(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("latency requires an exact float")
        return value

    @model_validator(mode="after")
    def status_shape(self) -> "ExternalReadAttempt":
        fixture_status = self.status in ("fixture_hit", "fixture_miss")
        if fixture_status and self.mode != "replay":
            raise ValueError("replay status/dispatch mismatch")
        if self.mode == "replay" and (
            self.status not in ("fixture_hit", "fixture_miss", "rejected_before_dispatch")
            or self.dispatched
        ):
            raise ValueError("replay status/dispatch mismatch")
        if self.status == "fixture_hit" and self.response_body_sha256 is None:
            raise ValueError("fixture hit requires response identity")
        if self.status == "fixture_miss" and self.sanitized_exception_class != "FixtureMiss":
            raise ValueError("fixture miss requires fixed sanitized class")
        if self.status == "completed" and (not self.dispatched or self.status_code != 200 or self.response_body_sha256 is None):
            raise ValueError("completed dispatch shape mismatch")
        if self.status == "completed" and self.response_byte_count is None:
            raise ValueError("completed dispatch requires response byte count")
        if self.status == "unknown_outcome" and not self.dispatched:
            raise ValueError("unknown outcome requires dispatch")
        if self.status == "rejected_before_dispatch" and self.dispatched:
            raise ValueError("rejected dispatch cannot be marked dispatched")
        return self


class FixtureCaptureRequestManifest(FrozenRecord):
    schema_version: Literal[1]
    adapter_version: Literal["toolsandbox-rapidapi-boundary-v1"]
    upstream_commit: Literal["165848b9a78cead7ca7fe7c89c688b58e6501219"]
    backend_manifest_sha256: Digest
    purpose: Literal["synthetic_contract_probe", "train_fixture_preparation"]
    train_manifest_sha256: Digest | None
    fixture_miss_audit_sha256: Digest | None
    requests: Annotated[tuple[FixtureRequest, ...], Field(min_length=1)]
    ordered_fixture_keys: tuple[Digest, ...]
    maximum_request_count: Annotated[int, Field(gt=0, le=1000)]
    requests_per_second: Annotated[float, Field(gt=0, le=100)]
    concurrency_limit: Literal[1]
    created_at_utc: UtcTimestamp

    @field_validator("requests_per_second", mode="before")
    @classmethod
    def exact_rate_type(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("rate requires an exact float")
        return value

    @model_validator(mode="after")
    def request_contract(self) -> "FixtureCaptureRequestManifest":
        keys = tuple(request.fixture_key for request in self.requests)
        if keys != self.ordered_fixture_keys or len(keys) > self.maximum_request_count:
            raise ValueError("capture request order/count mismatch")
        unique(keys, ordered=True)
        training = self.purpose == "train_fixture_preparation"
        if training != (self.train_manifest_sha256 is not None and self.fixture_miss_audit_sha256 is not None):
            raise ValueError("train capture requires train and miss-audit identities")
        return self


class CaptureReport(FrozenRecord):
    schema_version: Literal[1]
    status: Literal["completed", "failed"]
    request_count: Annotated[int, Field(ge=0)]
    success_count: Annotated[int, Field(ge=0)]
    failure_count: Annotated[int, Field(ge=0)]
    fixture_generation_id: str | None
    fixture_manifest_sha256: Digest | None
    total_running_time_seconds: Annotated[float, Field(ge=0)]
    status_counts: dict[str, int]
    latency_min_seconds: float | None
    latency_max_seconds: float | None
    latency_mean_seconds: float | None
    sanitized_error_classes: tuple[Identifier, ...]

    @model_validator(mode="after")
    def report_totals(self) -> "CaptureReport":
        if self.success_count + self.failure_count != self.request_count:
            raise ValueError("capture report counts mismatch")
        complete = self.status == "completed"
        if complete != (self.failure_count == 0 and self.fixture_generation_id is not None and self.fixture_manifest_sha256 is not None):
            raise ValueError("capture report completion mismatch")
        return self
