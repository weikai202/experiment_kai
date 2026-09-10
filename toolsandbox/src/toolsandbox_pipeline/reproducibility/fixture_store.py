"""Canonical RapidAPI requests and immutable fixture bundle storage."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.fixtures import (
    ADAPTER_VERSION,
    UPSTREAM_COMMIT,
    BackendRecord,
    FixtureEntry,
    FixtureFileRecord,
    FixtureManifest,
    FixtureRequest,
    RapidAPIBackendManifest,
)


EXPECTED_BACKENDS = (
    (
        "convert_currency",
        "currency-converter18-v1",
        "https://currency-converter18.p.rapidapi.com/api/v1/convert",
        "currency-converter18.p.rapidapi.com",
        ("from", "to", "amount"),
    ),
    (
        "search_lat_lon",
        "trueway-reverse-geocode-v1",
        "https://trueway-geocoding.p.rapidapi.com/ReverseGeocode",
        "trueway-geocoding.p.rapidapi.com",
        ("location", "language"),
    ),
    (
        "search_location_around_lat_lon",
        "maps-data-search-v1",
        "https://maps-data.p.rapidapi.com/searchmaps.php",
        "maps-data.p.rapidapi.com",
        ("query", "lat", "lng", "limit", "country", "lang"),
    ),
    (
        "search_stock",
        "real-time-finance-search-v1",
        "https://real-time-finance-data.p.rapidapi.com/search",
        "real-time-finance-data.p.rapidapi.com",
        ("query",),
    ),
    (
        "search_weather_around_lat_lon",
        "weatherapi-forecast-v1",
        "https://weatherapi-com.p.rapidapi.com/forecast.json",
        "weatherapi-com.p.rapidapi.com",
        ("q", "days"),
    ),
)
PINNED_BACKEND_CONFIG_SHA256 = "sha256:6259ca87dabb2f128a44837e1c7732431e660c2df659169cf3bd56ff67844fe3"
_SECRET_KEY_PARTS = frozenset(
    {"authorization", "cookie", "cookies", "header", "headers", "key", "secret", "token", "x-rapidapi-key"}
)


class FixtureValidationError(RuntimeError):
    """A fixture/config artifact failed closed validation."""


def sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def parse_strict_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs_no_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FixtureValidationError("invalid strict JSON") from exc


def _validate_exact_json(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise FixtureValidationError("non-finite fixture value")
        return
    if type(value) is list:
        for item in value:
            _validate_exact_json(item, path=path)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise FixtureValidationError("fixture object keys must be strings")
            normalized = key.lower().replace("-", "_")
            pieces = frozenset(part for part in normalized.split("_") if part)
            if key.lower() == "x-rapidapi-key" or pieces & _SECRET_KEY_PARTS:
                raise FixtureValidationError("secret/header-like fixture key")
            _validate_exact_json(item, path=path + (key,))
        return
    raise FixtureValidationError("unsupported fixture value type")


def _require_absolute_regular_file(path: Path) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise FixtureValidationError("absolute regular file required")
    for parent in path.parents:
        if parent.is_symlink():
            raise FixtureValidationError("symlink path is forbidden")


def load_backend_manifest(path: str | Path, *, expected_sha256: str) -> tuple[RapidAPIBackendManifest, str]:
    config_path = Path(path)
    _require_absolute_regular_file(config_path)
    raw = config_path.read_bytes()
    actual_sha256 = sha256_bytes(raw)
    if actual_sha256 != expected_sha256:
        raise FixtureValidationError("backend manifest hash mismatch")
    parse_strict_json(raw)
    try:
        manifest = RapidAPIBackendManifest.model_validate_json(raw)
    except Exception as exc:
        raise FixtureValidationError("invalid backend manifest") from exc
    actual = tuple(
        (
            item.canonical_tool_name,
            item.backend_version,
            item.url,
            item.host,
            item.permitted_parameter_names,
        )
        for item in manifest.backends
    )
    if (
        actual != EXPECTED_BACKENDS
        or manifest.upstream_commit != UPSTREAM_COMMIT
        or manifest.request_timeout_seconds != 30.0
        or actual_sha256 != PINNED_BACKEND_CONFIG_SHA256
    ):
        raise FixtureValidationError("pinned RapidAPI backend contract drift")
    return manifest, actual_sha256


def backend_by_name(manifest: RapidAPIBackendManifest, name: str) -> BackendRecord:
    if type(manifest) is not RapidAPIBackendManifest:
        raise TypeError("validated backend manifest required")
    matches = tuple(item for item in manifest.backends if item.canonical_tool_name == name)
    if len(matches) != 1:
        raise FixtureValidationError("unknown or duplicate backend")
    return matches[0]


def request_url_identity(record: BackendRecord) -> str:
    return canonical_sha256({"method": record.method, "url": record.url, "host": record.host})


def fixture_key(tool_name: str, arguments: Mapping[str, Any], backend_version: str) -> str:
    if type(tool_name) is not str or type(backend_version) is not str or type(arguments) is not dict:
        raise FixtureValidationError("fixture key inputs require exact types")
    _validate_exact_json(arguments)
    return canonical_sha256(
        {"tool_name": tool_name, "arguments": arguments, "backend_version": backend_version}
    )


def build_fixture_request(
    manifest: RapidAPIBackendManifest,
    *,
    url: Any,
    params: Any,
    headers: Any,
) -> FixtureRequest:
    if type(url) is not str or type(params) is not dict or type(headers) is not dict:
        raise FixtureValidationError("backend request requires exact URL/params/headers types")
    if tuple(headers) != ("X-RapidAPI-Host",) or type(headers.get("X-RapidAPI-Host")) is not str:
        raise FixtureValidationError("backend request host header mismatch")
    matches = tuple(
        item for item in manifest.backends if item.url == url and item.host == headers["X-RapidAPI-Host"]
    )
    if len(matches) != 1:
        raise FixtureValidationError("unknown or duplicate backend URL/host")
    record = matches[0]
    if tuple(params) != record.permitted_parameter_names:
        raise FixtureValidationError("backend parameter shape/order mismatch")
    _validate_exact_json(params)
    arguments = copy.deepcopy(params)
    return FixtureRequest(
        schema_version=1,
        fixture_key=fixture_key(record.canonical_tool_name, arguments, record.backend_version),
        canonical_tool_name=record.canonical_tool_name,
        effective_arguments=arguments,
        backend_version=record.backend_version,
        request_url_identity=request_url_identity(record),
    )


def validate_request_against_backend(request: FixtureRequest, manifest: RapidAPIBackendManifest) -> BackendRecord:
    record = backend_by_name(manifest, request.canonical_tool_name)
    if (
        request.backend_version != record.backend_version
        or request.request_url_identity != request_url_identity(record)
        or len(request.effective_arguments) != len(record.permitted_parameter_names)
        or set(request.effective_arguments) != set(record.permitted_parameter_names)
    ):
        raise FixtureValidationError("fixture request/backend mismatch")
    expected = fixture_key(request.canonical_tool_name, request.effective_arguments, request.backend_version)
    if expected != request.fixture_key:
        raise FixtureValidationError("fixture request key mismatch")
    return record


def _entry_set_sha256(entries: Iterable[FixtureEntry]) -> str:
    return canonical_sha256([entry.content_identity_payload() for entry in entries])


class FixtureStore:
    """A fully verified read-only fixture generation."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: FixtureManifest,
        manifest_sha256: str,
        entries: Mapping[str, FixtureEntry],
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self._entries = dict(entries)

    @classmethod
    def open(
        cls,
        manifest_path: str | Path,
        *,
        expected_manifest_sha256: str,
        backend_manifest: RapidAPIBackendManifest,
        backend_manifest_sha256: str,
    ) -> "FixtureStore":
        path = Path(manifest_path)
        _require_absolute_regular_file(path)
        if path.name != "manifest.json" or path.parent.name.startswith(".stage-"):
            raise FixtureValidationError("invalid fixture manifest layout")
        root = path.parent
        if root.name == "entries" or root.parent.name != "fixtures":
            raise FixtureValidationError("fixture bundle must be fixtures/<generation_id>")
        for directory in (root.parent.parent, root.parent, root, root / "entries"):
            if directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o777 != 0o700:
                raise FixtureValidationError("fixture directory permissions/layout mismatch")
        if path.stat().st_mode & 0o777 != 0o600:
            raise FixtureValidationError("fixture manifest must have mode 0600")
        raw_manifest = path.read_bytes()
        manifest_sha256 = sha256_bytes(raw_manifest)
        if manifest_sha256 != expected_manifest_sha256:
            raise FixtureValidationError("fixture manifest hash mismatch")
        try:
            parse_strict_json(raw_manifest)
            manifest = FixtureManifest.model_validate_json(raw_manifest)
        except Exception as exc:
            raise FixtureValidationError("invalid fixture manifest") from exc
        if raw_manifest != canonical_json_bytes(manifest.model_dump(mode="json")):
            raise FixtureValidationError("fixture manifest must use canonical JSON bytes")
        if (
            manifest.fixture_generation_id != root.name
            or manifest.backend_manifest_sha256 != backend_manifest_sha256
            or manifest.backend_versions != tuple(item.backend_version for item in backend_manifest.backends)
            or manifest.adapter_version != ADAPTER_VERSION
            or manifest.upstream_commit != UPSTREAM_COMMIT
        ):
            raise FixtureValidationError("fixture bundle identity drift")
        expected_paths = {"manifest.json", "entries"} | {
            f"entries/{key.removeprefix('sha256:')}.json" for key in manifest.ordered_fixture_keys
        }
        actual_paths = {item.relative_to(root).as_posix() for item in root.rglob("*")}
        if actual_paths != expected_paths:
            raise FixtureValidationError("unexpected or missing fixture bundle path")
        file_records = {record.fixture_key: record.entry_file_sha256 for record in manifest.entry_files}
        entries: dict[str, FixtureEntry] = {}
        for key in manifest.ordered_fixture_keys:
            entry_path = root / "entries" / f"{key.removeprefix('sha256:')}.json"
            if entry_path.is_symlink() or entry_path.stat().st_mode & 0o777 != 0o600:
                raise FixtureValidationError("fixture entry path/permissions mismatch")
            raw_entry = entry_path.read_bytes()
            if sha256_bytes(raw_entry) != file_records.get(key):
                raise FixtureValidationError("fixture entry file hash mismatch")
            try:
                parse_strict_json(raw_entry)
                entry = FixtureEntry.model_validate_json(raw_entry)
            except Exception as exc:
                raise FixtureValidationError("invalid fixture entry") from exc
            if raw_entry != canonical_json_bytes(entry.model_dump(mode="json")) or entry.fixture_key != key:
                raise FixtureValidationError("fixture filename/content mismatch")
            request = FixtureRequest(
                schema_version=1,
                fixture_key=entry.fixture_key,
                canonical_tool_name=entry.canonical_tool_name,
                effective_arguments=entry.effective_arguments,
                backend_version=entry.backend_version,
                request_url_identity=entry.request_url_identity,
            )
            validate_request_against_backend(request, backend_manifest)
            entries[key] = entry
        if _entry_set_sha256(entries.values()) != manifest.canonical_entry_set_sha256:
            raise FixtureValidationError("canonical fixture entry-set hash mismatch")
        return cls(root=root, manifest=manifest, manifest_sha256=manifest_sha256, entries=entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def lookup(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        return copy.deepcopy(entry.normalized_response_body)

    def lookup_entry(self, key: str) -> FixtureEntry | None:
        """Perform one immutable lookup and return an isolated validated entry."""

        entry = self._entries.get(key)
        return None if entry is None else entry.model_copy(deep=True)

    def entry(self, key: str) -> FixtureEntry | None:
        entry = self._entries.get(key)
        return None if entry is None else entry.model_copy(deep=True)


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def publish_fixture_bundle(
    output_root: str | Path,
    entries: Iterable[FixtureEntry],
    *,
    backend_manifest: RapidAPIBackendManifest,
    backend_manifest_sha256: str,
    created_at_utc: str,
) -> tuple[Path, str]:
    root = Path(output_root)
    if not root.is_absolute() or root.is_symlink():
        raise FixtureValidationError("absolute non-symlink output root required")
    ordered = tuple(sorted(entries, key=lambda entry: entry.fixture_key.encode("utf-8")))
    if not ordered or len({entry.fixture_key for entry in ordered}) != len(ordered):
        raise FixtureValidationError("nonempty unique fixture entries required")
    for entry in ordered:
        request = FixtureRequest(
            schema_version=1,
            fixture_key=entry.fixture_key,
            canonical_tool_name=entry.canonical_tool_name,
            effective_arguments=entry.effective_arguments,
            backend_version=entry.backend_version,
            request_url_identity=entry.request_url_identity,
        )
        validate_request_against_backend(request, backend_manifest)
    entry_set_sha256 = _entry_set_sha256(ordered)
    generation_id = "fg_" + entry_set_sha256.removeprefix("sha256:")
    entry_blobs = {entry.fixture_key: canonical_json_bytes(entry.model_dump(mode="json")) for entry in ordered}
    manifest = FixtureManifest(
        schema_version=1,
        adapter_version=ADAPTER_VERSION,
        upstream_commit=UPSTREAM_COMMIT,
        backend_manifest_sha256=backend_manifest_sha256,
        backend_versions=tuple(item.backend_version for item in backend_manifest.backends),
        fixture_generation_id=generation_id,
        entry_count=len(ordered),
        ordered_fixture_keys=tuple(entry.fixture_key for entry in ordered),
        entry_files=tuple(
            FixtureFileRecord(fixture_key=key, entry_file_sha256=sha256_bytes(raw))
            for key, raw in entry_blobs.items()
        ),
        canonical_entry_set_sha256=entry_set_sha256,
        created_at_utc=created_at_utc,
        publication_status="complete",
    )
    manifest_raw = canonical_json_bytes(manifest.model_dump(mode="json"))
    manifest_sha256 = sha256_bytes(manifest_raw)
    fixtures_root = root / "fixtures"
    target = fixtures_root / generation_id
    if target.exists():
        FixtureStore.open(
            target / "manifest.json",
            expected_manifest_sha256=manifest_sha256,
            backend_manifest=backend_manifest,
            backend_manifest_sha256=backend_manifest_sha256,
        )
        return target, manifest_sha256
    if root.exists():
        if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o777 != 0o700:
            raise FixtureValidationError("existing output root is not a private directory")
    else:
        root.mkdir(mode=0o700)
    if fixtures_root.exists():
        if fixtures_root.is_symlink() or not fixtures_root.is_dir() or fixtures_root.stat().st_mode & 0o777 != 0o700:
            raise FixtureValidationError("invalid fixtures directory")
    else:
        fixtures_root.mkdir(mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=fixtures_root))
    os.chmod(stage, 0o700)
    try:
        (stage / "entries").mkdir(mode=0o700)
        for key, raw in entry_blobs.items():
            _write_exclusive(stage / "entries" / f"{key.removeprefix('sha256:')}.json", raw)
        _write_exclusive(stage / "manifest.json", manifest_raw)
        for directory in (stage / "entries", stage):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.rename(stage, target)
        descriptor = os.open(fixtures_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    FixtureStore.open(
        target / "manifest.json",
        expected_manifest_sha256=manifest_sha256,
        backend_manifest=backend_manifest,
        backend_manifest_sha256=backend_manifest_sha256,
    )
    return target, manifest_sha256
