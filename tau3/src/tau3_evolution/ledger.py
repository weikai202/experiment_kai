"""Durable request, attempt, causal-application, and effect provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_sha256, read_json


@dataclass(frozen=True)
class LogicalRequest:
    logical_request_id: str
    scope_id: str
    provider_role: str
    purpose: str
    request_sha256: str
    model_id: str
    config_sha256: str


@dataclass(frozen=True)
class PhysicalAttempt:
    attempt_id: str
    logical_request_id: str
    input_tokens: int | None
    output_tokens: int | None
    completed: bool
    response_sha256: str


@dataclass(frozen=True)
class SubstantiveEffect:
    effect_id: str
    scope_id: str
    effect_kind: str
    artifact_sha256: str
    substantive: bool
    committed: bool


@dataclass(frozen=True)
class OutputApplication:
    application_id: str
    logical_request_id: str
    attempt_id: str
    effect_id: str | None
    application_kind: str
    committed: bool
    causal_order: int


@dataclass(frozen=True)
class AccountingSnapshot:
    scope_id: str
    ledger_sha256: str
    checkpoint_material_sha256: str
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_tokens: int | None
    usage_complete: bool
    total_cost: int | None
    cost_unit: str
    cost_complete: bool
    logical_request_count: int
    physical_attempt_count: int
    committed_application_count: int
    committed_substantive_effect_count: int


def _is_hash(value: str) -> bool:
    return type(value) is str and value.startswith("sha256:") and len(value) > len("sha256:")


class DurableAccountingLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self._document = read_json(self.path)
            self._validate_document()
        else:
            self._document: dict[str, list[dict[str, Any]] | int] = {
                "schema_version": 2,
                "logical_requests": [],
                "physical_attempts": [],
                "applications": [],
                "effects": [],
            }

    def _validate_document(self) -> None:
        if (
            set(self._document)
            != {
                "schema_version",
                "logical_requests",
                "physical_attempts",
                "applications",
                "effects",
            }
            or self._document["schema_version"] != 2
        ):
            raise ValueError("invalid accounting ledger document")

    def _append(self, collection: str, value: object, identity: str, field: str) -> None:
        rows = self._document[collection]
        assert isinstance(rows, list)
        payload = asdict(value)  # type: ignore[arg-type]
        existing = next((row for row in rows if row[field] == identity), None)
        if existing is not None:
            if existing != payload:
                raise ValueError(f"conflicting {collection} identity")
            return
        rows.append(payload)
        atomic_write_json(self.path, self._document)

    def record_request(self, value: LogicalRequest) -> None:
        if value.provider_role not in {"qwen", "embedding", "user_simulator"}:
            raise ValueError("unknown provider role")
        if (
            not all((value.logical_request_id, value.scope_id, value.purpose, value.model_id))
            or not _is_hash(value.request_sha256)
            or not _is_hash(value.config_sha256)
        ):
            raise ValueError("logical request lacks exact request/model/config provenance")
        self._append("logical_requests", value, value.logical_request_id, "logical_request_id")

    def record_attempt(self, value: PhysicalAttempt) -> None:
        for count in (value.input_tokens, value.output_tokens):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError("invalid token count")
        if not _is_hash(value.response_sha256):
            raise ValueError("physical attempt requires response or failure fingerprint")
        self._append("physical_attempts", value, value.attempt_id, "attempt_id")

    def record_effect(self, value: SubstantiveEffect) -> None:
        if not _is_hash(value.artifact_sha256):
            raise ValueError("effect artifact fingerprint required")
        self._append("effects", value, value.effect_id, "effect_id")

    def record_application(self, value: OutputApplication) -> None:
        if type(value.causal_order) is not int or value.causal_order < 0:
            raise ValueError("causal order must be a non-negative integer")
        self._append("applications", value, value.application_id, "application_id")

    @property
    def document(self) -> dict[str, object]:
        return read_json(self.path) if self.path.exists() else dict(self._document)

    @property
    def ledger_sha256(self) -> str:
        return canonical_sha256(self._document)

    def snapshot(self, *, scope_id: str, checkpoint_material_sha256: str) -> AccountingSnapshot:
        requests = [LogicalRequest(**row) for row in self._document["logical_requests"]]
        attempts = [PhysicalAttempt(**row) for row in self._document["physical_attempts"]]
        applications = [OutputApplication(**row) for row in self._document["applications"]]
        effects = [SubstantiveEffect(**row) for row in self._document["effects"]]
        if any(row.scope_id != scope_id for row in requests + effects):
            raise ValueError("ledger contains record outside bound scope")
        request_by_id = {row.logical_request_id: row for row in requests}
        attempt_by_id = {row.attempt_id: row for row in attempts}
        effect_by_id = {row.effect_id: row for row in effects}
        if len(request_by_id) != len(requests) or len(attempt_by_id) != len(attempts):
            raise ValueError("duplicate ledger identity")
        if any(row.logical_request_id not in request_by_id for row in attempts + applications):
            raise ValueError("attempt/application without logical request")
        for row in applications:
            attempt = attempt_by_id.get(row.attempt_id)
            if attempt is None or attempt.logical_request_id != row.logical_request_id:
                raise ValueError("application/physical attempt mismatch")
            if row.effect_id is not None and row.effect_id not in effect_by_id:
                raise ValueError("application references unknown effect")
        committed_effects = [row for row in effects if row.committed and row.substantive]
        committed_ids = {row.effect_id for row in committed_effects}
        effective = [
            row
            for row in applications
            if row.committed
            and row.effect_id in committed_ids
            and request_by_id[row.logical_request_id].provider_role == "qwen"
        ]
        orders = [(row.effect_id, row.causal_order) for row in effective]
        if len(orders) != len(set(orders)):
            raise ValueError("duplicate causal position within effect decision chain")
        for effect_id in committed_ids:
            chain = sorted(row.causal_order for row in effective if row.effect_id == effect_id)
            if chain and chain != list(range(len(chain))):
                raise ValueError("effect decision chain must use contiguous causal order")
        missing_effects = committed_ids - {row.effect_id for row in effective}
        usage_complete = all(
            row.completed and row.input_tokens is not None and row.output_tokens is not None
            for row in attempts
        ) and all(
            any(attempt.logical_request_id == request.logical_request_id for attempt in attempts)
            for request in requests
        )
        total_input = sum(row.input_tokens or 0 for row in attempts) if usage_complete else None
        total_output = sum(row.output_tokens or 0 for row in attempts) if usage_complete else None
        cost_attempt_ids = {row.attempt_id for row in effective}
        cost_attempts = [attempt_by_id[value] for value in cost_attempt_ids]
        cost_complete = not missing_effects and all(
            row.completed and row.output_tokens is not None for row in cost_attempts
        )
        return AccountingSnapshot(
            scope_id,
            self.ledger_sha256,
            checkpoint_material_sha256,
            total_input,
            total_output,
            total_input + total_output if usage_complete else None,
            usage_complete,
            sum(row.output_tokens or 0 for row in cost_attempts) if cost_complete else None,
            "qwen_effective_output_tokens",
            cost_complete,
            len(requests),
            len(attempts),
            sum(row.committed for row in applications),
            len(committed_effects),
        )
