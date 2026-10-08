from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .canonical import sha256_json


@dataclass(frozen=True)
class LogicalRequest:
    logical_request_id: str
    run_id: str
    round_index: int
    case_id: str
    turn_index: int
    provider: str
    model: str
    role: str
    canonical_input_sha256: str
    request_config_sha256: str

    @classmethod
    def build(cls, run_id: str, round_index: int, case_id: str, turn_index: int, provider: str, model: str, role: str, canonical_input_sha256: str, request_config_sha256: str) -> "LogicalRequest":
        core = {"run_id": run_id, "round_index": round_index, "case_id": case_id, "turn_index": turn_index, "provider": provider, "model": model, "role": role, "canonical_input_sha256": canonical_input_sha256, "request_config_sha256": request_config_sha256}
        return cls(sha256_json(core), **core)

    def validate(self) -> None:
        core = asdict(self); supplied = core.pop("logical_request_id")
        if supplied != sha256_json(core): raise ValueError("Logical request identity is not canonical")


@dataclass(frozen=True)
class AttemptUsage:
    attempt_id: str
    logical_request_id: str
    attempt_index: int
    input_tokens: int | None
    output_tokens: int | None
    usage_complete: bool
    response_sha256: str

    @classmethod
    def build(cls, logical_request_id: str, attempt_index: int, input_tokens: int | None, output_tokens: int | None, response_sha256: str) -> "AttemptUsage":
        core = {"logical_request_id": logical_request_id, "attempt_index": attempt_index, "response_sha256": response_sha256}
        return cls(sha256_json(core), logical_request_id, attempt_index, input_tokens, output_tokens, input_tokens is not None and output_tokens is not None, response_sha256)

    def validate(self) -> None:
        if self.usage_complete != (self.input_tokens is not None and self.output_tokens is not None): raise ValueError("usage_complete must agree with token fields")
        if any(x is not None and x < 0 for x in (self.input_tokens, self.output_tokens)): raise ValueError("Token counts cannot be negative")
        if self.attempt_index < 0 or self.attempt_id != sha256_json({"logical_request_id": self.logical_request_id, "attempt_index": self.attempt_index, "response_sha256": self.response_sha256}): raise ValueError("Attempt identity is not canonical")


@dataclass(frozen=True)
class ResponseApplication:
    application_id: str
    logical_request_id: str
    attempt_id: str
    response_sha256: str
    application_kind: str
    accepted: bool

    @classmethod
    def build(cls, logical_request_id: str, attempt_id: str, response_sha256: str, application_kind: str, accepted: bool) -> "ResponseApplication":
        core = {"logical_request_id": logical_request_id, "attempt_id": attempt_id, "response_sha256": response_sha256, "application_kind": application_kind, "accepted": accepted}
        return cls(sha256_json(core), **core)

    def validate(self) -> None:
        core = asdict(self); supplied = core.pop("application_id")
        if supplied != sha256_json(core): raise ValueError("Response application identity is not canonical")


@dataclass(frozen=True)
class CommittedEffect:
    effect_id: str
    effect_kind: str
    before_sha256: str
    after_sha256: str
    causal_application_ids: tuple[str, ...]
    subject_id: str

    @classmethod
    def build(cls, effect_kind: str, before_sha256: str, after_sha256: str, causal_application_ids: tuple[str, ...], subject_id: str = "") -> "CommittedEffect":
        core = {"effect_kind": effect_kind, "before_sha256": before_sha256, "after_sha256": after_sha256, "causal_application_ids": causal_application_ids, "subject_id": subject_id}
        return cls(sha256_json(core), **core)

    def validate(self) -> None:
        core = asdict(self); supplied = core.pop("effect_id")
        if supplied != sha256_json(core): raise ValueError("Committed effect identity is not canonical")

    @property
    def substantive(self) -> bool:
        return self.effect_kind not in {"NONE", "SKIP"} and self.before_sha256 != self.after_sha256


@dataclass(frozen=True)
class AccountingSnapshot:
    direct_latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_complete: bool
    missing_usage_attempt_ids: tuple[str, ...]
    qwen_effective_output_tokens: int | None
    attempt_ids: tuple[str, ...]
    application_ids: tuple[str, ...]
    committed_effect_ids: tuple[str, ...]
    ledger_sha256: str
    snapshot_sha256: str


class DirectTimer:
    def __init__(self, clock: Callable[[], float] = time.monotonic, started_at: float | None = None):
        self._clock = clock; self._started = clock() if started_at is None else started_at; self._stopped: float | None = None

    @property
    def started_at(self) -> float: return self._started

    def stop_after_durable_checkpoint(self) -> float:
        if self._stopped is not None: raise RuntimeError("Timer already stopped")
        self._stopped = self._clock(); value = self._stopped - self._started
        if not math.isfinite(value) or value < 0: raise ValueError("Invalid direct latency")
        return value


class UsageLedger:
    def __init__(self) -> None:
        self._requests: dict[str, LogicalRequest] = {}; self._attempts: dict[str, AttemptUsage] = {}; self._applications: dict[str, ResponseApplication] = {}; self._effects: dict[str, CommittedEffect] = {}

    def add_logical_request(self, request: LogicalRequest) -> None:
        request.validate()
        if request.logical_request_id in self._requests: raise ValueError("Duplicate logical request ID")
        self._requests[request.logical_request_id] = request

    def add_attempt(self, usage: AttemptUsage) -> None:
        usage.validate()
        if usage.attempt_id in self._attempts: raise ValueError("Duplicate attempt ID")
        if usage.logical_request_id not in self._requests: raise ValueError("Attempt cites an unknown logical request")
        if any(x.logical_request_id == usage.logical_request_id and x.attempt_index == usage.attempt_index for x in self._attempts.values()): raise ValueError("Duplicate physical attempt index")
        self._attempts[usage.attempt_id] = usage

    def apply_response(self, application: ResponseApplication) -> None:
        application.validate()
        if application.application_id in self._applications: raise ValueError("Duplicate application ID")
        attempt = self._attempts.get(application.attempt_id)
        if attempt is None or attempt.logical_request_id != application.logical_request_id: raise ValueError("Application does not bind one known request attempt")
        if attempt.response_sha256 != application.response_sha256: raise ValueError("Application response hash mismatch")
        self._applications[application.application_id] = application

    def commit_effect(self, effect: CommittedEffect) -> None:
        effect.validate()
        if effect.effect_id in self._effects: raise ValueError("Duplicate effect ID")
        if not set(effect.causal_application_ids) <= set(self._applications): raise ValueError("Effect cites an unknown response application")
        self._effects[effect.effect_id] = effect

    def to_payload(self) -> dict[str, Any]:
        core = {"requests": [asdict(x) for x in self._requests.values()], "attempts": [asdict(x) for x in self._attempts.values()], "applications": [asdict(x) for x in self._applications.values()], "effects": [asdict(x) for x in self._effects.values()]}
        return {"ledger": core, "ledger_sha256": sha256_json(core)}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "UsageLedger":
        core = payload.get("ledger")
        if not isinstance(core, dict) or payload.get("ledger_sha256") != sha256_json(core): raise ValueError("Durable usage ledger hash mismatch")
        ledger = cls()
        for row in core.get("requests", []): ledger.add_logical_request(LogicalRequest(**row))
        for row in core.get("attempts", []): ledger.add_attempt(AttemptUsage(**row))
        for row in core.get("applications", []): ledger.apply_response(ResponseApplication(**row))
        for row in core.get("effects", []):
            row = dict(row); row["causal_application_ids"] = tuple(row["causal_application_ids"]); ledger.commit_effect(CommittedEffect(**row))
        return ledger

    @property
    def ledger_sha256(self) -> str: return self.to_payload()["ledger_sha256"]

    def application(self, application_id: str) -> ResponseApplication:
        try: return self._applications[application_id]
        except KeyError as exc: raise ValueError("Unknown durable response application") from exc

    def request(self, logical_request_id: str) -> LogicalRequest:
        try: return self._requests[logical_request_id]
        except KeyError as exc: raise ValueError("Unknown durable logical request") from exc

    def effect(self, effect_id: str) -> CommittedEffect:
        try: return self._effects[effect_id]
        except KeyError as exc: raise ValueError("Unknown durable committed effect") from exc

    def snapshot(self, direct_latency_seconds: float) -> AccountingSnapshot:
        missing = tuple(x.attempt_id for x in self._attempts.values() if not x.usage_complete); complete = not missing
        inputs = sum(x.input_tokens or 0 for x in self._attempts.values()) if complete else None; outputs = sum(x.output_tokens or 0 for x in self._attempts.values()) if complete else None
        causal_app_ids = {a for effect in self._effects.values() if effect.substantive for a in effect.causal_application_ids}
        causal_attempt_ids = {app.attempt_id for aid, app in self._applications.items() if aid in causal_app_ids and app.accepted and self._requests[app.logical_request_id].provider == "qwen"}
        causal_attempts = [self._attempts[x] for x in causal_attempt_ids]; effective = None if any(not x.usage_complete for x in causal_attempts) else sum(x.output_tokens or 0 for x in causal_attempts)
        core = {"direct_latency_seconds": direct_latency_seconds, "input_tokens": inputs, "output_tokens": outputs, "total_tokens": None if inputs is None or outputs is None else inputs + outputs, "usage_complete": complete, "missing_usage_attempt_ids": missing, "qwen_effective_output_tokens": effective, "attempt_ids": tuple(self._attempts), "application_ids": tuple(self._applications), "committed_effect_ids": tuple(x.effect_id for x in self._effects.values() if x.substantive), "ledger_sha256": self.ledger_sha256}
        return AccountingSnapshot(**core, snapshot_sha256=sha256_json(core))
