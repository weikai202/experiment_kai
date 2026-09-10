"""Read-only verification of separately produced, manifest-bound preflights."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from pydantic import Field, model_validator

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.run import Digest, FrozenRunRecord, ResolvedRunManifest


PreflightKind = Literal[
    "qwen", "embedding", "user_simulator", "dataset_manifest",
    "fixture_mode", "checkpoint_filesystem",
]


class PreflightEvidence(FrozenRunRecord):
    schema_version: Literal[1] = 1
    kind: PreflightKind
    manifest_sha256: Digest
    configuration_sha256: Digest
    status: Literal["pass", "fail"]
    checked_at_utc: datetime
    setup_latency_seconds: Annotated[float, Field(ge=0)]
    returned_model: str | None = None
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    total_tokens: Annotated[int, Field(ge=0)] | None = None
    usage_complete: bool
    response_sha256: Digest

    @model_validator(mode="after")
    def coherent(self):
        if self.checked_at_utc.utcoffset() != timezone.utc.utcoffset(self.checked_at_utc):
            raise ValueError("preflight timestamp must be UTC")
        model_kinds = {"qwen", "embedding", "user_simulator"}
        if (self.returned_model is not None) != (self.kind in model_kinds):
            raise ValueError("model identity required exactly for model preflights")
        complete = all(value is not None for value in (
            self.input_tokens, self.output_tokens, self.total_tokens,
        ))
        if self.usage_complete != complete:
            raise ValueError("setup usage completeness mismatch")
        if self.kind not in model_kinds and (self.usage_complete or any(
            value is not None for value in (
                self.input_tokens, self.output_tokens, self.total_tokens,
            )
        )):
            raise ValueError("non-model preflight cannot report provider usage")
        return self


class PreflightGateResult(FrozenRunRecord):
    schema_version: Literal[1] = 1
    manifest_sha256: Digest
    evidence_sha256: Digest
    status: Literal["pass"] = "pass"
    setup_totals_excluded_from_experiment: Literal[True] = True


REQUIRED_KINDS = (
    "qwen", "embedding", "user_simulator", "dataset_manifest",
    "fixture_mode", "checkpoint_filesystem",
)


def verify_preflight_gate(
    manifest: ResolvedRunManifest,
    evidence: tuple[PreflightEvidence, ...],
    *,
    now_utc: datetime,
    maximum_age: timedelta = timedelta(hours=1),
) -> PreflightGateResult:
    if now_utc.utcoffset() != timezone.utc.utcoffset(now_utc):
        raise ValueError("gate clock must be UTC")
    if tuple(item.kind for item in evidence) != REQUIRED_KINDS:
        raise ValueError("all preflights are required in fixed order")
    manifest_hash = manifest.manifest_sha256
    configuration = {
        "qwen": canonical_sha256(manifest.qwen.model_dump(mode="json")),
        "embedding": canonical_sha256(manifest.embedding.model_dump(mode="json")),
        "user_simulator": canonical_sha256(manifest.user_simulator.model_dump(mode="json")),
        "dataset_manifest": manifest.dataset_manifest_sha256,
        "fixture_mode": canonical_sha256(manifest.fixture.model_dump(mode="json")),
        "checkpoint_filesystem": canonical_sha256(manifest.checkpoint.model_dump(mode="json")),
    }
    expected_models = {
        "qwen": manifest.qwen.model,
        "embedding": manifest.embedding.model,
        "user_simulator": manifest.user_simulator.model,
    }
    for item in evidence:
        if item.status != "pass" or item.manifest_sha256 != manifest_hash:
            raise ValueError("failed or mismatched preflight")
        if item.configuration_sha256 != configuration[item.kind]:
            raise ValueError("preflight configuration drift")
        age = now_utc - item.checked_at_utc
        if age < timedelta(0) or age > maximum_age:
            raise ValueError("stale preflight")
        if item.kind in expected_models and item.returned_model != expected_models[item.kind]:
            raise ValueError("returned model identity mismatch")
    evidence_hash = canonical_sha256([
        "preflight-gate-evidence-v1",
        [item.model_dump(mode="json") for item in evidence],
    ])
    return PreflightGateResult(
        manifest_sha256=manifest_hash,
        evidence_sha256=evidence_hash,
    )


__all__ = [
    "PreflightEvidence", "PreflightGateResult", "REQUIRED_KINDS",
    "verify_preflight_gate",
]
