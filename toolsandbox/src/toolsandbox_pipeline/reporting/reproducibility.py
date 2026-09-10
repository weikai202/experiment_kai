"""Evidence-bound reproducibility checklist evaluation."""

from __future__ import annotations

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.reporting import (
    ReproducibilityCheck,
    ReproducibilityChecklist,
    StrictReplaySmokeEvidence,
)


class ReproducibilityError(ValueError):
    """Required profile evidence is absent or internally inconsistent."""


def evaluate_reproducibility(
    *,
    profile: str,
    checks: tuple[ReproducibilityCheck, ...],
    strict_replay_smoke_evidence: StrictReplaySmokeEvidence | None = None,
) -> ReproducibilityChecklist:
    """Derive conclusion and claim scope; never accept unsupported free text."""

    if profile not in {"strict_replay", "official_live"}:
        raise ReproducibilityError("unsupported final evaluation profile")
    by_id = {item.item_id: item for item in checks}
    if len(by_id) != len(checks):
        raise ReproducibilityError("duplicate reproducibility item")
    expected_ids = tuple(f"section24_item_{index:02d}" for index in range(1, 15))
    if tuple(item.item_id for item in checks) != expected_ids:
        raise ReproducibilityError("exact ordered Section 24 checklist required")
    for index, item in enumerate(checks, start=1):
        expected_required = index != 13 or profile == "strict_replay"
        if item.required != expected_required:
            raise ReproducibilityError("Section 24 requiredness cannot be downgraded")
    smoke_check = by_id["section24_item_13"]
    if profile == "strict_replay":
        if strict_replay_smoke_evidence is None:
            raise ReproducibilityError("strict-replay structural smoke evidence required")
        expected_status = "pass" if strict_replay_smoke_evidence.identical else "fail"
        if smoke_check.status != expected_status or smoke_check.evidence_sha256 != strict_replay_smoke_evidence.evidence_sha256:
            raise ReproducibilityError("strict-replay smoke check disagrees with structural evidence")
    elif (
        strict_replay_smoke_evidence is not None
        or smoke_check.status != "not_applicable_with_reason"
        or smoke_check.reason_code != "official_live_profile"
        or smoke_check.evidence_sha256 is not None
    ):
        raise ReproducibilityError("official-live Section 24 item 13 must be not applicable")
    conclusion = (
        "partially_reproducible"
        if any(item.required and item.status in {"fail", "not_run"} for item in checks)
        else "reproducible"
    )
    values = dict(
        profile=profile,
        checks=checks,
        conclusion=conclusion,
        claim_scope=(
            "trajectory_reproducible"
            if profile == "strict_replay"
            else "configuration_traceable_or_statistically_reproducible"
        ),
        fixed_user_model_deviation_disclosed=profile == "official_live",
    )
    values["checklist_sha256"] = canonical_sha256({
        "schema_version": 1,
        **{
            key: [item.model_dump(mode="json") for item in value]
            if key == "checks" else value
            for key, value in values.items()
        },
    })
    return ReproducibilityChecklist(**values)


__all__ = ["ReproducibilityError", "evaluate_reproducibility"]
