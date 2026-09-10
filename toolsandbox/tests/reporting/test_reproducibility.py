import pytest

from toolsandbox_pipeline.reporting.reproducibility import (
    ReproducibilityError,
    evaluate_reproducibility,
)
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.reporting import ReproducibilityCheck, StrictReplaySmokeEvidence

D = "sha256:" + "4" * 64


def check(item_id, status="pass", required=True, evidence=D, reason=None):
    return ReproducibilityCheck(
        item_id=item_id, required=required, status=status,
        evidence_sha256=evidence, reason_code=reason,
    )


def checks(*, smoke_status="pass", smoke_evidence=D, official=False):
    values = []
    for index in range(1, 15):
        if index == 13 and official:
            values.append(check(
                "section24_item_13", status="not_applicable_with_reason",
                required=False, evidence=None, reason="official_live_profile",
            ))
        elif index == 13:
            values.append(check("section24_item_13", status=smoke_status, evidence=smoke_evidence))
        else:
            values.append(check(f"section24_item_{index:02d}"))
    return tuple(values)


def smoke(same=True):
    values = dict(
        first_final_context_hashes_sha256=D,
        second_final_context_hashes_sha256=D if same else "sha256:" + "5" * 64,
        first_native_scores_sha256=D, second_native_scores_sha256=D,
        first_retrieval_ids_sha256=D, second_retrieval_ids_sha256=D,
        first_request_output_hashes_sha256=D,
        second_request_output_hashes_sha256=D,
    )
    values["evidence_sha256"] = canonical_sha256(values)
    return StrictReplaySmokeEvidence(**values)


def test_required_not_run_forces_partial_reproducibility():
    result = evaluate_reproducibility(
        profile="strict_replay",
        checks=tuple(
            check(item.item_id, status="not_run", evidence=None)
            if item.item_id == "section24_item_14" else item
            for item in checks(smoke_evidence=smoke().evidence_sha256)
        ),
        strict_replay_smoke_evidence=smoke(),
    )
    assert result.conclusion == "partially_reproducible"
    assert result.claim_scope == "trajectory_reproducible"


def test_official_live_uses_traceability_claim_and_disclosure():
    result = evaluate_reproducibility(
        profile="official_live",
        checks=checks(official=True),
    )
    assert result.conclusion == "reproducible"
    assert result.claim_scope == "configuration_traceable_or_statistically_reproducible"
    assert result.fixed_user_model_deviation_disclosed


def test_profile_specific_evidence_is_mandatory():
    with pytest.raises(ReproducibilityError, match="exact ordered"):
        evaluate_reproducibility(profile="strict_replay", checks=())


def test_caller_cannot_downgrade_required_section_24_item():
    evidence = smoke()
    values = list(checks(smoke_evidence=evidence.evidence_sha256))
    values[0] = check("section24_item_01", required=False)
    with pytest.raises(ReproducibilityError, match="downgraded"):
        evaluate_reproducibility(
            profile="strict_replay", checks=tuple(values),
            strict_replay_smoke_evidence=evidence,
        )
