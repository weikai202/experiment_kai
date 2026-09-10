from datetime import datetime, timedelta, timezone

import pytest

from toolsandbox_pipeline.orchestration.preflight_gate import (
    PreflightEvidence,
    REQUIRED_KINDS,
    verify_preflight_gate,
)
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest

from .test_run_manifest import HASH, manifest_payload


def evidence(manifest, now):
    configurations = {
        "qwen": canonical_sha256(manifest.qwen.model_dump(mode="json")),
        "embedding": canonical_sha256(manifest.embedding.model_dump(mode="json")),
        "user_simulator": canonical_sha256(manifest.user_simulator.model_dump(mode="json")),
        "dataset_manifest": manifest.dataset_manifest_sha256,
        "fixture_mode": canonical_sha256(manifest.fixture.model_dump(mode="json")),
        "checkpoint_filesystem": canonical_sha256(manifest.checkpoint.model_dump(mode="json")),
    }
    models = {
        "qwen": manifest.qwen.model,
        "embedding": manifest.embedding.model,
        "user_simulator": manifest.user_simulator.model,
    }
    rows = []
    for kind in REQUIRED_KINDS:
        model = models.get(kind)
        rows.append(PreflightEvidence(
            kind=kind,
            manifest_sha256=manifest.manifest_sha256,
            configuration_sha256=configurations[kind],
            status="pass",
            checked_at_utc=now,
            setup_latency_seconds=0.25,
            returned_model=model,
            input_tokens=1 if model else None,
            output_tokens=1 if model else None,
            total_tokens=2 if model else None,
            usage_complete=model is not None,
            response_sha256=HASH,
        ))
    return tuple(rows)


def test_all_six_fresh_manifest_bound_preflights_pass_and_exclude_setup(tmp_path):
    manifest = ResolvedRunManifest.model_validate(manifest_payload(tmp_path), strict=True)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = verify_preflight_gate(manifest, evidence(manifest, now), now_utc=now)
    assert result.status == "pass"
    assert result.setup_totals_excluded_from_experiment is True


@pytest.mark.parametrize("failure", ["missing", "failed", "stale", "model", "config"])
def test_gate_fails_closed_without_implicit_probe(tmp_path, failure):
    manifest = ResolvedRunManifest.model_validate(manifest_payload(tmp_path), strict=True)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = list(evidence(manifest, now))
    if failure == "missing":
        rows.pop()
    elif failure == "failed":
        rows[0] = rows[0].model_copy(update={"status": "fail"})
    elif failure == "stale":
        rows[0] = rows[0].model_copy(update={"checked_at_utc": now - timedelta(hours=2)})
    elif failure == "model":
        rows[0] = rows[0].model_copy(update={"returned_model": "wrong"})
    else:
        rows[0] = rows[0].model_copy(update={"configuration_sha256": HASH})
    with pytest.raises(ValueError):
        verify_preflight_gate(manifest, tuple(rows), now_utc=now)
