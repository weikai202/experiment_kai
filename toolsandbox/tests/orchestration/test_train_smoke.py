import json
from pathlib import Path

import pytest

from toolsandbox_pipeline.orchestration.run_manifest import (
    RunManifestError,
    load_train_smoke_config,
    validate_train_smoke,
)
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest

from .test_run_manifest import manifest_payload


CONFIG = Path(__file__).parents[2] / "configs/run/train_smoke_v1.json"


def test_checked_in_smoke_is_fixed_train_fixture_nonformal(tmp_path):
    config, digest = load_train_smoke_config(CONFIG.resolve())
    manifest = ResolvedRunManifest.model_validate(manifest_payload(tmp_path), strict=True)
    validate_train_smoke(manifest, config)
    assert config.split == "train" and config.formal is False
    assert config.publish_formal_generation is False
    assert config.selection.reselection_from_outcomes_forbidden is True
    assert config.length_finish_reason_policy == "calibration_required"
    assert digest.startswith("sha256:")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("split",), "test"),
        (("publish_formal_generation",), True),
        (("selection", "family_ordinal"), 1),
        (("qwen_enable_thinking",), True),
    ],
)
def test_smoke_drift_fails_before_runtime(tmp_path, path, value):
    payload = json.loads(CONFIG.read_bytes())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunManifestError, match="invalid immutable"):
        load_train_smoke_config(drifted.resolve())
