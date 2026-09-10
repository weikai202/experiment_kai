import json
import pytest
from toolsandbox_pipeline.schemas.dataset import SplitManifest
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.reproducibility.dataset_manifest import publish_bundle
from toolsandbox_pipeline.reproducibility.dataset_access import DatasetAccessGate
from tests.reproducibility.test_dataset_manifest import blobs, synthetic_scenario, synthetic_hashes


def gate(tmp_path, split="train"):
    bundle = blobs()
    publish_bundle(tmp_path / "private", bundle)
    path = tmp_path / "private" / f"{split}_manifest.json"
    digest = file_hash(path.read_bytes())
    manifest = SplitManifest.model_validate_json(path.read_bytes())
    audit, constructed = [], []
    def reconstruct(record):
        constructed.append(record.scenario_id)
        return synthetic_scenario(record.variant)
    access = DatasetAccessGate(manifest_path=path, expected_manifest_sha256=digest,
                               reconstruct=reconstruct, audit=audit.append, hash_scenario=synthetic_hashes)
    return access, manifest, audit, constructed, dict(run_id="synthetic-run", phase="unit", manifest_sha256=digest)


def test_train_default_and_lease_isolation(tmp_path):
    access, manifest, audit, constructed, kwargs = gate(tmp_path)
    ids = tuple(s.scenario_id for s in manifest.scenarios[:2])
    leases = access.load(requested_ids=ids, **kwargs)
    assert constructed == list(ids) and audit[0]["accepted"]
    copy = leases[0].scenario
    copy.mutable.append(1)
    assert leases[0].scenario.mutable == []
    assert all(sid not in json.dumps(audit) for sid in ids)


@pytest.mark.parametrize("damage", ["test", "purpose", "hash", "duplicate", "unknown", "order", "shard"])
def test_rejection_before_content(tmp_path, damage):
    access, manifest, audit, constructed, kwargs = gate(tmp_path)
    ids = tuple(s.scenario_id for s in manifest.scenarios[:2])
    if damage == "test": kwargs["split"] = "test"
    elif damage == "purpose": kwargs["purpose"] = "skill_ab_validation"
    elif damage == "hash": kwargs["manifest_sha256"] = "sha256:" + "b" * 64
    elif damage == "duplicate": ids = (ids[0], ids[0])
    elif damage == "unknown": ids = ("unknown",)
    elif damage == "order": ids = tuple(reversed(ids))
    elif damage == "shard": kwargs.update(purpose="train_round", train_shard=2)
    with pytest.raises(ValueError):
        access.load(requested_ids=ids, **kwargs)
    assert not constructed and audit[-1]["accepted"] is False


def test_dev_requires_validation_purpose(tmp_path):
    access, manifest, audit, constructed, kwargs = gate(tmp_path, "dev")
    ids = (manifest.scenarios[0].scenario_id,)
    with pytest.raises(ValueError):
        access.load(requested_ids=ids, split="dev", **kwargs)
    assert len(access.load(requested_ids=ids, split="dev", purpose="skill_ab_validation", **kwargs)) == 1


def test_hash_mismatch_returns_no_partial_leases(tmp_path):
    access, manifest, audit, _, kwargs = gate(tmp_path)
    access.hash_scenario = lambda _: {}
    with pytest.raises(ValueError):
        access.load(requested_ids=(manifest.scenarios[0].scenario_id,), **kwargs)
    assert audit[-1]["accepted"] is False
