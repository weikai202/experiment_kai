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


def v2_gate(tmp_path, *, verified=True):
    access,manifest,audit,constructed,kwargs=gate(tmp_path)
    index_path=access.manifest_path.parent/'dataset_index.json'
    index=json.loads(index_path.read_bytes())
    # Synthetic fixture preserves the real setup-only build status.
    evidence=dict(version='dataset-build-execution-environment-v2',dataset_index_sha256=file_hash(index_path.read_bytes()),
        dataset_build_status=index['status'],dataset_build_environment=index['environment'],
        dataset_build_environment_sha256=index['environment_sha256'])
    calls=[]
    def verifier(value,**binding):
        calls.append((value,binding));return verified
    access=DatasetAccessGate(manifest_path=access.manifest_path,expected_manifest_sha256=access.expected_manifest_sha256,
        reconstruct=access.reconstruct,audit=access.audit,hash_scenario=access.hash_scenario,
        execution_attestation=evidence,verify_execution_attestation=verifier)
    return access,manifest,audit,constructed,kwargs,calls


def test_v2_verified_execution_allows_setup_build_but_preserves_manifest_and_shard(tmp_path):
    access,manifest,audit,constructed,kwargs,calls=v2_gate(tmp_path)
    sid=manifest.scenarios[0].scenario_id
    assert access.load(requested_ids=(sid,),purpose='train_round',train_shard=0,**kwargs)
    assert constructed==[sid] and calls[0][1]['dataset_index_path'].name=='dataset_index.json'
    index=json.loads((access.manifest_path.parent/'dataset_index.json').read_bytes())
    assert index['status']=='setup_only'


@pytest.mark.parametrize('damage',['test','hash','evidence','verifier','shard'])
def test_v2_cannot_bypass_split_manifest_or_runtime_validation(tmp_path,damage):
    access,manifest,audit,constructed,kwargs,calls=v2_gate(tmp_path,verified=damage!='verifier')
    sid=manifest.scenarios[0].scenario_id
    kwargs.update(purpose='train_round',train_shard=0)
    if damage=='test':kwargs['split']='test'
    elif damage=='hash':kwargs['manifest_sha256']='sha256:'+'b'*64
    elif damage=='evidence':access.execution_attestation['dataset_index_sha256']='sha256:'+'b'*64
    elif damage=='shard':kwargs['train_shard']=2
    with pytest.raises(ValueError):access.load(requested_ids=(sid,),**kwargs)
    assert not constructed and not audit[-1]['accepted']


def test_original_gate_still_rejects_setup_only_formal_train(tmp_path):
    access,manifest,audit,constructed,kwargs=gate(tmp_path)
    with pytest.raises(ValueError):
        access.load(requested_ids=(manifest.scenarios[0].scenario_id,),purpose='train_round',train_shard=0,**kwargs)
    assert not constructed
