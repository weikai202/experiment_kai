"""Ordinary access accepts purpose-authorized train/dev only; test is absent."""
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.schemas.dataset import DatasetAccessRequest, DatasetIndex, ScenarioRecord, SplitManifest


class DatasetAccessError(ValueError):
    pass


@dataclass(frozen=True)
class ScenarioLease:
    record: ScenarioRecord
    _scenario: object = field(repr=False)

    @property
    def scenario(self):
        return copy.deepcopy(self._scenario)

    def __reduce_ex__(self, protocol):
        raise TypeError("host-only Scenario lease cannot be serialized")


class DatasetAccessGate:
    def __init__(self, *, manifest_path, expected_manifest_sha256, reconstruct, audit, hash_scenario=None,
                 execution_attestation=None, verify_execution_attestation=None):
        self.manifest_path = Path(manifest_path)
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.reconstruct = reconstruct
        self.audit = audit
        if (execution_attestation is None) != (verify_execution_attestation is None):
            raise TypeError("v2 execution evidence and verifier must be supplied together")
        if execution_attestation is not None and (type(execution_attestation) is not dict or not callable(verify_execution_attestation)):
            raise TypeError("explicit v2 execution evidence and callable verifier required")
        self.execution_attestation = copy.deepcopy(execution_attestation)
        self.verify_execution_attestation = verify_execution_attestation
        if hash_scenario is None:
            from .scenario_hashes import scenario_hashes
            hash_scenario = scenario_hashes
        self.hash_scenario = hash_scenario

    def load(self, *, requested_ids, run_id, phase, manifest_sha256, split="train", purpose="development", train_shard=None):
        accepted = False
        ids = list(requested_ids) if type(requested_ids) in (tuple, list) and all(type(v) is str for v in requested_ids) else []
        try:
            request = DatasetAccessRequest(split=split, purpose=purpose, requested_ids=tuple(ids), run_id=run_id, phase=phase,
                                           manifest_sha256=manifest_sha256, train_shard=train_shard)
            # Reject test and mismatched path/purpose before reading any manifest bytes.
            path = self.manifest_path
            if path.name != request.split + "_manifest.json" or not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
                raise DatasetAccessError("invalid split manifest path")
            if manifest_sha256 != self.expected_manifest_sha256:
                raise DatasetAccessError("unexpected manifest identity")
            raw = path.read_bytes()
            if file_hash(raw) != manifest_sha256:
                raise DatasetAccessError("manifest content identity mismatch")
            manifest = SplitManifest.model_validate_json(raw)
            if manifest.split != request.split:
                raise DatasetAccessError("manifest split mismatch")
            if request.purpose == "train_round":
                index_path = path.parent / "dataset_index.json"
                if index_path.is_symlink():
                    raise DatasetAccessError("invalid dataset index path")
                index_raw = index_path.read_bytes()
                index = DatasetIndex.model_validate_json(index_raw)
                if index.train_manifest_sha256 != manifest_sha256 or index.environment_sha256 != manifest.environment_sha256:
                    raise DatasetAccessError("formal train manifest/build environment mismatch")
                evidence = self.execution_attestation
                if evidence is None:
                    if index.status != "complete":
                        raise DatasetAccessError("formal train requires a complete pinned environment")
                else:
                    if (evidence.get('version') != 'dataset-build-execution-environment-v2'
                            or evidence.get('dataset_index_sha256') != file_hash(index_raw)
                            or evidence.get('dataset_build_status') != index.status
                            or evidence.get('dataset_build_environment_sha256') != index.environment_sha256
                            or evidence.get('dataset_build_environment') != index.environment.model_dump(mode='json')):
                        raise DatasetAccessError("v2 execution evidence does not match immutable dataset build")
                    if self.verify_execution_attestation(copy.deepcopy(evidence), dataset_index_path=index_path,
                            expected_dataset_index_sha256=file_hash(index_raw)) is not True:
                        raise DatasetAccessError("current v2 execution environment is not verified")
            by_id = {r.scenario_id: r for r in manifest.scenarios}
            chosen = set(ids)
            if any(sid not in by_id for sid in ids) or tuple(r.scenario_id for r in manifest.scenarios if r.scenario_id in chosen) != request.requested_ids:
                raise DatasetAccessError("unknown or out-of-order scenario selection")
            families = {f.family_id: f for f in manifest.families}
            if request.train_shard is not None and any(families[by_id[sid].scenario_family_id].train_shard != request.train_shard for sid in ids):
                raise DatasetAccessError("cross-shard scenario selection")
            leases = []
            for sid in ids:
                record = by_id[sid]
                scenario = copy.deepcopy(self.reconstruct(record))
                expected = {name: getattr(record, name) for name in (
                    "starting_context_sha256", "evaluation_definition_sha256", "agent_facing_tool_names_sha256", "agent_facing_tool_schema_sha256")}
                if self.hash_scenario(scenario) != expected or tuple(str(v) for v in scenario.categories) != record.categories or scenario.max_messages != record.max_messages:
                    raise DatasetAccessError("reconstructed scenario identity mismatch")
                leases.append(ScenarioLease(record, scenario))
            accepted = True
            return tuple(leases)
        except Exception:
            raise DatasetAccessError("dataset access rejected") from None
        finally:
            self.audit(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                split=split if split in ("train", "dev", "test") else "invalid",
                purpose=purpose if purpose in ("development", "retrieval_smoke", "online_token_calibration", "train_round", "skill_ab_validation") else "invalid",
                run_id=run_id if type(run_id) is str else "invalid", phase=phase if type(phase) is str else "invalid",
                manifest_sha256=manifest_sha256 if type(manifest_sha256) is str and len(manifest_sha256) == 71 else None,
                requested_count=len(ids), requested_ids_sha256=canonical_sha256(ids), accepted=accepted))
