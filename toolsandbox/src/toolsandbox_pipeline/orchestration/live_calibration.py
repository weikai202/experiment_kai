"""Bounded natural train corpus collection; execution is explicitly injected.

The coordinator's trusted pilot captures requests that its normal Policy,
Controller, Critic and Revision routing actually made. This service never requests
an additional role to fill a quota and never promotes the resulting recommendation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from toolsandbox_pipeline.online.token_limit_calibration import (
    CalibrationRuntimeInputs, CapturedCalibrationRequest, calibrate_and_write,
    select_calibration_sample,
)
from toolsandbox_pipeline.online.token_limits import BOOTSTRAP, ROLES

PURPOSE = 'online_token_calibration'
PHASE = 'online_token_calibration_pilot'


@dataclass(frozen=True)
class PilotCapturedRequest:
    captured: CapturedCalibrationRequest
    logical_request_id: str
    source_attempt_id: str

    def __post_init__(self):
        if type(self.captured) is not CapturedCalibrationRequest:
            raise TypeError('exact captured calibration request required')
        if any(type(value) is not str or not value or any(c.isspace() for c in value)
               for value in (self.logical_request_id, self.source_attempt_id)):
            raise ValueError('durable pilot request identities required')


@dataclass(frozen=True)
class PilotReceipt:
    scenario_id: str
    scenario_family_id: str
    episode_id: str
    status: Literal['completed_evaluated', 'failed']
    requests: tuple[PilotCapturedRequest, ...] = field(repr=False)
    checkpoint_id: str

    def __post_init__(self):
        if self.status not in ('completed_evaluated', 'failed'):
            raise ValueError('explicit terminal pilot status required')
        if type(self.requests) is not tuple or any(type(r) is not PilotCapturedRequest for r in self.requests):
            raise TypeError('ordered captured pilot requests required')
        if any(type(value) is not str or not value for value in
               (self.scenario_id, self.scenario_family_id, self.episode_id, self.checkpoint_id)):
            raise ValueError('durable episode receipt identity required')


class CalibrationCollectionError(RuntimeError):
    """Sanitized failure with prior receipts retained for coordinator recovery."""
    def __init__(self, code, receipts, counts):
        self.code, self.receipts, self.role_counts = code, tuple(receipts), dict(counts)
        super().__init__(code)


@dataclass(frozen=True)
class CalibrationCollectionResult:
    receipts: tuple[PilotReceipt, ...] = field(repr=False)
    role_counts: dict[str, int]
    recommended_config: object
    artifact_hashes: dict[str, str]


def collect_and_calibrate(*, records, train_manifest_sha256, run_id, gate,
                          pilot_executor, persist_receipt, runtime_inputs, qwen,
                          hard_ceiling, invoke, output_directory):
    """Collect initial 32 + at most 32 reserve pilots before any replay.

    ``gate.load`` is the existing DatasetAccessGate interface. The authorized
    ``pilot_executor(lease=..., run_id=..., purpose=...)`` returns only naturally
    dispatched requests with durable source IDs. ``persist_receipt(receipt)`` must
    durably record each returned terminal receipt before collection continues.
    Callbacks own secret injection, authorization, request validity and accounting.
    """
    if type(records) is not tuple or type(runtime_inputs) is not CalibrationRuntimeInputs:
        raise TypeError('validated train records and runtime bindings required')
    if type(run_id) is not str or not run_id:
        raise ValueError('explicit calibration run ID required')
    root = Path(output_directory)
    if not root.is_absolute() or '..' in root.parts or root.exists() or not root.parent.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('new explicit calibration output directory required')
    if type(hard_ceiling) is not int or hard_ceiling < max(BOOTSTRAP) or (qwen.output_limit is not None and hard_ceiling > qwen.output_limit):
        raise ValueError('calibration hard ceiling is incompatible with server limits')
    sample = select_calibration_sample(records, train_manifest_sha256)
    receipts, captured = [], []
    role_fingerprints = {role: set() for role in ROLES}
    logical_ids, attempt_ids, episodes = set(), set(), set()
    fingerprint_requests = {}

    def counts():
        return {role: len(values) for role, values in role_fingerprints.items()}

    def fail(code):
        raise CalibrationCollectionError(code, receipts, counts())

    for index, scenario_id in enumerate((*sample.initial, *sample.reserve)):
        if index >= len(sample.initial) and all(value >= 32 for value in counts().values()):
            break
        try:
            leases = gate.load(requested_ids=(scenario_id,), run_id=run_id,
                phase=PHASE, manifest_sha256=train_manifest_sha256, split='train', purpose=PURPOSE)
            if len(leases) != 1 or leases[0].record.scenario_id != scenario_id:
                fail('pilot_lease_binding_mismatch')
            receipt = pilot_executor(lease=leases[0], run_id=run_id, purpose=PURPOSE)
            if type(receipt) is not PilotReceipt:
                fail('invalid_pilot_receipt')
            persist_receipt(receipt)
            receipts.append(receipt)
        except CalibrationCollectionError:
            raise
        except Exception:
            fail('pilot_collection_failed')
        if (receipt.scenario_id != scenario_id
            or receipt.scenario_family_id != leases[0].record.scenario_family_id
            or receipt.episode_id in episodes):
            fail('pilot_receipt_binding_mismatch')
        episodes.add(receipt.episode_id)
        if receipt.status != 'completed_evaluated':
            fail('pilot_did_not_complete')
        if not receipt.requests:
            fail('pilot_has_no_captured_requests')
        for request in receipt.requests:
            item = request.captured
            if (item.scenario_id != scenario_id or item.scenario_family_id != receipt.scenario_family_id
                or request.logical_request_id in logical_ids or request.source_attempt_id in attempt_ids
                or item.prepared.role not in role_fingerprints):
                fail('captured_request_binding_mismatch')
            logical_ids.add(request.logical_request_id)
            attempt_ids.add(request.source_attempt_id)
            key = (item.prepared.role, item.prepared.canonical_input_fingerprint)
            if key in fingerprint_requests and fingerprint_requests[key] != item.prepared:
                fail('captured_fingerprint_conflict')
            fingerprint_requests[key] = item.prepared
            fingerprints = role_fingerprints[item.prepared.role]
            if item.prepared.canonical_input_fingerprint not in fingerprints:
                fingerprints.add(item.prepared.canonical_input_fingerprint)
                captured.append(item)
    if any(value < 32 for value in counts().values()):
        fail('insufficient_natural_role_requests')
    # Validate every role before calibrate_and_write can replay even Policy once.
    for role in ROLES:
        corpus = [r.prepared for r in captured if r.prepared.role == role]
        identities = {(p.prompt_sha256, p.output_schema_sha256, p.generation_id) for p in corpus}
        if len(identities) != 1:
            fail('mixed_role_corpus_identity')
    if len({request.prepared.generation_id for request in captured}) != 1:
        fail('mixed_generation_corpus_identity')
    executed = tuple(receipt.scenario_id for receipt in receipts)
    if {request.scenario_id for request in captured} != set(executed):
        fail('deduplicated_corpus_missing_pilot')
    config, hashes = calibrate_and_write(tuple(captured), sample=sample,
        executed_scenario_ids=executed, runtime_inputs=runtime_inputs, qwen=qwen,
        hard_ceiling=hard_ceiling, invoke=invoke, output_directory=root)
    return CalibrationCollectionResult(tuple(receipts), counts(), config, hashes)
