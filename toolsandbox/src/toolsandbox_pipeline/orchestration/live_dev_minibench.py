"""Injected native Dev A/B driver; no data/model access during construction."""
from dataclasses import dataclass
from copy import deepcopy
import re
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from toolsandbox_pipeline.offline.dev_selector import DevFamilySelectorView, select_dev_scenarios
from toolsandbox_pipeline.offline.dev_minibench import evaluate_dev_branches
from toolsandbox_pipeline.offline.skill_orchestrator import MiniBenchExecution
from toolsandbox_pipeline.offline.skill_rewrite import next_skill_version, validate_skill_candidate
from toolsandbox_pipeline.reproducibility import canonical_sha256, canonical_json_bytes
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.offline_skill import DevBranchResult, SkillContent
from toolsandbox_pipeline.schemas.skill import SkillRecord


@dataclass(frozen=True)
class DevSelectorMetadata:
    manifest_sha256: str
    families: tuple[DevFamilySelectorView, ...]


class NativeDevBranchRequest(StrictModel):
    model_config = ConfigDict(frozen=True)
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=0, le=2)
    unit_id: str = Field(min_length=1)
    phase: Literal['dev_minibench'] = 'dev_minibench'
    scenario_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    branch: Literal['previous', 'candidate']
    evaluated_skill: SkillContent
    evaluated_skill_version: str
    earlier_accepted_skills: tuple[SkillRecord, ...]
    shared_configuration_sha256: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    dev_manifest_sha256: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')

    @property
    def input_sha256(self):
        return canonical_sha256(self.model_dump(mode='json'))


class StoredDevBranch(StrictModel):
    model_config = ConfigDict(frozen=True)
    input_sha256: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    result: DevBranchResult
    ordered_qwen_application_ids: tuple[str, ...] = ()

    @model_validator(mode='after')
    def unique_applications(self):
        values = self.ordered_qwen_application_ids
        if len(values) != len(set(values)) or any(not value or any(c.isspace() for c in value) for value in values):
            raise ValueError('unique nonempty Qwen application IDs required')
        return self


class LedgerDevBranchStore:
    """Immutable Task011 checkpoints; branch outputs contain no dev trajectories."""
    def __init__(self, ledger):
        self.ledger = ledger

    def bind_unit(self, unit_id, identity):
        if identity.get('run_id') != self.ledger.store.identity.run_id:
            raise ValueError('Dev branch store run mismatch')
        cid = 'dev-unit-' + canonical_sha256([self.ledger.store.identity.run_id, unit_id])[7:]
        self.ledger.commit_checkpoint(cid, 'dev_unit_identity', identity)

    def load(self, request):
        if request.run_id != self.ledger.store.identity.run_id:
            raise ValueError('Dev branch store run mismatch')
        event = self.ledger.get_checkpoint('dev-branch-' + request.episode_id)
        if event is None:
            return None
        record = StoredDevBranch.model_validate_json(canonical_json_bytes(event.payload))
        if record.input_sha256 != request.input_sha256:
            raise ValueError('stored Dev branch input mismatch')
        return record

    def commit(self, request, record):
        if request.run_id != self.ledger.store.identity.run_id or record.input_sha256 != request.input_sha256:
            raise ValueError('Dev branch store identity mismatch')
        self.ledger.commit_checkpoint('dev-branch-' + request.episode_id, 'dev_branch_completed', record.model_dump(mode='json'))
        return record


class LiveDevMiniBenchExecutor:
    """Runs only an already-generated candidate against its exact previous Skill.

    metadata_loader.load returns DevSelectorMetadata for the requested manifest.
    dev_gate is the existing DatasetAccessGate interface. branch_executor.run_branch
    must execute fresh native episodes, bind request.phase/episode/round to its
    provider ledger, and return StoredDevBranch only after the native evaluator and
    final episode checkpoint. Its application IDs remain provenance until the
    outer Skill orchestrator commits an accepted mutation.
    """
    def __init__(self, *, run_id, round_index, dev_manifest_sha256,
                 shared_configuration_sha256, public_tool_inventory,
                 metadata_loader, dev_gate, branch_executor, branch_store):
        if (type(run_id) is not str or not run_id or any(c.isspace() for c in run_id)
                or any(type(value) is not str or re.fullmatch(r'sha256:[0-9a-f]{64}', value) is None
                       for value in (dev_manifest_sha256, shared_configuration_sha256))):
            raise ValueError('explicit Dev run and manifest identities required')
        if type(round_index) is not int or round_index not in (0, 1, 2):
            raise ValueError('invalid Dev producing round')
        for dependency, methods in ((metadata_loader, ('load',)), (dev_gate, ('load',)),
                                    (branch_executor, ('run_branch',)),
                                    (branch_store, ('bind_unit', 'load', 'commit'))):
            if any(not callable(getattr(dependency, method, None)) for method in methods):
                raise ValueError('explicit Dev metadata/gate/native driver/durability required')
        self.run_id, self.round_index = run_id, round_index
        self.dev_manifest_sha256 = dev_manifest_sha256
        self.shared_configuration_sha256 = shared_configuration_sha256
        self.public_tool_inventory = public_tool_inventory
        self.metadata_loader, self.dev_gate = metadata_loader, dev_gate
        self.branch_executor, self.branch_store = branch_executor, branch_store

    def evaluate(self, *, current_skill, candidate, earlier_accepted_skills, unit_id):
        if type(current_skill) is not SkillRecord or type(candidate) is not SkillContent:
            raise TypeError('validated previous Skill and generated candidate required')
        if current_skill.status != 'active' or type(unit_id) is not str or not unit_id or any(c.isspace() for c in unit_id):
            raise ValueError('active Skill and stable unit ID required')
        current_skill = SkillRecord.model_validate_json(current_skill.model_dump_json())
        candidate = SkillContent.model_validate_json(candidate.model_dump_json())
        validate_skill_candidate(current=current_skill, candidate=candidate, public_tool_inventory=self.public_tool_inventory)
        if (type(earlier_accepted_skills) is not tuple
                or any(type(skill) is not SkillRecord or skill.status != 'active' for skill in earlier_accepted_skills)
                or len({skill.skill_id for skill in earlier_accepted_skills}) != len(earlier_accepted_skills)
                or any(skill.skill_id == current_skill.skill_id for skill in earlier_accepted_skills)):
            raise ValueError('distinct active earlier accepted Skills required')
        earlier_accepted_skills = tuple(SkillRecord.model_validate_json(skill.model_dump_json()) for skill in earlier_accepted_skills)
        common = canonical_sha256({'configuration': self.shared_configuration_sha256,
            'earlier_accepted_skills': [skill.model_dump(mode='json') for skill in earlier_accepted_skills]})
        self.branch_store.bind_unit(unit_id, dict(run_id=self.run_id, round_index=self.round_index,
            dev_manifest_sha256=self.dev_manifest_sha256, shared_configuration_sha256=common,
            previous=current_skill.model_dump(mode='json'), candidate=candidate.model_dump(mode='json')))
        metadata = self.metadata_loader.load(run_id=self.run_id, phase='dev_minibench', split='dev',
            purpose='skill_ab_validation', manifest_sha256=self.dev_manifest_sha256)
        if (type(metadata) is not DevSelectorMetadata or metadata.manifest_sha256 != self.dev_manifest_sha256
                or type(metadata.families) is not tuple
                or any(type(family) is not DevFamilySelectorView for family in metadata.families)):
            raise ValueError('verified Dev selector metadata required')
        selection = select_dev_scenarios(data_seed=0, skill_id=current_skill.skill_id,
            tool_dependencies=current_skill.tool_dependencies, dataset_manifest_sha256=self.dev_manifest_sha256,
            families=metadata.families)
        results, applications = [], []
        for scenario_id in selection.selected_scenario_ids:
            for branch, content, version in (
                ('previous', SkillContent.from_record(current_skill), current_skill.version),
                ('candidate', candidate, next_skill_version(current_skill.version)),
            ):
                episode_id = 'dev-' + canonical_sha256([self.run_id, self.round_index, unit_id, scenario_id, branch])[7:]
                request = NativeDevBranchRequest(run_id=self.run_id, round_index=self.round_index,
                    unit_id=unit_id, scenario_id=scenario_id, episode_id=episode_id, branch=branch,
                    evaluated_skill=content, evaluated_skill_version=version,
                    earlier_accepted_skills=earlier_accepted_skills, shared_configuration_sha256=common,
                    dev_manifest_sha256=self.dev_manifest_sha256)
                record = self.branch_store.load(request)
                if record is None:
                    leases = self.dev_gate.load(requested_ids=(scenario_id,), run_id=self.run_id, phase='dev_minibench',
                        manifest_sha256=self.dev_manifest_sha256, split='dev', purpose='skill_ab_validation')
                    if len(leases) != 1 or leases[0].record.scenario_id != scenario_id:
                        raise ValueError('Dev gate returned a different scenario')
                    record = self.branch_executor.run_branch(request=request, scenario=deepcopy(leases[0].scenario))
                    self._validate(request, record)
                    committed = self.branch_store.commit(request, record)
                    if committed != record:
                        raise ValueError('Dev branch commit mismatch')
                self._validate(request, record)
                results.append(record.result)
                applications.extend(record.ordered_qwen_application_ids)
        if len(applications) != len(set(applications)):
            raise ValueError('Qwen application reused across Dev episodes')
        return MiniBenchExecution(evaluate_dev_branches(selection, tuple(results)), tuple(applications))

    @staticmethod
    def _validate(request, record):
        if type(record) is not StoredDevBranch:
            raise TypeError('durable native Dev branch record required')
        record = StoredDevBranch.model_validate_json(record.model_dump_json())
        result = record.result
        if (record.input_sha256 != request.input_sha256
                or result.scenario_id != request.scenario_id or result.branch != request.branch
                or result.episode_id != request.episode_id
                or result.evaluated_skill_id != request.evaluated_skill.skill_id
                or result.evaluated_skill_version != request.evaluated_skill_version
                or result.shared_configuration_sha256 != request.shared_configuration_sha256):
            raise ValueError('native Dev branch identity mismatch')


__all__ = ["DevSelectorMetadata", "NativeDevBranchRequest", "StoredDevBranch",
           "LedgerDevBranchStore", "LiveDevMiniBenchExecutor"]
