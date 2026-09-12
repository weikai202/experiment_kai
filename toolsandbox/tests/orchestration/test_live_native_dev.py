from types import SimpleNamespace
import pytest
from toolsandbox_pipeline.orchestration.live_native_dev import branch_skills, _AuthorizedDevGate, NativeDevBranchExecutor
from toolsandbox_pipeline.orchestration.live_dev_minibench import NativeDevBranchRequest
from toolsandbox_pipeline.schemas.offline_skill import SkillContent
from tests.offline_skill.test_orchestrator import current_skill
from tests.orchestration.test_live_episode import executor, record, D
from toolsandbox_pipeline.reproducibility.dataset_access import ScenarioLease
from toolsandbox_pipeline.schemas.trajectory import EpisodeIdentity


def request():
    skill = current_skill()
    content = SkillContent.from_record(skill).model_copy(update={'instruction': 'Check visible evidence before acting'})
    return NativeDevBranchRequest(run_id='run',round_index=1,unit_id='unit',scenario_id='family',episode_id='dev-branch',
        branch='candidate',evaluated_skill=content,evaluated_skill_version='v1.1',earlier_accepted_skills=(),
        shared_configuration_sha256=D,dev_manifest_sha256=D)


def test_branch_overlay_preserves_statistics_and_original_snapshot():
    original = current_skill()
    result, = branch_skills(SimpleNamespace(skills=(original,)), request())
    assert result.version == 'v1.1' and result.instruction == request().evaluated_skill.instruction
    assert result.online_statistics == original.online_statistics
    assert original.version == 'v1.0'


def test_dev_gate_cannot_load_train_or_test():
    gate = _AuthorizedDevGate(SimpleNamespace(load=lambda **_: pytest.fail('must fail before data access')))
    for split in ('train','test'):
        with pytest.raises(PermissionError):
            gate.load(split=split,purpose='skill_ab_validation')


def test_dev_entry_uses_independent_marker_and_noneligible_native_path(monkeypatch):
    service, calls, _, _ = executor(monkeypatch)
    fields = service._identity(record(),0).model_dump(mode='json')
    fields.update(phase='dev_minibench',episode_id='dev-branch')
    identity = EpisodeIdentity.model_validate(fields)
    seen=[]
    monkeypatch.setattr(service,'_run_lease',lambda lease, identity, **kwargs: seen.append(kwargs) or 'native-result')
    args=dict(lease=ScenarioLease(record(),object()),identity=identity,authorization=lambda **_:None,
              purpose='skill_ab_validation',dev_manifest_sha256=D)
    assert service.run_dev_lease(**args) == 'native-result'
    assert seen == [{'offline_eligible':False}]
    with pytest.raises(RuntimeError,match='recovery'):
        service.run_dev_lease(**args)
    assert any(key.startswith('live-dev-dispatch-') for kind,key in calls if kind=='capture_checkpoint')


def test_dev_entry_rejects_wrong_purpose_before_native(monkeypatch):
    service, calls, _, _ = executor(monkeypatch)
    fields=service._identity(record(),0).model_dump(mode='json')
    fields.update(phase='dev_minibench')
    with pytest.raises(PermissionError):
        service.run_dev_lease(lease=ScenarioLease(record(),object()),identity=EpisodeIdentity.model_validate(fields),
            authorization=lambda **_:pytest.fail('authorization should not run'),purpose='train_round',dev_manifest_sha256=D)
    assert calls == []


def test_selector_metadata_permission_precedes_manifest_or_native_access():
    from toolsandbox_pipeline.orchestration.live_native_dev import NativeDevSelectorMetadataLoader
    deny=lambda *args,**kwargs:pytest.fail('source accessed before authorization')
    loader=NativeDevSelectorMetadataLoader(run_id='run',dev_manifest_sha256=D,manifest_loader=deny,base_family_loader=deny,authorize=deny,audit=deny)
    with pytest.raises(PermissionError):
        loader.load(run_id='run',phase='train_round',split='dev',purpose='skill_ab_validation',manifest_sha256=D)


def test_selector_returns_only_manifest_bound_public_metadata():
    from hashlib import sha256
    from toolsandbox_pipeline.orchestration.live_native_dev import NativeDevSelectorMetadataLoader
    from tests.reproducibility.test_dataset_manifest import blobs
    bundle=blobs()
    raw=bundle['dev_manifest.json']
    digest='sha256:'+sha256(raw).hexdigest()
    calls=[]
    class Base:
        starting_context=SimpleNamespace(tool_allow_list=['search_stock','end_conversation'])
        @property
        def evaluation(self):pytest.fail('hidden evaluator access')
    loader=NativeDevSelectorMetadataLoader(run_id='run',dev_manifest_sha256=digest,manifest_loader=lambda:raw,
        base_family_loader=lambda family:calls.append(family) or Base(),authorize=lambda **_:None,audit=lambda payload:calls.append(payload))
    result=loader.load(run_id='run',phase='dev_minibench',split='dev',purpose='skill_ab_validation',manifest_sha256=digest)
    assert result.families and result.manifest_sha256==digest
    assert all(f.necessary_canonical_tools==('search_stock','end_conversation') for f in result.families)
    assert set(result.families[0].__dict__)=={'family_id','split','necessary_canonical_tools','expanded_scenario_ids'}
    assert calls[-1]['family_count']==len(result.families)
