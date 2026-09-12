from types import SimpleNamespace

import pytest

from toolsandbox_pipeline.orchestration.live_dev_minibench import (
    DevSelectorMetadata, LiveDevMiniBenchExecutor, LedgerDevBranchStore, StoredDevBranch,
)
from toolsandbox_pipeline.offline.dev_selector import DevFamilySelectorView, select_dev_scenarios
from toolsandbox_pipeline.schemas.offline_skill import SkillContent, DevBranchResult
from toolsandbox_pipeline.reproducibility.dataset_access import ScenarioLease
from tests.offline_skill.test_orchestrator import current_skill
from tests.orchestration.test_live_offline import live, D


class Metadata:
    def __init__(self):
        self.calls = []
        self.families = (DevFamilySelectorView('family', 'dev', ('search_stock',), ('scenario-a','scenario-b')),)
    def load(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs['split'] == 'dev' and kwargs['purpose'] == 'skill_ab_validation'
        return DevSelectorMetadata(D, self.families)


class Gate:
    def __init__(self):
        self.calls = []
        self.initial = {'mutations': []}
    def load(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs['split'] == 'dev' and kwargs['purpose'] == 'skill_ab_validation'
        assert kwargs['phase'] == 'dev_minibench'
        sid, = kwargs['requested_ids']
        return (ScenarioLease(SimpleNamespace(scenario_id=sid), self.initial),)


class NativeDriver:
    def __init__(self):
        self.requests, self.scenarios = [], []
        self.fail_once = False
        self.wrong_identity = False
        self.incomplete = False
    def run_branch(self, *, request, scenario):
        assert not scenario['mutations']
        scenario['mutations'].append(request.episode_id)
        self.requests.append(request)
        self.scenarios.append(scenario)
        assert request.phase == 'dev_minibench'
        if self.fail_once and request.branch == 'candidate':
            self.fail_once = False
            raise RuntimeError('synthetic native driver interruption')
        complete = not self.incomplete
        result = DevBranchResult(
            scenario_id='wrong' if self.wrong_identity else request.scenario_id,
            branch=request.branch, episode_id=request.episode_id,
            evaluated_skill_id=request.evaluated_skill.skill_id,
            evaluated_skill_version=request.evaluated_skill_version,
            shared_configuration_sha256=request.shared_configuration_sha256,
            complete=complete,
            fully_successful=(request.branch == 'candidate') if complete else None,
            similarity=(1.0 if request.branch == 'candidate' else 0.0) if complete else None,
            minefield_hit=False if complete else None, evaluator_record_sha256=D if complete else None)
        return StoredDevBranch(input_sha256=request.input_sha256, result=result,
                               ordered_qwen_application_ids=('application-' + request.episode_id,))


def setup(live, **changes):
    metadata, gate, driver = Metadata(), Gate(), NativeDriver()
    values = dict(run_id='run', round_index=0, dev_manifest_sha256=D,
        shared_configuration_sha256=D, public_tool_inventory=('search_stock',),
        metadata_loader=metadata, dev_gate=gate, branch_executor=driver,
        branch_store=LedgerDevBranchStore(live.ledger))
    values.update(changes)
    service = LiveDevMiniBenchExecutor(**values)
    current = current_skill()
    content = SkillContent.from_record(current).model_dump()
    content['instruction'] = 'Check prerequisites before using verified inputs.'
    candidate = SkillContent.model_validate(content)
    return SimpleNamespace(service=service, metadata=metadata, gate=gate, driver=driver,
                           current=current, candidate=candidate)


def evaluate(case, earlier=()):
    return case.service.evaluate(current_skill=case.current, candidate=case.candidate,
        earlier_accepted_skills=earlier, unit_id='rewrite-unit')


def test_paired_native_driver_exact_order_fresh_contexts_and_durable_reuse(live):
    case = setup(live)
    result = evaluate(case)
    selection = select_dev_scenarios(data_seed=0, skill_id='skill', tool_dependencies=('search_stock',),
        dataset_manifest_sha256=D, families=case.metadata.families)
    assert [(r.scenario_id,r.branch) for r in case.driver.requests] == [
        (sid, branch) for sid in selection.selected_scenario_ids for branch in ('previous','candidate')]
    assert result.result.accepted and result.result.scenario_count == 2
    assert len({r.episode_id for r in case.driver.requests}) == 4
    assert len({id(s) for s in case.driver.scenarios}) == 4
    assert case.gate.initial == {'mutations': []}
    first, second = case.driver.requests[:2]
    assert first.shared_configuration_sha256 == second.shared_configuration_sha256
    assert first.evaluated_skill == SkillContent.from_record(case.current)
    assert second.evaluated_skill == case.candidate
    assert first.earlier_accepted_skills == second.earlier_accepted_skills == ()
    assert len(result.ordered_qwen_application_ids) == 4
    assert evaluate(case) == result
    assert len(case.driver.requests) == len(case.gate.calls) == 4
    assert live.ledger.effective_output_cost() == (0, True)  # No acceptance effect is committed here.


def test_earlier_accepted_skills_identical_on_both_branches(live):
    case = setup(live)
    earlier = current_skill().model_copy(update={'skill_id':'earlier'})
    evaluate(case, (earlier,))
    assert all(request.earlier_accepted_skills == (earlier,) for request in case.driver.requests)


def test_interrupted_candidate_does_not_rerun_completed_previous(live):
    case = setup(live)
    case.driver.fail_once = True
    with pytest.raises(RuntimeError, match='interruption'):
        evaluate(case)
    first = case.driver.requests[0].episode_id
    result = evaluate(case)
    assert result.result.accepted
    assert sum(request.episode_id == first for request in case.driver.requests) == 1


def test_invalid_candidate_before_metadata_or_gate(live):
    case = setup(live)
    case.candidate = SkillContent.from_record(case.current)
    with pytest.raises(ValueError, match='no-op'):
        evaluate(case)
    assert not case.metadata.calls and not case.gate.calls


def test_missing_native_driver_rejected_before_any_metadata(live):
    with pytest.raises(ValueError, match='explicit Dev'):
        setup(live, branch_executor=None)


def test_branch_identity_mismatch_is_not_saved_or_aggregated(live):
    case = setup(live)
    case.driver.wrong_identity = True
    with pytest.raises(ValueError, match='identity mismatch'):
        evaluate(case)
    assert case.service.branch_store.load(case.driver.requests[0]) is None


def test_changed_candidate_cannot_reuse_unit(live):
    case = setup(live)
    evaluate(case)
    before = len(case.metadata.calls)
    case.candidate = case.candidate.model_copy(update={'instruction':'A different candidate.'})
    with pytest.raises(Exception, match='conflict'):
        evaluate(case)
    assert len(case.metadata.calls) == before


def test_no_relevant_selection_never_opens_scenario(live):
    case = setup(live)
    case.metadata.families = (DevFamilySelectorView('unrelated', 'dev', ('other_tool',), ('scenario-other',)),)
    result = evaluate(case)
    assert not result.result.accepted
    assert result.result.reason == 'no_relevant_dev_scenarios'
    assert not case.gate.calls and not case.driver.requests


def test_incomplete_native_result_is_rejected_by_existing_acceptance_rule(live):
    case = setup(live)
    case.driver.incomplete = True
    result = evaluate(case)
    assert not result.result.accepted and result.result.reason == 'incomplete_branch'



def test_gate_denial_never_calls_native_driver(live):
    case = setup(live)
    class Denied:
        def load(self, **kwargs):
            raise PermissionError('synthetic denied lease')
    case.service.dev_gate = Denied()
    with pytest.raises(PermissionError):
        evaluate(case)
    assert not case.driver.requests


def test_invalid_manifest_identity_fails_before_metadata(live):
    with pytest.raises(ValueError, match='manifest identities'):
        setup(live, dev_manifest_sha256='unverified')
