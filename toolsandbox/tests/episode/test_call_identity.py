import copy
import json

import pytest
from pydantic import ValidationError
from tool_sandbox.common.execution_context import RoleType, set_current_context
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.checkpointing import ToolLedger
from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnDecision
from toolsandbox_pipeline.toolsandbox_adapter.call_identity import execution_action, execution_call_ids
from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import (
    ToolActionBinding, TransactionalExecutionEnvironment, TransactionalRoleError,
)


def decision(turn=0, *, parallel=False):
    digest = 'sha256:' + 'a' * 64
    call = {'call_id': 'call_3', 'selected_skill_id': 'skill-1', 'name': 'alpha', 'arguments': {}}
    action = {'type': 'parallel_batch', 'calls': [call, {**call, 'call_id': 'call_4'}]} if parallel else {'type': 'function_call', **call}
    identity = dict(run_id='run-1', profile='official_live', phase='train', family_id='family', scenario_id='scenario', episode_id='episode', agent_turn_index=turn, expected_generation_id='g000', environment_identity='environment')
    identity.update({name: digest for name in ('dataset_manifest_sha256', 'runtime_config_sha256', 'prompt_manifest_sha256', 'token_limit_config_sha256')})
    return OnlineTurnDecision.model_validate_json(json.dumps(dict(identity=identity, state_id='state', generation_id='g000', final_action={'action': action}, initial_policy_logical_request_id='request', initial_controller_decision={'blocking_codes': [], 'critic_trigger_codes': [], 'evidence': []}, revision_count=0, retrieval_references=[], audit_record_sha256=digest)))


def test_model_ids_remain_local_but_host_ids_distinguish_turns_and_restore():
    first, second = decision(), decision(1)
    original = first.model_dump_json()
    assert first.final_action.action.call_id == second.final_action.action.call_id == 'call_3'
    assert execution_call_ids(first) != execution_call_ids(second)
    restored = OnlineTurnDecision.model_validate_json(original)
    assert execution_call_ids(restored) == execution_call_ids(first)
    assert first.model_dump_json() == original
    dispatched = execution_action(first)
    assert dispatched.action.arguments == first.final_action.action.arguments
    assert dispatched.action.name == first.final_action.action.name
    assert dispatched.action.call_id.isidentifier()
    assert first.audit_record_sha256 == restored.audit_record_sha256


def test_parallel_mapping_is_stable_unique_and_does_not_accept_ambiguous_raw_batch():
    item = decision(parallel=True)
    assert len(set(execution_call_ids(item))) == 2
    assert execution_call_ids(item) == execution_call_ids(OnlineTurnDecision.model_validate_json(item.model_dump_json()))
    payload = item.final_action.model_dump(mode='json')
    payload['action']['calls'][1]['call_id'] = 'call_3'
    with pytest.raises(ValidationError, match='must be unique'):
        ActionEnvelope.model_validate_json(json.dumps(payload))


class EchoEnvironment(BaseRole):
    role_type = RoleType.EXECUTION_ENVIRONMENT
    def __init__(self):
        self.executions = 0
    def respond(self, ending_index=None):
        self.executions += 1
        call = self.get_messages()[-1]
        self.add_messages([Message(RoleType.EXECUTION_ENVIRONMENT, RoleType.AGENT,
                                   f'result-{self.executions}', openai_tool_call_id=call.openai_tool_call_id,
                                   openai_function_name='alpha')])


def test_repeated_model_id_native_results_and_committed_checkpoint_restore(
    trajectory_store, start_context, episode_identity
):
    set_current_context(start_context)
    delegate = EchoEnvironment()
    ledger = ToolLedger(trajectory_store.store)
    records = []
    for turn in (0, 1):
        item = decision(turn)
        call_id, = execution_call_ids(item)
        BaseRole.add_messages([Message(RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT,
                                       'alpha()', openai_tool_call_id=call_id, openai_function_name='alpha')])
        from tool_sandbox.common.execution_context import get_current_context
        pre = copy.deepcopy(get_current_context())
        binding = ToolActionBinding(action_sha256=canonical_sha256(item.final_action.model_dump(mode='json')), call_ids=(call_id,), selected_skill_ids=('skill-1',), canonical_tool_ids=('alpha',), effect_classes=('sandbox_read',), action_ordinal=turn + 1)
        def role():
            return TransactionalExecutionEnvironment(delegate, identity=episode_identity, tool_ledger=ledger, trajectory_store=trajectory_store, binding_provider=lambda messages: binding, backend_manifest_sha256='sha256:' + 'c' * 64)
        original = role(); original.respond()
        record = original.records[0]
        assert delegate.executions == turn + 1
        set_current_context(pre)
        recovered = role(); recovered.respond()
        assert delegate.executions == turn + 1  # restore committed context, never dispatch again
        assert recovered.records[0].transaction_id == record.transaction_id
        assert recovered.records[0].result_message_indices == record.result_message_indices
        from toolsandbox_pipeline.toolsandbox_adapter.messages import extract_visible_messages
        from toolsandbox_pipeline.toolsandbox_adapter.pipeline_agent import PipelineAgent
        visible = {message.source_message_index: message for message in extract_visible_messages(PipelineAgent)}
        result, = (visible[index] for index in record.result_message_indices)
        assert result.openai_tool_call_id == call_id
        assert result.content == f'result-{turn + 1}'
        records.append(record)
    assert records[0].call_ids != records[1].call_ids
    assert set(records[0].result_message_indices).isdisjoint(records[1].result_message_indices)


def test_dispatch_uses_host_ids_in_python_and_metadata_keeps_decision_original():
    from types import SimpleNamespace
    from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import _CapturingResponder
    from toolsandbox_pipeline.toolsandbox_adapter import ControllerToolContext, action_to_messages
    item = decision(parallel=True)
    raw_json = item.model_dump_json()
    capture = _CapturingResponder(SimpleNamespace(respond_decision=lambda turn: item))
    dispatched = capture.respond(object())
    def alpha():
        return 'ok'
    context = ControllerToolContext({'alpha': 'alpha'}, 'sha256:' + '0' * 64, {'alpha': alpha})
    messages = action_to_messages(dispatched, context)
    for message, call_id in zip(messages, execution_call_ids(item)):
        assert message.openai_tool_call_id == call_id
        assert f'alpha(**{call_id}_parameters)' in message.content
        assert 'call_3_parameters' not in message.content
    assert capture.decision is item
    assert item.model_dump_json() == raw_json


def test_mismatched_binding_rejected_before_tool_dispatch(trajectory_store, start_context, episode_identity):
    set_current_context(start_context)
    BaseRole.add_messages([Message(RoleType.AGENT, RoleType.EXECUTION_ENVIRONMENT,
                                   'alpha()', openai_tool_call_id='actual', openai_function_name='alpha')])
    delegate = EchoEnvironment()
    binding = ToolActionBinding(action_sha256='sha256:' + 'b' * 64, call_ids=('wrong',), selected_skill_ids=('skill-1',), canonical_tool_ids=('alpha',), effect_classes=('sandbox_read',), action_ordinal=1)
    role = TransactionalExecutionEnvironment(delegate, identity=episode_identity, tool_ledger=ToolLedger(trajectory_store.store), trajectory_store=trajectory_store, binding_provider=lambda messages: binding, backend_manifest_sha256='sha256:' + 'c' * 64)
    with pytest.raises(TransactionalRoleError, match='execution IDs differ'):
        role.respond()
    assert delegate.executions == 0
