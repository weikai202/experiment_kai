"""Real native execution/transactions over synthetic duplicate model labels."""
import copy
import json
import pytest

from tool_sandbox.common.execution_context import get_current_context, set_current_context
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.execution_environment import ExecutionEnvironment

from toolsandbox_pipeline.checkpointing import ToolLedger
from toolsandbox_pipeline.orchestration.reflection_support import committed_outcomes
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnDecision
from toolsandbox_pipeline.toolsandbox_adapter import ControllerToolContext, action_to_messages
from toolsandbox_pipeline.toolsandbox_adapter.call_identity import execution_action, execution_call_ids
from toolsandbox_pipeline.toolsandbox_adapter.messages import extract_visible_messages
from toolsandbox_pipeline.toolsandbox_adapter.pipeline_agent import PipelineAgent
from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import ToolActionBinding, TransactionalExecutionEnvironment
from tests.episode.test_call_identity import decision
from tests.providers.test_request_identity import context


alpha = pow  # Stable builtin reference; no test-module globals in console pickle.


class CountingNative(ExecutionEnvironment):
    def __init__(self):
        self.dispatches = 0
    def respond(self, ending_index=None):
        self.dispatches += 1
        return super().respond(ending_index=ending_index)


@pytest.mark.parametrize("first_value", [11, 0])
def test_duplicate_raw_labels_different_arguments_execute_associate_and_restore(
    trajectory_store, start_context, episode_identity, first_value,
):
    from toolsandbox_pipeline.toolsandbox_adapter.action_decode import decode_action
    raw = json.dumps({'action': {'type': 'parallel_batch', 'calls': [
        {'call_id': 'same', 'selected_skill_id': None, 'name': 'alpha', 'arguments': {'base': value, 'exp': -1}}
        for value in (first_value, 29)]}})
    decoded = decode_action(raw, context())
    assert [call.arguments for call in decoded.action.action.calls] == [{'base':first_value, 'exp':-1}, {'base':29, 'exp':-1}]
    assert len({call.call_id for call in decoded.action.action.calls}) == 2
    assert [entry.model_call_id for entry in decoded.call_identities] == ['same','same']
    assert [entry.host_action_call_id for entry in decoded.call_identities] == [call.call_id for call in decoded.action.action.calls]
    item = decision(parallel=True).model_copy(update={'final_action':decoded.action})
    # Persisting/reloading the decision is enough to reconstruct the native layer.
    ref = trajectory_store.persist_online_decision(item)
    restored_decision = OnlineTurnDecision.model_validate_json(trajectory_store.store.blobs.read(ref))
    assert execution_call_ids(item) == execution_call_ids(restored_decision)
    assert decode_action(raw, context(attempt_id='attempt-recovery')).action == decoded.action
    native_ids = execution_call_ids(item)
    assert len(set(native_ids)) == 2
    assert set(native_ids).isdisjoint(call.call_id for call in decoded.action.action.calls)
    set_current_context(start_context)
    start_context.interactive_console.locals['alpha'] = alpha
    mapping = {'alpha':'alpha'}
    adapter = ControllerToolContext(mapping, canonical_sha256(mapping), {'alpha':alpha})
    messages = action_to_messages(execution_action(item), adapter)
    BaseRole.add_messages(list(messages))
    pre = copy.deepcopy(get_current_context())
    binding = ToolActionBinding(action_sha256=canonical_sha256(item.final_action.model_dump(mode='json')),
        call_ids=native_ids,selected_skill_ids=(None,None),canonical_tool_ids=('alpha','alpha'),
        effect_classes=('sandbox_read','sandbox_read'),action_ordinal=1)
    delegate, ledger = CountingNative(), ToolLedger(trajectory_store.store)
    def wrapper():
        return TransactionalExecutionEnvironment(delegate,identity=episode_identity,tool_ledger=ledger,
            trajectory_store=trajectory_store,binding_provider=lambda _:binding,
            backend_manifest_sha256='sha256:'+'c'*64)
    first=wrapper(); first.respond()
    record,=first.records
    assert delegate.dispatches == 1 and record.committed
    assert record.failed is (first_value == 0)
    assert record.rolled_back is (first_value == 0)
    visible=extract_visible_messages(PipelineAgent)
    rows={row.openai_tool_call_id:row for row in visible if row.source_message_index in record.result_message_indices}
    assert set(rows) == set(native_ids), (record.result_message_indices, [(r.source_message_index,r.openai_tool_call_id,r.content) for r in visible])
    assert {key: rows[key].content for key in native_ids} == {native_ids[0]:str(pow(first_value,-1)) if first_value else 'ZeroDivisionError: 0.0 cannot be raised to a negative power',native_ids[1]:str(pow(29,-1))}
    outcomes=committed_outcomes([(binding,decoded.action.action.calls,mapping)],first.records,visible)
    assert [(row.call_id,row.arguments,row.result_source_message_index) for row in outcomes] == [
        (native_ids[0],{'base':first_value, 'exp':-1},rows[native_ids[0]].source_message_index),
        (native_ids[1],{'base':29, 'exp':-1},rows[native_ids[1]].source_message_index)]
    from toolsandbox_pipeline.toolsandbox_adapter.trajectory_store import StoredContext
    set_current_context(trajectory_store.load_context(StoredContext(
        record.pre_context_reference, record.pre_context_sha256)))
    replay=wrapper(); replay.respond()
    assert delegate.dispatches == 1
    assert replay.records[0].transaction_id == record.transaction_id
    assert replay.records[0].result_message_indices == record.result_message_indices
    replay_visible=extract_visible_messages(PipelineAgent)
    assert committed_outcomes([(binding,decoded.action.action.calls,mapping)],replay.records,replay_visible)==outcomes
    # No caller-owned raw payload or strict normalized decision was overwritten.
    assert [call['call_id'] for call in json.loads(raw)['action']['calls']] == ['same','same']
    assert restored_decision.final_action == decoded.action

    # A later turn may reuse both raw labels without importing old result rows.
    next_decoded = decode_action(raw, context(logical_request_id='logical-next',unit_reference='state-next'))
    next_item = decision(turn=1,parallel=True).model_copy(update={'final_action':next_decoded.action})
    next_ids=execution_call_ids(next_item)
    assert set(next_ids).isdisjoint(native_ids)
    BaseRole.add_messages(list(action_to_messages(execution_action(next_item),adapter)))
    binding=ToolActionBinding(action_sha256=canonical_sha256(next_item.final_action.model_dump(mode='json')),
        call_ids=next_ids,selected_skill_ids=(None,None),canonical_tool_ids=('alpha','alpha'),
        effect_classes=('sandbox_read','sandbox_read'),action_ordinal=2)
    later=wrapper(); later.respond()
    assert delegate.dispatches==2
    assert len(later.records[0].result_message_indices)==2
    assert set(later.records[0].result_message_indices).isdisjoint(record.result_message_indices)
    current=extract_visible_messages(PipelineAgent)
    current_rows={row.openai_tool_call_id:row for row in current if row.source_message_index in later.records[0].result_message_indices}
    assert set(current_rows)==set(next_ids)
    assert current_rows[next_ids[1]].content==str(pow(29,-1))
