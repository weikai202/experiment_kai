from types import SimpleNamespace as NS
import pytest
from toolsandbox_pipeline.orchestration.reflection_support import policy_projection, LedgerMemoryDurability
from toolsandbox_pipeline.schemas.trajectory import TrustedTrajectory, TrustedEvaluatorRecord
from toolsandbox_pipeline.offline.memory_projection import validate_projection_visibility
from toolsandbox_pipeline.checkpointing import LLMLedger


def sample(trajectory_store, start_context, episode_identity):
    end=trajectory_store.persist_context(start_context)
    identity=episode_identity.model_copy(update={'shard_id':'train-shard-0'})
    ev=TrustedEvaluatorRecord(milestone_similarity=0.,minefield_similarity=0.,similarity=0.,turn_count=0,
        milestone_mapping=(),minefield_mapping=(),fully_successful=False,
        evaluation_definition_sha256=identity.evaluation_definition_sha256,ending_context_sha256=end.context_sha256)
    ref,_=trajectory_store.persist_evaluator(identity,ev)
    trajectory=TrustedTrajectory.build(identity=identity,messages=(),online_turns=(),tool_actions=(),logical_request_ids=(),physical_attempt_ids=(),
        ending_context_reference=end.reference,ending_context_sha256=end.context_sha256,evaluator_record_reference=ref,
        evaluator_record_sha256=ref.sha256,skill_attributions=(),eligible_for_train_offline_consumption=True)
    return trajectory,ev


def test_projection_uses_only_permitted_labels(trajectory_store,start_context,episode_identity):
    trajectory,ev=sample(trajectory_store,start_context,episode_identity)
    projection=policy_projection(trajectory,ev,(),(),())
    validate_projection_visibility(projection)
    assert projection.native_similarity==0. and projection.host_attribution=='unsuccessful'
    assert 'milestone_mapping' not in projection.model_dump_json()
    assert 'ending_context_reference' not in projection.model_dump_json()


@pytest.mark.parametrize('change',[{'eligible_for_train_offline_consumption':False}])
def test_old_diagnostic_trajectories_cannot_be_relabelled(trajectory_store,start_context,episode_identity,change):
    trajectory,ev=sample(trajectory_store,start_context,episode_identity)
    with pytest.raises(ValueError,match='fresh eligible'):
        policy_projection(trajectory.model_copy(update=change),ev,(),(),())


def test_projection_rejects_wrong_evaluator_and_incomplete_turns(trajectory_store,start_context,episode_identity):
    trajectory,ev=sample(trajectory_store,start_context,episode_identity)
    with pytest.raises(ValueError,match='context mismatch'):
        policy_projection(trajectory,ev.model_copy(update={'ending_context_sha256':'sha256:'+'0'*64}),(),(),())
    with pytest.raises(ValueError,match='complete ordered'):
        policy_projection(trajectory,ev,({'state':{}},),(),())


def test_durability_checkpoint_is_idempotent(trajectory_store):
    durability=LedgerMemoryDurability(LLMLedger(trajectory_store.store))
    first=durability.checkpoint('reflection_probe',{'value':'unit'})
    assert first==durability.checkpoint('reflection_probe',{'value':'unit'})
    assert durability.load_unit('absent') is None
    assert durability.high_water_marks()['checkpoint_event_ordinal']==1


def test_committed_outcomes_matches_only_transaction_result_positions():
    from types import SimpleNamespace as N
    from toolsandbox_pipeline.orchestration.reflection_support import committed_outcomes
    from toolsandbox_pipeline.schemas.state import VisibleMessageInput, VisibleRole
    call = N(name='search_messages', arguments={})
    bindings = [(N(action_ordinal=i, action_sha256='same-action', call_ids=('reused-local-id',)),
                 (call,), {'search_messages':'search_messages'}) for i in (1,2)]
    records = [N(committed=True, action_sha256='same-action', call_ids=('reused-local-id',),
                 result_message_indices=(index,)) for index in (10,12)]
    messages = [VisibleMessageInput(source_message_index=index, sender=VisibleRole.EXECUTION_ENVIRONMENT,
                recipient=VisibleRole.AGENT, content='[]', openai_tool_call_id='reused-local-id',
                openai_function_name='search_messages') for index in (10,12)]
    # Even legacy repeated IDs never multiply result rows across transactions.
    outcomes = committed_outcomes(bindings, records, messages)
    assert [x.result_source_message_index for x in outcomes] == [10,12]
    records[1].result_message_indices = (10,12)
    with pytest.raises(ValueError, match='one committed visible result'):
        committed_outcomes(bindings, records, messages)


def test_committed_outcomes_keeps_model_id_distinct_from_execution_id():
    from types import SimpleNamespace as N
    from toolsandbox_pipeline.orchestration.reflection_support import committed_outcomes
    from toolsandbox_pipeline.schemas.state import VisibleMessageInput, VisibleRole
    bound=N(action_ordinal=1, action_sha256='action', call_ids=('exec-unique',))
    calls=(N(call_id='call_3', name='search_messages', arguments={}),)
    record=N(committed=True, action_sha256='action', call_ids=('exec-unique',), result_message_indices=(2,))
    message=VisibleMessageInput(source_message_index=2, sender=VisibleRole.EXECUTION_ENVIRONMENT,
               recipient=VisibleRole.AGENT, content='[]', openai_tool_call_id='exec-unique')
    result=committed_outcomes([(bound,calls,{'search_messages':'search_messages'})], [record], [message])
    assert result[0].call_id == 'exec-unique'
    assert calls[0].call_id == 'call_3'
    record.call_ids=('wrong-execution',)
    with pytest.raises(ValueError, match='binding mismatch'):
        committed_outcomes([(bound,calls,{'search_messages':'search_messages'})], [record], [message])
