"""Development-only adapters for one fresh Policy-memory reflection round."""
import json
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.offline_memory import PolicyTrajectoryProjection, MemoryUpdateUnitResult
from toolsandbox_pipeline.schemas.checkpoint import QwenEffectKind, NonSubstantiveOutcome
from toolsandbox_pipeline.offline.memory_projection import validate_projection_visibility


def policy_projection(trajectory, evaluator, envelopes, proposed_actions, decisions):
    identity = trajectory.identity
    if (not trajectory.eligible_for_train_offline_consumption or identity.round_index != 0
            or identity.shard_id != 'train-shard-0' or not identity.phase.startswith('train')
            or identity.generation_id != 'g000'):
        raise ValueError('fresh eligible round-zero train trajectory required')
    if evaluator.ending_context_sha256 != trajectory.ending_context_sha256:
        raise ValueError('evaluator context mismatch')
    if not (len(envelopes) == len(proposed_actions) == len(decisions) == len(trajectory.online_turns)):
        raise ValueError('complete ordered online inputs required')
    for envelope, turn, decision in zip(envelopes, trajectory.online_turns, decisions):
        if envelope['state']['state_id'] != turn.state_id or decision.state_id != turn.state_id:
            raise ValueError('online state mismatch')
    codes = tuple(dict.fromkeys(code.value for d in decisions
                               for code in (*d.initial_controller_decision.blocking_codes,
                                            *d.initial_controller_decision.critic_trigger_codes)))
    projection = PolicyTrajectoryProjection(
        trajectory_id=trajectory.trajectory_id, manifest_position=identity.manifest_position,
        visible_states=tuple(e['state'] for e in envelopes),
        retrieved_policy_memory=tuple(m for e in envelopes for m in e['policy_memory']),
        retrieved_skills=tuple(s for e in envelopes for s in e['skills']),
        proposed_actions=tuple(proposed_actions), final_actions=tuple(d.final_action for d in decisions),
        controller_codes=codes,
        visible_tool_outcomes=tuple({'content':m.content,'openai_tool_call_id':m.openai_tool_call_id,
                                     'tool_call_exception':m.tool_call_exception}
                                    for m in trajectory.messages
                                    if m.sender=='EXECUTION_ENVIRONMENT' and m.recipient=='AGENT'
                                    and 'AGENT' in m.visible_to),
        native_similarity=evaluator.similarity, fully_successful=evaluator.fully_successful,
        host_attribution='successful' if evaluator.fully_successful else 'unsuccessful')
    validate_projection_visibility(projection)
    return projection


class LedgerMemoryDurability:
    def __init__(self, ledger):
        self.ledger = ledger

    def load_unit(self, unit_reference):
        event = self.ledger.get_checkpoint('reflection-unit-' + unit_reference)
        return None if event is None else MemoryUpdateUnitResult.model_validate_json(json.dumps(event.payload))

    def checkpoint(self, event_kind, payload):
        checkpoint_id = 'reflection-' + canonical_sha256([event_kind,payload])[7:]
        return self.ledger.commit_checkpoint(checkpoint_id,event_kind,payload).checkpoint_id

    def commit_mutation(self, mutation, ordered_application_ids):
        payload=mutation.model_dump(mode='json');digest=canonical_sha256(payload)
        checkpoint_id='reflection-mutation-'+digest[7:]
        effect=self.ledger.commit_checkpoint_and_effect(
            checkpoint_id=checkpoint_id,event_kind='reflection_memory_mutation',checkpoint_payload=payload,
            effect_kind=QwenEffectKind.POLICY_MEMORY_MUTATION if mutation.role=='policy' else QwenEffectKind.WORLD_MEMORY_MUTATION,
            effect_artifact_id='reflection-memory-'+digest[7:],effect_artifact_sha256=digest,
            ordered_application_ids=ordered_application_ids)
        return effect.effect_id,checkpoint_id

    def record_non_substantive(self, outcome, application_ids):
        self.ledger.record_non_substantive_outcome(outcome=NonSubstantiveOutcome(outcome),ordered_application_ids=application_ids)

    def commit_unit(self, result):
        self.ledger.commit_checkpoint('reflection-unit-'+result.unit_id,'reflection_unit_completed',result.model_dump(mode='json'))
        return result

    def high_water_marks(self):
        return self.ledger.store.high_water_marks()


def committed_outcomes(bindings, records, visible_messages):
    """Join each execution to its committed result rows, never all call-ID history."""
    from toolsandbox_pipeline.schemas.state import CommittedToolOutcomeInput
    by_index = {m.source_message_index: m for m in visible_messages}
    if len(by_index) != len(visible_messages):
        raise ValueError('duplicate visible message index')
    outcomes = []
    for bound, calls, mapping in bindings:
        if not 1 <= bound.action_ordinal <= len(records):
            raise ValueError('missing committed tool transaction')
        record = records[bound.action_ordinal - 1]
        if (not record.committed or record.action_sha256 != bound.action_sha256
                or record.call_ids != bound.call_ids or len(calls) != len(bound.call_ids)):
            raise ValueError('tool transaction binding mismatch')
        for execution_id, call in zip(bound.call_ids, calls):
            matches = [by_index[index] for index in record.result_message_indices
                       if index in by_index
                       and by_index[index].sender.value == 'EXECUTION_ENVIRONMENT'
                       and by_index[index].openai_tool_call_id == execution_id]
            if len(matches) != 1:
                raise ValueError('one committed visible result per tool call required')
            message = matches[0]
            outcomes.append(CommittedToolOutcomeInput(
                call_id=execution_id, agent_facing_tool_name=call.name,
                arguments=call.arguments, result_source_message_index=message.source_message_index,
                public_return_contract_id=mapping[call.name], canonical_tool_name=mapping[call.name],
                tool_mapping_manifest_hash=canonical_sha256(mapping)))
    return tuple(outcomes)
