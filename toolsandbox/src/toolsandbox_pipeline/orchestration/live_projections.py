"""Restricted current-round projections for the deployment coordinator.

Captured inputs come from the same durable online role requests as the trajectory.
No scenario, evaluator definition, model, or network access occurs here.
"""
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable

from toolsandbox_pipeline.offline.memory_projection import validate_projection_visibility
from toolsandbox_pipeline.offline.skill_projection import failure_update_projection
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnDecision
from toolsandbox_pipeline.schemas.offline_memory import PolicyTrajectoryProjection, WorldTrajectoryProjection
from toolsandbox_pipeline.schemas.offline_skill import SkillFailureEvidence
from toolsandbox_pipeline.schemas.trajectory import TrustedTrajectory, TrustedEvaluatorRecord


@dataclass(frozen=True)
class CapturedOnlineTurn:
    policy_envelope: dict
    proposed_action: ActionEnvelope
    decision: OnlineTurnDecision


@dataclass(frozen=True)
class CapturedEpisodeInputs:
    evaluator: TrustedEvaluatorRecord
    turns: tuple[CapturedOnlineTurn, ...]


@dataclass(frozen=True)
class CurrentRoundScope:
    run_id: str
    round_index: int
    dataset_manifest_sha256: str
    runtime_config_sha256: str

    def validate(self, trajectory: TrustedTrajectory) -> None:
        if self.round_index not in (0, 1, 2):
            raise ValueError('current round must be 0, 1, or 2')
        if type(trajectory) is not TrustedTrajectory:
            raise TypeError('exact trusted trajectory required')
        # Revalidate hashes, even if a caller used model_copy/model_construct.
        TrustedTrajectory.model_validate_json(trajectory.model_dump_json())
        identity = trajectory.identity
        if (not trajectory.eligible_for_train_offline_consumption
            or identity.run_id != self.run_id or identity.round_index != self.round_index
            or identity.shard_id != f'train-shard-{self.round_index}'
            or identity.generation_id != f'g{self.round_index:03d}'
            or identity.dataset_manifest_sha256 != self.dataset_manifest_sha256
            or identity.runtime_config_sha256 != self.runtime_config_sha256
            or not identity.phase.startswith('train')):
            raise ValueError('trajectory is outside the current train round')


class LiveTrajectoryLoader:
    def __init__(self, *, blob_store, scope: CurrentRoundScope):
        self.blob_store, self.scope = blob_store, scope

    def load_trajectory(self, reference) -> TrustedTrajectory:
        if reference.schema_name != 'TrustedTrajectory' or reference.schema_version != 1:
            raise ValueError('trusted trajectory blob required')
        raw = self.blob_store.read(reference)
        if reference.sha256 != 'sha256:' + sha256(raw).hexdigest() or len(raw) != reference.byte_count:
            raise ValueError('trajectory blob identity mismatch')
        trajectory = TrustedTrajectory.model_validate_json(raw)
        self.scope.validate(trajectory)
        return trajectory


class _Inputs:
    def __init__(self, *, scope: CurrentRoundScope, capture_loader: Callable):
        self.scope, self.capture_loader = scope, capture_loader

    def checked(self, trajectory):
        self.scope.validate(trajectory)
        captured = self.capture_loader(trajectory)
        if type(captured) is not CapturedEpisodeInputs or type(captured.evaluator) is not TrustedEvaluatorRecord:
            raise TypeError('exact captured episode inputs required')
        evaluator = captured.evaluator
        TrustedEvaluatorRecord.model_validate_json(evaluator.model_dump_json())
        if (canonical_sha256(evaluator.model_dump(mode='json')) != trajectory.evaluator_record_sha256
            or evaluator.ending_context_sha256 != trajectory.ending_context_sha256
            or evaluator.evaluation_definition_sha256 != trajectory.identity.evaluation_definition_sha256):
            raise ValueError('evaluator binding mismatch')
        if len(captured.turns) != len(trajectory.online_turns):
            raise ValueError('complete ordered online inputs required')
        for item, turn in zip(captured.turns, trajectory.online_turns):
            if type(item) is not CapturedOnlineTurn or type(item.decision) is not OnlineTurnDecision:
                raise TypeError('exact captured online turn required')
            decision = item.decision
            OnlineTurnDecision.model_validate_json(decision.model_dump_json())
            if set(item.policy_envelope) != {'state', 'policy_memory', 'skills'}:
                raise ValueError('exact Policy envelope required')
            if (item.policy_envelope['state']['state_id'] != turn.state_id
                or decision.state_id != turn.state_id
                or decision.identity.agent_turn_index != turn.agent_turn_index
                or decision.identity.episode_id != trajectory.identity.episode_id
                or decision.identity.run_id != trajectory.identity.run_id
                or decision.generation_id != trajectory.identity.generation_id
                or canonical_sha256(decision.model_dump(mode='json')) != turn.decision_sha256
                or canonical_sha256(decision.final_action.model_dump(mode='json')) != turn.final_action_sha256
                or decision.initial_policy_logical_request_id not in turn.logical_request_ids
                or (decision.critic_logical_request_id is not None and decision.critic_logical_request_id not in turn.logical_request_ids)):
                raise ValueError('online decision binding mismatch')
            ActionEnvelope.model_validate_json(item.proposed_action.model_dump_json())
        return captured


def _visible_outcomes(trajectory, indices=None, call_ids=None):
    return tuple(message for message in trajectory.messages
        if message.sender == 'EXECUTION_ENVIRONMENT' and message.recipient == 'AGENT'
        and 'AGENT' in message.visible_to
        and (indices is None or message.sandbox_message_index in indices)
        and (call_ids is None or message.openai_tool_call_id in call_ids))


def _outcome_payload(messages):
    return tuple({'content': message.content, 'openai_tool_call_id': message.openai_tool_call_id,
                  'tool_call_exception': message.tool_call_exception} for message in messages)


def _codes(decision):
    return tuple(dict.fromkeys(code.value for code in (
        *decision.initial_controller_decision.blocking_codes,
        *decision.initial_controller_decision.critic_trigger_codes)))


class LiveMemoryProjectionAdapter(_Inputs):
    def project(self, trajectory):
        captured = self.checked(trajectory)
        policy = PolicyTrajectoryProjection(
            trajectory_id=trajectory.trajectory_id, manifest_position=trajectory.identity.manifest_position,
            visible_states=tuple(item.policy_envelope['state'] for item in captured.turns),
            retrieved_policy_memory=tuple(value for item in captured.turns for value in item.policy_envelope['policy_memory']),
            retrieved_skills=tuple(value for item in captured.turns for value in item.policy_envelope['skills']),
            proposed_actions=tuple(item.proposed_action for item in captured.turns),
            final_actions=tuple(item.decision.final_action for item in captured.turns),
            controller_codes=tuple(dict.fromkeys(code for item in captured.turns for code in _codes(item.decision))),
            visible_tool_outcomes=_outcome_payload(_visible_outcomes(trajectory)),
            native_similarity=captured.evaluator.similarity, fully_successful=captured.evaluator.fully_successful,
            host_attribution='successful' if captured.evaluator.fully_successful else 'unsuccessful')
        validate_projection_visibility(policy)
        world = None
        # One candidate per trajectory: deterministic first directly attributable draft.
        for item, turn in zip(captured.turns, trajectory.online_turns):
            decision = item.decision
            if decision.critic_verdict is None or canonical_sha256(item.proposed_action.model_dump(mode='json')) != turn.final_action_sha256:
                continue
            actions = [action for action in trajectory.tool_actions if action.committed and action.executed
                       and not action.rolled_back and action.call_ids == turn.executed_call_ids
                       and action.action_sha256 == turn.final_action_sha256]
            if len(actions) != 1:
                continue
            action = actions[0]
            outcomes = _visible_outcomes(trajectory, action.result_message_indices, action.call_ids)
            if any(sum(message.openai_tool_call_id == call_id for message in outcomes) != 1
                   for call_id in action.call_ids):
                continue  # Missing/ambiguous visible evidence cannot label the draft.
            if not any(message.tool_call_exception for message in outcomes):
                continue
            world = WorldTrajectoryProjection(
                trajectory_id=trajectory.trajectory_id, manifest_position=trajectory.identity.manifest_position,
                visible_states=(item.policy_envelope['state'],), draft_action=item.proposed_action,
                controller_codes=_codes(decision), visible_tool_outcomes=_outcome_payload(outcomes),
                critic_output=decision.critic_verdict, revision_occurred=bool(decision.revision_count),
                attributable_failure=True, attribution_kind='real_tool_exception')
            validate_projection_visibility(world)
            break
        return policy, world


class LiveFailureEvidenceAdapter(_Inputs):
    def project(self, trajectory):
        captured = self.checked(trajectory)
        evidence = []
        for attribution in trajectory.skill_attributions:
            if (attribution.generation_id != trajectory.identity.generation_id
                or attribution.evaluator_record_sha256 != trajectory.evaluator_record_sha256
                or attribution.fully_successful != captured.evaluator.fully_successful):
                raise ValueError('Skill evaluator binding mismatch')
            expected = []
            indices = set()
            for action in trajectory.tool_actions:
                if not action.executed or not action.committed or action.rolled_back or action.is_user_conversation_control:
                    continue
                for call_id, skill_id, tool in zip(action.call_ids, action.selected_skill_ids, action.canonical_tool_ids):
                    if skill_id == attribution.skill_id:
                        expected.append((call_id, tool))
                        indices.update(action.result_message_indices)
            if tuple(expected) != tuple(zip(attribution.executed_call_ids, attribution.canonical_tool_ids)):
                raise ValueError('Skill executed-call binding mismatch')
            if captured.evaluator.fully_successful:
                continue
            outcomes = _visible_outcomes(trajectory, indices, attribution.executed_call_ids)
            if any(message.tool_call_exception for message in outcomes):
                kind, outcome, description = ('visible_tool_exception', 'tool_execution_exception',
                    'An executed tool associated with this Skill raised a visible exception in an unsuccessful task.')
            elif any(message.sender == 'AGENT' and message.recipient == 'USER'
                     and 'AGENT' in message.visible_to and message.sandbox_message_index > max(indices, default=-1)
                     for message in trajectory.messages):
                kind, outcome, description = ('visible_terminal_state_failure', 'task_not_fully_successful',
                    'The task ended without full success after executing this Skill; no particular tool error is inferred.')
            else:
                continue
            fields = dict(skill_id=attribution.skill_id, skill_version=attribution.skill_version,
                trajectory_id=trajectory.trajectory_id, episode_id=trajectory.identity.episode_id,
                manifest_position=trajectory.identity.manifest_position, evidence_kind=kind,
                canonical_tool_dependencies=tuple(sorted(set(attribution.canonical_tool_ids), key=lambda value: value.encode('utf-8'))),
                sanitized_outcome_class=outcome, generalized_failure=description)
            result = SkillFailureEvidence(**fields, source_evidence_sha256=canonical_sha256(
                {'protocol': 'skill-failure-evidence-v1', **fields,
                 'canonical_tool_dependencies': list(fields['canonical_tool_dependencies'])}))
            failure_update_projection(result, ())
            evidence.append(result)
        return tuple(evidence)
