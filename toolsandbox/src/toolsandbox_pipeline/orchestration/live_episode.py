"""Coordinator-injected formal train episode wiring; no implicit providers/access.

One instance belongs to one authorized round. Native roles, evaluation and durable
online routing are reused. A coordinator must supply calibrated context builders,
a durable User, fixture boundary, and a round-scoped restricted capture sink.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Mapping

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.roles.execution_environment import ExecutionEnvironment

from toolsandbox_pipeline.checkpointing.tool_ledger import ToolLedger
from toolsandbox_pipeline.online.turn_context import TurnContext, TurnRoleRequestBuilders
from toolsandbox_pipeline.online.controller_inputs import ReproducibilityProfile
from toolsandbox_pipeline.online.turn_responder import DurableTurnResponder
from toolsandbox_pipeline.online.durable_roles import DurableRoleExecution
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnIdentity, OnlineTurnAuditRecord
from toolsandbox_pipeline.schemas.trajectory import EpisodeIdentity, EpisodeResult
from toolsandbox_pipeline.toolsandbox_adapter.call_identity import execution_call_ids
from toolsandbox_pipeline.toolsandbox_adapter.episode_runner import EpisodeRunner, EpisodeRunInput
from toolsandbox_pipeline.toolsandbox_adapter.messages import extract_visible_messages
from toolsandbox_pipeline.toolsandbox_adapter.native_evaluator import NativeEvaluator
from toolsandbox_pipeline.toolsandbox_adapter.pipeline_agent import PipelineAgent
from toolsandbox_pipeline.toolsandbox_adapter.tools import build_adapter_turn
from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import (
    TransactionalAgentRole, TransactionalUserRole, TransactionalExecutionEnvironment, ToolActionBinding,
)
from toolsandbox_pipeline.orchestration.reflection_support import committed_outcomes


@dataclass(frozen=True)
class LiveEpisodeScope:
    run_id: str
    round_index: int
    shard_id: str
    generation_id: str
    profile: str
    dataset_manifest_sha256: str
    runtime_config_sha256: str
    prompt_manifest_sha256: str
    token_limit_config_sha256: str
    fixture_manifest_sha256: str
    environment_sha256: str
    scenario_positions: Mapping[str, int]
    phase: str = 'train_round'

    def __post_init__(self):
        if (type(self.round_index) is not int or self.round_index not in (0, 1, 2)
                or self.shard_id != f'train-shard-{self.round_index}'
                or self.generation_id != f'g{self.round_index:03d}'
                or self.profile not in ('strict_replay', 'official_live')
                or not self.run_id or self.phase != 'train_round'):
            raise ValueError('formal train episode scope mismatch')
        positions = dict(self.scenario_positions)
        if (not positions or any(type(k) is not str or not k or type(v) is not int or v < 0
                                 for k, v in positions.items())
                or len(set(positions.values())) != len(positions)):
            raise ValueError('unique manifest scenario positions required')
        for key in ('dataset_manifest_sha256', 'runtime_config_sha256', 'prompt_manifest_sha256',
                    'token_limit_config_sha256', 'fixture_manifest_sha256', 'environment_sha256'):
            value = getattr(self, key)
            if not isinstance(value, str) or re.fullmatch(r'sha256:[0-9a-f]{64}', value) is None:
                raise ValueError('manifest digest required')
        object.__setattr__(self, 'scenario_positions', MappingProxyType(positions))


@dataclass(frozen=True)
class LiveTurnRequest:
    identity: OnlineTurnIdentity
    accounting_scope: AccountingScope
    adapter_turn: object
    committed_tool_outcomes: tuple


@dataclass(frozen=True)
class LiveEpisodeCapture:
    """Restricted current-round input, retained by a revocable caller-owned sink."""
    identity: EpisodeIdentity
    initial_envelopes: tuple[tuple[int, str], ...]
    decisions: tuple
    proposed_actions: tuple[ActionEnvelope, ...]
    result: EpisodeResult


class LiveEpisodeResumeRequired(RuntimeError):
    """An existing dispatch requires explicit Task014 resume coordination."""


class LiveEpisodeExecutor:
    """Implements RoundRunner's run_scenario signature and returns EpisodeResult.

    context_factory(LiveTurnRequest) supplies a complete TurnContext using the
    caller's generation retrieval, ledger/gateways and formal calibrated builders.
    user_factory(identity, trajectory_store, ledger) supplies the instrumented
    durable User (or authorized strict local User). authorization(scope) must
    verify resolved formal manifest/preflights/calibrations before dataset access.
    physical_attempt_provider(identity, logical_request_ids) supplies all actual
    episode attempt IDs, including User and embeddings, from the authoritative ledger.

    This entry point starts fresh episodes only. A persisted dispatch marker makes
    re-entry after crashes or capture-sink failure fail closed. Complete mid-episode
    resume requires separate reconstruction of the native roles and tool bindings;
    this service does not claim to implement that recovery path.
    """
    def __init__(self, *, scope, gate, generation, ledger, trajectory_store,
                 metadata, context_factory, user_factory, authorization,
                 physical_attempt_provider, boot_id, capture_sink,
                 external_boundary_factory=None, contextual_external_boundary_factory=None,
                 backend_manifest_sha256=None):
        if type(scope) is not LiveEpisodeScope:
            raise TypeError('LiveEpisodeScope required')
        if generation.manifest.generation_id != scope.generation_id:
            raise ValueError('pinned generation mismatch')
        if not boot_id:
            raise ValueError('boot identity required')
        if any(not callable(value) for value in (context_factory, user_factory, authorization,
                                                  physical_attempt_provider, capture_sink)):
            raise TypeError('explicit coordinator services required')
        self.scope, self.gate, self.generation = scope, gate, generation
        self.ledger, self.trajectory_store = ledger, trajectory_store
        self.metadata = tuple(metadata)
        self.context_factory, self.user_factory = context_factory, user_factory
        self.authorization, self.physical_attempt_provider = authorization, physical_attempt_provider
        self.boot_id, self.capture_sink = boot_id, capture_sink
        if external_boundary_factory is not None and contextual_external_boundary_factory is not None:
            raise ValueError('choose one external boundary factory contract')
        if contextual_external_boundary_factory is not None and not callable(contextual_external_boundary_factory):
            raise TypeError('contextual external boundary factory must be callable')
        if backend_manifest_sha256 is not None and re.fullmatch(r'sha256:[0-9a-f]{64}', backend_manifest_sha256) is None:
            raise ValueError('actual backend manifest digest required')
        if contextual_external_boundary_factory is not None and backend_manifest_sha256 is None:
            raise ValueError('contextual boundary requires actual backend manifest digest')
        self.external_boundary_factory = external_boundary_factory
        self.contextual_external_boundary_factory = contextual_external_boundary_factory
        self.backend_manifest_sha256 = backend_manifest_sha256 or scope.fixture_manifest_sha256
        self._results = {}
        self._active = False

    def _identity(self, record, position):
        scope = self.scope
        return EpisodeIdentity(run_id=scope.run_id, profile=scope.profile, phase=scope.phase,
            round_index=scope.round_index, shard_id=scope.shard_id,
            family_id=record.scenario_family_id, scenario_id=record.scenario_id,
            episode_id=f'{scope.run_id}-{scope.generation_id}-episode-{position}',
            manifest_position=position, system_variant='generation_0' if scope.round_index == 0 else 'updated',
            generation_id=scope.generation_id, starting_context_sha256=record.starting_context_sha256,
            evaluation_definition_sha256=record.evaluation_definition_sha256,
            agent_tool_schema_sha256=record.agent_facing_tool_schema_sha256,
            dataset_manifest_sha256=scope.dataset_manifest_sha256,
            runtime_config_sha256=scope.runtime_config_sha256, prompt_manifest_sha256=scope.prompt_manifest_sha256,
            token_limit_config_sha256=scope.token_limit_config_sha256,
            fixture_manifest_sha256=scope.fixture_manifest_sha256, environment_sha256=scope.environment_sha256,
            max_messages=record.max_messages)

    def run_scenario(self, *, scenario_id, manifest_position, generation_id):
        scope = self.scope
        if (generation_id != scope.generation_id or scenario_id not in scope.scenario_positions
                or type(manifest_position) is not int or scope.scenario_positions[scenario_id] != manifest_position):
            raise PermissionError('scenario outside authorized round manifest')
        if self._active:
            raise RuntimeError('concurrent native episodes are forbidden')
        if scenario_id in self._results:
            return self._results[scenario_id]
        dispatch_key = 'live-episode-dispatch-' + canonical_sha256(
            [scope.run_id, scope.generation_id, scenario_id, manifest_position])[7:]
        if self.ledger.get_checkpoint(dispatch_key) is not None:
            raise LiveEpisodeResumeRequired('existing episode requires explicit resume coordinator')
        self.authorization(scope)
        leases = self.gate.load(requested_ids=[scenario_id], run_id=scope.run_id, phase=scope.phase,
            manifest_sha256=scope.dataset_manifest_sha256, split='train', purpose='train_round',
            train_shard=scope.round_index)
        if len(leases) != 1 or leases[0].record.scenario_id != scenario_id:
            raise ValueError('dataset lease identity mismatch')
        identity = self._identity(leases[0].record, manifest_position)
        self.ledger.commit_checkpoint(dispatch_key, 'live_episode_dispatch_reserved',
                                     {'identity': identity.model_dump(mode='json')})
        self._active = True
        try:
            result = self._run_lease(leases[0], identity)
            self._results[scenario_id] = result
            return result
        finally:
            self._active = False

    def run_calibration_lease(self, *, lease, identity, authorization, purpose,
                              prepared_request_sink, role_execution_sink):
        """Execute one already-gated development lease through the native loop.

        This separate entry never admits calibration trajectories to offline train
        updates and never weakens the calibrated-limit guard on run_scenario.
        """
        scope = self.scope
        if (purpose != 'online_token_calibration' or type(identity) is not EpisodeIdentity
                or identity.phase != 'online_token_calibration_pilot'
                or identity.round_index is not None or identity.shard_id is not None
                or identity.generation_id != 'g000' or scope.generation_id != 'g000'
                or identity.system_variant != 'generation_0'
                or identity.run_id != scope.run_id or identity.profile != scope.profile
                or scope.scenario_positions.get(identity.scenario_id) != identity.manifest_position):
            raise PermissionError('calibration lease scope mismatch')
        for name in ('dataset_manifest_sha256', 'runtime_config_sha256', 'prompt_manifest_sha256',
                     'token_limit_config_sha256', 'fixture_manifest_sha256', 'environment_sha256'):
            if getattr(identity, name) != getattr(scope, name):
                raise PermissionError('calibration manifest binding mismatch')
        if any(not callable(value) for value in (authorization, prepared_request_sink, role_execution_sink)):
            raise TypeError('explicit calibration authorization and capture required')
        identity.validate_manifest_record(lease.record)
        if self._active:
            raise RuntimeError('concurrent native episodes are forbidden')
        dispatch_key = 'live-calibration-dispatch-' + canonical_sha256(
            [identity.run_id, identity.episode_id, identity.scenario_id, identity.manifest_position])[7:]
        if self.ledger.get_checkpoint(dispatch_key) is not None:
            raise LiveEpisodeResumeRequired('existing calibration episode requires explicit recovery')
        authorization(identity=identity, lease=lease, purpose=purpose)
        self.ledger.commit_checkpoint(dispatch_key, 'live_calibration_dispatch_reserved',
            {'identity': identity.model_dump(mode='json'), 'purpose': purpose,
             'eligible_for_train_offline_consumption': False})
        self._active = True
        try:
            return self._run_lease(lease, identity, calibration=True,
                prepared_request_sink=prepared_request_sink, role_execution_sink=role_execution_sink)
        finally:
            self._active = False

    def run_dev_lease(self, *, lease, identity, authorization, purpose, dev_manifest_sha256):
        """Only an already-selected Skill A/B branch may use this non-training entry."""
        scope = self.scope
        if (purpose != 'skill_ab_validation' or type(identity) is not EpisodeIdentity
                or identity.phase != 'dev_minibench' or identity.run_id != scope.run_id
                or identity.round_index != scope.round_index or identity.shard_id != scope.shard_id
                or identity.generation_id != scope.generation_id or identity.profile != scope.profile
                or identity.dataset_manifest_sha256 != dev_manifest_sha256
                or scope.scenario_positions.get(identity.scenario_id) != identity.manifest_position):
            raise PermissionError('Dev mini-bench scope mismatch')
        for name in ('runtime_config_sha256', 'prompt_manifest_sha256', 'token_limit_config_sha256',
                     'fixture_manifest_sha256', 'environment_sha256'):
            if getattr(identity, name) != getattr(scope, name):
                raise PermissionError('Dev mini-bench manifest mismatch')
        identity.validate_manifest_record(lease.record)
        if self._active:
            raise RuntimeError('concurrent native episodes are forbidden')
        key = 'live-dev-dispatch-' + canonical_sha256([identity.run_id, identity.episode_id])[7:]
        if self.ledger.get_checkpoint(key) is not None:
            raise LiveEpisodeResumeRequired('existing Dev branch requires explicit recovery')
        authorization(identity=identity, lease=lease, purpose=purpose)
        self.ledger.commit_checkpoint(key, 'live_dev_dispatch_reserved',
            {'identity': identity.model_dump(mode='json'), 'purpose': purpose,
             'eligible_for_train_offline_consumption': False})
        self._active = True
        try:
            return self._run_lease(lease, identity, offline_eligible=False)
        finally:
            self._active = False

    def _run_lease(self, lease, identity, *, calibration=False,
                   prepared_request_sink=None, role_execution_sink=None, offline_eligible=True):
        ledger, store = self.ledger, self.trajectory_store
        responders, bindings, captured = {}, [], {}
        active_external = {}
        by_name = {item.canonical_tool_name: item for item in self.metadata}
        accounting = AccountingScope(run_id=identity.run_id, round_index=identity.round_index,
            task_id=identity.episode_id, scenario_family_id=identity.family_id,
            scenario_id=identity.scenario_id, system_variant=identity.system_variant)

        def factory(turn_index):
            visible = extract_visible_messages(PipelineAgent)
            adapter = build_adapter_turn(PipelineAgent, visible)
            mapping = dict(adapter.controller_context.agent_to_execution_name)
            outcomes = committed_outcomes(bindings, environment.records, visible)
            online_identity = OnlineTurnIdentity(run_id=identity.run_id, profile=ReproducibilityProfile(identity.profile),
                phase=identity.phase, round_index=identity.round_index, shard_id=identity.shard_id,
                family_id=identity.family_id, scenario_id=identity.scenario_id, episode_id=identity.episode_id,
                agent_turn_index=turn_index, expected_generation_id=identity.generation_id,
                dataset_manifest_sha256=identity.dataset_manifest_sha256, runtime_config_sha256=identity.runtime_config_sha256,
                prompt_manifest_sha256=identity.prompt_manifest_sha256, token_limit_config_sha256=identity.token_limit_config_sha256,
                fixture_manifest_sha256=identity.fixture_manifest_sha256, environment_identity=identity.environment_sha256)
            context = self.context_factory(LiveTurnRequest(online_identity, accounting, adapter, outcomes))
            if (type(context) is not TurnContext or context.identity != online_identity
                    or context.generation is not self.generation or context.committed_tool_outcomes != outcomes):
                raise ValueError('coordinator turn context identity mismatch')
            original = context.role_request_builders
            def validate(prepared):
                if calibration:
                    if (prepared.token_limit_config_status not in ('provisional', 'calibration')
                            or prepared.token_limit_config_sha256 != identity.token_limit_config_sha256):
                        raise ValueError('pinned calibration online limits required')
                    prepared_request_sink(turn_index, prepared)
                elif (prepared.token_limit_config_status != 'calibrated'
                      or prepared.token_limit_config_sha256 != identity.token_limit_config_sha256):
                    raise ValueError('formal calibrated online limits required')
                return prepared
            def initial(state, retrieval):
                prepared = original.initial_policy(state, retrieval)
                validate(prepared.request)
                text = prepared.context.user_envelope
                checkpoint_id = 'live-initial-' + canonical_sha256([identity.episode_id, turn_index])[7:]
                ledger.commit_checkpoint(checkpoint_id, 'live_initial_envelope',
                    {'episode_id': identity.episode_id, 'turn_index': turn_index, 'user_envelope': text})
                if turn_index in captured and captured[turn_index] != text:
                    raise ValueError('initial envelope changed for same turn')
                captured[turn_index] = text
                return prepared
            def critic(*args):
                prepared = original.critic(*args); validate(prepared.request); return prepared
            def revision(*args):
                return validate(original.revision(*args))
            context = replace(context, role_request_builders=TurnRoleRequestBuilders(initial, critic, revision))
            if calibration:
                class CapturingRoles:
                    def __init__(self, delegate):
                        self.delegate = delegate
                    def execute(self, prepared):
                        execution = self.delegate.execute(prepared)
                        if type(execution) is not DurableRoleExecution or execution.prepared_request != prepared:
                            raise ValueError('calibration execution capture identity mismatch')
                        role_execution_sink(turn_index, execution)
                        return execution
                    def apply(self, *args, **kwargs):
                        return self.delegate.apply(*args, **kwargs)
                    def commit_online_action(self, *args, **kwargs):
                        return self.delegate.commit_online_action(*args, **kwargs)
                context = replace(context, durable_roles=CapturingRoles(context.durable_roles))
            responder = DurableTurnResponder(context)
            responders[turn_index] = (responder, mapping)
            return responder

        def audit_loader(decision):
            responder, _ = responders[decision.identity.agent_turn_index]
            event = ledger.get_checkpoint(responder._final_checkpoint_id(responder._input_fingerprint))
            return OnlineTurnAuditRecord.model_validate_json(json.dumps(event.payload['audit_record']))

        agent = TransactionalAgentRole(identity=identity, trajectory_store=store,
                                       responder_factory=factory, audit_loader=audit_loader)
        def binding(messages):
            ordinal = len(environment.records) + 1
            if messages[0].sender == RoleType.USER:
                return ToolActionBinding(action_sha256=canonical_sha256([m.content for m in messages]),
                    call_ids=tuple(m.openai_tool_call_id or 'user-control-' + canonical_sha256(
                        [identity.episode_id, ordinal, index, m.content])[7:] for index, m in enumerate(messages)),
                    selected_skill_ids=(None,) * len(messages), canonical_tool_ids=('end_conversation',) * len(messages),
                    effect_classes=('conversation_control',) * len(messages), action_ordinal=ordinal)
            decision = agent.decisions[-1]
            action = decision.final_action.action
            calls = tuple(action.calls) if hasattr(action, 'calls') else (action,)
            _, mapping = responders[decision.identity.agent_turn_index]
            canonical = tuple(mapping[call.name] for call in calls)
            bound = ToolActionBinding(action_sha256=canonical_sha256(decision.final_action.model_dump(mode='json')),
                call_ids=execution_call_ids(decision), selected_skill_ids=tuple(call.selected_skill_id for call in calls),
                canonical_tool_ids=canonical, effect_classes=tuple(by_name[name].effect.value for name in canonical),
                action_ordinal=ordinal)
            bindings.append((bound, calls, mapping))
            active_external.update(binding=bound, state_id=decision.state_id)
            return bound
        external_factory = self.external_boundary_factory
        if self.contextual_external_boundary_factory is not None:
            def external_factory(attempt_sink):
                if not active_external:
                    raise ValueError('external boundary requires an actual current action binding')
                return self.contextual_external_boundary_factory(identity=identity,
                    binding=active_external['binding'], state_id=active_external['state_id'],
                    attempt_sink=attempt_sink)
        environment = TransactionalExecutionEnvironment(ExecutionEnvironment(), identity=identity,
            tool_ledger=ToolLedger(ledger.store), trajectory_store=store, binding_provider=binding,
            backend_manifest_sha256=self.backend_manifest_sha256,
            external_boundary_factory=external_factory)
        user = self.user_factory(identity, store, ledger)
        roles = {RoleType.AGENT: agent, RoleType.EXECUTION_ENVIRONMENT: environment,
                 RoleType.USER: TransactionalUserRole(user, identity=identity, trajectory_store=store)}
        runner = EpisodeRunner(trajectory_store=store, native_evaluator=NativeEvaluator(store),
            physical_attempt_provider=lambda ids: self.physical_attempt_provider(identity, ids), boot_id=self.boot_id)
        result = runner.run(EpisodeRunInput(scenario=lease.scenario, manifest_record=lease.record,
            identity=identity, roles=roles, skill_versions={skill.skill_id: skill.version for skill in self.generation.skills},
            eligible_for_train_offline_consumption=offline_eligible and not calibration))
        decisions = tuple(agent.decisions)
        # A recovered final turn may not call the Initial builder again. Restore its
        # exact prior envelope from the deterministic checkpoint, never regenerate.
        for decision in decisions:
            index = decision.identity.agent_turn_index
            if index not in captured:
                event = ledger.get_checkpoint('live-initial-' + canonical_sha256([identity.episode_id, index])[7:])
                if event is None:
                    raise ValueError('missing durable initial envelope')
                captured[index] = event.payload['user_envelope']
        proposed = tuple(ActionEnvelope.model_validate_json(json.dumps(
            ledger.load_completed_output(decision.initial_policy_logical_request_id).output)) for decision in decisions)
        self.capture_sink(LiveEpisodeCapture(identity, tuple(sorted(captured.items())), decisions, proposed, result))
        return result


__all__ = ['LiveEpisodeScope', 'LiveTurnRequest', 'LiveEpisodeCapture', 'LiveEpisodeExecutor', 'LiveEpisodeResumeRequired']
