"""Explicit real retrieval, grounded state and durable online-role wiring.

No provider is constructed from ambient configuration. Calibration runs the same
online algorithm; only request-limit eligibility differs from formal execution.
"""
from __future__ import annotations

import typing
from types import MappingProxyType

from pydantic import TypeAdapter

from toolsandbox_pipeline.online.controller import Controller
from toolsandbox_pipeline.online.durable_roles import (
    DurableRoleExecutor, LedgerQwenResponseSeam, Task011CheckpointEventSink,
    Task011DurableRoleBackend,
)
from toolsandbox_pipeline.online.prompt_builder import initial_context, critic_context, prepare_request
from toolsandbox_pipeline.online.prompt_contracts import RevisionContext
from toolsandbox_pipeline.online.qwen_roles import InitialPolicyRunner, CriticRunner, RevisionRunner
from toolsandbox_pipeline.online.state_builder import StateBuilder
from toolsandbox_pipeline.online.turn_context import (
    TurnContext, TurnRoleRequestBuilders, PreparedInitialPolicy, PreparedCritic,
)
from toolsandbox_pipeline.orchestration.live_episode import LiveTurnRequest
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.retrieval.service import RetrievalService


class LiveTurnContextFactory:
    """Callable for LiveEpisodeExecutor.context_factory.

    embedding_context_factory(accounting_scope, ordinal, texts) must reserve a
    real ledger request. embedding_record_durable(response) verifies its durable
    commit. Each turn closes over its own accounting scope. Prepared-request
    capture belongs to the executor, so every natural request is captured once.
    """

    def __init__(self, *, generation, ledger, qwen_config, gateways, prompts,
                 token_limits, token_limit_config_sha256, manifest_identity,
                 metadata, embedding_cache, embedding_gateway,
                 embedding_context_factory, embedding_record_durable,
                 count_prompt_tokens, mode, runtime_inputs_sha256=None):
        if mode not in ('calibration', 'formal'):
            raise ValueError('explicit calibration or formal mode required')
        if set(gateways) != {'policy', 'critic', 'revision'} or set(prompts) != set(gateways):
            raise ValueError('exact online role mappings required')
        if any(prompts[role].entry.role != role for role in prompts):
            raise ValueError('prompt role mismatch')
        if any(gateway.config != qwen_config for gateway in gateways.values()):
            raise ValueError('gateway decoding configuration mismatch')
        if any(not callable(value) for value in (
                embedding_context_factory, embedding_record_durable, count_prompt_tokens)):
            raise TypeError('explicit embedding durability and token counter required')
        if mode == 'formal':
            qwen_config.validate_external()
            if token_limits.status != 'calibrated' or runtime_inputs_sha256 is None:
                raise ValueError('formal calibrated limits and runtime bindings required')
            if token_limits.runtime_inputs_sha256 != runtime_inputs_sha256:
                raise ValueError('formal calibration runtime bindings mismatch')
        self.generation, self.ledger, self.qwen_config = generation, ledger, qwen_config
        self.prompts = MappingProxyType(dict(prompts))
        self.token_limits, self.token_limit_config_sha256 = token_limits, token_limit_config_sha256
        self.manifest_identity, self.metadata = manifest_identity, tuple(metadata)
        self.embedding_cache, self.embedding_gateway = embedding_cache, embedding_gateway
        self.embedding_context_factory = embedding_context_factory
        self.embedding_record_durable = embedding_record_durable
        self.count_prompt_tokens, self.mode = count_prompt_tokens, mode
        self.runtime_inputs_sha256 = runtime_inputs_sha256
        self._contracts = {}
        seam = LedgerQwenResponseSeam(ledger)
        runner_mode = 'formal' if mode == 'formal' else 'offline'
        self.runners = {
            role: runner(gateways[role], manifest_identity=manifest_identity,
                         mode=runner_mode, durability_seam=seam)
            for role, runner in (('policy', InitialPolicyRunner), ('critic', CriticRunner),
                                 ('revision', RevisionRunner))
        }

    def __call__(self, request: LiveTurnRequest) -> TurnContext:
        if type(request) is not LiveTurnRequest:
            raise TypeError('LiveTurnRequest required')
        identity, scope = request.identity, request.accounting_scope
        if (identity.expected_generation_id != self.generation.manifest.generation_id
                or identity.runtime_config_sha256 != self.manifest_identity
                or identity.token_limit_config_sha256 != self.token_limit_config_sha256
                or identity.run_id != self.ledger.store.identity.run_id
                or scope.run_id != identity.run_id or scope.task_id != identity.episode_id
                or scope.scenario_id != identity.scenario_id
                or scope.scenario_family_id != identity.family_id
                or scope.round_index != identity.round_index
                or scope.system_variant != ('generation_0' if identity.expected_generation_id == 'g000' else 'updated')):
            raise ValueError('live turn manifest or accounting scope mismatch')
        if self.mode == 'calibration' and (identity.phase != 'online_token_calibration_pilot'
                or identity.round_index is not None or identity.shard_id is not None
                or identity.expected_generation_id != 'g000'):
            raise ValueError('calibration requires an isolated generation-zero pilot')
        if self.mode == 'formal' and identity.phase == 'online_token_calibration_pilot':
            raise ValueError('formal factory cannot execute a calibration pilot')
        mapping = dict(request.adapter_turn.controller_context.agent_to_execution_name)
        reverse = {value: key for key, value in mapping.items()}
        if len(reverse) != len(mapping):
            raise ValueError('bijective visible tool mapping required')
        contracts = self._contracts.setdefault(identity.episode_id, {})
        for name, fn in request.adapter_turn.controller_context.tool_objects.items():
            result = typing.get_type_hints(fn).get('return')
            if result is not None:
                contracts[mapping[name]] = TypeAdapter(result)
        retrieval = RetrievalService(self.generation, cache=self.embedding_cache,
            gateway=self.embedding_gateway,
            context_factory=lambda ordinal, texts: self.embedding_context_factory(scope, ordinal, texts),
            record_durable=self.embedding_record_durable)

        def prepared(context, role):
            return prepare_request(context, prompt=self.prompts[role], token_limits=self.token_limits,
                token_limit_config_sha256=self.token_limit_config_sha256, qwen_config=self.qwen_config,
                canonical_to_agent=reverse, mode=self.mode, count_prompt_tokens=self.count_prompt_tokens,
                runtime_inputs_sha256=self.runtime_inputs_sha256)

        def initial(state, bundle):
            context = initial_context(state, bundle, self.prompts['policy'])
            return PreparedInitialPolicy(context, prepared(context, 'policy'))

        def critic(initial, action, decision, bundle):
            context = critic_context(initial, action, decision, bundle)
            return PreparedCritic(context, prepared(context, 'critic'))

        def revision(initial, critic, output):
            context = RevisionContext(initial=initial, critic=critic,
                                      critic_feedback_json=output.model_dump_json())
            return prepared(context, 'revision')

        backend = Task011DurableRoleBackend(ledger=self.ledger, runners=self.runners,
            run_id=identity.run_id, phase=identity.phase, qwen_model=self.qwen_config.model,
            decoding_configuration_sha256=canonical_sha256(self.qwen_config.model_dump(mode='json')),
            manifest_identity=self.manifest_identity, accounting_scope=scope)
        return TurnContext(identity=identity, generation=self.generation, retrieval=retrieval,
            state_builder=StateBuilder(dict(contracts)), controller=Controller(),
            controller_tool_metadata=self.metadata,
            role_request_builders=TurnRoleRequestBuilders(initial, critic, revision),
            durable_roles=DurableRoleExecutor(backend), checkpoint_sink=Task011CheckpointEventSink(self.ledger),
            committed_tool_outcomes=request.committed_tool_outcomes)
