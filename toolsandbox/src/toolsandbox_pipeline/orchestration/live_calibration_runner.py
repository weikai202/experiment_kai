"""Coordinator entry for real natural pilots and durable frozen-input replay.

Call inside the authorized fixed world clock with an already validated train gate,
G000 snapshot, ledger, token counter and external boundary. No ambient credentials,
registry access, automatic promotion, or fabricated role inputs are provided here.
"""
from __future__ import annotations

from toolsandbox_pipeline.checkpointing import LogicalLLMRequestIdentity
from toolsandbox_pipeline.online.durable_roles import LedgerQwenResponseSeam
from toolsandbox_pipeline.online.qwen_roles import InitialPolicyRunner, CriticRunner, RevisionRunner
from toolsandbox_pipeline.orchestration.live_calibration import collect_and_calibrate, PHASE
from toolsandbox_pipeline.orchestration.live_episode import LiveEpisodeExecutor
from toolsandbox_pipeline.orchestration.live_pilot import LiveCalibrationPilot
from toolsandbox_pipeline.orchestration.live_providers import LiveProviderServices
from toolsandbox_pipeline.orchestration.live_turn_context import LiveTurnContextFactory
from toolsandbox_pipeline.providers.contracts import ProviderRole, ProviderRequestError
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope


class DurableCalibrationReplay:
    """Fresh replay IDs; all failures including length remain ledger-accounted.

    State bindings come only from completed natural pilot receipts. The outer
    entry is fresh-only; crash recovery must be explicitly coordinated, not
    silently replay a prefix of paid calls under recycled attempt identities.
    """
    def __init__(self, *, ledger, gateways, qwen_config, manifest_identity, run_id):
        if ledger.store.identity.run_id != run_id or set(gateways) != {'policy', 'critic', 'revision'}:
            raise ValueError('calibration replay ledger/role identity mismatch')
        if any(gateway.config != qwen_config for gateway in gateways.values()):
            raise ValueError('calibration gateway configuration mismatch')
        self.ledger, self.gateways, self.qwen = ledger, gateways, qwen_config
        self.manifest_identity, self.run_id = manifest_identity, run_id
        self._bindings, self._sealed, self._ordinal = {}, set(), 0

    def persist_receipt(self, receipt):
        requests = []
        for item in receipt.requests:
            captured = item.captured
            stored = self.ledger.load_completed_output(item.logical_request_id)
            material = self.ledger.load_completed_response_material(item.logical_request_id)
            if (stored.source_attempt_id != item.source_attempt_id
                    or material.attempt.context.input_fingerprint != captured.prepared.canonical_input_fingerprint):
                raise ValueError('pilot receipt source attempt mismatch')
            binding = (captured.scenario_id, captured.scenario_family_id)
            state_id = captured.prepared.state_id
            if state_id in self._bindings and self._bindings[state_id] != binding:
                raise ValueError('ambiguous captured state binding')
            self._bindings[state_id] = binding
            self._sealed.add(self._content_identity(captured.prepared))
            requests.append({'captured': captured.model_dump(mode='json'),
                             'logical_request_id': item.logical_request_id,
                             'source_attempt_id': item.source_attempt_id})
        payload = dict(episode_id=receipt.episode_id, scenario_id=receipt.scenario_id,
            scenario_family_id=receipt.scenario_family_id, status=receipt.status,
            pilot_checkpoint_id=receipt.checkpoint_id, requests=requests)
        self.ledger.commit_checkpoint('calibration-corpus-receipt-' + canonical_sha256(payload)[7:],
                                      'calibration_corpus_receipt', payload)

    @staticmethod
    def _content_identity(prepared):
        return canonical_sha256({key: value for key, value in prepared.model_dump(mode="json").items()
            if key not in {"max_tokens", "selected_limit", "token_limit_config_status",
                           "token_limit_config_sha256", "canonical_input_fingerprint"}})

    def __call__(self, prepared):
        if (prepared.state_id not in self._bindings or prepared.generation_id != 'g000'
                or self._content_identity(prepared) not in self._sealed):
            raise ValueError('replay request lacks natural G000 pilot binding')
        sid, family = self._bindings[prepared.state_id]
        ordinal = self._ordinal
        self._ordinal += 1
        unit = f'online-calibration-replay-{ordinal}'
        identity = LogicalLLMRequestIdentity(run_id=self.run_id, role=ProviderRole(prepared.role),
            phase=f'online_token_calibration_replay_{ordinal}', unit_reference=prepared.state_id,
            input_fingerprint=prepared.canonical_input_fingerprint, model=self.qwen.model,
            decoding_configuration_sha256=canonical_sha256(self.qwen.model_dump(mode='json')),
            output_schema_sha256=prepared.output_schema_sha256)
        record = self.ledger.prepare_request(identity)
        self.ledger.bind_accounting_scope(record.logical_request_id, AccountingScope(run_id=self.run_id,
            round_index=None, task_id=unit, scenario_id=sid, scenario_family_id=family,
            system_variant='generation_0'))
        self.ledger.commit_checkpoint('calibration-replay-request-' + canonical_sha256([self.run_id, ordinal])[7:],
            'calibration_replay_request', {'ordinal': ordinal, 'logical_request_id': record.logical_request_id,
                                         'prepared_request': prepared.model_dump(mode='json')})
        context = self.ledger.allocate_attempt(record.logical_request_id,
            manifest_identity=self.manifest_identity, replayed_after_unknown_outcome=False)
        # Role runners bind unit_reference to the original state. The logical
        # dispatch identity is unique independently through its replay phase.
        if context.unit_reference != prepared.state_id:
            raise ValueError('replay ledger unit must match original prepared state')
        self.ledger.mark_in_flight(context)
        cls = {'policy': InitialPolicyRunner, 'critic': CriticRunner, 'revision': RevisionRunner}[prepared.role]
        runner = cls(self.gateways[prepared.role], manifest_identity=self.manifest_identity,
                     mode='calibration', durability_seam=LedgerQwenResponseSeam(self.ledger))
        try:
            result = runner.run(prepared, context)
        except ProviderRequestError as error:
            self.ledger.record_failure(error)
            raise
        if result.truncated:
            self.ledger.record_failure(ProviderRequestError(result.attempt))
        self.ledger.commit_checkpoint('calibration-replay-result-' + canonical_sha256([self.run_id, ordinal])[7:],
            'calibration_replay_result', result.model_dump(mode='json'))
        return result


def run_live_calibration(*, scope, records, gate, generation, ledger, trajectory_store,
        qwen_config, embedding_config, user_config, prompts, token_limits, metadata,
        embedding_cache, count_prompt_tokens, runtime_inputs, authorization,
        physical_attempt_provider, boot_id, output_directory, hard_ceiling,
        external_boundary_factory=None, contextual_external_boundary_factory=None,
        backend_manifest_sha256=None, qwen_transport=None, embedding_transport=None, user_transport=None):
    """Run actual wiring; returns CalibrationCollectionResult, never promotes it.

    Caller owns runtime loading, G000 construction, fixed world clock, provider
    preflights and authorization. Optional transports support offline tests only;
    omission uses the configured real providers with no fallback.
    """
    if scope.generation_id != 'g000' or scope.round_index != 0:
        raise ValueError('calibration entry requires generation-zero scope')
    if ledger.store.identity.config_manifest_sha256 != scope.runtime_config_sha256:
        raise ValueError('calibration runtime manifest mismatch')
    authorization(scope)
    checkpoint = 'live-calibration-entry-' + canonical_sha256(scope.run_id)[7:]
    if ledger.get_checkpoint(checkpoint) is not None:
        raise RuntimeError('existing calibration entry requires explicit recovery')
    services = LiveProviderServices(ledger=ledger, qwen_config=qwen_config,
        embedding_config=embedding_config, user_config=user_config,
        manifest_identity=scope.runtime_config_sha256, phase=PHASE,
        qwen_transport=qwen_transport, embedding_transport=embedding_transport, user_transport=user_transport)
    factory = LiveTurnContextFactory(generation=generation, ledger=ledger,
        qwen_config=qwen_config, gateways=services.gateways, prompts=prompts,
        token_limits=token_limits, token_limit_config_sha256=scope.token_limit_config_sha256,
        manifest_identity=scope.runtime_config_sha256, metadata=metadata,
        embedding_cache=embedding_cache, embedding_gateway=services.embedding_gateway,
        embedding_context_factory=services.embedding_context_factory,
        embedding_record_durable=services.embedding_record_durable,
        count_prompt_tokens=count_prompt_tokens, mode='calibration')
    def capture(captured):
        if captured.identity.round_index is not None or captured.identity.phase != PHASE:
            raise ValueError('calibration capture cannot enter a train update round')
        payload = {'identity': captured.identity.model_dump(mode='json'),
            'initial_envelopes': [list(item) for item in captured.initial_envelopes],
            'decisions': [item.model_dump(mode='json') for item in captured.decisions],
            'proposed_actions': [item.model_dump(mode='json') for item in captured.proposed_actions],
            'result': captured.result.model_dump(mode='json')}
        ledger.commit_checkpoint('calibration-episode-capture-' + canonical_sha256(captured.identity.episode_id)[7:],
                                 'calibration_episode_capture', payload)
    executor = LiveEpisodeExecutor(scope=scope, gate=gate, generation=generation,
        ledger=ledger, trajectory_store=trajectory_store, metadata=metadata, context_factory=factory,
        user_factory=services.user_factory, authorization=authorization,
        physical_attempt_provider=physical_attempt_provider, boot_id=boot_id, capture_sink=capture,
        external_boundary_factory=external_boundary_factory,
        contextual_external_boundary_factory=contextual_external_boundary_factory,
        backend_manifest_sha256=backend_manifest_sha256)
    replay = DurableCalibrationReplay(ledger=ledger, gateways=services.gateways,
        qwen_config=qwen_config, manifest_identity=scope.runtime_config_sha256, run_id=scope.run_id)
    ledger.commit_checkpoint(checkpoint, 'live_calibration_entry_reserved',
        {'run_id': scope.run_id, 'runtime_inputs': runtime_inputs.model_dump(mode='json'),
         'token_limit_config_sha256': scope.token_limit_config_sha256})
    return collect_and_calibrate(records=records, train_manifest_sha256=scope.dataset_manifest_sha256,
        run_id=scope.run_id, gate=gate, pilot_executor=LiveCalibrationPilot(executor=executor, authorization=authorization),
        persist_receipt=replay.persist_receipt, runtime_inputs=runtime_inputs, qwen=qwen_config,
        hard_ceiling=hard_ceiling, invoke=replay, output_directory=output_directory)
