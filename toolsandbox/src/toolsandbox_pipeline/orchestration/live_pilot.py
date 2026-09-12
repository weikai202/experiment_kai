"""Natural calibration pilot adapter over the shared native episode executor."""
from __future__ import annotations

from toolsandbox_pipeline.online.durable_roles import DurableRoleExecution
from toolsandbox_pipeline.online.prompt_contracts import PreparedRoleRequest
from toolsandbox_pipeline.online.token_limit_calibration import CapturedCalibrationRequest
from toolsandbox_pipeline.orchestration.live_calibration import PilotCapturedRequest, PilotReceipt, PURPOSE, PHASE
from toolsandbox_pipeline.orchestration.live_episode import LiveEpisodeExecutor
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.trajectory import EpisodeExecutionStatus, EpisodeResult


class LiveCalibrationPilot:
    """Callable accepted by collect_and_calibrate's pilot_executor parameter.

    The supplied executor must use the actual native loop and authorized providers
    injected by the coordinator. Completed role executions originate from the
    durable executor; this adapter creates no prompts, role calls, or model results.
    Failed attempts remain in that executor's authoritative ledger, while all
    prepared requests are separately checkpointed before role dispatch.
    """
    def __init__(self, *, executor: LiveEpisodeExecutor, authorization):
        if type(executor) is not LiveEpisodeExecutor or not callable(authorization):
            raise TypeError('shared native executor and explicit pilot authorization required')
        if executor.scope.generation_id != 'g000':
            raise ValueError('calibration pilot requires pinned Generation 0')
        self.executor, self.authorization = executor, authorization

    def __call__(self, *, lease, run_id, purpose):
        engine, scope = self.executor, self.executor.scope
        if purpose != PURPOSE or run_id != scope.run_id or lease.record.scenario_id not in scope.scenario_positions:
            raise PermissionError('calibration pilot scope mismatch')
        position = scope.scenario_positions[lease.record.scenario_id]
        base = engine._identity(lease.record, position)
        identity = type(base).model_validate({**base.model_dump(),
            'phase': PHASE, 'round_index': None, 'shard_id': None,
            'episode_id': f'{run_id}-online-calibration-episode-{position}'})
        prepared_by_key, ordered_prepared, executions = {}, [], []

        def prepared_sink(turn_index, prepared):
            if type(prepared) is not PreparedRoleRequest:
                raise TypeError('exact naturally prepared role request required')
            key = (turn_index, prepared.role)
            if key in prepared_by_key:
                if prepared_by_key[key] != prepared:
                    raise ValueError('pilot prepared request changed within one role turn')
                return
            payload = {'episode_id': identity.episode_id, 'turn_index': turn_index,
                       'prepared_request': prepared.model_dump(mode='json')}
            checkpoint_id = 'pilot-prepared-' + canonical_sha256([identity.episode_id, turn_index, prepared.role])[7:]
            engine.ledger.commit_checkpoint(checkpoint_id, 'online_calibration_prepared_request', payload)
            prepared_by_key[key] = prepared
            ordered_prepared.append(key)

        def execution_sink(turn_index, execution):
            if type(execution) is not DurableRoleExecution:
                raise TypeError('exact durable role execution required')
            key = (turn_index, execution.prepared_request.role)
            if prepared_by_key.get(key) != execution.prepared_request:
                raise ValueError('pilot execution did not match captured natural request')
            if any(previous[0] == key for previous in executions):
                raise ValueError('duplicate natural role execution')
            stored = engine.ledger.load_completed_output(execution.logical_request_id)
            if (stored.logical_request_id != execution.logical_request_id
                or stored.source_attempt_id != execution.source_attempt_id
                or stored.output != execution.output.model_dump(mode='json')):
                raise ValueError('pilot source attempt does not match durable output')
            executions.append((key, execution))

        result = engine.run_calibration_lease(lease=lease, identity=identity,
            authorization=self.authorization, purpose=purpose,
            prepared_request_sink=prepared_sink, role_execution_sink=execution_sink)
        if type(result) is not EpisodeResult or result.identity != identity:
            raise ValueError('native pilot result identity mismatch')
        completed = result.status is EpisodeExecutionStatus.COMPLETED_EVALUATED
        if completed and [key for key, _ in executions] != ordered_prepared:
            raise ValueError('completed pilot is missing durable natural role executions')
        requests = tuple(PilotCapturedRequest(CapturedCalibrationRequest(
            scenario_id=identity.scenario_id, scenario_family_id=identity.family_id,
            prepared=execution.prepared_request), execution.logical_request_id, execution.source_attempt_id)
            for _, execution in executions)
        payload = {'purpose': purpose, 'episode_id': identity.episode_id,
            'scenario_id': identity.scenario_id, 'status': result.status.value,
            'last_checkpoint_ordinal': result.last_checkpoint_ordinal,
            'prepared_request_count': len(ordered_prepared), 'completed_request_count': len(executions),
            'requests': [{'logical_request_id': request.logical_request_id,
                'source_attempt_id': request.source_attempt_id,
                'input_fingerprint': request.captured.prepared.canonical_input_fingerprint}
                for request in requests]}
        checkpoint_id = 'pilot-receipt-' + canonical_sha256([identity.run_id, identity.episode_id])[7:]
        engine.ledger.commit_checkpoint(checkpoint_id, 'online_calibration_pilot_receipt', payload)
        return PilotReceipt(identity.scenario_id, identity.family_id, identity.episode_id,
            'completed_evaluated' if completed else 'failed', requests, checkpoint_id)


__all__ = ['LiveCalibrationPilot']
