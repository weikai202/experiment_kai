"""Direct round/run timing plus authoritative ledger usage and effective cost."""
from pathlib import Path

from toolsandbox_pipeline.metrics import MetricsAggregator, MetricsArtifactWriter
from toolsandbox_pipeline.schemas.accounting import (
    TaskMetricRecord, RoundMetricRecord, RunMetricRecord, RoundAccountingInput, RunAccountingInput,
)
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.checkpoint import QwenEffectKind


class LiveTrainingMetrics:
    def __init__(self, *, ledger, manifest):
        self.ledger, self.manifest = ledger, manifest
        self.aggregator = MetricsAggregator(qwen_provider='vllm_openai_compatible', qwen_model=manifest.qwen.model)
        self.episodes, self.rounds, self.requests, self.tasks = {}, [], {}, {}

    def record_episode(self, result):
        previous = self.episodes.get(result.identity.episode_id)
        if previous is not None and previous != result:
            raise ValueError('episode metrics identity conflict')
        self.episodes[result.identity.episode_id] = result

    def _snapshot(self, round_index):
        rows = self.ledger.store._connection.execute('SELECT DISTINCT scope FROM llm_accounting_scopes').fetchall()
        scopes = [AccountingScope.model_validate_json(bytes(row['scope'])) for row in rows]
        snapshots = [self.ledger.snapshot_accounting(scope) for scope in scopes
                     if scope.run_id == self.manifest.run_id and scope.round_index == round_index]
        def metric_scope(item):
            # Offline bookkeeping scopes are not scenario task identities.
            if item.phase in ('offline_update', 'generation_publication'):
                return item.model_copy(update={'task_id': None, 'scenario_family_id': None, 'scenario_id': None})
            return item
        attempts = tuple(metric_scope(item) for snapshot in snapshots for item in snapshot.physical_attempts)
        logical = tuple(metric_scope(item) for snapshot in snapshots for item in snapshot.logical_requests)
        dev_apps = {row.application_id for row in logical if row.phase == 'dev_minibench' and row.application_id}
        effects = tuple(effect for snapshot in snapshots for effect in snapshot.substantive_effects
                        if not (effect.effect_kind is QwenEffectKind.COMMITTED_ONLINE_ACTION
                                and set(effect.ordered_application_ids) <= dev_apps))
        # Dev online actions alone have no accepted Skill mutation. Their output
        # becomes cost-eligible only through an accepted Skill effect link.
        return attempts, logical, effects

    def finalize_round(self, *, round_index, input_generation_id, train_shard_id,
                       published_generation_id, timing, completion_status,
                       episode_results, memory_result, skill_result):
        for result in episode_results:
            self.record_episode(result)
        attempts, logical, effects = self._snapshot(round_index)
        totals = self.aggregator.aggregate(attempts, logical, effects)
        now = timing.ended_at_utc
        requests = self.aggregator.request_records(attempts, logical, effects, recorded_at_utc=now)
        tasks = []
        for result in self.episodes.values():
            if result.identity.round_index != round_index:
                continue
            task_attempts = tuple(row for row in attempts if row.task_id == result.identity.episode_id)
            task_logical = tuple(row for row in logical if row.task_id == result.identity.episode_id)
            applications = {row.application_id for row in task_logical if row.application_id}
            # Per-task views include only effects wholly inside this episode.
            task_effects = tuple(effect for effect in effects if set(effect.ordered_application_ids) <= applications)
            counts = {role: sum(row.role == role for row in task_logical) for role in ('policy','critic','revision','user_simulator')}
            task = TaskMetricRecord.build(recorded_at_utc=now, run_id=self.manifest.run_id,
                task=result.task_accounting_input,
                totals=self.aggregator.aggregate(task_attempts, task_logical, task_effects),
                breakdowns=self.aggregator.breakdowns(task_attempts, task_logical, task_effects), model_call_counts=counts)
            tasks.append(task)
        round_input = RoundAccountingInput(run_id=self.manifest.run_id, round_index=round_index,
            input_generation_id=input_generation_id, train_shard_id=train_shard_id,
            published_generation_id=published_generation_id, total_timing=timing, completion_status=completion_status)
        record = RoundMetricRecord.build(recorded_at_utc=now, run_id=self.manifest.run_id,
            round=round_input, totals=totals, started_at_utc=timing.started_at_utc, ended_at_utc=timing.ended_at_utc,
            total_running_time_seconds=timing.total_running_time_seconds, scenario_count=len(episode_results),
            **{name: getattr(totals, name) for name in ('total_tokens','usage_complete','total_cost','cost_unit','cost_complete')})
        self.ledger.commit_checkpoint(f'training-round-metrics-{round_index}', 'training_round_metrics', record.model_dump(mode='json'))
        self.rounds.append(record)
        self.requests.update({request.attempt.attempt_id: request for request in requests})
        self.tasks.update({task.task.task_id: task for task in tasks})
        root = Path(self.manifest.run_root) / f'round-{round_index}-metrics'
        root.mkdir(mode=0o700)
        self._write(root, requests, tasks, (record,), None)
        return record

    def _write(self, root, requests, tasks, rounds, run):
        return MetricsArtifactWriter(root).materialize(run_id=self.manifest.run_id,
            request_records=tuple(requests), task_records=tuple(tasks), round_records=tuple(rounds),
            run_record=run, ledger_high_water_marks=self.ledger.store.high_water_marks(),
            pinned_qwen_provider='vllm_openai_compatible', pinned_qwen_model=self.manifest.qwen.model)

    def finalize_training(self, *, timing, completion_status, round_results):
        all_snapshots = [self._snapshot(record.round.round_index) for record in self.rounds]
        joined = tuple(tuple(item for snapshot in all_snapshots for item in snapshot[column]) for column in range(3))
        totals = self.aggregator.aggregate(*joined)
        run = RunAccountingInput(run_id=self.manifest.run_id, run_kind='training', timing=timing,
            profile=self.manifest.profile, manifest_sha256=self.manifest.manifest_sha256,
            config_sha256=self.manifest.manifest_sha256, completion_status=completion_status)
        record = RunMetricRecord.build(recorded_at_utc=timing.ended_at_utc, run_id=self.manifest.run_id,
            run=run, totals=totals, total_running_time_seconds=timing.total_running_time_seconds,
            round_record_ids=tuple(item.record_id for item in self.rounds), request_record_count=len(self.requests),
            task_record_count=len(self.tasks), round_record_count=len(self.rounds),
            **{name: getattr(totals, name) for name in ('total_tokens','usage_complete','total_cost','cost_unit','cost_complete')})
        self._write(Path(self.manifest.run_root), self.requests.values(), self.tasks.values(), self.rounds, record)
        return record
