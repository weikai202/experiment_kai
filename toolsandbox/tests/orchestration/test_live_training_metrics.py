from types import SimpleNamespace
from toolsandbox_pipeline.orchestration.live_training_metrics import LiveTrainingMetrics
from toolsandbox_pipeline.metrics.timing import ScopeTimer
from tests.orchestration.test_live_offline import live, prepared, D


def test_real_ledger_usage_persists_three_round_and_run_artifacts(live, tmp_path):
    live.requests.memory_executor().execute_and_apply(prepared(live))
    manifest = SimpleNamespace(run_id='run',run_root=tmp_path / "run", qwen=SimpleNamespace(model=live.requests.gateway.config.model),
        profile='official_live', manifest_sha256=D)
    metrics = LiveTrainingMetrics(ledger=live.ledger, manifest=manifest)
    timer = ScopeTimer(scope_kind='training_run',scope_id='run',boot_id='boot')
    for index in range(3):
        scope = ScopeTimer(scope_kind='round_total',scope_id=f'round-{index}',boot_id='boot')
        record = metrics.finalize_round(round_index=index,input_generation_id=f'g{index:03d}',train_shard_id=f'train-shard-{index}',
            published_generation_id=f'g{index+1:03d}',timing=scope.close(),completion_status='complete',
            episode_results=(),memory_result=None,skill_result=None)
        assert record.totals.total_cost == 0
    result = metrics.finalize_training(timing=timer.close(),completion_status='complete',round_results=())
    assert result.round_record_count == 3
    assert result.request_record_count == 1


def test_episode_metrics_and_offline_scopes_remain_distinct(live,tmp_path):
    from toolsandbox_pipeline.schemas.accounting import TaskAccountingInput
    live.requests.memory_executor().execute_and_apply(prepared(live))
    manifest=SimpleNamespace(run_id='run',run_root=tmp_path/'run',qwen=SimpleNamespace(model=live.requests.gateway.config.model),profile='official_live',manifest_sha256=D)
    metrics=LiveTrainingMetrics(ledger=live.ledger,manifest=manifest)
    attempts,logical,_=metrics._snapshot(0)
    assert all(row.task_id is None and row.scenario_id is None for row in (*attempts,*logical))
    task=TaskAccountingInput(run_id='run',task_id='episode',scenario_family_id='family',scenario_id='scenario',system_variant='generation_0',
        timing=ScopeTimer('scenario_task','episode','boot').close(),completion_status='complete')
    result=SimpleNamespace(identity=SimpleNamespace(episode_id='episode',round_index=0),task_accounting_input=task)
    record=metrics.finalize_round(round_index=0,input_generation_id='g000',train_shard_id='train-shard-0',published_generation_id='g001',
        timing=ScopeTimer('round_total','round','boot').close(),completion_status='complete',episode_results=(result,),memory_result=None,skill_result=None)
    assert metrics.tasks['episode'].totals.total_tokens==0
    assert record.totals.total_tokens>0
