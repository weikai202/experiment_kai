"""Self-contained HTML snapshot; no external scripts, requests, or private data."""
from html import escape

LABELS={'online_token_calibration_pilot':'在线校准 · 真实轨迹','online_token_calibration':'在线校准 · 重放', 'offline_token_calibration':'离线校准','train_round':'正式训练','formal_training':'正式训练','online_calibration':'在线校准','offline_calibration':'离线校准','train_round_0':'正式训练 · 第 0 轮','train_round_1':'正式训练 · 第 1 轮','train_round_2':'正式训练 · 第 2 轮'}
def fmt(value,places=3):
    if value is None:return '—'
    if isinstance(value,bool):return '是' if value else '否'
    if isinstance(value,float):return f'{value:.{places}f}'
    return escape(str(value))
def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+(''.join('<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>' for row in rows) or '<tr><td colspan="20">尚无已完成记录</td></tr>')+'</tbody></table></div>'
def render(data):
    phases=data.get('phases',[])
    body='<h1>ToolSandbox 实时进度</h1><p>更新时间（UTC）：'+fmt(data.get('updated_at_utc'))+' · 服务：'+fmt(data.get('service',{}).get('state'))+'</p>'
    if data.get('stale'):body+='<p>当前为上一份已验证快照；数据库正在写入，本次跳过采样。最后验证：'+fmt(data.get('last_verified_at_utc'))+'</p>'
    body+='<p class="note">监测进程运行时每 15 秒生成快照；实验停止后保留最后快照。本页尝试每 15 秒刷新。Tokens 只累计已返回的 usage。校准分数仅用于诊断；不同场景或不同训练分片的均分不能直接解释为性能提升。未完成场景不记为零分。</p>'
    body+=table(['阶段','状态','已完成场景','失败','等待中','原生相似度均值'],[(LABELS.get(p.get('phase'),p.get('phase')),p.get('status'),p.get('episodes_completed'),p.get('episodes_failed',0),p.get('episodes_pending',0),p.get('cumulative_mean_similarity')) for p in phases])
    for phase in phases:
        name=phase.get('phase','unknown');body+='<h2>'+escape(LABELS.get(name,name))+'</h2>'
        collection=phase.get('collection')
        if collection:body+='<p>初始校准目标：'+fmt(collection.get('initial_target'))+' 条；最多扩展至 '+fmt(collection.get('reserve_limit'))+' 条。Policy/Critic/Revision 各需至少 32 个自然请求输入。</p>'
        roles=phase.get('roles',[])
        body+=table(['角色','逻辑请求','物理尝试','已记录 tokens','Usage 完整','最长输出 tokens','截断次数','自然唯一输入'],[(r.get('role'),r.get('logical_requests'),r.get('physical_attempts'),r.get('observed_total_tokens'),r.get('usage_complete'),r.get('max_output_tokens'),r.get('length_finishes'),r.get('natural_unique_inputs')) for r in roles])
        eps=phase.get('latest_episodes',[])
        body+=table(['场景','原生相似度','完全成功','耗时（秒）','累计均分'],[(e.get('scenario_id'),e.get('similarity'),e.get('fully_successful'),e.get('elapsed_seconds'),e.get('cumulative_mean_similarity')) for e in eps])
        failed=phase.get('unscored_episodes',[])
        if failed:body+=table(['未评分场景','状态','失败角色','错误类型'],[(e.get('scenario_id'),e.get('status'),(e.get('failure') or {}).get('role'),(e.get('failure') or {}).get('error_class')) for e in failed])
        if phase.get('round_metrics'):body+='<p>正式轮次指标请同时查看 current.json 中的原始汇总；不同训练分片不直接可比。</p>'
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="15"><title>ToolSandbox 实时进度</title><style>body{font:15px/1.6 system-ui,sans-serif;color:#172334;background:#f4f7fb;max-width:1250px;margin:30px auto;padding:0 20px}h1{font-size:28px}h2{margin-top:32px;font-size:20px}.note{color:#536277}.scroll{overflow:auto;background:white;border:1px solid #d8e0ea;border-radius:8px;margin:16px 0}table{border-collapse:collapse;width:100%;text-align:left}th,td{padding:11px 14px;border-bottom:1px solid #e4e9ef}th{background:#edf2f8;white-space:nowrap}td:first-child{max-width:550px;overflow-wrap:anywhere}tr:last-child td{border:0}</style>'+body+'</html>'
