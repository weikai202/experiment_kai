#!/usr/bin/env python3
"""Read-only, fixed-campaign progress snapshots. No provider or secret access."""
import argparse
from contextlib import contextmanager
import os
import tempfile
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import time

CAMPAIGN=Path('/root/toolsandbox-runtime/full-runs/full-live-20260912T202445-014054')
ONLINE=Path('/root/toolsandbox-runtime/runs/live-calibration-20260912T202446-72cbf3')
OUTPUT=Path('/root/toolsandbox-runtime/progress')
SERVICE='toolsandbox-full'
RUNTIME_ROOT=Path('/root/toolsandbox-runtime')
TERMINAL_STATES=('STOPPED','EXITED','FATAL')
SAFE_ERRORS={'ProviderOutputError','OutputTruncated','RateLimitError','APIConnectionError','TimeoutError','AuthenticationError','PermissionDeniedError','RejectedBeforeDispatch'}
ROLES=('policy','critic','revision','memory_candidate','memory_review','failure_mode_update','skill_candidate','embedding','user_simulator')


def phase_name(value,round_index=None):
    if value.startswith('online_'):return 'online_calibration'
    if 'calibration' in value and value.startswith('offline'):return 'offline_calibration'
    if value=='dev_minibench':return f'dev_minibench_round_{round_index}'
    if value in ('train_round','offline_update','generation_publication'):return f'train_round_{round_index}'
    return value


def service_status():
    try:
        result=subprocess.run(['supervisorctl','status',SERVICE],capture_output=True,text=True,timeout=5)
        words=result.stdout.split()
        state=words[1] if len(words)>1 and words[0]==SERVICE else 'UNKNOWN'
        if state not in ('RUNNING','STARTING','STOPPED','EXITED','FATAL','BACKOFF','STOPPING','UNKNOWN'):
            state='UNKNOWN'
    except (OSError,subprocess.TimeoutExpired):
        state='UNKNOWN'
    return {'name':SERVICE,'state':state}


def evaluator_score(blob_root,reference):
    """Hash bytes; extract only top-level score scalars, never return mappings."""
    if not reference or reference.get('schema_name')!='TrustedEvaluatorRecord':
        return None
    digest=reference.get('sha256','')
    if re.fullmatch(r'sha256:[a-f0-9]{64}',digest) is None:
        return None
    path=blob_root/'sha256'/digest[7:9]/digest[9:]
    if path.is_symlink():return None
    raw=path.read_bytes()
    if 'sha256:'+hashlib.sha256(raw).hexdigest()!=digest:return None
    # Track object/array nesting while scanning string boundaries. Values of
    # unselected keys (including all hidden mappings) are never decoded.
    text=raw.decode();depth=0;i=0;values={};decoder=json.JSONDecoder()
    while i<len(text):
        char=text[i]
        if char=='"':
            start=i;i+=1
            while i<len(text):
                if text[i]=='\\':i+=2;continue
                if text[i]=='"':break
                i+=1
            end=i+1
            if depth==1 and text[end:].lstrip().startswith(':'):
                key=text[start:end]
                if key in ('"similarity"','"fully_successful"'):
                    value_start=end
                    while text[value_start].isspace():value_start+=1
                    value_start+=1
                    while text[value_start].isspace():value_start+=1
                    value,_=decoder.raw_decode(text,value_start)
                    values[key[1:-1]]=value
            i=end;continue
        if char in '{[':depth+=1
        elif char in '}]':depth-=1
        i+=1
    score=values.get('similarity');success=values.get('fully_successful')
    if type(score) not in (float,int) or not 0<=score<=1 or type(success) is not bool:return None
    return {'similarity':float(score),'fully_successful':success}


def aggregate_role(rows):
    logical={r['logical_request_id']:r['logical_status'] for r in rows}
    attempts=[r for r in rows if r['attempt_id'] is not None]
    complete=all(r['usage_complete']==1 and r['total_tokens'] is not None for r in attempts)
    observed=sum(r['total_tokens'] or 0 for r in attempts)
    outputs=[r['output_tokens'] for r in attempts if r['output_tokens'] is not None]
    natural={r['input_fingerprint'] for r in rows if r['phase']=='online_token_calibration_pilot' and r['logical_status'] in ('applied','response_completed')}
    return dict(logical_requests=len(logical),status_counts=dict(Counter(logical.values())),
        physical_attempts=len(attempts),attempt_status_counts=dict(Counter(r['attempt_status'] for r in attempts)),
        total_tokens=observed if complete else None,observed_total_tokens=observed,usage_complete=complete,
        max_output_tokens=max(outputs) if outputs else None,
        length_finishes=sum(r['finish_reason']=='length' for r in attempts),natural_unique_inputs=len(natural),target_inputs=32)


def episode_summary(episodes):
    ordered=sorted(episodes,key=lambda e:(e['completed_at_utc'],e['episode_id']))
    total=0.;count=0;successful=0
    for entry in ordered:
        if entry['similarity'] is not None:
            total+=entry['similarity'];count+=1;successful+=int(entry['fully_successful'])
        entry['cumulative_mean_similarity']=total/count if count else None
    return dict(episodes_completed=len(ordered),episodes_scored=count,
        cumulative_mean_similarity=total/count if count else None,
        fully_successful_count=successful,latest_episodes=ordered[-20:])


class SnapshotUnavailable(RuntimeError):
    """The source was changing; retain the last verified monitor snapshot."""


def _stat_identity(value):
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def _journal_present(path):
    return any(Path(str(path)+suffix).exists() for suffix in ('-journal','-wal','-shm'))


@contextmanager
def isolated_database(path):
    """Never connect SQLite to the live database: writers have timeout=0.

    Copy opaque bytes with no SQLite locks. Any observed journal, file change,
    replacement or inconsistent copy rejects the sample. Only the private copy
    is opened by SQLite. DELETE mode is required; WAL sidecars fail closed.
    """
    path=Path(path)
    with tempfile.TemporaryDirectory(prefix='toolsandbox-monitor-') as directory:
        target=Path(directory)/'snapshot.sqlite3'
        if _journal_present(path):raise SnapshotUnavailable('source_journal_present')
        descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            before=os.fstat(descriptor)
            if _stat_identity(before)!=_stat_identity(os.stat(path,follow_symlinks=False)):
                raise SnapshotUnavailable('source_replaced')
            with os.fdopen(os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as output:
                while True:
                    chunk=os.read(descriptor,1024*1024)
                    if not chunk:break
                    output.write(chunk)
                    if _journal_present(path):raise SnapshotUnavailable('source_changed')
            after=os.fstat(descriptor)
            if (_stat_identity(before)!=_stat_identity(after)
                    or _stat_identity(after)!=_stat_identity(os.stat(path,follow_symlinks=False))
                    or _journal_present(path)):
                raise SnapshotUnavailable('source_changed')
        finally:
            os.close(descriptor)
        connection=sqlite3.connect('file:'+str(target)+'?mode=ro&immutable=1',uri=True,timeout=0)
        try:
            connection.execute('PRAGMA query_only=ON')
            if connection.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:
                raise SnapshotUnavailable('copy_check_failed')
            yield connection
        finally:
            connection.close()


def read_database(path):
    phases={};errors=[]
    with isolated_database(path) as db:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        run_id=db.execute("SELECT json_extract(payload,'$.run_id') FROM run_identity").fetchone()[0]
        rows=db.execute("""SELECT r.logical_request_id,r.role,r.status logical_status,
            json_extract(r.identity,'$.phase') phase,json_extract(r.identity,'$.input_fingerprint') input_fingerprint,
            json_extract(sc.scope,'$.round_index') round_index,json_extract(sc.scope,'$.task_id') task_id,
            a.attempt_id,a.status attempt_status,a.usage_complete,a.output_tokens,
            json_extract(a.result,'$.finish_reason') finish_reason,json_extract(a.result,'$.exception_class') exception_class,
            json_extract(a.result,'$.metrics.usage.total_tokens') total_tokens
            FROM logical_llm_requests r LEFT JOIN physical_llm_attempts a USING(logical_request_id)
            LEFT JOIN llm_accounting_scopes sc USING(logical_request_id)""").fetchall()
        def phase(name):
            key=name
            return phases.setdefault(key,dict(phase=key,run_id=run_id,roles=[],episodes=[],round_metrics=[]))
        grouped=defaultdict(list)
        for row in rows:grouped[(phase_name(row['phase'],row['round_index']),row['role'])].append(dict(row))
        for (name,role),entries in grouped.items():
            stats=aggregate_role(entries)
            if name!='online_calibration' or role not in ('policy','critic','revision'):
                stats.update(natural_unique_inputs=None,target_inputs=None)
            phase(name)['roles'].append(dict(role=role,**stats))
        identities={}
        for r in db.execute("""SELECT json_extract(payload,'$.identity.episode_id') episode_id,
                json_extract(payload,'$.identity.phase') phase,json_extract(payload,'$.identity.scenario_id') scenario_id,
                json_extract(payload,'$.identity.round_index') round_index,
                created_at_utc FROM checkpoint_events WHERE json_type(payload,'$.identity.episode_id')='text'
                ORDER BY event_ordinal"""):
            identities.setdefault(r['episode_id'],dict(r))
        captures={}
        for r in db.execute("""SELECT json_extract(payload,'$.identity.episode_id') episode_id,
                json_extract(payload,'$.result.task_accounting_input.timing.total_running_time_seconds') elapsed
                FROM checkpoint_events WHERE event_kind='calibration_episode_capture'"""):
            captures[r['episode_id']]=r['elapsed']
        finals=db.execute("""SELECT json_extract(payload,'$.episode_id') episode_id,
            json_extract(payload,'$.evaluator_record_reference') reference,created_at_utc
            FROM checkpoint_events WHERE event_kind='episode_completed_evaluated' ORDER BY event_ordinal""").fetchall()
        seen=set()
        for row in finals:
            eid=row['episode_id']
            if eid in seen:continue
            seen.add(eid);identity=identities.get(eid)
            if identity is None:errors.append('completed_episode_identity_unavailable');continue
            try:score=evaluator_score(path.parent/'blobs',json.loads(row['reference']) if row['reference'] else None)
            except (OSError,ValueError,IndexError):score=None
            elapsed=captures.get(eid);timing_source='authoritative_episode_timing'
            if elapsed is None:
                try:elapsed=(datetime.fromisoformat(row['created_at_utc'])-datetime.fromisoformat(identity['created_at_utc'])).total_seconds()
                except ValueError:elapsed=None
                timing_source='checkpoint_wall_interval'
            if score is None:errors.append('completed_episode_score_unavailable')
            phase(phase_name(identity['phase'],identity['round_index']))['episodes'].append(dict(episode_id=eid,scenario_id=identity['scenario_id'],
                completed_at_utc=row['created_at_utc'],elapsed_seconds=float(elapsed) if elapsed is not None else None,
                timing_source=timing_source,similarity=score['similarity'] if score else None,
                fully_successful=score['fully_successful'] if score else None))
        for r in db.execute("""SELECT json_extract(payload,'$.round.round_index') round_index,
            json_extract(payload,'$.round.completion_status') completion_status,
            json_extract(payload,'$.scenario_count') scenario_count,
            json_extract(payload,'$.total_running_time_seconds') elapsed,
            json_extract(payload,'$.total_tokens') total_tokens,json_extract(payload,'$.usage_complete') usage_complete,
            json_extract(payload,'$.total_cost') total_cost,json_extract(payload,'$.cost_complete') cost_complete
            FROM checkpoint_events WHERE event_kind='training_round_metrics' ORDER BY event_ordinal"""):
            phase('train_round_'+str(r['round_index']))['round_metrics'].append(dict(r))
        failures={}
        for row in rows:
            if row['logical_status']=='terminal_failure' and row['task_id'] in identities and row['task_id'] not in seen:
                failures[row['task_id']]=dict(role=row['role'] if row['role'] in ROLES else 'unknown',
                    error_class=row['exception_class'] if row['exception_class'] in SAFE_ERRORS else 'OtherError',
                    reason_code='terminal_request_failure_observed')
        for item in phases.values():
            item.update(episode_summary(item.pop('episodes')))
            item['episodes_started']=sum(phase_name(v['phase'],v['round_index'])==item['phase'] for v in identities.values())
            item['unscored_episodes']=[dict(episode_id=eid,scenario_id=v['scenario_id'],similarity=None,fully_successful=None,
                status='failed' if eid in failures else 'pending',failure=failures.get(eid))
                for eid,v in identities.items() if eid not in seen and phase_name(v['phase'],v['round_index'])==item['phase']]
            item['episodes_failed']=sum(e['status']=='failed' for e in item['unscored_episodes'])
            item['episodes_interrupted']=0
            item['episodes_pending']=sum(e['status']=='pending' for e in item['unscored_episodes'])
            item['roles'].sort(key=lambda r:ROLES.index(r['role']) if r['role'] in ROLES else 99)
            if item['phase']=='online_calibration':
                item['collection']=dict(initial_target=32,reserve_limit=64,completed=item['episodes_completed'],role_quota_target=32)
            if item['phase'].startswith('train_round_'):item['comparability']='different_train_shards_not_directly_comparable'
    return list(phases.values()),errors


def snapshot(campaign=CAMPAIGN,online=ONLINE,status=None):
    result=dict(schema_version=1,updated_at_utc=datetime.now(timezone.utc).isoformat(),campaign_id=campaign.name,
        service=service_status() if status is None else status,phases=[],errors=[],stale=False,
        monitored_online_run_id=online.name)
    paths={online/'checkpoint'/'checkpointing'/'ledger.sqlite3'}
    if campaign.exists():paths.update(campaign.glob('**/ledger.sqlite3'))
    for path in sorted(paths):
        if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):continue
        try:
            phases,errors=read_database(path);result['phases'].extend(phases);result['errors'].extend(errors)
        except (sqlite3.Error,OSError,ValueError,KeyError,SnapshotUnavailable):
            result['errors'].append('database_snapshot_unavailable');result['stale']=True
    present={p['phase'] for p in result['phases']}
    for name in ('online_calibration','offline_calibration','train_round_0','train_round_1','train_round_2'):
        if name not in present:
            result['phases'].append(dict(phase=name,run_id=None,status='not_started',roles=[],round_metrics=[],
                episodes_started=0,episodes_pending=0,episodes_failed=0,episodes_interrupted=0,unscored_episodes=[],**episode_summary([])))
    for item in result['phases']:
        if result['service']['state'] in TERMINAL_STATES:
            for episode in item['unscored_episodes']:
                if episode['status']=='pending':episode['status']='interrupted'
            item['episodes_interrupted']=sum(e['status']=='interrupted' for e in item['unscored_episodes'])
            item['episodes_pending']=0
        if item.get('status')=='not_started':continue
        item['status']='running' if result['service']['state']=='RUNNING' else 'stopped'
        if item['episodes_failed'] and result['service']['state'] in TERMINAL_STATES:item['status']='failed'
        if item['round_metrics'] and item['round_metrics'][-1]['completion_status']=='complete':item['status']='completed'
        if item['phase']=='online_calibration' and ('offline_calibration' in present or any(n.startswith('train_round_') for n in present)):item['status']='completed'
        if item['phase']=='offline_calibration' and any(n.startswith('train_round_') for n in present):item['status']='completed'
    result['phases'].sort(key=lambda p:({'online_calibration':0,'offline_calibration':1,'train_round_0':2,'train_round_1':3,'train_round_2':4}.get(p['phase'],5),p['phase']))
    result['service']['stop_reason']=None
    if result['service']['state'] in TERMINAL_STATES:
        failures=[dict(phase=p['phase'],episode_id=e['episode_id'],**e['failure']) for p in result['phases'] for e in p['unscored_episodes'] if e['failure']]
        result['service']['stop_reason']=failures[-1] if failures else {'reason_code':'service_stopped_without_recorded_terminal_request_failure'}
    result['errors']=sorted(set(result['errors']))
    return result


def markdown(data):
    lines=['# Pipeline progress',f"UTC: {data['updated_at_utc']}",f"Stale snapshot: {data.get('stale',False)}; last verified: {data.get('last_verified_at_utc','—')}",f"Service: {data['service']['name']} {data['service']['state']}",f"Stop evidence: {data['service'].get('stop_reason') or '—'}",'']
    def f(v):return '—' if v is None else str(round(v,4)) if isinstance(v,float) else str(v)
    for phase in data['phases']:
        lines.extend([f"## {phase['phase']}",f"Completed: {phase['episodes_completed']}; failed: {phase['episodes_failed']}; interrupted: {phase['episodes_interrupted']}; pending: {phase['episodes_pending']}; mean similarity: {f(phase['cumulative_mean_similarity'])}",'',
            '| Role | Requests | Attempts | Tokens | Complete usage | Max output | Length | Natural / 32 |',
            '|---|---:|---:|---:|---|---:|---:|---:|'])
        for role in phase['roles']:lines.append('| '+' | '.join(f(role.get(k)) for k in ('role','logical_requests','physical_attempts','total_tokens','usage_complete','max_output_tokens','length_finishes','natural_unique_inputs'))+' |')
        lines.extend(['','| Episode | Scenario | Similarity | Success | Seconds | Cumulative mean |','|---|---|---:|---|---:|---:|'])
        for e in phase['latest_episodes']:lines.append('| '+' | '.join(f(e[k]) for k in ('episode_id','scenario_id','similarity','fully_successful','elapsed_seconds','cumulative_mean_similarity'))+' |')
        if phase['unscored_episodes']:
            lines.extend(['','| Unscored episode | State | Failure evidence |','|---|---|---|'])
            for episode in phase['unscored_episodes']:
                failure=episode['failure']
                reason=(failure['role']+': '+failure['error_class']) if failure else '—'
                lines.append(f"| {episode['episode_id']} | {episode['status']} | {reason} |")
        if phase.get('comparability'):lines.append('\nDifferent train shards: performance is not directly comparable across rounds.')
        lines.append('')
    return '\n'.join(lines)+'\n'


def write_snapshot(data,output=OUTPUT):
    if data.get('stale'):
        previous_path=output/'current.json'
        try:previous=json.loads(previous_path.read_text())
        except (OSError,ValueError):previous=None
        if previous and previous.get('campaign_id')==data['campaign_id'] and previous.get('monitored_online_run_id')==data.get('monitored_online_run_id'):
            previous['last_verified_at_utc']=previous.get('last_verified_at_utc',previous['updated_at_utc'])
            previous['updated_at_utc']=data['updated_at_utc']
            previous['service']=data['service']
            previous['stale']=True
            previous['errors']=data['errors']
            data=previous
    else:data['last_verified_at_utc']=data['updated_at_utc']
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    from progress_view import render
    for name,text in (('current.json',json.dumps(data,ensure_ascii=False,indent=2)+'\n'),('current.md',markdown(data)),('dashboard.html',render(data))):
        temp=output/(name+'.tmp');temp.write_text(text);temp.chmod(0o600);temp.replace(output/name)


def runtime_path(value):
    path=Path(value)
    if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(RUNTIME_ROOT) or path==RUNTIME_ROOT:
        raise argparse.ArgumentTypeError('explicit directory under /root/toolsandbox-runtime required')
    if not path.is_dir() or any(p.is_symlink() for p in (path,*path.parents)):
        raise argparse.ArgumentTypeError('existing non-symlink runtime directory required')
    return path

def parse_args(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true')
    parser.add_argument('--campaign',type=runtime_path);parser.add_argument('--online',type=runtime_path)
    args=parser.parse_args(argv)
    if (args.campaign is None)!=(args.online is None):parser.error('--campaign and --online must be provided together')
    if args.campaign is None:
        args.campaign=runtime_path(CAMPAIGN);args.online=runtime_path(ONLINE)
    return args

def main():
    args=parse_args()
    while True:
        value=snapshot(args.campaign,args.online);write_snapshot(value)
        if args.once or value['service']['state'] in ('STOPPED','EXITED','FATAL'):return
        try:time.sleep(15)
        except KeyboardInterrupt:
            write_snapshot(snapshot(args.campaign,args.online));return

if __name__=='__main__':main()
