import argparse,hashlib,json,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from progress_monitor import aggregate_role,episode_summary,evaluator_score,read_database,snapshot,runtime_path,parse_args,RUNTIME_ROOT,isolated_database,SnapshotUnavailable,write_snapshot


class MonitorTests(unittest.TestCase):
    def test_incomplete_usage_is_not_reported_as_complete_zero(self):
        base=dict(logical_request_id='q',logical_status='prepared',attempt_id='a',attempt_status='in_flight',
            usage_complete=None,total_tokens=None,output_tokens=None,finish_reason=None,
            phase='online_token_calibration_pilot',input_fingerprint='fingerprint')
        result=aggregate_role([base])
        self.assertIsNone(result['total_tokens']);self.assertFalse(result['usage_complete'])
        self.assertEqual(result['natural_unique_inputs'],0)
        self.assertEqual(result['observed_total_tokens'],0)

    def test_missing_score_is_not_zero_and_cumulative_uses_only_observations(self):
        rows=[dict(episode_id='a',completed_at_utc='1',similarity=.5,fully_successful=False),
              dict(episode_id='b',completed_at_utc='2',similarity=None,fully_successful=None)]
        value=episode_summary(rows)
        self.assertEqual(value['cumulative_mean_similarity'],.5)
        self.assertEqual(value['episodes_scored'],1)
        self.assertEqual(value['latest_episodes'][1]['cumulative_mean_similarity'],.5)
        self.assertIsNone(episode_summary([])['cumulative_mean_similarity'])

    def test_only_top_level_score_is_decoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=json.dumps(dict(milestone_mapping=[{'similarity':0.8,'secret_fixture':'DO_NOT_EXPOSE'}],
                similarity=.25,fully_successful=False)).encode()
            digest=hashlib.sha256(raw).hexdigest();root=Path(tmp);path=root/'sha256'/digest[:2]/digest[2:]
            path.parent.mkdir(parents=True);path.write_bytes(raw)
            output=evaluator_score(root,dict(schema_name='TrustedEvaluatorRecord',sha256='sha256:'+digest))
            self.assertEqual(output,dict(similarity=.25,fully_successful=False))
            self.assertNotIn('DO_NOT_EXPOSE',json.dumps(output))

    def test_sqlite_phases_separate_and_unfinished_episode_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'ledger.sqlite3'
            db=sqlite3.connect(path)
            db.executescript('''CREATE TABLE run_identity(payload BLOB);
              CREATE TABLE logical_llm_requests(logical_request_id TEXT,identity BLOB,role TEXT,status TEXT);
              CREATE TABLE physical_llm_attempts(logical_request_id TEXT,attempt_id TEXT,status TEXT,usage_complete INT,output_tokens INT,result BLOB);
              CREATE TABLE llm_accounting_scopes(logical_request_id TEXT,scope BLOB);
              CREATE TABLE checkpoint_events(event_ordinal INT,event_kind TEXT,payload BLOB,created_at_utc TEXT);''')
            db.execute('INSERT INTO run_identity VALUES(?)',(json.dumps({'run_id':'synthetic'}),))
            for i,phase in enumerate(('online_token_calibration_pilot','offline_token_calibration','train_round')):
                db.execute('INSERT INTO logical_llm_requests VALUES(?,?,?,?)',(str(i),json.dumps({'phase':phase,'input_fingerprint':str(i)}),'policy','prepared'))
                db.execute('INSERT INTO llm_accounting_scopes VALUES(?,?)',(str(i),json.dumps({'round_index':0 if phase=='train_round' else None})))
            identity=dict(episode_id='unfinished',phase='online_token_calibration_pilot',scenario_id='synthetic',round_index=None)
            db.execute('INSERT INTO checkpoint_events VALUES(?,?,?,?)',(1,'dispatch',json.dumps({'identity':identity}),'2026-01-01T00:00:00+00:00'))
            db.commit();db.close()
            phases,errors=read_database(path);by={p['phase']:p for p in phases}
            self.assertEqual(set(by),{'online_calibration','offline_calibration','train_round_0'})
            self.assertEqual(by['online_calibration']['episodes_pending'],1)
            self.assertEqual(by['online_calibration']['episodes_completed'],0)
            self.assertIsNone(by['online_calibration']['cumulative_mean_similarity'])
            self.assertEqual(by['train_round_0']['comparability'],'different_train_shards_not_directly_comparable')
            self.assertEqual(errors,[])
            db=sqlite3.connect(path)
            db.execute("UPDATE logical_llm_requests SET status='terminal_failure' WHERE logical_request_id='0'")
            db.execute("UPDATE llm_accounting_scopes SET scope=? WHERE logical_request_id='0'",(json.dumps({'round_index':None,'task_id':'unfinished'}),))
            db.execute('INSERT INTO physical_llm_attempts VALUES(?,?,?,?,?,?)',('0','attempt','failed',0,None,
                json.dumps({'exception_class':'Do not leak arbitrary provider details','finish_reason':'stop','metrics':{'usage':{'total_tokens':None}}})))
            db.commit();db.close()
            stopped=snapshot(root,root/'absent',dict(name='toolsandbox-full',state='EXITED'))
            phase=next(p for p in stopped['phases'] if p['phase']=='online_calibration')
            self.assertEqual((phase['episodes_failed'],phase['episodes_pending'],phase['episodes_completed']),(1,0,0))
            self.assertIsNone(phase['cumulative_mean_similarity'])
            self.assertEqual(phase['unscored_episodes'][0]['status'],'failed')
            self.assertEqual(stopped['service']['stop_reason']['error_class'],'OtherError')
            self.assertNotIn('Do not leak',json.dumps(stopped))

    def test_absent_phases_remain_not_started_without_zero_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=snapshot(Path(tmp)/'campaign',Path(tmp)/'online',dict(name='toolsandbox-full',state='RUNNING'))
            self.assertEqual(len(value['phases']),5)
            self.assertTrue(all(p['status']=='not_started' and p['cumulative_mean_similarity'] is None for p in value['phases']))

    def test_explicit_paths_are_bounded_and_paired(self):
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as tmp:
            root=Path(tmp);campaign=root/'campaign';online=root/'online';campaign.mkdir();online.mkdir()
            args=parse_args(['--campaign',str(campaign),'--online',str(online),'--once'])
            self.assertEqual((args.campaign,args.online),(campaign,online))
            link=root/'link';link.symlink_to(campaign,target_is_directory=True)
            for value in ('/etc','relative',str(link),str(root/'missing'),str(root/'..'/'anything')):
                with self.assertRaises(argparse.ArgumentTypeError):runtime_path(value)
            with self.assertRaises(SystemExit):parse_args(['--campaign',str(campaign)])

    def test_source_writer_never_waits_for_long_snapshot_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'live.sqlite3'
            writer=sqlite3.connect(path,timeout=0,isolation_level=None)
            writer.execute('PRAGMA journal_mode=DELETE')
            writer.execute('CREATE TABLE sample(value INTEGER)')
            writer.execute('INSERT INTO sample VALUES(1)')
            # Reproduce the previous implementation's lock failure mechanism.
            reader=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,timeout=0)
            reader.execute('BEGIN');reader.execute('SELECT * FROM sample').fetchall()
            with self.assertRaises(sqlite3.OperationalError):writer.execute('UPDATE sample SET value=2')
            reader.close()
            real_connect=sqlite3.connect
            opened=[]
            def checked_connect(database,*args,**kwargs):
                opened.append(str(database))
                self.assertNotIn(str(path),str(database))
                return real_connect(database,*args,**kwargs)
            with patch('progress_monitor.sqlite3.connect',side_effect=checked_connect):
                with isolated_database(path) as copy:
                    self.assertEqual(copy.execute('SELECT value FROM sample').fetchone()[0],1)
                    # Monitor may spend arbitrarily long processing the copied DB.
                    for value in range(2,20):writer.execute('UPDATE sample SET value=?',(value,))
                    self.assertEqual(copy.execute('SELECT value FROM sample').fetchone()[0],1)
            self.assertEqual(writer.execute('SELECT value FROM sample').fetchone()[0],19)
            self.assertEqual(len(opened),1);writer.close()

    def test_source_journal_or_concurrent_change_rejects_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'live.sqlite3';writer=sqlite3.connect(path,timeout=0,isolation_level=None)
            writer.execute('CREATE TABLE sample(value INTEGER)');writer.execute('INSERT INTO sample VALUES(1)')
            journal=Path(str(path)+'-journal');journal.write_bytes(b'active')
            with self.assertRaises(SnapshotUnavailable):
                with isolated_database(path):pass
            journal.unlink()
            import os
            real_read=os.read;changed=[]
            def race_read(*args):
                raw=real_read(*args)
                if raw and not changed:
                    writer.execute('UPDATE sample SET value=2');changed.append(True)
                return raw
            with patch('progress_monitor.os.read',side_effect=race_read):
                with self.assertRaises(SnapshotUnavailable):
                    with isolated_database(path):pass
            self.assertEqual(writer.execute('SELECT value FROM sample').fetchone()[0],2);writer.close()

    def test_stale_snapshot_retains_verified_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);value=snapshot(root/'campaign',root/'online',dict(name='toolsandbox-full',state='RUNNING'))
            value['phases'][0]['cumulative_mean_similarity']=.75
            write_snapshot(value,root/'progress')
            stale=snapshot(root/'campaign',root/'online',dict(name='toolsandbox-full',state='RUNNING'))
            stale.update(stale=True,errors=['database_snapshot_unavailable'])
            write_snapshot(stale,root/'progress')
            stored=json.loads((root/'progress'/'current.json').read_text())
            self.assertTrue(stored['stale'])
            self.assertEqual(stored['phases'][0]['cumulative_mean_similarity'],.75)
            self.assertEqual(stored['last_verified_at_utc'],value['updated_at_utc'])

if __name__=='__main__':unittest.main()
