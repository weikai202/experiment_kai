"""CLI/settings checks perform no runtime/provider bootstrap."""
import json
import pytest
from toolsandbox_pipeline.orchestration.live_calibration_bootstrap import load_settings, DEFAULT_SETTINGS, _write


def test_default_settings_are_explicit_and_versioned():
    settings = load_settings(DEFAULT_SETTINGS)
    assert settings['schema_version'] == 1
    assert settings['train_manifest'].endswith('/dataset/train_manifest.json')
    assert 'frankfurter_v2' in settings['backend_manifest']


def test_settings_reject_unknown_fields_and_relative_paths(tmp_path):
    settings = load_settings(DEFAULT_SETTINGS)
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({**settings, 'arbitrary_command': 'no'}))
    with pytest.raises(ValueError):
        load_settings(path)
    path.write_text(json.dumps({**settings, 'train_manifest': 'relative.json'}))
    with pytest.raises(ValueError):
        load_settings(path)


def test_artifacts_are_exclusive_not_overwritten(tmp_path):
    path = tmp_path / 'artifact.json'
    _write(path, {'value': 1})
    with pytest.raises(FileExistsError):
        _write(path, {'value': 2})
    assert json.loads(path.read_bytes()) == {'value': 1}


def test_failed_preflight_writes_only_sanitized_phase(tmp_path):
    from toolsandbox_pipeline.orchestration.live_calibration_bootstrap import _run_provider_preflights
    def failed(*args):
        raise ValueError('sensitive provider response must not be persisted')
    with pytest.raises(ValueError):
        _run_provider_preflights(tmp_path, [('embedding', object())], failed)
    assert json.loads((tmp_path / 'failure.json').read_bytes()) == {
        'exception_class': 'ValueError', 'phase': 'provider_preflight', 'provider': 'embedding'}


def test_main_suppresses_exception_details(monkeypatch, capsys):
    from toolsandbox_pipeline.orchestration import live_calibration_bootstrap as module
    import sys
    monkeypatch.setattr(sys, 'argv', ['calibration'])
    def failed(*args):
        raise ValueError('sensitive detail')
    monkeypatch.setattr(module, 'run_live_calibration_from_settings', failed)
    assert module.main() == 1
    output = capsys.readouterr()
    assert 'sensitive' not in output.out + output.err
    assert json.loads(output.out)['exception_class'] == 'ValueError'


def test_authorization_accepts_scope_and_exact_pilot_keyword_contract(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from toolsandbox_pipeline.orchestration.live_calibration_bootstrap import calibration_authorizer
    from toolsandbox_pipeline.retrieval.index import file_hash
    from tests.orchestration.test_live_episode import scope, record, D
    from toolsandbox_pipeline.schemas.trajectory import EpisodeIdentity
    backend=tmp_path/'backend.json';backend.write_text('{}')
    rec=record()
    authorize=calibration_authorizer(run_id='run',runtime_manifest_sha256=D,backend_path=backend,
        backend_sha256=file_hash(backend.read_bytes()),records=(rec,))
    authorize(replace(scope(),round_index=0,shard_id='train-shard-0',generation_id='g000'))
    identity=EpisodeIdentity(run_id='run',profile='official_live',phase='online_token_calibration_pilot',
        family_id=rec.scenario_family_id,scenario_id=rec.scenario_id,episode_id='pilot',manifest_position=0,
        system_variant='generation_0',generation_id='g000',starting_context_sha256=D,evaluation_definition_sha256=D,
        agent_tool_schema_sha256=D,dataset_manifest_sha256=D,runtime_config_sha256=D,prompt_manifest_sha256=D,
        token_limit_config_sha256=D,fixture_manifest_sha256=D,environment_sha256=D,max_messages=rec.max_messages)
    lease=SimpleNamespace(record=rec)
    authorize(identity=identity,lease=lease,purpose='online_token_calibration')
    with pytest.raises(PermissionError):authorize(identity=identity,lease=lease,purpose='train_round')
    with pytest.raises(PermissionError):
        authorize(identity=identity.model_copy(update={'starting_context_sha256':'sha256:'+'2'*64}),lease=lease,purpose='online_token_calibration')


def test_collection_failure_retains_only_sanitized_cause_class():
    from toolsandbox_pipeline.orchestration.live_calibration_bootstrap import _failure_summary
    from toolsandbox_pipeline.orchestration.live_calibration import CalibrationCollectionError
    try:
        try:raise TypeError('sensitive original detail')
        except TypeError:raise CalibrationCollectionError('pilot_collection_failed',(),{})
    except CalibrationCollectionError as error:
        result=_failure_summary(error,'pilot')
    assert result == {'exception_class':'CalibrationCollectionError','phase':'pilot',
                      'code':'pilot_collection_failed','cause_classes':['TypeError']}
