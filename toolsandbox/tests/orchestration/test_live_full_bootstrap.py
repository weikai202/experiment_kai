"""Offline full-entry integration contracts; no model or dataset scenarios."""
import ast
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from toolsandbox_pipeline.orchestration import live_full_bootstrap as full


def test_all_explicit_runtime_imports_and_constructor_keywords_exist():
    tree = ast.parse(Path(full.__file__).read_text())
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('toolsandbox_pipeline'):
            module = importlib.import_module(node.module)
            for alias in node.names:
                imported[alias.asname or alias.name] = getattr(module, alias.name)
    # Validate complete keyword call shapes for every real top-level imported
    # callable. Nested callbacks retain runtime validation in their own modules.
    checked = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in imported:
            continue
        target = imported[node.func.id]
        if not callable(target) or any(isinstance(arg, ast.Starred) for arg in node.args) or any(k.arg is None for k in node.keywords):
            continue
        try:
            signature = inspect.signature(target)
        except ValueError:
            continue
        try:
            signature.bind(*([None]*len(node.args)), **{key.arg:None for key in node.keywords})
        except TypeError as error:
            raise AssertionError(f'{node.func.id} line {node.lineno}: {error}') from error
        checked.append(node.func.id)
    assert {'LiveTrainingRuntime','LiveTrainingAssets','NativeDevMiniBenchFactory',
            'NativeDevSelectorMetadataLoader','collect_offline_calibration_requests',
            'calibrate_offline_requests','offline_limit_recommendations'} <= set(checked)


def test_failed_online_stage_does_not_call_offline_or_train(tmp_path,monkeypatch):
    settings = full.load_settings(full.DEFAULT_SETTINGS)
    settings.update(output_root=str(tmp_path/'campaigns'),online_run_directory=None)
    path = tmp_path/'settings.json';path.write_text(json.dumps(settings))
    monkeypatch.setenv('PYTHONHASHSEED','0')
    monkeypatch.setattr(full,'validate_settings',lambda _: {'status':'pass'})
    def fail(*args, **kwargs):raise ValueError('private provider response')
    monkeypatch.setattr(full,'run_live_calibration_from_settings',fail)
    monkeypatch.setattr(full,'_offline_calibrate',lambda **kwargs:pytest.fail('offline must not dispatch'))
    monkeypatch.setattr(full,'_train',lambda **kwargs:pytest.fail('train must not dispatch'))
    with pytest.raises(ValueError):full.run_full_from_settings(path)
    failure, = (tmp_path/'campaigns').glob('*/failure.json')
    assert json.loads(failure.read_bytes()) == {'phase':'online_calibration','exception_class':'ValueError'}


def test_completed_online_source_runs_stages_in_order_without_repeating_pilot(tmp_path,monkeypatch):
    source=tmp_path/'online';source.mkdir();(source/'completed.json').write_text('{}')
    settings=full.load_settings(full.DEFAULT_SETTINGS)
    settings.update(output_root=str(tmp_path/'campaigns'),online_run_directory=str(source))
    path=tmp_path/'settings.json';path.write_text(json.dumps(settings))
    monkeypatch.setenv('PYTHONHASHSEED','0')
    monkeypatch.setattr(full,'validate_settings',lambda _: {'status':'pass'})
    monkeypatch.setattr(full,'run_live_calibration_from_settings',lambda _:pytest.fail('pilot must not repeat'))
    calls=[]
    material=object();result=object()
    def offline(**kwargs):calls.append('offline');assert kwargs['source']==source;return material
    def train(**kwargs):calls.append('train');assert kwargs['material'] is material;return result
    monkeypatch.setattr(full,'_offline_calibrate',offline);monkeypatch.setattr(full,'_train',train)
    assert full.run_full_from_settings(path) is result
    assert calls==['offline','train']


def test_full_settings_reject_unreviewed_fields(tmp_path):
    payload=full.load_settings(full.DEFAULT_SETTINGS);payload['test_run']=True
    path=tmp_path/'invalid.json';path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):full.load_settings(path)


def test_actual_calibration_artifacts_build_honest_live_v2_manifest(tmp_path, monkeypatch):
    from tests.online.test_token_limit_calibration import test_calibration_artifact_contains_all_roles_and_hashes
    from tests.reproducibility.test_dataset_manifest import blobs
    from tests.memory.test_generation_store import load
    from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig, UserSimulatorConfig
    from toolsandbox_pipeline.offline.memory_prompts import load_memory_prompts
    from toolsandbox_pipeline.retrieval.index import file_hash
    from toolsandbox_pipeline.orchestration.run_manifest import validate_component_files
    source=tmp_path/'source';source.mkdir()
    # Runs the production calibrator with real role gateways and fake transport;
    # produces its actual 192-attempt artifact/config, not a model_construct shell.
    test_calibration_artifact_contains_all_roles_and_hashes(source)
    (source/'calibrated').rename(source/'recommendation')
    hashes={p.name:file_hash(p.read_bytes()) for p in (source/'recommendation').iterdir()}
    (source/'completed.json').write_text(json.dumps({'artifact_hashes':hashes}))
    (source/'seed-provenance.json').write_text('{}')
    dataset=blobs()
    train=tmp_path/'train.json';train.write_bytes(dataset['train_manifest.json'])
    index=tmp_path/'index.json';index.write_bytes(dataset['dataset_index.json'])
    image=tmp_path/'image.json';image.write_text(json.dumps({'registry_resolved_digest':'sha256:'+'a'*64,
        'running_container_digest_verified':False}))
    settings=full.load_settings(full.DEFAULT_SETTINGS)
    settings.update(dataset_index=str(index),dataset_index_sha256=file_hash(index.read_bytes()),image_provenance=str(image))
    from toolsandbox_pipeline.orchestration import live_environment_attestation
    synthetic_environment = json.loads(index.read_bytes())['environment']
    monkeypatch.setattr(live_environment_attestation, 'attest_execution_environment', lambda **kwargs: {
        'version':'synthetic-attestation', 'dataset_build_environment':synthetic_environment,
        'execution_environment':{**synthetic_environment,'dependency_lock_sha256':'sha256:'+'b'*64}})
    online=full.load_online_settings(settings['online_settings']);online['train_manifest']=str(train)
    skill_prompts,skill_hashes,skill_manifest=full._load_skill_prompts()
    material=dict(qwen=QwenConfig(structured_output_wire_mode='guided_json'),embedding=EmbeddingConfig(expected_dimension=2),
        user=UserSimulatorConfig(),generation=load(source/'g000'),source_store=SimpleNamespace(identity=SimpleNamespace(run_id='synthetic')),
        memory_prompts=load_memory_prompts(full.PROJECT,full.PROJECT/'prompts/offline/memory_manifest.json'),
        skill_manifest_hash=skill_manifest,source_manifest={'user_prompt_binding':{'protocol':'synthetic'},
            'fixture_scope':{'profile':'official_live','fixture_store':None}},
        offline={'hashes':{name:'sha256:'+'a'*64 for name in ('offline_memory_token_limits.calibrated.json',
            'offline_skill_token_limits.calibrated.json','offline_calibration_evidence.json')}})
    directory=tmp_path/'formal';directory.mkdir()
    manifest,*_=full._formal_manifest(directory=directory,settings=settings,online=online,source=source,material=material)
    validate_component_files(manifest)
    assert manifest.protocol_version=='toolsandbox-evolution-live-v2'
    assert manifest.container_image_digest is None and manifest.qwen.container_digest is None
    assert manifest.image_verification=='unverified' and manifest.test_access_policy=='forbidden'
    assert manifest.dependency_lock_sha256 == 'sha256:'+'b'*64
    evidence=json.loads((directory/'environment-provenance.json').read_bytes())
    assert evidence['attestation']['dataset_build_environment']['dependency_lock_sha256'] == synthetic_environment['dependency_lock_sha256']


def test_online_stage_uses_explicit_completed_directory_not_episode_names(tmp_path, monkeypatch):
    source = tmp_path/'opaque-actual-run-location'
    source.mkdir();(source/'completed.json').write_text('{}')
    settings = full.load_settings(full.DEFAULT_SETTINGS)
    settings.update(output_root=str(tmp_path/'campaigns'), online_run_directory=None)
    path = tmp_path/'settings.json';path.write_text(json.dumps(settings))
    monkeypatch.setenv('PYTHONHASHSEED','0')
    monkeypatch.setattr(full,'validate_settings',lambda _: {'status':'pass'})
    def online(*args, completed_run_sink):
        completed_run_sink(source)
        return SimpleNamespace(receipts=[SimpleNamespace(episode_id='arbitrary-no-run-directory')])
    monkeypatch.setattr(full,'run_live_calibration_from_settings',online)
    def offline(**kwargs):
        assert kwargs['source'] == source
        return object()
    monkeypatch.setattr(full,'_offline_calibrate',offline)
    monkeypatch.setattr(full,'_train',lambda **kwargs: 'completed')
    assert full.run_full_from_settings(path) == 'completed'


def test_full_train_gate_passes_pinned_execution_proof_to_real_gate(tmp_path, monkeypatch):
    from tests.reproducibility.test_dataset_access import gate
    from tests.reproducibility.test_dataset_manifest import synthetic_hashes
    from toolsandbox_pipeline.orchestration import live_environment_attestation as attest
    from toolsandbox_pipeline.retrieval.index import file_hash
    base,manifest,_,_,args=gate(tmp_path)
    index_path=base.manifest_path.parent/'dataset_index.json'
    index=json.loads(index_path.read_bytes())
    evidence=dict(version='dataset-build-execution-environment-v2',dataset_index_sha256=file_hash(index_path.read_bytes()),
        dataset_build_status=index['status'],dataset_build_environment_sha256=index['environment_sha256'],
        dataset_build_environment=index['environment'])
    verified=[]
    def verify(value, **kwargs):
        verified.append((value,kwargs))
        return value == evidence and kwargs['expected_dataset_index_sha256'] == evidence['dataset_index_sha256']
    monkeypatch.setattr(attest,'validate_execution_attestation',verify)
    current=full._attested_train_gate(manifest_path=base.manifest_path,manifest_sha256=base.expected_manifest_sha256,
        reconstruct=base.reconstruct,audit=lambda event:None,attestation=evidence)
    current.hash_scenario=synthetic_hashes
    selected=manifest.scenarios[0]
    family=next(item for item in manifest.families if item.family_id==selected.scenario_family_id)
    result=current.load(requested_ids=(selected.scenario_id,),**{**args,'phase':'train_round'},
        split='train',purpose='train_round',train_shard=family.train_shard)
    assert result[0].record==selected
    assert len(verified)==1 and verified[0][1]['project_root']==full.PROJECT
    assert json.loads(index_path.read_bytes())==index
