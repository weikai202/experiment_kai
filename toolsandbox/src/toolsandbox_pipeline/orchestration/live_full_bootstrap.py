"""Fixed live-v2 campaign: online calibration, offline calibration, three train rounds.

Dev content is acquired only by the Skill selector/mini-bench. No final-test entry
exists here. Registry-declared image identity is explicitly not a verified running
image identity. Missing natural calibration roles remain fatal, never synthesized.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.orchestration.live_calibration_bootstrap import (
    PROJECT, _write, load_settings as load_online_settings, run_live_calibration_from_settings,
)
DEFAULT_SETTINGS = PROJECT / 'configs/run/live_full_v2.json'
CHECKPOINT_CONFIG = PROJECT / 'configs/reproducibility/checkpointing_v1.json'


def load_settings(path):
    data = json.loads(Path(path).read_bytes())
    keys = {'schema_version', 'online_settings', 'online_run_directory', 'output_root',
            'dataset_index', 'dataset_index_sha256', 'image_provenance', 'dev_manifest', 'offline_memory_limits'}
    if set(data) != keys or data['schema_version'] != 2:
        raise ValueError('exact live full v2 settings required')
    for key in keys - {'schema_version', 'online_run_directory', 'dataset_index_sha256'}:
        if not Path(data[key]).is_absolute():
            raise ValueError('absolute full runtime paths required')
    if data['online_run_directory'] is not None and not Path(data['online_run_directory']).is_absolute():
        raise ValueError('explicit completed online calibration directory required')
    return data


def validate_settings(path=DEFAULT_SETTINGS):
    """Offline validation; does not read dev/test scenario contents or credentials."""
    data = load_settings(path)
    online = load_online_settings(data['online_settings'])
    for key in ('dataset_index', 'image_provenance', 'offline_memory_limits'):
        if not Path(data[key]).is_file():
            raise ValueError('required full runtime file missing: ' + key)
    if not Path(data['dev_manifest']).is_file():
        raise ValueError('selector Dev manifest file missing')
    index = json.loads(Path(data['dataset_index']).read_bytes())
    image = json.loads(Path(data['image_provenance']).read_bytes())
    if index['train_manifest_sha256'] != file_hash(Path(online['train_manifest']).read_bytes()):
        raise ValueError('train/index mismatch')
    from toolsandbox_pipeline.orchestration.live_environment_attestation import attest_execution_environment
    attest_execution_environment(project_root=PROJECT, dataset_index_path=Path(data['dataset_index']),
        expected_dataset_index_sha256=data['dataset_index_sha256'])
    if not image.get('registry_resolved_digest'):
        raise ValueError('registry image declaration missing')
    return {'status': 'pass', 'protocol': 'toolsandbox-evolution-live-v2',
            'running_image_verified': image.get('running_container_digest_verified') is True,
            'test_access': 'forbidden', 'provider_calls': 0}


def _open_store(path):
    from toolsandbox_pipeline.checkpointing import CheckpointStore, RunIdentity
    database = Path(path) / 'checkpointing/ledger.sqlite3'
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        row = connection.execute('SELECT payload FROM run_identity WHERE singleton=1').fetchone()
        identity = RunIdentity.model_validate_json(row[0])
    finally:
        connection.close()
    return CheckpointStore.open(Path(path), identity, CHECKPOINT_CONFIG)


def _load_skill_prompts():
    raw = (PROJECT / 'prompts/offline/skill_manifest.json').read_bytes()
    manifest = json.loads(raw)
    prompts, hashes = {}, {}
    for item in manifest['prompts']:
        path = PROJECT / 'prompts/offline' / item['path']
        content = path.read_bytes()
        if file_hash(content) != item['sha256']:
            raise ValueError('offline Skill prompt changed')
        prompts[item['role']], hashes[item['role']] = content.decode(), file_hash(content)
    if set(prompts) != {'failure_mode_update', 'skill_candidate'}:
        raise ValueError('exact offline Skill prompts required')
    return prompts, hashes, file_hash(raw)


def _token_counter(qwen):
    def count(messages):
        import httpx
        rows = [m.model_dump(mode='json') if hasattr(m, 'model_dump') else m for m in messages]
        result = httpx.post('http://127.0.0.1:18080/tokenize', json={'model': qwen.model,
            'messages': rows, 'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)
        result.raise_for_status()
        return result.json()['count']
    return count


def _offline_calibrate(*, source, directory, settings, online, resources):
    from toolsandbox_pipeline.checkpointing import CheckpointStore, RunIdentity, LLMLedger
    from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig, UserSimulatorConfig
    from toolsandbox_pipeline.online.tool_metadata import public_tool_inventory
    from toolsandbox_pipeline.memory.store import load_generation
    from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
    from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
    from toolsandbox_pipeline.orchestration.live_providers import LiveProviderServices
    from toolsandbox_pipeline.orchestration.live_offline_calibration import (OfflineCalibrationScope,
        load_pilot_calibration_inputs, DurableOfflineCalibrationReplay, collect_offline_calibration_requests,
        calibrate_offline_requests, verify_collected_source, offline_limit_recommendations)
    from toolsandbox_pipeline.offline.memory_prompts import load_memory_prompts
    from toolsandbox_pipeline.offline.memory_roles import load_token_limits as load_memory_limits
    from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
    from toolsandbox_pipeline.orchestration.seed_skill_builder import extract_public_tool_schemas
    from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
    source_manifest = json.loads((source / 'manifest.json').read_bytes())
    if not (source / 'completed.json').is_file():
        raise ValueError('completed online calibration required before offline collection')
    source_store = resources.enter_context(_open_store(source / 'checkpoint'))
    source_ledger = LLMLedger(source_store)
    events = source_store._connection.execute("SELECT checkpoint_id FROM checkpoint_events WHERE event_kind='online_calibration_pilot_receipt' ORDER BY event_ordinal").fetchall()
    episodes = tuple(source_ledger.get_checkpoint(row['checkpoint_id']).payload['episode_id'] for row in events)
    qc = QwenConfig.model_validate(source_manifest['qwen'])
    ec = EmbeddingConfig.model_validate(source_manifest['embedding'])
    uc = UserSimulatorConfig.model_validate(source_manifest['user'])
    current_qc = QwenConfig.model_validate_json(Path(online['qwen_config']).read_bytes())
    if qc != current_qc:
        raise ValueError('online calibration Qwen configuration drift')
    inventory = public_tool_inventory()
    gen = load_generation(source / 'generation-staging/g000', expected_generation_id='g000',
        expected_embedding=EmbeddingIdentity(), tool_inventory=inventory, tool_inventory_sha256=canonical_sha256(list(inventory)))
    scope = OfflineCalibrationScope(run_id=source_store.identity.run_id,
        dataset_manifest_sha256=source_store.identity.dataset_manifest_sha256,
        runtime_config_sha256=source_store.identity.config_manifest_sha256,
        train_scenario_ids=tuple(source_ledger.get_checkpoint(row['checkpoint_id']).payload['scenario_id'] for row in events))
    trajectories, capture_loader = load_pilot_calibration_inputs(ledger=source_ledger, episode_ids=episodes, scope=scope)
    memory_prompts = load_memory_prompts(PROJECT, PROJECT / 'prompts/offline/memory_manifest.json')
    memory_limits, memory_limits_hash = load_memory_limits(Path(settings['offline_memory_limits']))
    skill_prompts, skill_hashes, skill_manifest_hash = _load_skill_prompts()
    bindings = {'source_online_run_manifest_sha256': source_store.identity.config_manifest_sha256,
        'source_dataset_manifest_sha256': source_store.identity.dataset_manifest_sha256,
        'source_generation_manifest_sha256': canonical_sha256(gen.manifest.model_dump(mode='json')),
        'qwen': qc.model_dump(mode='json'), 'embedding': ec.model_dump(mode='json'),
        'memory_prompt_manifest_sha256': memory_prompts.manifest_sha256,
        'skill_prompt_manifest_sha256': skill_manifest_hash,
        'source_hashes': {str(p.relative_to(PROJECT)): file_hash(p.read_bytes())
            for p in sorted((PROJECT / 'src/toolsandbox_pipeline').rglob('*.py'))}}
    _write(directory / 'offline-runtime.json', bindings)
    runtime_hash = canonical_sha256(bindings)
    run_identity = source_store.identity.model_copy(update={'run_id': directory.name + '-offline', 'config_manifest_sha256': runtime_hash})
    store = resources.enter_context(CheckpointStore.create(directory / 'offline-checkpoint', run_identity, CHECKPOINT_CONFIG))
    ledger = LLMLedger(store)
    services = LiveProviderServices(ledger=ledger, qwen_config=qc, embedding_config=ec, user_config=uc,
        manifest_identity=runtime_hash, phase='offline_calibration_collection')
    cache = resources.enter_context(EmbeddingCache(directory / 'offline-embedding-cache.sqlite', EmbeddingIdentity(), ec.expected_dimension))
    accounting = AccountingScope(run_id=run_identity.run_id, task_id='offline-calibration-retrieval',
        scenario_family_id='offline', scenario_id='offline', system_variant='generation_0')
    class Resolver:
        def resolve_one(self, text):
            return cache.resolve((text,), gateway=services.embedding_gateway,
                context_factory=lambda ordinal, texts: services.embedding_context_factory(accounting, ordinal, texts),
                record_durable=services.embedding_record_durable)[0]
    retriever = MemoryCandidateRetriever(generation_id='g000', policy_records=gen.policy_memory,
        world_records=gen.world_memory, indexes=gen.indexes, embedding_resolver=Resolver())
    replay = DurableOfflineCalibrationReplay(ledger=ledger, gateway=services.qwen_gateway,
        manifest_identity=runtime_hash, count_tokens=_token_counter(qc))
    captured = collect_offline_calibration_requests(trajectories=trajectories, capture_loader=capture_loader,
        scope=scope, replay=replay, prompts=memory_prompts, limits=memory_limits, limits_sha256=memory_limits_hash,
        retriever=retriever, current_skills=gen.skills, skill_prompts=skill_prompts,
        public_tool_schemas=extract_public_tool_schemas(), runtime_inputs=bindings, hard_ceiling=online['hard_ceiling'])
    evidence = calibrate_offline_requests(captured=captured, replay=replay,
        verify_source=lambda item: verify_collected_source(ledger, item), runtime_inputs=bindings, hard_ceiling=online['hard_ceiling'])
    recommendation = offline_limit_recommendations(evidence=evidence, replay=replay, runtime_inputs=bindings)
    output = directory / 'offline-recommendation'
    output.mkdir(mode=0o700)
    for name, raw in recommendation['blobs'].items():
        with open(output / name, 'xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(source_manifest=source_manifest, source_store=source_store, generation=gen,
        qwen=qc, embedding=ec, user=uc, memory_prompts=memory_prompts,
        skill_prompts=skill_prompts, skill_hashes=skill_hashes, skill_manifest_hash=skill_manifest_hash,
        offline=recommendation, offline_evidence=evidence, offline_bindings=bindings)


def _formal_manifest(*, directory, settings, online, source, material):
    from toolsandbox_pipeline.schemas.run import ResolvedRunManifest, CheckpointObservationRegistry
    from toolsandbox_pipeline.schemas.checkpoint import CheckpointConfig
    from toolsandbox_pipeline.schemas.dataset import SplitManifest
    from toolsandbox_pipeline.orchestration.seed_skill_builder import seed_skill_bytes, extract_public_tool_schemas, GENERATOR_VERSION
    from toolsandbox_pipeline.online.prompt_loader import load_prompts
    from toolsandbox_pipeline.online.token_limits import load_token_limits
    from toolsandbox_pipeline.online.tool_metadata import public_tool_inventory
    from toolsandbox_pipeline.offline.memory_prompts import load_memory_prompts
    index = json.loads(Path(settings['dataset_index']).read_bytes())
    image = json.loads(Path(settings['image_provenance']).read_bytes())
    from toolsandbox_pipeline.orchestration.live_environment_attestation import attest_execution_environment
    attestation = attest_execution_environment(project_root=PROJECT, dataset_index_path=Path(settings['dataset_index']),
        expected_dataset_index_sha256=settings['dataset_index_sha256'])
    environment = attestation['execution_environment']
    qc, ec, uc = material['qwen'], material['embedding'], material['user']
    train_raw = Path(online['train_manifest']).read_bytes()
    train = SplitManifest.model_validate_json(train_raw)
    online_path = source / 'recommendation/online_token_limits.recommended.json'
    online_hash = file_hash(online_path.read_bytes())
    limits = load_token_limits(online_path, expected_sha256=online_hash)
    # The protocol publishes exact filenames; discover them from the completed
    # receipt, never by selecting whichever calibration happens to pass.
    completed = json.loads((source / 'completed.json').read_bytes())
    for name, digest in completed['artifact_hashes'].items():
        if file_hash((source / 'recommendation' / name).read_bytes()) != digest:
            raise ValueError('online calibration artifact changed')
    prompts = {p.entry.role: p for p in load_prompts(PROJECT)}
    for role in limits.roles:
        if role.prompt_sha256 != prompts[role.role].entry.sha256:
            raise ValueError('online calibration prompt drift')
    current_memory = load_memory_prompts(PROJECT, PROJECT / 'prompts/offline/memory_manifest.json')
    if current_memory != material['memory_prompts'] or _load_skill_prompts()[2] != material['skill_manifest_hash']:
        raise ValueError('offline calibrated prompt drift')
    seed_root = directory / 'seeds'
    seed_root.mkdir(mode=0o700)
    for name, raw in (('policy_memory.jsonl', b''), ('world_memory.jsonl', b''),
                      ('skills.jsonl', seed_skill_bytes(material['generation'].skills))):
        with open(seed_root / name, 'xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    provenance = (source / 'seed-provenance.json').read_bytes()
    public_schemas = extract_public_tool_schemas()
    seed_manifest = {'protocol': 'calibrated-g000-source-v1',
        'generation_manifest_sha256': canonical_sha256(material['generation'].manifest.model_dump(mode='json')),
        'source_run_id': material['source_store'].identity.run_id, 'provenance_sha256': file_hash(provenance)}
    _write(seed_root / 'source_manifest.json', seed_manifest)
    _write(directory / 'environment-provenance.json', {'attestation': attestation, 'image': image,
        'limitation': 'Registry tag digest is not proof of the running container image.'})
    image_verified = image.get('running_container_digest_verified') is True
    actual_image = image.get('running_container_digest') if image_verified else None
    if image_verified and actual_image is None:
        raise ValueError('verified image claim lacks actual running digest')
    declared = image['registry_resolved_digest']
    source_binding = material['source_manifest']['user_prompt_binding']
    offline_bundle = {'memory': material['offline']['hashes']['offline_memory_token_limits.calibrated.json'],
                      'skill': material['offline']['hashes']['offline_skill_token_limits.calibrated.json'],
                      'evidence': material['offline']['hashes']['offline_calibration_evidence.json']}
    _write(directory / 'offline-limit-bundle.json', offline_bundle)
    inventory = public_tool_inventory()
    config = CheckpointConfig.model_validate_json(CHECKPOINT_CONFIG.read_bytes())
    payload = dict(protocol_version='toolsandbox-evolution-live-v2', run_id=directory.name,
        purpose='formal_training', profile='official_live', dataset_manifest_path=online['train_manifest'],
        dataset_manifest_sha256=file_hash(train_raw), ordered_train_shard_ids=['train-shard-0','train-shard-1','train-shard-2'],
        upstream_source_sha256=environment['upstream_source_sha256'], dependency_lock_sha256=environment['dependency_lock_sha256'],
        container_image_digest=actual_image, registry_declared_digest=declared,
        image_verification='verified' if image_verified else 'unverified',
        environment_sha256=canonical_sha256({'attestation': attestation, 'image': image}),
        python_patch_version=environment['python_patch_version'],
        scenario_registry_sha256=canonical_sha256([r.scenario_id for r in train.scenarios]),
        tool_inventory_sha256=canonical_sha256(list(inventory)),
        evaluator_registry_sha256=canonical_sha256([r.evaluation_definition_sha256 for r in train.scenarios]),
        schema_bundle_sha256=canonical_sha256({str(p.relative_to(PROJECT)):file_hash(p.read_bytes())
            for p in sorted((PROJECT/'src/toolsandbox_pipeline/schemas').glob('*.py'))}),
        qwen=dict(endpoint_identity_sha256=canonical_sha256(online['qwen_base_url']), container_digest=actual_image,
            registry_declared_digest=declared, image_verification='verified' if image_verified else 'unverified',
            server_configuration_sha256=file_hash(Path(online['qwen_config']).read_bytes()),
            decoding_configuration_sha256=canonical_sha256(qc.model_dump(mode='json')),
            structured_output_wire_mode=qc.structured_output_wire_mode),
        embedding=dict(client_configuration_sha256=canonical_sha256(ec.model_dump(mode='json')), expected_dimension=ec.expected_dimension),
        user_simulator=dict(profile_sha256=canonical_sha256(uc.model_dump(mode='json')),
            prompt_sha256=canonical_sha256(source_binding), few_shot_sha256=canonical_sha256(source_binding),
            tool_schema_sha256=canonical_sha256(list(public_schemas)),
            stop_configuration_sha256=canonical_sha256({'protocol':source_binding,'model':uc.model})),
        online_prompt_manifest_sha256=file_hash((PROJECT/'prompts/manifest.json').read_bytes()),
        offline_prompt_manifest_sha256=canonical_sha256({'memory':current_memory.manifest_sha256,'skill':material['skill_manifest_hash']}),
        online_token_limits_sha256=online_hash, offline_token_limits_sha256=canonical_sha256(offline_bundle), calibrated_token_limits=True,
        seeds=dict(policy_memory_path=str(seed_root/'policy_memory.jsonl'), policy_memory_sha256=file_hash(b''),
            world_memory_path=str(seed_root/'world_memory.jsonl'), world_memory_sha256=file_hash(b''),
            skills_path=str(seed_root/'skills.jsonl'), skills_sha256=file_hash((seed_root/'skills.jsonl').read_bytes()),
            source_manifest_path=str(seed_root/'source_manifest.json'), source_manifest_sha256=file_hash((seed_root/'source_manifest.json').read_bytes()),
            public_tool_schema_inventory_sha256=canonical_sha256(list(public_schemas)), skill_generator_version=GENERATOR_VERSION,
            skill_generator_sha256=file_hash((PROJECT/'src/toolsandbox_pipeline/orchestration/seed_skill_builder.py').read_bytes()),
            skill_provenance_sha256=file_hash(provenance)),
        fixture=dict(mode='live', manifest_sha256=canonical_sha256(material['source_manifest']['fixture_scope']),
            backend_configuration_sha256=online['backend_manifest_sha256']), checkpoint=config.model_dump(mode='json'),
        checkpoint_registry_schema_sha256=canonical_sha256(CheckpointObservationRegistry.model_json_schema()),
        metrics_schema_sha256=file_hash((PROJECT/'src/toolsandbox_pipeline/schemas/usage.py').read_bytes()), run_root=str(directory))
    manifest = ResolvedRunManifest.model_validate_json(canonical_json_bytes(payload))
    _write(directory/'resolved-run.json', manifest.model_dump(mode='json'))
    return manifest, train, limits, online_hash, prompts


def _formal_preflight(manifest, ledger, material, online, directory):
    from toolsandbox_pipeline.orchestration.preflight_gate import PreflightEvidence, verify_preflight_gate
    from toolsandbox_pipeline.providers.preflight import run_preflight
    from toolsandbox_pipeline.orchestration.run_manifest import validate_component_files
    evidence = []
    for role, mode, cfg, identity in (('qwen','qwen',material['qwen'],manifest.qwen),
            ('embedding','embedding',material['embedding'],manifest.embedding),
            ('user_simulator','user-simulator',material['user'],manifest.user_simulator)):
        result = run_preflight(mode, cfg)
        _write(directory/f'formal-preflight-{role}.json', result)
        if result['status'] != 'pass':
            raise RuntimeError('formal provider preflight failed')
        evidence.append(PreflightEvidence(kind=role, manifest_sha256=manifest.manifest_sha256,
            configuration_sha256=canonical_sha256(identity.model_dump(mode='json')), status='pass',
            checked_at_utc=datetime.now(timezone.utc), setup_latency_seconds=float(result['latency_seconds']),
            returned_model=result['returned_model'], input_tokens=result['input_tokens'], output_tokens=result['output_tokens'],
            total_tokens=result['total_tokens'], usage_complete=result['usage_complete'], response_sha256=result['response_hash']))
    started = time.monotonic()
    validate_component_files(manifest)
    dataset_latency = time.monotonic() - started
    started = time.monotonic()
    external = json.loads(Path(online['external_preflight']).read_bytes())
    if external['status'] != 'pass' or external['backend_manifest_sha256'] != manifest.fixture.backend_configuration_sha256:
        raise ValueError('formal external preflight binding mismatch')
    fixture_latency = time.monotonic() - started
    started = time.monotonic()
    probe = {'manifest_sha256':manifest.manifest_sha256, 'checkpoint_config':manifest.checkpoint.model_dump(mode='json')}
    ledger.commit_checkpoint('formal-filesystem-preflight','formal_filesystem_preflight',probe)
    if ledger.get_checkpoint('formal-filesystem-preflight').payload != probe:
        raise ValueError('checkpoint filesystem readback mismatch')
    filesystem_latency = time.monotonic() - started
    for role, digest, response, latency in (('dataset_manifest', manifest.dataset_manifest_sha256,
            {'verified_component_files':True}, dataset_latency), ('fixture_mode',canonical_sha256(manifest.fixture.model_dump(mode='json')),external, fixture_latency),
            ('checkpoint_filesystem',canonical_sha256(manifest.checkpoint.model_dump(mode='json')),probe, filesystem_latency)):
        evidence.append(PreflightEvidence(kind=role,manifest_sha256=manifest.manifest_sha256,configuration_sha256=digest,
            status='pass',checked_at_utc=datetime.now(timezone.utc),setup_latency_seconds=latency,usage_complete=False,
            response_sha256=canonical_sha256(response)))
    gate = verify_preflight_gate(manifest, tuple(evidence), now_utc=datetime.now(timezone.utc))
    _write(directory/'formal-preflight-evidence.json', [item.model_dump(mode='json') for item in evidence])
    return gate


def _attested_train_gate(*, manifest_path, manifest_sha256, reconstruct, audit, attestation):
    from toolsandbox_pipeline.reproducibility.dataset_access import DatasetAccessGate
    from toolsandbox_pipeline.orchestration.live_environment_attestation import validate_execution_attestation
    return DatasetAccessGate(manifest_path=manifest_path, expected_manifest_sha256=manifest_sha256,
        reconstruct=reconstruct, audit=audit, execution_attestation=attestation,
        verify_execution_attestation=lambda evidence, **kwargs: validate_execution_attestation(
            evidence, project_root=PROJECT, **kwargs))


def _train(*, source, directory, settings, online, material, resources):
    from toolsandbox_pipeline.checkpointing import CheckpointStore, RunIdentity, LLMLedger
    from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
    from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
    from toolsandbox_pipeline.orchestration.live_training import LiveTrainingAssets, LiveTrainingRuntime
    from toolsandbox_pipeline.orchestration.live_providers import LiveProviderServices
    from toolsandbox_pipeline.orchestration.live_native_dev import NativeDevMiniBenchFactory, NativeDevSelectorMetadataLoader
    from toolsandbox_pipeline.orchestration.generation_publisher import publish_generation_zero
    from toolsandbox_pipeline.toolsandbox_adapter.trajectory_store import TrajectoryStore
    from toolsandbox_pipeline.toolsandbox_adapter.native_external_scope import NativeExternalCallScope
    from toolsandbox_pipeline.reproducibility.fixture_store import load_backend_manifest
    from toolsandbox_pipeline.reproducibility.fixture_cli import RequestsTransport
    from toolsandbox_pipeline.reproducibility.rapidapi_boundary import RapidAPIBoundary
    from toolsandbox_pipeline.reproducibility.dataset_access import DatasetAccessGate
    from toolsandbox_pipeline.reproducibility.dataset_manifest import load_build_config
    from toolsandbox_pipeline.reproducibility.clock import FixedWorldClock
    from toolsandbox_pipeline.reproducibility.splits import build_family_registry, build_augmented_registry
    from toolsandbox_pipeline.schemas.fixtures import ExternalReadContext
    from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
    from toolsandbox_pipeline.schemas.dataset import REGISTRIES
    from toolsandbox_pipeline.online.tool_metadata import public_tool_inventory, load_tool_metadata
    from toolsandbox_pipeline.orchestration.seed_skill_builder import extract_public_tool_schemas
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import (named_scenarios, named_single_tool_call_scenarios,
        named_multiple_tool_call_scenarios, named_multiple_user_turn_scenarios, named_insufficient_information_scenarios)
    manifest, train, online_limits, online_hash, prompts = _formal_manifest(
        directory=directory, settings=settings, online=online, source=source, material=material)
    source_gen = source / 'generation-staging/g000'
    generation_root = directory / 'generations'
    generation_root.mkdir(mode=0o700)
    stage = directory / 'initial-generation-stage'
    stage.mkdir(mode=0o700)
    shutil.copytree(source_gen, stage / 'g000')
    publish_generation_zero(stage / 'g000', generation_root)
    identity = RunIdentity(run_id=manifest.run_id,profile=manifest.profile,
        environment_identity=manifest.environment_sha256, dataset_manifest_sha256=manifest.dataset_manifest_sha256,
        config_manifest_sha256=manifest.manifest_sha256,prompt_manifest_sha256=manifest.online_prompt_manifest_sha256,
        generation_manifest_sha256=canonical_sha256(material['generation'].manifest.model_dump(mode='json')),
        fixture_manifest_sha256=manifest.fixture.manifest_sha256)
    store = resources.enter_context(CheckpointStore.create(directory/'formal-checkpoint', identity, CHECKPOINT_CONFIG))
    ledger = LLMLedger(store)
    cache = resources.enter_context(EmbeddingCache(directory/'formal-embedding-cache.sqlite',EmbeddingIdentity(),material['embedding'].expected_dimension))
    def providers(phase):
        return LiveProviderServices(ledger=ledger,qwen_config=material['qwen'],embedding_config=material['embedding'],
            user_config=material['user'],manifest_identity=manifest.manifest_sha256,phase=phase)
    offline = material['offline']
    assets = LiveTrainingAssets(online_prompts=prompts,online_limits=online_limits,online_limits_sha256=online_hash,
        online_runtime_inputs_sha256=online_limits.runtime_inputs_sha256,memory_prompts=material['memory_prompts'],
        memory_limits=offline['memory_limits'],memory_limits_sha256=offline['hashes']['offline_memory_token_limits.calibrated.json'],
        skill_prompts=material['skill_prompts'],skill_prompt_hashes=material['skill_hashes'],
        skill_prompt_manifest_sha256=material['skill_manifest_hash'],
        skill_limits_sha256=offline['hashes']['offline_skill_token_limits.calibrated.json'],
        skill_token_limits={role:offline['selections'][role] for role in ('failure_mode_update','skill_candidate')},
        public_tool_inventory=public_tool_inventory(),public_tool_schemas=extract_public_tool_schemas(),
        metadata=load_tool_metadata(Path(online['tool_metadata']),Path(online['tool_metadata_manifest'])))
    backend, backend_hash = load_backend_manifest(online['backend_manifest'],expected_sha256=online['backend_manifest_sha256'])
    preflight = _formal_preflight(manifest,ledger,material,online,directory)
    def verify_calibration(current, target):
        if target.manifest_sha256 != manifest.manifest_sha256 or current is not assets:
            return False
        if online_limits.qwen_config_sha256 != canonical_sha256(material['qwen'].model_dump(mode='json')):
            return False
        if online_limits.calibration_artifact_sha256 != file_hash((source/'recommendation/calibration.json').read_bytes()):
            return False
        artifact = json.loads((source/'recommendation/calibration.json').read_bytes())
        if online_limits.runtime_inputs_sha256 != canonical_sha256(artifact['runtime_inputs']):
            return False
        if artifact['runtime_inputs']['generation_manifest_sha256'] != canonical_sha256(material['generation'].manifest.model_dump(mode='json')):
            return False
        if online_limits.train_manifest_sha256 != manifest.dataset_manifest_sha256:
            return False
        if any(file_hash((PROJECT/name).read_bytes()) != digest for name,digest in material['offline_bindings']['source_hashes'].items()):
            return False
        return all(file_hash((directory/'offline-recommendation'/name).read_bytes()) == digest
                   for name,digest in offline['hashes'].items())
    def authorize(current):
        if current.manifest_sha256 != manifest.manifest_sha256 or not verify_calibration(assets,manifest):
            raise PermissionError('formal manifest/calibration drift')
        if file_hash((PROJECT/'prompts/manifest.json').read_bytes()) != manifest.online_prompt_manifest_sha256:
            raise PermissionError('formal prompt drift')
        if file_hash(Path(online['backend_manifest']).read_bytes()) != backend_hash:
            raise PermissionError('formal backend drift')
    @contextmanager
    def external_boundary(*,identity,binding,state_id,attempt_sink):
        with NativeExternalCallScope(binding.call_ids) as calls:
            def context():
                return ExternalReadContext(schema_version=1,run_id=identity.run_id,profile='official_live',phase=identity.phase,
                    scenario_family_id=identity.family_id,scenario_id=identity.scenario_id,state_id=state_id,
                    logical_tool_call_id=calls.current_call_id(),backend_manifest_sha256=backend_hash,fixture_manifest_sha256=None)
            with RapidAPIBoundary(mode='official_live',profile='official_live',backend_manifest=backend,
                    backend_manifest_sha256=backend_hash,context_provider=context,attempt_sink=attempt_sink,
                    transport=RequestsTransport(),credential_provider=lambda:os.environ['RAPID_API_KEY']):
                yield
    def attempts(identity,logical_ids):
        scope = AccountingScope(run_id=identity.run_id,round_index=identity.round_index,task_id=identity.episode_id,
            scenario_family_id=identity.family_id,scenario_id=identity.scenario_id,system_variant=identity.system_variant)
        return ledger.snapshot_accounting(scope).population.physical_attempt_ids
    def audit(event):
        ledger.commit_checkpoint('formal-dataset-'+canonical_sha256(event)[7:],'formal_dataset_access',event)
    build_config,_ = load_build_config(PROJECT/'configs/reproducibility/dataset_build_v1.json')
    index = json.loads(Path(settings['dataset_index']).read_bytes())
    with FixedWorldClock(build_config):
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            families = build_family_registry(dict(zip(REGISTRIES,(named_single_tool_call_scenarios,
                named_multiple_tool_call_scenarios,named_multiple_user_turn_scenarios,named_insufficient_information_scenarios))),ToolBackend.DEFAULT)
            registry = build_augmented_registry(families,named_scenarios,ToolBackend.DEFAULT)
        environment_proof = json.loads((directory/'environment-provenance.json').read_bytes())
        if canonical_sha256({'attestation':environment_proof['attestation'], 'image':environment_proof['image']}) != manifest.environment_sha256:
            raise ValueError('formal execution environment provenance changed')
        train_gate = _attested_train_gate(manifest_path=Path(online['train_manifest']), manifest_sha256=manifest.dataset_manifest_sha256,
            reconstruct=lambda record:registry[record.scenario_id], audit=audit, attestation=environment_proof['attestation'])
        # Constructing a selector-only gate does not authorize arbitrary Dev reads.
        dev_gate = DatasetAccessGate(manifest_path=Path(settings['dev_manifest']),expected_manifest_sha256=index['dev_manifest_sha256'],
            reconstruct=lambda record:registry[record.scenario_id],audit=audit)
        def authorize_selector(**request):
            authorize(manifest)
            if (request['run_id'] != manifest.run_id or request['split'] != 'dev'
                    or request['purpose'] != 'skill_ab_validation' or request['phase'] != 'dev_minibench'):
                raise PermissionError('Dev metadata requires selected mini-bench scope')
        metadata_loader = NativeDevSelectorMetadataLoader(run_id=manifest.run_id,dev_manifest_sha256=index['dev_manifest_sha256'],
            manifest_loader=lambda:Path(settings['dev_manifest']).read_bytes(),base_family_loader=lambda family_id:registry[family_id],
            authorize=authorize_selector,audit=audit)
        dev = NativeDevMiniBenchFactory(dev_gate=dev_gate,metadata_loader=metadata_loader,
            dev_manifest_sha256=index['dev_manifest_sha256'],shared_configuration_sha256=manifest.manifest_sha256)
        runtime = LiveTrainingRuntime(manifest=manifest,preflight=preflight,train_manifest=train,assets=assets,ledger=ledger,
            trajectory_store=TrajectoryStore(store,environment_identity=manifest.environment_sha256),embedding_cache=cache,
            provider_factory=providers,train_gate=train_gate,dev_mini_bench_factory=dev,authorization=authorize,
            verify_calibration=verify_calibration,verify_skill_limit=offline['verify_token_limit_evidence'],
            count_prompt_tokens=_token_counter(material['qwen']),physical_attempt_provider=attempts,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),generation_root=generation_root,
            contextual_external_boundary_factory=external_boundary,backend_manifest_sha256=backend_hash)
        result = runtime.run()
    _write(directory/'training-completed.json', {'run_id':result.run_id,'status':result.completion_status,
        'updated_generation_id':result.updated_generation_id,'metrics':result.metric_record.model_dump(mode='json'),
        'image_verification':manifest.image_verification,'test_access':'forbidden'})
    return result


def run_full_from_settings(path=DEFAULT_SETTINGS):
    validate_settings(path)
    settings = load_settings(path)
    online = load_online_settings(settings['online_settings'])
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('launcher must set PYTHONHASHSEED=0')
    os.umask(0o077)
    os.environ['QWEN_BASE_URL'],os.environ['QWEN_API_KEY'] = online['qwen_base_url'],'EMPTY'
    os.environ['TZ']='UTC'
    time.tzset()
    import locale
    locale.setlocale(locale.LC_ALL,'C.UTF-8')
    root = Path(settings['output_root'])
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    directory = root / ('full-live-'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(mode=0o700)
    _write(directory/'settings.json',settings)
    phase = 'online_calibration'
    print(json.dumps({'phase':phase,'campaign_directory':str(directory)}),flush=True)
    try:
        if settings['online_run_directory'] is None:
            completed_runs = []
            run_live_calibration_from_settings(settings['online_settings'], completed_run_sink=completed_runs.append)
            if len(completed_runs) != 1 or not isinstance(completed_runs[0], Path) or not completed_runs[0].is_absolute():
                raise ValueError('one explicit completed calibration run directory required')
            source = completed_runs[0]
        else:
            source = Path(settings['online_run_directory'])
        _write(directory/'online-source.json',{'run_directory':str(source),'completed_sha256':file_hash((source/'completed.json').read_bytes())})
        phase = 'offline_calibration'
        print(json.dumps({'phase':phase,'campaign_directory':str(directory)}),flush=True)
        with ExitStack() as resources:
            material = _offline_calibrate(source=source,directory=directory,settings=settings,online=online,resources=resources)
            phase = 'formal_training'
            print(json.dumps({'phase':phase,'campaign_directory':str(directory)}),flush=True)
            return _train(source=source,directory=directory,settings=settings,online=online,material=material,resources=resources)
    except Exception as error:
        _write(directory/'failure.json',{'phase':phase,'exception_class':type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings',type=Path,default=DEFAULT_SETTINGS)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    try:
        if args.validate_only:
            print(json.dumps(validate_settings(args.settings)),flush=True)
        else:
            run_full_from_settings(args.settings)
        return 0
    except Exception as error:
        print(json.dumps({'phase':'full_failed','exception_class':type(error).__name__}),flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
