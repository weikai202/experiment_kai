"""Allowlisted coordinator CLI: fresh real online calibration, train only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import inspect
import io
import json
import locale
import os
from pathlib import Path
import time
import uuid

from toolsandbox_pipeline.reproducibility import canonical_sha256, canonical_json_bytes
from toolsandbox_pipeline.retrieval.index import file_hash

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS = PROJECT / 'configs/run/live_calibration_v1.json'


def _write(path, payload):
    raw = canonical_json_bytes(payload)
    with open(path, 'xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def load_settings(path):
    value = json.loads(Path(path).read_bytes())
    expected = {'schema_version', 'runtime_root', 'qwen_config', 'embedding_config',
        'train_manifest', 'online_limits', 'tool_metadata', 'tool_metadata_manifest',
        'backend_manifest', 'backend_manifest_sha256', 'external_preflight', 'hard_ceiling', 'qwen_base_url'}
    if set(value) != expected or value['schema_version'] != 1:
        raise ValueError('unsupported live calibration settings')
    if type(value['hard_ceiling']) is not int or value['hard_ceiling'] < 384:
        raise ValueError('invalid hard ceiling')
    for key in expected - {'schema_version', 'backend_manifest_sha256', 'hard_ceiling', 'qwen_base_url'}:
        if not Path(value[key]).is_absolute():
            raise ValueError('explicit absolute runtime paths required')
    if value['qwen_base_url'] != 'http://127.0.0.1:18080/v1':
        raise ValueError('reviewed local Qwen endpoint required')
    return value


def _run_provider_preflights(run, configs, preflight):
    results = []
    for name, config in configs:
        try:
            result = preflight(name, config)
            results.append(result)
            _write(run / f'preflight-{name}.json', result)
            if result['status'] != 'pass':
                raise RuntimeError('provider preflight failed')
        except Exception as error:
            _write(run / 'failure.json', {'exception_class': type(error).__name__,
                                         'phase': 'provider_preflight', 'provider': name})
            raise
    return results


def calibration_authorizer(*, run_id, runtime_manifest_sha256, backend_path, backend_sha256, records):
    """Authorize both the initial scope and the exact leased pilot dispatch."""
    records_by_id = {record.scenario_id: record for record in records}
    def authorize(current=None, *, identity=None, lease=None, purpose=None):
        if current is not None:
            if identity is not None or lease is not None or purpose is not None:
                raise PermissionError('mixed calibration authorization forms')
            if (current.run_id != run_id or current.runtime_config_sha256 != runtime_manifest_sha256
                    or current.generation_id != 'g000' or current.round_index != 0):
                raise PermissionError('calibration scope identity mismatch')
        else:
            if identity is None or lease is None or purpose != 'online_token_calibration':
                raise PermissionError('exact calibration lease authorization required')
            record = records_by_id.get(identity.scenario_id)
            if (record is None or lease.record != record or identity.run_id != run_id
                    or identity.runtime_config_sha256 != runtime_manifest_sha256
                    or identity.phase != 'online_token_calibration_pilot' or identity.generation_id != 'g000'
                    or identity.round_index is not None or identity.shard_id is not None
                    or identity.family_id != record.scenario_family_id
                    or identity.starting_context_sha256 != record.starting_context_sha256
                    or identity.evaluation_definition_sha256 != record.evaluation_definition_sha256
                    or identity.agent_tool_schema_sha256 != record.agent_facing_tool_schema_sha256
                    or identity.max_messages != record.max_messages):
                raise PermissionError('calibration lease identity mismatch')
        if file_hash(Path(backend_path).read_bytes()) != backend_sha256:
            raise PermissionError('external backend changed since authorization')
    return authorize


def _failure_summary(error, phase):
    from toolsandbox_pipeline.orchestration.live_calibration import CalibrationCollectionError
    result = {'exception_class': type(error).__name__, 'phase': phase}
    if isinstance(error, CalibrationCollectionError):
        result['code'] = error.code
    causes, seen = [], set()
    current = error.__cause__ or error.__context__
    while current is not None and id(current) not in seen and len(causes) < 5:
        seen.add(id(current))
        causes.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    if causes:
        result['cause_classes'] = causes
    return result


def run_live_calibration_from_settings(settings_path=DEFAULT_SETTINGS, *, completed_run_sink=None):
    if completed_run_sink is not None and not callable(completed_run_sink):
        raise TypeError("completed_run_sink must be callable")
    # Imports stay inside the explicit operation; module import performs no work.
    import httpx
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.roles import openai_api_user
    from tool_sandbox.scenarios import (named_scenarios, named_single_tool_call_scenarios,
        named_multiple_tool_call_scenarios, named_multiple_user_turn_scenarios,
        named_insufficient_information_scenarios)
    from toolsandbox_pipeline.checkpointing import CheckpointStore, RunIdentity, LLMLedger
    from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig, UserSimulatorConfig
    from toolsandbox_pipeline.schemas.dataset import SplitManifest, REGISTRIES
    from toolsandbox_pipeline.schemas.fixtures import ExternalReadContext
    from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
    from toolsandbox_pipeline.online.prompt_loader import load_prompts
    from toolsandbox_pipeline.online.token_limits import load_token_limits
    from toolsandbox_pipeline.online.token_limit_calibration import CalibrationRuntimeInputs, CalibrationScenario
    from toolsandbox_pipeline.online.tool_metadata import load_tool_metadata, public_tool_inventory
    from toolsandbox_pipeline.providers.preflight import run_preflight
    from toolsandbox_pipeline.retrieval.contracts import EmbeddingIdentity
    from toolsandbox_pipeline.retrieval.embedding_cache import EmbeddingCache
    from toolsandbox_pipeline.orchestration.seed_skill_builder import build_seed_skills
    from toolsandbox_pipeline.orchestration.generation_builder import build_generation
    from toolsandbox_pipeline.orchestration.live_providers import LiveProviderServices
    from toolsandbox_pipeline.orchestration.live_episode import LiveEpisodeScope
    from toolsandbox_pipeline.orchestration.live_calibration_runner import run_live_calibration
    from toolsandbox_pipeline.reproducibility.dataset_access import DatasetAccessGate
    from toolsandbox_pipeline.reproducibility.dataset_manifest import load_build_config
    from toolsandbox_pipeline.reproducibility.clock import FixedWorldClock
    from toolsandbox_pipeline.reproducibility.splits import build_family_registry, build_augmented_registry
    from toolsandbox_pipeline.reproducibility.fixture_store import load_backend_manifest
    from toolsandbox_pipeline.reproducibility.fixture_cli import RequestsTransport
    from toolsandbox_pipeline.reproducibility.rapidapi_boundary import RapidAPIBoundary
    from toolsandbox_pipeline.toolsandbox_adapter.native_external_scope import NativeExternalCallScope
    from toolsandbox_pipeline.toolsandbox_adapter.trajectory_store import TrajectoryStore

    settings = load_settings(settings_path)
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('launcher must set PYTHONHASHSEED=0 before interpreter startup')
    os.environ['QWEN_BASE_URL'] = settings['qwen_base_url']
    os.environ['QWEN_API_KEY'] = 'EMPTY'
    os.umask(0o077)
    os.environ['TZ'] = 'UTC'
    time.tzset()
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    qc = QwenConfig.model_validate_json(Path(settings['qwen_config']).read_bytes())
    ec = EmbeddingConfig.model_validate_json(Path(settings['embedding_config']).read_bytes())
    uc = UserSimulatorConfig()
    train_raw = Path(settings['train_manifest']).read_bytes()
    train = SplitManifest.model_validate_json(train_raw)
    if train.split != 'train':
        raise ValueError('calibration requires train manifest')
    train_hash = file_hash(train_raw)
    backend, backend_hash = load_backend_manifest(settings['backend_manifest'],
        expected_sha256=settings['backend_manifest_sha256'])
    preflight_raw = Path(settings['external_preflight']).read_bytes()
    external = json.loads(preflight_raw)
    if (external.get('backend_manifest_sha256') != backend_hash
            or {row.get('tool') for row in external.get('results', [])} != {item.canonical_tool_name for item in backend.backends}
            or external.get('status') != 'pass' or len(external.get('results', [])) != 5 or any(
            row.get('status') != 'pass' for row in external['results'])):
        raise ValueError('approved five-service external preflight required')
    prompts = {p.entry.role: p for p in load_prompts(PROJECT)}
    prompt_hash = file_hash((PROJECT / 'prompts/manifest.json').read_bytes())
    limit_hash = file_hash(Path(settings['online_limits']).read_bytes())
    limits = load_token_limits(Path(settings['online_limits']), expected_sha256=limit_hash)
    metadata = load_tool_metadata(Path(settings['tool_metadata']), Path(settings['tool_metadata_manifest']))
    build_config, clock_hash = load_build_config(PROJECT / 'configs/reproducibility/dataset_build_v1.json')
    inventory = public_tool_inventory()
    inventory_hash = canonical_sha256(list(inventory))
    no_fixture = {'profile': 'official_live', 'fixture_store': None, 'backend_manifest_sha256': backend_hash}
    no_fixture_hash = canonical_sha256(no_fixture)
    root = Path(settings['runtime_root'])
    run_id = 'live-calibration-' + time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:6]
    run = root / 'runs' / run_id
    run.mkdir(mode=0o700)
    _write(run / 'settings.json', settings)
    print(json.dumps({'phase': 'calibration_bootstrap', 'run_dir': str(run)}), flush=True)
    preflights = _run_provider_preflights(run, (('qwen', qc), ('embedding', ec),
        ('user-simulator', uc)), run_preflight)
    ec = ec.model_copy(update={'expected_dimension': preflights[1]['vector_dimension']})
    seed, provenance = build_seed_skills()
    _write(run / 'seed-provenance.json', provenance)
    source_hashes = {str(p.relative_to(PROJECT)): file_hash(p.read_bytes())
        for p in sorted((PROJECT / 'src/toolsandbox_pipeline').rglob('*.py'))}
    user_protocol = {'definition': 'generic User protocol source plus train manifest; not one literal prompt',
        'user_role_source_sha256': file_hash(Path(inspect.getfile(openai_api_user)).read_bytes()),
        'train_manifest_sha256': train_hash, 'upstream_commit': '165848b9a78cead7ca7fe7c89c688b58e6501219'}
    manifest = {'schema_version': 1, 'purpose': 'online_token_calibration', 'profile': 'official_live',
        'settings': settings, 'qwen': qc.model_dump(mode='json'), 'embedding': ec.model_dump(mode='json'),
        'user': uc.model_dump(mode='json'), 'dataset_manifest_sha256': train_hash,
        'prompt_manifest_sha256': prompt_hash, 'token_limit_config_sha256': limit_hash,
        'source_hashes': source_hashes, 'dependency_lock_sha256': file_hash((PROJECT / 'uv.lock').read_bytes()),
        'external_preflight_sha256': file_hash(preflight_raw), 'fixture_scope': no_fixture,
        'user_prompt_binding': user_protocol,
        'notes': ['Actual live external tools, no replay fixtures.', 'Calibration recommendation only; no automatic formal promotion.']}
    manifest_hash = canonical_sha256(manifest)
    environment_hash = canonical_sha256({'image': json.loads((root / 'image-provenance.json').read_bytes()),
                                        'qwen': qc.model_dump(mode='json')})
    _write(run / 'manifest.json', manifest)
    identity = RunIdentity(run_id=run_id, profile='official_live', environment_identity=environment_hash,
        dataset_manifest_sha256=train_hash, config_manifest_sha256=manifest_hash,
        prompt_manifest_sha256=prompt_hash, generation_manifest_sha256=canonical_sha256([s.model_dump(mode='json') for s in seed]),
        fixture_manifest_sha256=no_fixture_hash)
    store = CheckpointStore.create(run / 'checkpoint', identity, PROJECT / 'configs/reproducibility/checkpointing_v1.json')
    ledger = LLMLedger(store)
    cache = EmbeddingCache(run / 'embedding-cache.sqlite', EmbeddingIdentity(), ec.expected_dimension)
    try:
        services = LiveProviderServices(ledger=ledger, qwen_config=qc, embedding_config=ec, user_config=uc,
            manifest_identity=manifest_hash, phase='online_calibration_generation_zero')
        generation_scope = AccountingScope(run_id=run_id, task_id='generation-zero',
            scenario_family_id='setup', scenario_id='setup', system_variant='generation_0')
        staging = run / 'generation-staging'
        staging.mkdir()
        gen = build_generation(staging, generation_id='g000', parent_generation_id=None,
            policy_memory=(), world_memory=(), skills=seed, tool_inventory=inventory,
            tool_inventory_sha256=inventory_hash, cache=cache, gateway=services.embedding_gateway,
            context_factory=lambda ordinal, texts: services.embedding_context_factory(generation_scope, ordinal, texts),
            record_durable=services.embedding_record_durable)
        scope = LiveEpisodeScope(run_id=run_id, round_index=0, shard_id='train-shard-0', generation_id='g000',
            profile='official_live', dataset_manifest_sha256=train_hash, runtime_config_sha256=manifest_hash,
            prompt_manifest_sha256=prompt_hash, token_limit_config_sha256=limit_hash,
            fixture_manifest_sha256=no_fixture_hash, environment_sha256=environment_hash,
            scenario_positions={r.scenario_id: i for i, r in enumerate(train.scenarios)})
        runtime_inputs = CalibrationRuntimeInputs(generation_manifest_sha256=canonical_sha256(gen.manifest.model_dump(mode='json')),
            embedding_config_sha256=canonical_sha256(ec.model_dump(mode='json')), user_simulator_model=uc.model,
            user_simulator_prompt_sha256=canonical_sha256(user_protocol),
            controller_config_sha256=file_hash(Path(settings['tool_metadata']).read_bytes()),
            tool_schema_manifest_sha256=canonical_sha256([r.agent_facing_tool_schema_sha256 for r in train.scenarios]),
            fixture_store_manifest_sha256=no_fixture_hash, world_clock_config_sha256=clock_hash,
            online_orchestrator_version='live-calibration-bootstrap-v1')
        def count_tokens(messages):
            # Same validated local vLLM tokenizer used by reflection smoke.
            response = httpx.post('http://127.0.0.1:18080/tokenize', json={'model': qc.model,
                'messages': [m.model_dump(mode='json') for m in messages], 'add_generation_prompt': True,
                'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)
            response.raise_for_status()
            return response.json()['count']
        authorize = calibration_authorizer(run_id=run_id, runtime_manifest_sha256=manifest_hash,
            backend_path=settings['backend_manifest'], backend_sha256=backend_hash, records=train.scenarios)
        @contextmanager
        def external_boundary(*, identity, binding, state_id, attempt_sink):
            with NativeExternalCallScope(binding.call_ids) as calls:
                def context():
                    return ExternalReadContext(schema_version=1, run_id=identity.run_id, profile='official_live',
                        phase=identity.phase, scenario_family_id=identity.family_id, scenario_id=identity.scenario_id,
                        state_id=state_id, logical_tool_call_id=calls.current_call_id(),
                        backend_manifest_sha256=backend_hash, fixture_manifest_sha256=None)
                with RapidAPIBoundary(mode='official_live', profile='official_live', backend_manifest=backend,
                    backend_manifest_sha256=backend_hash, context_provider=context, attempt_sink=attempt_sink,
                    transport=RequestsTransport(), credential_provider=lambda: os.environ['RAPID_API_KEY']):
                    yield
        def physical_attempts(identity, logical_ids):
            accounting = AccountingScope(run_id=identity.run_id, round_index=None, task_id=identity.episode_id,
                scenario_family_id=identity.family_id, scenario_id=identity.scenario_id, system_variant='generation_0')
            return ledger.snapshot_accounting(accounting).population.physical_attempt_ids
        with FixedWorldClock(build_config):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                families = build_family_registry(dict(zip(REGISTRIES, (named_single_tool_call_scenarios,
                    named_multiple_tool_call_scenarios, named_multiple_user_turn_scenarios,
                    named_insufficient_information_scenarios))), ToolBackend.DEFAULT)
                registry = build_augmented_registry(families, named_scenarios, ToolBackend.DEFAULT)
            def audit(event):
                ledger.commit_checkpoint('calibration-dataset-' + canonical_sha256(event)[7:], 'calibration_dataset_access', event)
            gate = DatasetAccessGate(manifest_path=Path(settings['train_manifest']), expected_manifest_sha256=train_hash,
                reconstruct=lambda rec: registry[rec.scenario_id], audit=audit)
            result = run_live_calibration(scope=scope,
                records=tuple(CalibrationScenario(scenario_id=r.scenario_id, variant=r.variant, split='train') for r in train.scenarios),
                gate=gate, generation=gen, ledger=ledger, trajectory_store=TrajectoryStore(store, environment_identity=environment_hash),
                qwen_config=qc, embedding_config=ec, user_config=uc, prompts=prompts, token_limits=limits,
                metadata=metadata, embedding_cache=cache, count_prompt_tokens=count_tokens, runtime_inputs=runtime_inputs,
                authorization=authorize, physical_attempt_provider=physical_attempts,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), output_directory=run / 'recommendation',
                hard_ceiling=settings['hard_ceiling'], contextual_external_boundary_factory=external_boundary,
                backend_manifest_sha256=backend_hash)
        _write(run / 'completed.json', {'role_counts': result.role_counts, 'artifact_hashes': result.artifact_hashes})
        if completed_run_sink is not None:
            completed_run_sink(run)
        return result
    except Exception as error:
        _write(run / 'failure.json', _failure_summary(error, 'generation_pilot_or_replay'))
        raise
    finally:
        cache.close()
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, default=DEFAULT_SETTINGS)
    args = parser.parse_args()
    try:
        run_live_calibration_from_settings(args.settings)
    except Exception as error:
        print(json.dumps({'phase': 'calibration_failed', 'exception_class': type(error).__name__}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
