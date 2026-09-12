"""Concrete three-round runtime assembly using the reviewed native services.

Providers, authenticated gates, calibrated assets and token counting are supplied
explicitly. There is no alternate fake runtime, implicit calibration, or test access.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json

from toolsandbox_pipeline.checkpointing import UnitLedger
from toolsandbox_pipeline.memory.store import load_generation
from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
from toolsandbox_pipeline.orchestration.generation_builder import RoundGenerationCoordinator
from toolsandbox_pipeline.orchestration.live_episode import LiveEpisodeExecutor, LiveEpisodeScope
from toolsandbox_pipeline.orchestration.live_offline import LiveOfflineRequests, LiveMemoryUpdater, LiveSkillUpdater
from toolsandbox_pipeline.orchestration.live_projections import (
    CurrentRoundScope, CapturedOnlineTurn, CapturedEpisodeInputs, LiveTrajectoryLoader,
    LiveMemoryProjectionAdapter, LiveFailureEvidenceAdapter,
)
from toolsandbox_pipeline.orchestration.live_training_metrics import LiveTrainingMetrics
from toolsandbox_pipeline.orchestration.live_turn_context import LiveTurnContextFactory
from toolsandbox_pipeline.orchestration.round_runner import RoundRunner, Task014RoundBuffer
from toolsandbox_pipeline.orchestration.training_runner import TrainingRunner
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.dataset import SplitManifest
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.run import ResolvedRunManifest


@dataclass(frozen=True)
class LiveTrainingAssets:
    online_prompts: dict
    online_limits: object
    online_limits_sha256: str
    online_runtime_inputs_sha256: str
    memory_prompts: object
    memory_limits: object
    memory_limits_sha256: str
    skill_prompts: dict
    skill_prompt_hashes: dict
    skill_prompt_manifest_sha256: str
    skill_limits_sha256: str
    skill_token_limits: dict
    public_tool_inventory: tuple[str, ...]
    public_tool_schemas: tuple[dict, ...]
    metadata: tuple
    policy_input_representation: str = 'packed-v2'

    def verify_formal(self, manifest, verify_calibration):
        if self.online_limits.status != 'calibrated' or self.memory_limits.status != 'calibrated':
            raise ValueError('actual online and memory calibration required')
        if self.online_limits_sha256 != manifest.online_token_limits_sha256:
            raise ValueError('online calibrated config manifest mismatch')
        if set(self.skill_token_limits) != {'failure_mode_update', 'skill_candidate'}:
            raise ValueError('both Skill roles require calibrated selections')
        if any(limit.stage != 'calibrated' for limit in self.skill_token_limits.values()):
            raise ValueError('actual Skill calibration required')
        if canonical_sha256(list(self.public_tool_inventory)) != manifest.tool_inventory_sha256:
            raise ValueError('public tool inventory mismatch')
        if not callable(verify_calibration) or verify_calibration(self, manifest) is not True:
            raise ValueError('manifest-bound calibration evidence verification failed')


class _RevocableCaptures:
    def __init__(self, store, metrics):
        self.store, self.metrics, self.entries, self.revoked = store, metrics, {}, False
    def accept(self, capture):
        if self.revoked:
            raise PermissionError('round capture access revoked')
        self.metrics.record_episode(capture.result)
        if capture.result.evaluator_record_reference is None:
            return
        envelope_by_turn = dict(capture.initial_envelopes)
        value = CapturedEpisodeInputs(self.store.load_evaluator(capture.result.evaluator_record_reference),
            tuple(CapturedOnlineTurn(json.loads(envelope_by_turn[decision.identity.agent_turn_index]), proposed, decision)
                  for proposed, decision in zip(capture.proposed_actions, capture.decisions)))
        if len(capture.proposed_actions) != len(capture.decisions):
            raise ValueError('complete initial actions required')
        self.entries[capture.identity.episode_id] = value
    def load(self, trajectory):
        if self.revoked:
            raise PermissionError('round capture access revoked')
        return self.entries[trajectory.identity.episode_id]
    def revoke(self):
        self.entries.clear()
        self.revoked = True


class _RoundBuffer(Task014RoundBuffer):
    def __init__(self, *, capture_registry, **kwargs):
        super().__init__(**kwargs)
        self.capture_registry = capture_registry
    def archive_and_revoke(self, **kwargs):
        reference = super().archive_and_revoke(**kwargs)
        self.capture_registry.revoke()
        return reference


class _Finalizer:
    def __init__(self, ledger):
        self.ledger = ledger
    def commit_final_round_checkpoint(self, identity):
        return self.ledger.commit_checkpoint(f'live-final-round-{identity.round_index}',
            'formal_round_completed', dict(identity.__dict__))


class LiveTrainingRuntime:
    def __init__(self, *, manifest, preflight, train_manifest, assets, ledger,
                 trajectory_store, embedding_cache, provider_factory, train_gate,
                 dev_mini_bench_factory, authorization, verify_calibration,
                 verify_skill_limit, count_prompt_tokens, physical_attempt_provider,
                 boot_id, generation_root, contextual_external_boundary_factory=None,
                 backend_manifest_sha256=None):
        if type(manifest) is not ResolvedRunManifest or type(train_manifest) is not SplitManifest:
            raise TypeError('resolved formal run and exact train manifest required')
        if manifest.purpose != 'formal_training' or train_manifest.split != 'train':
            raise PermissionError('formal training accepts only the sealed train manifest')
        train_manifest = SplitManifest.model_validate_json(train_manifest.model_dump_json())
        # Dataset manifests carry a terminal LF; verify the actual input file too.
        from hashlib import sha256
        raw = Path(manifest.dataset_manifest_path).read_bytes()
        if 'sha256:' + sha256(raw).hexdigest() != manifest.dataset_manifest_sha256 or SplitManifest.model_validate_json(raw) != train_manifest:
            raise ValueError('train manifest identity mismatch')
        if ledger.store.identity.run_id != manifest.run_id or ledger.store.identity.config_manifest_sha256 != manifest.manifest_sha256:
            raise ValueError('formal ledger/run manifest mismatch')
        for service in (provider_factory, dev_mini_bench_factory, authorization, verify_skill_limit,
                        count_prompt_tokens, physical_attempt_provider):
            if not callable(service):
                raise TypeError('all concrete runtime services are required')
        assets.verify_formal(manifest, verify_calibration)
        if preflight.status != 'pass' or preflight.manifest_sha256 != manifest.manifest_sha256:
            raise ValueError('manifest-bound formal preflight pass required')
        if backend_manifest_sha256 != manifest.fixture.backend_configuration_sha256:
            raise ValueError('actual external backend configuration hash required')
        self.manifest, self.assets, self.ledger = manifest, assets, ledger
        self.store, self.cache, self.provider_factory = trajectory_store, embedding_cache, provider_factory
        self.train_gate, self.dev_factory = train_gate, dev_mini_bench_factory
        self.authorization, self.verify_skill_limit = authorization, verify_skill_limit
        self.count_tokens, self.attempts, self.boot_id = count_prompt_tokens, physical_attempt_provider, boot_id
        self.generation_root = Path(generation_root)
        if not self.generation_root.is_absolute() or self.generation_root.resolve() != self.generation_root or not self.generation_root.is_dir():
            raise ValueError('existing private generation root required')
        self.external_factory, self.backend_sha = contextual_external_boundary_factory, backend_manifest_sha256
        families = {family.family_id: family.train_shard for family in train_manifest.families}
        self.shards = tuple(tuple(record.scenario_id for record in train_manifest.scenarios
                           if families[record.scenario_family_id] == index) for index in range(3))
        self.metrics = LiveTrainingMetrics(ledger=ledger, manifest=manifest)
        self._built = set()
        self.runner = TrainingRunner(manifest=manifest, preflight=preflight, boot_id=boot_id,
            round_factory=self.build_round, metrics=self.metrics)
        self._load('g000')  # Validate the published, attested bootstrap before any episode.

    def _load(self, generation_id):
        return load_generation(self.generation_root / generation_id,
            expected_generation_id=generation_id, expected_embedding=self.cache.identity,
            tool_inventory=self.assets.public_tool_inventory,
            tool_inventory_sha256=self.manifest.tool_inventory_sha256)

    def build_round(self, index):
        if index not in (0, 1, 2) or index in self._built or index != len(self._built):
            raise ValueError('fresh consecutive round required; recovery must be explicit')
        self.authorization(self.manifest)
        manifest, assets = self.manifest, self.assets
        snapshot = self._load(f'g{index:03d}')
        scope = LiveEpisodeScope(run_id=manifest.run_id, round_index=index, shard_id=f'train-shard-{index}',
            generation_id=f'g{index:03d}', profile=manifest.profile, dataset_manifest_sha256=manifest.dataset_manifest_sha256,
            runtime_config_sha256=manifest.manifest_sha256, prompt_manifest_sha256=manifest.online_prompt_manifest_sha256,
            token_limit_config_sha256=assets.online_limits_sha256, fixture_manifest_sha256=manifest.fixture.manifest_sha256,
            environment_sha256=manifest.environment_sha256, scenario_positions={sid: pos for pos, sid in enumerate(self.shards[index])})
        online = self.provider_factory('train_round')
        offline = self.provider_factory('offline_update')
        publication = self.provider_factory('generation_publication')
        for provider in (online, offline, publication):
            if provider.ledger is not self.ledger or provider.manifest_identity != manifest.manifest_sha256:
                raise ValueError('provider ledger identity mismatch')
        context_factory = LiveTurnContextFactory(generation=snapshot, ledger=self.ledger,
            qwen_config=online.qwen_config, gateways=online.gateways, prompts=assets.online_prompts,
            token_limits=assets.online_limits, token_limit_config_sha256=assets.online_limits_sha256,
            manifest_identity=manifest.manifest_sha256, metadata=assets.metadata,
            embedding_cache=self.cache, embedding_gateway=online.embedding_gateway,
            embedding_context_factory=online.embedding_context_factory,
            embedding_record_durable=online.embedding_record_durable, count_prompt_tokens=self.count_tokens,
            mode='formal', runtime_inputs_sha256=assets.online_runtime_inputs_sha256)
        captures = _RevocableCaptures(self.store, self.metrics)
        episodes = LiveEpisodeExecutor(scope=scope, gate=self.train_gate, generation=snapshot,
            ledger=self.ledger, trajectory_store=self.store, metadata=assets.metadata,
            context_factory=context_factory, user_factory=online.user_factory,
            authorization=lambda _: self.authorization(manifest), physical_attempt_provider=self.attempts,
            boot_id=self.boot_id, capture_sink=captures.accept,
            contextual_external_boundary_factory=self.external_factory, backend_manifest_sha256=self.backend_sha)
        projection_scope = CurrentRoundScope(manifest.run_id, index, manifest.dataset_manifest_sha256, manifest.manifest_sha256)
        archive = Path(manifest.run_root) / f'round-{index}-raw-archive'
        buffer = _RoundBuffer(capture_registry=captures, run_id=manifest.run_id, round_index=index,
            shard_id=f'train-shard-{index}', generation_id=f'g{index:03d}',
            dataset_manifest_sha256=manifest.dataset_manifest_sha256, config_manifest_sha256=manifest.manifest_sha256,
            memory_prompt_manifest_sha256=assets.memory_prompts.manifest_sha256,
            skill_prompt_manifest_sha256=assets.skill_prompt_manifest_sha256,
            skill_token_limit_config_sha256=assets.skill_limits_sha256, archive_directory=archive,
            trajectory_loader=LiveTrajectoryLoader(blob_store=self.ledger.store.blobs, scope=projection_scope),
            memory_projection=LiveMemoryProjectionAdapter(scope=projection_scope, capture_loader=captures.load),
            failure_evidence=LiveFailureEvidenceAdapter(scope=projection_scope, capture_loader=captures.load))
        offline_scope = AccountingScope(run_id=manifest.run_id, round_index=index,
            task_id=f'{manifest.run_id}-offline-{index}', scenario_family_id='offline', scenario_id='offline',
            system_variant='generation_0' if index == 0 else 'updated')
        requests = LiveOfflineRequests(ledger=self.ledger, gateway=offline.qwen_gateway, scope=offline_scope,
            manifest_identity=manifest.manifest_sha256, count_tokens=self.count_tokens)
        cache = self.cache
        class Resolver:
            def resolve_one(self, text):
                return cache.resolve((text,), gateway=offline.embedding_gateway,
                    context_factory=lambda ordinal, texts: offline.embedding_context_factory(offline_scope, ordinal, texts),
                    record_durable=offline.embedding_record_durable)[0]
        retriever = MemoryCandidateRetriever(generation_id=scope.generation_id, policy_records=snapshot.policy_memory,
            world_records=snapshot.world_memory, indexes=snapshot.indexes, embedding_resolver=Resolver())
        memory = LiveMemoryUpdater(requests=requests, snapshot=snapshot, prompts=assets.memory_prompts,
            limits=assets.memory_limits, limits_sha256=assets.memory_limits_sha256, retriever=retriever,
            policy_input_representation=assets.policy_input_representation, world_input_representation='v1')
        mini_bench = self.dev_factory(index=index, snapshot=snapshot, runtime=self)
        if not callable(getattr(mini_bench, 'evaluate', None)):
            raise TypeError('native Dev mini-bench executor required before train starts')
        skills = LiveSkillUpdater(requests=requests, current_skills=snapshot.skills,
            public_tool_inventory=assets.public_tool_inventory, public_tool_schemas=assets.public_tool_schemas,
            prompts=assets.skill_prompts, prompt_hashes=assets.skill_prompt_hashes,
            prompt_manifest_sha256=assets.skill_prompt_manifest_sha256, token_limits_sha256=assets.skill_limits_sha256,
            max_tokens={role: limit.max_tokens for role, limit in assets.skill_token_limits.items()},
            token_limit_configs=assets.skill_token_limits, verify_token_limit_evidence=self.verify_skill_limit,
            mini_bench_executor=mini_bench)
        staging = Path(manifest.run_root) / f'round-{index}-generation-staging'
        staging.mkdir(mode=0o700)
        coordinator = RoundGenerationCoordinator(run_id=manifest.run_id, round_index=index,
            dataset_manifest_sha256=manifest.dataset_manifest_sha256, config_manifest_sha256=manifest.manifest_sha256,
            parent_manifest_sha256=canonical_sha256(snapshot.manifest.model_dump(mode='json')),
            generation_root=self.generation_root, staging_parent=staging, tool_inventory=assets.public_tool_inventory,
            tool_inventory_sha256=manifest.tool_inventory_sha256, cache=self.cache,
            gateway=publication.embedding_gateway,
            context_factory=lambda ordinal, texts: publication.embedding_context_factory(offline_scope, ordinal, texts),
            record_durable=publication.embedding_record_durable, ledger=UnitLedger(self.ledger.store), checkpoints=self.ledger)
        self._built.add(index)
        return RoundRunner(run_id=manifest.run_id, boot_id=self.boot_id, episodes=episodes,
            buffer=buffer, memory=memory, skills=skills, generations=coordinator,
            metrics=self.metrics, finalizer=_Finalizer(self.ledger))

    def run(self):
        return self.runner.run(self.shards)


__all__ = ['LiveTrainingAssets', 'LiveTrainingRuntime']
