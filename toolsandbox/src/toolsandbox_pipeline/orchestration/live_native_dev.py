"""Native, non-training Dev branches over exact authorized dataset leases."""
from pathlib import Path

from toolsandbox_pipeline.orchestration.generation_builder import build_generation
from toolsandbox_pipeline.orchestration.live_dev_minibench import LiveDevMiniBenchExecutor, LedgerDevBranchStore, StoredDevBranch
from toolsandbox_pipeline.orchestration.live_episode import LiveEpisodeExecutor, LiveEpisodeScope
from toolsandbox_pipeline.orchestration.live_turn_context import LiveTurnContextFactory
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.reproducibility.dataset_access import ScenarioLease
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.offline_skill import DevBranchResult
from toolsandbox_pipeline.schemas.skill import SkillRecord
from toolsandbox_pipeline.schemas.trajectory import EpisodeIdentity, EpisodeExecutionStatus


class _AuthorizedDevGate:
    def __init__(self, gate):
        self.gate, self.records = gate, {}

    def load(self, **kwargs):
        if kwargs.get('split') != 'dev' or kwargs.get('purpose') != 'skill_ab_validation':
            raise PermissionError('only selected Skill Dev branches allowed')
        leases = self.gate.load(**kwargs)
        for lease in leases:
            previous = self.records.get(lease.record.scenario_id)
            if previous is not None and previous != lease.record:
                raise ValueError('authorized Dev record changed')
            self.records[lease.record.scenario_id] = lease.record
        return leases


def branch_skills(snapshot, request):
    """Apply the selected semantic version while retaining host-owned statistics."""
    active = {item.skill_id: item for item in snapshot.skills if item.status == 'active'}
    for item in request.earlier_accepted_skills:
        if item.skill_id not in active or item.status != 'active':
            raise ValueError('earlier accepted Skill must replace an active Skill')
        active[item.skill_id] = item
    target = active.get(request.evaluated_skill.skill_id)
    if target is None:
        raise ValueError('evaluated Skill must already exist')
    value = target.model_dump()
    value.update(request.evaluated_skill.model_dump())
    value['version'] = request.evaluated_skill_version
    active[target.skill_id] = SkillRecord.model_validate(value)
    # Superseded historical records remain available to the generation validator.
    return tuple(item for item in snapshot.skills if item.status != 'active') + tuple(active.values())


class NativeDevBranchExecutor:
    def __init__(self, *, runtime, index, snapshot, gate, dev_manifest_sha256):
        self.runtime, self.index, self.snapshot, self.gate = runtime, index, snapshot, gate
        self.dev_sha, self.snapshots = dev_manifest_sha256, {}

    def run_branch(self, *, request, scenario):
        runtime, manifest, assets = self.runtime, self.runtime.manifest, self.runtime.assets
        if request.run_id != manifest.run_id or request.round_index != self.index or request.dev_manifest_sha256 != self.dev_sha:
            raise PermissionError('native Dev request scope mismatch')
        runtime.authorization(manifest)
        record = self.gate.records.get(request.scenario_id)
        if record is None:
            raise PermissionError('native Dev requires an authorized selected lease')
        providers = runtime.provider_factory('dev_minibench')
        if providers.ledger is not runtime.ledger or providers.manifest_identity != manifest.manifest_sha256:
            raise ValueError('Dev provider ledger mismatch')
        skills = branch_skills(self.snapshot, request)
        key = canonical_sha256([request.unit_id, request.branch, request.shared_configuration_sha256,
                                [skill.model_dump(mode='json') for skill in skills]])[7:]
        if key not in self.snapshots:
            staging = Path(manifest.run_root) / 'dev-branch-snapshots' / key
            staging.mkdir(parents=True, mode=0o700)
            accounting = AccountingScope(run_id=manifest.run_id, round_index=self.index,
                task_id=request.episode_id, scenario_family_id=record.scenario_family_id,
                scenario_id=record.scenario_id, system_variant='generation_0' if self.index == 0 else 'updated')
            self.snapshots[key] = build_generation(staging, generation_id=f'g{self.index:03d}',
                parent_generation_id=None if self.index == 0 else f'g{self.index-1:03d}',
                policy_memory=self.snapshot.policy_memory, world_memory=self.snapshot.world_memory, skills=skills,
                tool_inventory=assets.public_tool_inventory, tool_inventory_sha256=manifest.tool_inventory_sha256,
                cache=runtime.cache, gateway=providers.embedding_gateway,
                context_factory=lambda ordinal, texts: providers.embedding_context_factory(accounting, ordinal, texts),
                record_durable=providers.embedding_record_durable)
        generation = self.snapshots[key]
        scope = LiveEpisodeScope(run_id=manifest.run_id, round_index=self.index, shard_id=f'train-shard-{self.index}',
            generation_id=f'g{self.index:03d}', profile=manifest.profile,
            dataset_manifest_sha256=manifest.dataset_manifest_sha256, runtime_config_sha256=manifest.manifest_sha256,
            prompt_manifest_sha256=manifest.online_prompt_manifest_sha256,
            token_limit_config_sha256=assets.online_limits_sha256, fixture_manifest_sha256=manifest.fixture.manifest_sha256,
            environment_sha256=manifest.environment_sha256, scenario_positions={record.scenario_id: 0})
        contexts = LiveTurnContextFactory(generation=generation, ledger=runtime.ledger,
            qwen_config=providers.qwen_config, gateways=providers.gateways, prompts=assets.online_prompts,
            token_limits=assets.online_limits, token_limit_config_sha256=assets.online_limits_sha256,
            manifest_identity=manifest.manifest_sha256, metadata=assets.metadata,
            embedding_cache=runtime.cache, embedding_gateway=providers.embedding_gateway,
            embedding_context_factory=providers.embedding_context_factory,
            embedding_record_durable=providers.embedding_record_durable, count_prompt_tokens=runtime.count_tokens,
            mode='formal', runtime_inputs_sha256=assets.online_runtime_inputs_sha256)
        executor = LiveEpisodeExecutor(scope=scope, gate=self.gate, generation=generation,
            ledger=runtime.ledger, trajectory_store=runtime.store, metadata=assets.metadata,
            context_factory=contexts, user_factory=providers.user_factory,
            authorization=lambda _: runtime.authorization(manifest), physical_attempt_provider=runtime.attempts,
            boot_id=runtime.boot_id, capture_sink=lambda capture: runtime.metrics.record_episode(capture.result),
            contextual_external_boundary_factory=runtime.external_factory, backend_manifest_sha256=runtime.backend_sha)
        fields = executor._identity(record, 0).model_dump(mode='json')
        fields.update(phase='dev_minibench', episode_id=request.episode_id, dataset_manifest_sha256=self.dev_sha)
        identity = EpisodeIdentity.model_validate(fields)
        result = executor.run_dev_lease(lease=ScenarioLease(record, scenario), identity=identity,
            authorization=lambda **_: runtime.authorization(manifest), purpose='skill_ab_validation',
            dev_manifest_sha256=self.dev_sha)
        complete = result.status is EpisodeExecutionStatus.COMPLETED_EVALUATED
        details, applications = {}, ()
        if complete:
            evaluator = runtime.store.load_evaluator(result.evaluator_record_reference)
            trajectory = runtime.store.load_trajectory(result.trusted_trajectory_reference)
            if trajectory.eligible_for_train_offline_consumption:
                raise ValueError('Dev trajectory incorrectly eligible for training')
            details = dict(fully_successful=evaluator.fully_successful, similarity=evaluator.similarity,
                minefield_hit=evaluator.minefield_similarity != 0.0, evaluator_record_sha256=trajectory.evaluator_record_sha256)
            applications = tuple(app for turn in trajectory.online_turns for app in turn.application_ids)
        return StoredDevBranch(input_sha256=request.input_sha256, ordered_qwen_application_ids=applications,
            result=DevBranchResult(scenario_id=request.scenario_id, branch=request.branch,
                episode_id=request.episode_id, evaluated_skill_id=request.evaluated_skill.skill_id,
                evaluated_skill_version=request.evaluated_skill_version,
                shared_configuration_sha256=request.shared_configuration_sha256, complete=complete, **details))


class NativeDevMiniBenchFactory:
    def __init__(self, *, dev_gate, metadata_loader, dev_manifest_sha256, shared_configuration_sha256):
        self.gate, self.metadata = dev_gate, metadata_loader
        self.dev_sha, self.configuration_sha = dev_manifest_sha256, shared_configuration_sha256

    def __call__(self, *, index, snapshot, runtime):
        gate = _AuthorizedDevGate(self.gate)
        branches = NativeDevBranchExecutor(runtime=runtime, index=index, snapshot=snapshot,
            gate=gate, dev_manifest_sha256=self.dev_sha)
        return LiveDevMiniBenchExecutor(run_id=runtime.manifest.run_id, round_index=index,
            dev_manifest_sha256=self.dev_sha, shared_configuration_sha256=self.configuration_sha,
            public_tool_inventory=runtime.assets.public_tool_inventory, metadata_loader=self.metadata,
            dev_gate=gate, branch_executor=branches, branch_store=LedgerDevBranchStore(runtime.ledger))


class NativeDevSelectorMetadataLoader:
    """A selector-only capability; native base scenarios never escape this method.

    base_family_loader must reconstruct the unaugmented native family. Only its
    public tool allow-list is accessed, and no scenario object is retained.
    """
    def __init__(self, *, run_id, dev_manifest_sha256, manifest_loader,
                 base_family_loader, authorize, audit):
        if any(not callable(item) for item in (manifest_loader, base_family_loader, authorize, audit)):
            raise TypeError('explicit authorized native selector metadata sources required')
        self.run_id, self.dev_sha = run_id, dev_manifest_sha256
        self.manifest_loader, self.base_loader = manifest_loader, base_family_loader
        self.authorize, self.audit = authorize, audit

    def load(self, *, run_id, phase, split, purpose, manifest_sha256):
        from hashlib import sha256
        from toolsandbox_pipeline.schemas.dataset import SplitManifest
        from toolsandbox_pipeline.offline.dev_selector import DevFamilySelectorView
        from toolsandbox_pipeline.orchestration.live_dev_minibench import DevSelectorMetadata
        request = dict(run_id=run_id, phase=phase, split=split, purpose=purpose, manifest_sha256=manifest_sha256)
        if (run_id != self.run_id or phase != 'dev_minibench' or split != 'dev'
                or purpose != 'skill_ab_validation' or manifest_sha256 != self.dev_sha):
            raise PermissionError('Dev metadata is restricted to the current Skill selector')
        self.authorize(**request)
        raw = self.manifest_loader()
        if type(raw) is not bytes or 'sha256:' + sha256(raw).hexdigest() != self.dev_sha:
            raise ValueError('exact Dev manifest bytes required')
        manifest = SplitManifest.model_validate_json(raw)
        if manifest.split != 'dev':
            raise PermissionError('only Dev manifest metadata allowed')
        families = []
        for family in manifest.families:
            base = self.base_loader(family.family_id)
            allow_list = base.starting_context.tool_allow_list
            if not isinstance(allow_list, (list, tuple)) or not allow_list:
                raise ValueError('explicit native base-family necessary tool allow-list required')
            tools = tuple(allow_list)
            if any(type(tool) is not str or not tool for tool in tools) or len(set(tools)) != len(tools):
                raise ValueError('unique canonical necessary tools required')
            ids = tuple(record.scenario_id for record in manifest.scenarios if record.scenario_family_id == family.family_id)
            families.append(DevFamilySelectorView(family.family_id, 'dev', tools, ids))
            del base
        result = DevSelectorMetadata(self.dev_sha, tuple(families))
        self.audit({**request, 'metadata_kind': 'native_base_family_necessary_tools',
                    'family_count': len(families), 'selector_metadata_sha256': canonical_sha256([
                        dict(family_id=f.family_id, necessary_canonical_tools=list(f.necessary_canonical_tools),
                             expanded_scenario_ids=list(f.expanded_scenario_ids)) for f in families])})
        return result
