"""Train-only frozen offline-request replay, with durable measured evidence.

The coordinator captures and verifies natural offline requests in a separate
calibration corpus. This module neither constructs trajectories nor mutates a
training update buffer; all calibration calls use round_index=None.
"""
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic

from toolsandbox_pipeline.checkpointing import LogicalLLMRequestIdentity
from toolsandbox_pipeline.offline.memory_roles import PreparedMemoryRequest, OfflineMemoryTokenLimits
from toolsandbox_pipeline.offline.skill_roles import PreparedOfflineSkillRequest, FailureModeDecisionOutput
from toolsandbox_pipeline.online.token_limits import recommended_limit
from toolsandbox_pipeline.providers.contracts import ProviderRole, ProviderRequestError, ROLE_BOOTSTRAP_MAX_TOKENS
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.offline_memory import PolicyMemoryCandidateDecision, WorldMemoryCandidateDecision, MemoryReviewOutput
from toolsandbox_pipeline.schemas.offline_skill import SkillContentCandidate
from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig

PROTOCOL = 'offline-token-calibration-v1'
ROLES = ('memory_candidate', 'memory_review', 'failure_mode_update', 'skill_candidate')
MODELS = {m.__name__: m for m in (PolicyMemoryCandidateDecision, WorldMemoryCandidateDecision,
                                MemoryReviewOutput, FailureModeDecisionOutput, SkillContentCandidate)}


@dataclass(frozen=True)
class CapturedOfflineCalibrationRequest:
    """Source IDs are verified by the coordinator before any replay is permitted."""
    scenario_id: str
    scenario_family_id: str
    source_logical_request_id: str
    source_attempt_id: str
    source_manifest_sha256: str
    prepared: object
    split: str = 'train'

    def payload(self):
        p = self.prepared
        if self.split != 'train' or any(type(v) is not str or not v for v in
                (self.scenario_id, self.scenario_family_id, self.source_logical_request_id,
                 self.source_attempt_id, self.source_manifest_sha256)):
            raise ValueError('explicit train source identity required')
        if type(p) is PreparedMemoryRequest:
            role, messages, model = p.role, [m.model_dump() for m in p.messages], p.output_model_name
        elif type(p) is PreparedOfflineSkillRequest:
            role, messages, model = p.role.value, p.messages, p.output_model.__name__
        else:
            raise TypeError('captured offline prepared request required')
        expected = {'memory_candidate': ('PolicyMemoryCandidateDecision','WorldMemoryCandidateDecision'),
                    'memory_review': ('MemoryReviewOutput',),
                    'failure_mode_update': ('FailureModeDecisionOutput',),
                    'skill_candidate': ('SkillContentCandidate',)}
        if role not in expected or model not in expected[role]:
            raise ValueError('offline role/schema mismatch')
        if not messages or any(set(m) != {'role','content'} or m['role'] not in ('system','user')
                               or type(m['content']) is not str or not m['content'] for m in messages):
            raise ValueError('captured frozen messages required')
        return dict(scenario_id=self.scenario_id, scenario_family_id=self.scenario_family_id,
            source_logical_request_id=self.source_logical_request_id, source_attempt_id=self.source_attempt_id,
            source_manifest_sha256=self.source_manifest_sha256, split=self.split, role=role,
            messages=messages, output_model_name=model,
            output_schema_sha256=canonical_sha256(MODELS[model].model_json_schema()))


class DurableOfflineCalibrationReplay:
    def __init__(self, *, ledger, gateway, manifest_identity, count_tokens):
        if manifest_identity != ledger.store.identity.config_manifest_sha256:
            raise ValueError('calibration ledger manifest mismatch')
        self.ledger, self.gateway = ledger, gateway
        self.manifest_identity, self.count_tokens = manifest_identity, count_tokens

    def invoke(self, payload, *, corpus_sha256, ordinal, ceiling):
        messages = payload['messages']
        tokens = self.count_tokens(messages)
        if type(tokens) is not int or tokens < 0 or self.gateway.config.context_limit is None or tokens + ceiling > self.gateway.config.context_limit:
            raise ValueError('offline calibration context ceiling exceeded')
        selection = RoleTokenLimitConfig(version=PROTOCOL, role=payload['role'], stage='calibration',
            max_tokens=ceiling, evidence_manifest_identity=corpus_sha256)
        decoding = dict(qwen=self.gateway.config.model_dump(mode='json'), selection=selection.model_dump(mode='json'))
        input_hash = canonical_sha256(dict(payload=payload, decoding=decoding, corpus_sha256=corpus_sha256))
        unit = 'offline-calibration-' + str(ordinal)
        record = self.ledger.prepare_request(LogicalLLMRequestIdentity(
            run_id=self.ledger.store.identity.run_id, role=ProviderRole(payload['role']),
            phase='offline_token_calibration', unit_reference=unit, input_fingerprint=input_hash,
            model=self.gateway.config.model, decoding_configuration_sha256=canonical_sha256(decoding),
            output_schema_sha256=payload['output_schema_sha256']))
        rid = record.logical_request_id
        self.ledger.bind_accounting_scope(rid, AccountingScope(run_id=self.ledger.store.identity.run_id,
            round_index=None, task_id=unit, scenario_id=payload['scenario_id'],
            scenario_family_id=payload['scenario_family_id'], system_variant='generation_0'))
        self.ledger.commit_checkpoint('offline-calibration-input-' + rid, 'offline_calibration_request',
            dict(payload=payload, selection=selection.model_dump(mode='json'), ordinal=ordinal, corpus_sha256=corpus_sha256))
        checkpoint_id = 'offline-calibration-result-' + rid
        prior = self.ledger.get_checkpoint(checkpoint_id)
        if prior is not None:
            return dict(prior.payload)
        model = MODELS[payload['output_model_name']]
        plan = self.ledger.plan_recovery(rid)
        if plan.action.value in ('apply_stored_llm_response', 'restore_applied_llm_checkpoint'):
            saved = self.ledger.load_completed_response_material(rid)
            model.model_validate_json(canonical_json_bytes(saved.validated_output))
            attempt, valid = saved.attempt, True
        elif plan.action.value == 'terminal_failure':
            raise RuntimeError('offline calibration failed request requires explicit recovery')
        else:
            context = self.ledger.reconcile_and_allocate_attempt(rid, manifest_identity=self.manifest_identity)
            self.ledger.mark_in_flight(context)
            try:
                response = self.gateway.generate(context, messages, model, max_tokens=ceiling, token_limit_config=selection)
            except ProviderRequestError as error:
                self.ledger.record_failure(error)
                if error.attempt.finish_reason != 'length':
                    raise
                attempt, valid = error.attempt, False
            else:
                self.ledger.complete_response(response,
                    validated_output=response.value.model_dump(mode='json', exclude_unset=True),
                    output_schema_name=model.__name__, output_schema_version=1)
                attempt, valid = response.attempt, True
        usage = attempt.metrics.usage
        if not usage.usage_complete or usage.output_tokens is None or usage.total_tokens is None or attempt.finish_reason not in ('stop','length'):
            raise ValueError('complete actual calibration usage required')
        result = dict(logical_request_id=rid, ceiling=ceiling, strict_valid=valid,
                      attempt=attempt.model_dump(mode='json'))
        self.ledger.commit_checkpoint(checkpoint_id, 'offline_calibration_result', result)
        return result


def calibrate_offline_requests(*, captured, replay, verify_source, runtime_inputs, hard_ceiling):
    """Seal all four role corpora before dispatch; return evidence, not promotion.

    verify_source(item) must verify exact captured request bytes, train provenance
    and durable source IDs against source artifacts. A bool inferred from caller
    labels is not sufficient. No source or calibration output enters a round.
    """
    if type(captured) is not tuple or not captured or not callable(verify_source):
        raise ValueError('sealed corpus and source verifier required')
    if type(runtime_inputs) is not dict or not runtime_inputs:
        raise ValueError('manifest-bound runtime inputs required')
    if type(hard_ceiling) is not int or hard_ceiling < max(ROLE_BOOTSTRAP_MAX_TOKENS[ProviderRole(r)] for r in ROLES):
        raise ValueError('invalid offline calibration hard ceiling')
    if replay.gateway.config.output_limit is not None and hard_ceiling > replay.gateway.config.output_limit:
        raise ValueError('offline calibration exceeds server output ceiling')
    payloads, source_ids, inputs_seen = [], set(), set()
    for item in captured:
        if type(item) is not CapturedOfflineCalibrationRequest:
            raise TypeError('captured offline request required')
        payload = item.payload()
        source = (item.source_logical_request_id, item.source_attempt_id)
        if source in source_ids or verify_source(item) is not True:
            raise ValueError('duplicate or unverified source request')
        source_ids.add(source)
        input_key = canonical_sha256([payload['role'], payload['messages'], payload['output_schema_sha256']])
        if input_key not in inputs_seen:
            payloads.append(payload)
            inputs_seen.add(input_key)
    for role in ROLES:
        inputs = [p for p in payloads if p['role'] == role]
        fingerprints = {canonical_sha256([p['messages'],p['output_schema_sha256']]) for p in inputs}
        if not inputs or len(fingerprints) != len(inputs):
            raise ValueError('each offline role needs nonempty unique natural inputs')
    corpus = dict(protocol=PROTOCOL, runtime_inputs=runtime_inputs,
        qwen_config=replay.gateway.config.model_dump(mode='json'), hard_ceiling=hard_ceiling, captured=payloads)
    corpus_sha = canonical_sha256(corpus)
    replay.ledger.commit_checkpoint('offline-calibration-corpus-' + corpus_sha[7:], 'offline_calibration_corpus', corpus)
    evidence_id = 'offline-calibration-evidence-' + corpus_sha[7:]
    prior = replay.ledger.get_checkpoint(evidence_id)
    if prior is not None:
        return dict(prior.payload)
    started, results, summaries = monotonic(), [], {}
    def call(payload, ceiling):
        if ceiling > hard_ceiling:
            raise ValueError('offline calibration hard ceiling exhausted')
        result = replay.invoke(payload, corpus_sha256=corpus_sha, ordinal=len(results), ceiling=ceiling)
        results.append(result)
        return result
    for role in ROLES:
        inputs = [p for p in payloads if p['role'] == role]
        bootstrap = ROLE_BOOTSTRAP_MAX_TOKENS[ProviderRole(role)]
        ceiling, observed = bootstrap, 0
        for payload in inputs:
            while True:
                result = call(payload, ceiling)
                if result['strict_valid']:
                    observed = max(observed, result['attempt']['metrics']['usage']['output_tokens'])
                    break
                ceiling = ((ceiling * 2 + 63) // 64) * 64
        recommended = recommended_limit(observed)
        while True:
            final = [call(payload, recommended) for payload in inputs]
            if all(r['strict_valid'] for r in final):
                break
            recommended += 64
        summaries[role] = dict(bootstrap=bootstrap, observed_max_completion_tokens=observed,
            recommended_max_tokens=recommended, corpus_size=len(inputs), final_zero_length=True,
            final_strict_valid_count=len(inputs), final_logical_request_ids=[r['logical_request_id'] for r in final])
    evidence = dict(protocol=PROTOCOL, status='measured', corpus_sha256=corpus_sha,
        runtime_inputs=runtime_inputs, qwen_config=corpus['qwen_config'], roles=summaries, results=results,
        total_tokens=sum(r['attempt']['metrics']['usage']['total_tokens'] for r in results),
        usage_complete=True, total_cost=0, cost_complete=True,
        accounting_scope='offline_calibration_replay_only',
        total_running_time_seconds=monotonic()-started)
    replay.ledger.commit_checkpoint(evidence_id, 'offline_calibration_measured', evidence)
    return evidence


@dataclass(frozen=True)
class OfflineCalibrationScope:
    """Projection-only scope; never sets train-offline eligibility on a pilot."""
    run_id: str
    dataset_manifest_sha256: str
    runtime_config_sha256: str
    train_scenario_ids: tuple[str, ...]

    def validate(self, trajectory):
        from toolsandbox_pipeline.schemas.trajectory import TrustedTrajectory
        if type(trajectory) is not TrustedTrajectory:
            raise TypeError('trusted calibration trajectory required')
        TrustedTrajectory.model_validate_json(trajectory.model_dump_json())
        i = trajectory.identity
        if (i.run_id != self.run_id or i.scenario_id not in self.train_scenario_ids
                or i.dataset_manifest_sha256 != self.dataset_manifest_sha256
                or i.runtime_config_sha256 != self.runtime_config_sha256
                or i.generation_id != 'g000' or i.round_index is not None or i.shard_id is not None
                or i.phase != 'online_token_calibration_pilot'
                or trajectory.eligible_for_train_offline_consumption):
            raise ValueError('only isolated completed train pilot trajectories permitted')


def collect_offline_calibration_requests(*, trajectories, capture_loader, scope, replay,
        prompts, limits, limits_sha256, retriever, current_skills, skill_prompts,
        public_tool_schemas, runtime_inputs, hard_ceiling):
    """Execute natural offline branches over isolated train pilot projections.

    Reuses production projection, candidate/reviewer preparation, Skill statistics,
    failure-mode updates and rewrite trigger. Local failure buffers/statistics are
    calibration artifacts only. No Mini-Bench/dev selection or generation publish
    occurs, and no pilot is converted into a formal update-buffer reference.
    """
    from toolsandbox_pipeline.orchestration.live_projections import LiveMemoryProjectionAdapter, LiveFailureEvidenceAdapter
    from toolsandbox_pipeline.offline.memory_roles import prepare_candidate_request, prepare_review_request
    from toolsandbox_pipeline.offline.memory_projection import reject_sensitive_candidate
    from toolsandbox_pipeline.schemas.offline_memory import MemoryCandidateNone
    from toolsandbox_pipeline.offline.skill_roles import OfflineSkillRoleRunner
    from toolsandbox_pipeline.offline.skill_projection import failure_update_projection, skill_rewrite_projection
    from toolsandbox_pipeline.offline.skill_statistics import ManifestSkillUse, stage_skill_statistics, rewrite_triggered, mark_rewrite_attempt
    from toolsandbox_pipeline.offline.failure_modes import apply_failure_mode_decision
    from toolsandbox_pipeline.schemas.skill import SkillRecord
    if type(scope) is not OfflineCalibrationScope or type(trajectories) is not tuple or not trajectories:
        raise ValueError('explicit isolated calibration scope and ordered trajectories required')
    if limits.status != 'provisional' or limits.memory_candidate != 512 or limits.memory_review != 256:
        raise ValueError('natural collection starts at actual bootstrap limits')
    positions = tuple(t.identity.manifest_position for t in trajectories)
    if positions != tuple(sorted(set(positions))):
        raise ValueError('unique manifest-ordered calibration trajectories required')
    memory = LiveMemoryProjectionAdapter(scope=scope, capture_loader=capture_loader)
    failures = LiveFailureEvidenceAdapter(scope=scope, capture_loader=capture_loader)
    # Validate every source/projection before dispatch. Production checks bind
    # trusted evaluator, decisions, visible inputs and executed Skill attribution.
    projected = [(t, memory.project(t), failures.project(t)) for t in trajectories]
    corpus_seed = dict(protocol=PROTOCOL + '-natural-collection', runtime_inputs=runtime_inputs,
        trajectories=[t.trajectory_id for t in trajectories],
        memory_prompt_manifest_sha256=prompts.manifest_sha256,
        skill_prompts=skill_prompts, current_skills=[s.model_dump(mode='json') for s in current_skills],
        qwen_config=replay.gateway.config.model_dump(mode='json'),
        input_representation={'policy':'packed-v2','world':'v1'})
    seed_hash = canonical_sha256(corpus_seed)
    replay.ledger.commit_checkpoint('offline-natural-corpus-' + seed_hash[7:], 'offline_natural_corpus', corpus_seed)
    captured, ordinal, collection_attempts = [], 0, []

    def execute(prepared, trajectory):
        nonlocal ordinal
        if type(prepared) is PreparedMemoryRequest:
            role, messages, model = prepared.role, [m.model_dump() for m in prepared.messages], prepared.output_model_name
        else:
            role, messages, model = prepared.role.value, prepared.messages, prepared.output_model.__name__
        payload = dict(role=role, messages=messages, output_model_name=model,
            output_schema_sha256=canonical_sha256(MODELS[model].model_json_schema()),
            scenario_id=trajectory.identity.scenario_id, scenario_family_id=trajectory.identity.family_id,
            source_trajectory_id=trajectory.trajectory_id, source_manifest_sha256=scope.runtime_config_sha256)
        ceiling = ROLE_BOOTSTRAP_MAX_TOKENS[ProviderRole(role)]
        while True:
            if ceiling > hard_ceiling:
                raise ValueError('natural offline collection hard ceiling exhausted')
            result = replay.invoke(payload, corpus_sha256=seed_hash, ordinal=ordinal, ceiling=ceiling)
            ordinal += 1
            collection_attempts.append(result)
            if result['strict_valid']:
                break
            ceiling = ((2*ceiling+63)//64)*64
        rid = result['logical_request_id']
        saved = replay.ledger.load_completed_output(rid)
        item = CapturedOfflineCalibrationRequest(trajectory.identity.scenario_id, trajectory.identity.family_id,
            rid, saved.source_attempt_id, replay.manifest_identity, prepared)
        # Freeze source request and binding now; later replay source verification
        # checks this receipt against completed response and original input.
        replay.ledger.commit_checkpoint('offline-natural-request-' + rid, 'offline_natural_request', item.payload())
        captured.append(item)
        return MODELS[model].model_validate_json(canonical_json_bytes(saved.output))

    for t, (policy, world), _ in projected:
        for projection, representation in ((policy,'packed-v2'),(world,'v1')):
            if projection is None:
                continue
            candidate_request = prepare_candidate_request(projection, unit_reference='offline-natural-' + str(len(captured)),
                prompts=prompts, limits=limits, limits_sha256=limits_sha256,
                structured_output_wire_mode=replay.gateway.config.structured_output_wire_mode,
                input_representation=representation)
            candidate = execute(candidate_request,t).decision()
            reject_sensitive_candidate(candidate,projection)
            if isinstance(candidate,MemoryCandidateNone):
                continue
            matches = retriever.retrieve(candidate)
            review = prepare_review_request(candidate,matches.records,
                unit_reference='offline-natural-' + str(len(captured)), prompts=prompts,limits=limits,
                limits_sha256=limits_sha256,structured_output_wire_mode=replay.gateway.config.structured_output_wire_mode)
            execute(review,t)
    uses = tuple(ManifestSkillUse(t.identity.manifest_position,t.identity.episode_id,a)
        for t in trajectories for a in sorted(t.skill_attributions,key=lambda a:a.skill_id.encode()))
    active = {s.skill_id:s for s in current_skills if s.status == 'active'}
    seq = max((m.last_observed_seq for s in active.values() for m in s.failure_mode_buffer),default=0)
    for sid in sorted({u.attribution.skill_id for u in uses},key=str.encode):
        if sid not in active:
            raise ValueError('pilot used unknown active Skill')
        skill = active[sid]
        statistics = stage_skill_statistics(skill.online_statistics,uses,skill_id=sid)
        buffer = skill.failure_mode_buffer
        relevant = [(t,e) for t,_,entries in projected for e in entries if e.skill_id == sid]
        for t,evidence in relevant:
            projection = failure_update_projection(evidence,buffer)
            runner = OfflineSkillRoleRunner(replay.gateway,role=ProviderRole.FAILURE_MODE_UPDATE,max_tokens=512)
            request = runner.prepare(unit_id='offline-natural-' + str(len(captured)),messages=[
                {'role':'system','content':skill_prompts['failure_mode_update']},
                {'role':'user','content':canonical_json_bytes(projection).decode()}])
            decision = execute(request,t).as_decision()
            next_seq = None
            if decision.decision != 'SKIP':
                seq += 1
                next_seq = seq
            buffer = apply_failure_mode_decision(skill_id=sid,buffer=buffer,decision=decision,next_observed_seq=next_seq).after
        if not rewrite_triggered(statistics):
            continue
        statistics = mark_rewrite_attempt(statistics)
        staged = SkillRecord(**skill.model_dump(mode='python',exclude={'online_statistics','failure_mode_buffer'}),
            online_statistics=statistics,failure_mode_buffer=buffer)
        projection = skill_rewrite_projection(skill=staged,statistics=statistics,failure_modes=buffer,
            public_tool_schemas=public_tool_schemas,
            generalized_train_trajectories=tuple(failure_update_projection(e,())['failure_evidence'] for _,e in relevant))
        runner = OfflineSkillRoleRunner(replay.gateway,role=ProviderRole.SKILL_CANDIDATE,max_tokens=2048)
        request = runner.prepare(unit_id='offline-natural-' + str(len(captured)),messages=[
            {'role':'system','content':skill_prompts['skill_candidate']},
            {'role':'user','content':canonical_json_bytes(projection).decode()}])
        source = next(t for t in reversed(trajectories) if any(a.skill_id == sid for a in t.skill_attributions))
        execute(request,source)
    replay.ledger.commit_checkpoint('offline-natural-summary-' + seed_hash[7:], 'offline_natural_collection_summary',
        dict(corpus_seed_sha256=seed_hash, requests=[item.payload() for item in captured],
             results=collection_attempts,
             total_tokens=sum(r['attempt']['metrics']['usage']['total_tokens'] for r in collection_attempts),
             usage_complete=True, total_cost=0, cost_complete=True))
    return tuple(captured)


def verify_collected_source(ledger, item):
    payload = item.payload()
    checkpoint = ledger.get_checkpoint('offline-natural-request-' + item.source_logical_request_id)
    if checkpoint is None or dict(checkpoint.payload) != payload:
        return False
    saved = ledger.load_completed_response_material(item.source_logical_request_id)
    input_checkpoint = ledger.get_checkpoint('offline-calibration-input-' + item.source_logical_request_id)
    source = input_checkpoint.payload['payload']
    return (saved.attempt.context.attempt_id == item.source_attempt_id
        and saved.attempt.context.manifest_identity == item.source_manifest_sha256
        and source['messages'] == payload['messages'] and source['role'] == payload['role']
        and source['output_schema_sha256'] == payload['output_schema_sha256']
        and source['scenario_id'] == payload['scenario_id']
        and source['scenario_family_id'] == payload['scenario_family_id'])


def offline_limit_recommendations(*, evidence, replay, runtime_inputs):
    """Materialize compatible configs only from this ledger's measured evidence.

    The coordinator atomically writes and pins these bytes in the new full-run
    manifest. This function cannot mutate a running generation or runtime config.
    """
    if evidence.get('protocol') != PROTOCOL or evidence.get('status') != 'measured':
        raise ValueError('measured offline evidence required')
    saved = replay.ledger.get_checkpoint('offline-calibration-evidence-' + evidence['corpus_sha256'][7:])
    if (saved is None or dict(saved.payload) != evidence or runtime_inputs != evidence['runtime_inputs']
            or replay.gateway.config.model_dump(mode='json') != evidence['qwen_config']):
        raise ValueError('offline evidence/runtime identity mismatch')
    evidence_hash = canonical_sha256(evidence)
    caps = {role:evidence['roles'][role]['recommended_max_tokens'] for role in ROLES}
    memory = OfflineMemoryTokenLimits(schema_version=1,status='calibrated',
        memory_candidate=caps['memory_candidate'],memory_review=caps['memory_review'])
    skill = dict(schema_version=1,status='calibrated',failure_mode_update=caps['failure_mode_update'],skill_candidate=caps['skill_candidate'])
    blobs = {'offline_memory_token_limits.calibrated.json':canonical_json_bytes(memory.model_dump(mode='json')),
             'offline_skill_token_limits.calibrated.json':canonical_json_bytes(skill),
             'offline_calibration_evidence.json':canonical_json_bytes(evidence)}
    hashes = {name:'sha256:'+sha256(raw).hexdigest() for name,raw in blobs.items()}
    selections = {role:RoleTokenLimitConfig(version=PROTOCOL,role=role,stage='calibrated',
        max_tokens=caps[role],evidence_manifest_identity=evidence_hash) for role in ROLES}
    def verify(selection, config_sha256):
        name = 'offline_memory_token_limits.calibrated.json' if selection.role.startswith('memory_') else 'offline_skill_token_limits.calibrated.json'
        return selection == selections.get(selection.role) and config_sha256 == hashes[name]
    return dict(blobs=blobs,hashes=hashes,memory_limits=memory,selections=selections,verify_token_limit_evidence=verify)


def load_pilot_calibration_inputs(*, ledger, episode_ids, scope):
    """Read the exact capture checkpoints emitted by run_live_calibration.

    This reads only caller-selected train pilot IDs, never enumerates dev/test.
    The source checkpoint/blob store must remain available during collection.
    """
    from toolsandbox_pipeline.schemas.trajectory import EpisodeResult, EpisodeExecutionStatus, TrustedTrajectory, TrustedEvaluatorRecord
    from toolsandbox_pipeline.schemas.online_turn import OnlineTurnDecision
    from toolsandbox_pipeline.schemas.action import ActionEnvelope
    from toolsandbox_pipeline.orchestration.live_projections import CapturedEpisodeInputs, CapturedOnlineTurn, LiveMemoryProjectionAdapter
    import json
    if type(episode_ids) is not tuple or not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError('explicit unique pilot episode IDs required')
    trajectories, captures = [], {}
    for episode_id in episode_ids:
        checkpoint = ledger.get_checkpoint('calibration-episode-capture-' + canonical_sha256(episode_id)[7:])
        if checkpoint is None:
            raise ValueError('durable pilot capture missing')
        payload = checkpoint.payload
        result = EpisodeResult.model_validate_json(canonical_json_bytes(payload['result']))
        if (result.status is not EpisodeExecutionStatus.COMPLETED_EVALUATED
                or result.identity.episode_id != episode_id or result.identity.model_dump(mode='json') != payload['identity']):
            raise ValueError('completed matching pilot result required')
        tref, eref = result.trusted_trajectory_reference, result.evaluator_record_reference
        if tref.schema_name != 'TrustedTrajectory' or eref.schema_name != 'TrustedEvaluatorRecord':
            raise ValueError('trusted pilot blob schemas required')
        trajectory = TrustedTrajectory.model_validate_json(ledger.store.blobs.read(tref))
        evaluation = TrustedEvaluatorRecord.model_validate_json(ledger.store.blobs.read(eref))
        if trajectory.identity != result.identity:
            raise ValueError('pilot trajectory/result mismatch')
        scope.validate(trajectory)
        envelopes = payload['initial_envelopes']
        if tuple(pair[0] for pair in envelopes) != tuple(t.agent_turn_index for t in trajectory.online_turns):
            raise ValueError('ordered complete pilot envelopes required')
        decisions = tuple(OnlineTurnDecision.model_validate_json(canonical_json_bytes(value)) for value in payload['decisions'])
        actions = tuple(ActionEnvelope.model_validate_json(canonical_json_bytes(value)) for value in payload['proposed_actions'])
        if len(decisions) != len(actions) or len(actions) != len(envelopes):
            raise ValueError('complete pilot decisions/actions required')
        captured = CapturedEpisodeInputs(evaluation,tuple(CapturedOnlineTurn(json.loads(envelope),action,decision)
            for (_,envelope),action,decision in zip(envelopes,actions,decisions)))
        LiveMemoryProjectionAdapter(scope=scope,capture_loader=lambda _:captured).checked(trajectory)
        captures[trajectory.trajectory_id] = captured
        trajectories.append(trajectory)
    ordered = tuple(sorted(trajectories,key=lambda t:t.identity.manifest_position))
    if len({t.identity.manifest_position for t in ordered}) != len(ordered):
        raise ValueError('duplicate pilot manifest position')
    return ordered, lambda trajectory: captures[trajectory.trajectory_id]
