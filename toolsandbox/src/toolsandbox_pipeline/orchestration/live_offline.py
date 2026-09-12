"""Explicit ledger-backed offline services; construction never dispatches requests.

These adapters do not attest calibration or select dev data. The caller supplies
manifest-bound prompts, limits, token counting, retrieval and DevMiniBenchExecutor.
Policy packed-v2 / World v1 is explicitly selected and bound to a round checkpoint.
Non-bootstrap Skill limits require an externally supplied evidence verifier.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json

from toolsandbox_pipeline.checkpointing import LogicalLLMRequestIdentity
from toolsandbox_pipeline.offline.memory_orchestrator import AppliedMemoryDecision, MemoryUpdateOrchestrator
from toolsandbox_pipeline.offline.memory_roles import MemoryRoleRunner
from toolsandbox_pipeline.offline.skill_orchestrator import AppliedFailureDecision, AppliedSkillCandidate, SkillUpdateOrchestrator
from toolsandbox_pipeline.offline.skill_projection import failure_update_projection, skill_rewrite_projection
from toolsandbox_pipeline.offline.skill_roles import OfflineSkillRoleRunner
from toolsandbox_pipeline.providers.contracts import ProviderRole, ProviderRequestError, ROLE_BOOTSTRAP_MAX_TOKENS
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.checkpoint import QwenEffectKind, NonSubstantiveOutcome
from toolsandbox_pipeline.schemas.offline_memory import MemoryUpdateUnitResult, PolicyMemoryCandidateDecision, WorldMemoryCandidateDecision, MemoryReviewOutput


@dataclass(frozen=True)
class AppliedOfflineOutput:
    output: object = field(repr=False)
    logical_request_id: str
    source_attempt_id: str
    application_id: str
    checkpoint_id: str


class LiveOfflineRequests:
    """One dispatcher and durable application path for a single round/scope."""

    def __init__(self, *, ledger, gateway, scope, manifest_identity, count_tokens):
        if scope.run_id != ledger.store.identity.run_id or scope.round_index is None:
            raise ValueError('offline scope/run/round mismatch')
        if manifest_identity != ledger.store.identity.config_manifest_sha256:
            raise ValueError('offline manifest mismatch')
        self.ledger, self.gateway, self.scope = ledger, gateway, scope
        self.manifest_identity, self.count_tokens = manifest_identity, count_tokens

    def validate_buffer(self, identity):
        expected = self.ledger.store.identity
        if (identity.run_id != self.scope.run_id or identity.round_index != self.scope.round_index
                or identity.config_manifest_sha256 != self.manifest_identity
                or identity.dataset_manifest_sha256 != expected.dataset_manifest_sha256
                or identity.current_generation_id != f'g{self.scope.round_index:03d}'):
            raise ValueError('offline buffer identity mismatch')

    def persist(self, response):
        self.ledger.complete_response(
            response, validated_output=response.value.model_dump(mode='json', exclude_unset=True),
            output_schema_name=type(response.value).__name__, output_schema_version=1)

    def execute(self, *, role, unit_id, fingerprint, messages, output_model, max_tokens, invoke):
        tokens = self.count_tokens(messages)
        if type(tokens) is not int or tokens < 0:
            raise ValueError('exact prompt token count required')
        if self.gateway.config.context_limit is None or tokens + max_tokens > self.gateway.config.context_limit:
            raise ValueError('offline context limit exceeded before dispatch')
        request = self.ledger.prepare_request(LogicalLLMRequestIdentity(
            run_id=self.scope.run_id, role=role, phase='offline_update', unit_reference=unit_id,
            input_fingerprint=fingerprint, model=self.gateway.config.model,
            decoding_configuration_sha256=canonical_sha256({'qwen': self.gateway.config.model_dump(mode='json'), 'max_tokens': max_tokens}),
            output_schema_sha256=canonical_sha256(output_model.model_json_schema())))
        rid = request.logical_request_id
        self.ledger.commit_checkpoint('offline-input-' + rid, 'offline_request_identity',
            {'messages': messages, 'output_schema': output_model.model_json_schema(),
             'max_tokens': max_tokens, 'unit_id': unit_id, 'role': role.value})
        self.ledger.bind_accounting_scope(rid, self.scope)
        plan = self.ledger.plan_recovery(rid)
        if plan.action.value in ('apply_stored_llm_response', 'restore_applied_llm_checkpoint'):
            saved = self.ledger.load_completed_output(rid)
            output = output_model.model_validate_json(canonical_json_bytes(saved.output))
            source_id = saved.source_attempt_id
        elif plan.action.value == 'terminal_failure':
            raise RuntimeError('offline request terminal failure')
        else:
            context = self.ledger.reconcile_and_allocate_attempt(rid, manifest_identity=self.manifest_identity)
            self.ledger.mark_in_flight(context)
            try:
                response = invoke(context)
                if hasattr(response, 'raw_response_body'):
                    self.persist(response)
                    output = response.value
                else:
                    output = response.output  # MemoryRoleRunner persisted through its seam.
                source_id = response.attempt.context.attempt_id
            except ProviderRequestError as error:
                self.ledger.record_failure(error)
                raise
        payload = output.model_dump(mode='json', exclude_unset=True)
        checkpoint_id = 'offline-apply-' + rid
        app = self.ledger.commit_checkpoint_and_apply(
            checkpoint_id=checkpoint_id, event_kind='offline_response_applied',
            checkpoint_payload={'output': payload}, logical_request_id=rid,
            source_attempt_id=source_id, application_artifact_id='offline-output-' + rid,
            application_artifact_sha256=canonical_sha256(payload))
        return AppliedOfflineOutput(output, rid, source_id, app.application_id, checkpoint_id)

    def memory_executor(self):
        requests = self
        class Seam:
            def load_completed_response(self, context):
                return None  # Stored responses are resolved before entering this runner.
            def persist_completed_response(self, response):
                requests.persist(response)
        runner = MemoryRoleRunner(self.gateway, manifest_identity=self.manifest_identity, durability_seam=Seam())
        class Executor:
            def execute_and_apply(self, prepared):
                model = {'PolicyMemoryCandidateDecision': PolicyMemoryCandidateDecision,
                         'WorldMemoryCandidateDecision': WorldMemoryCandidateDecision,
                         'MemoryReviewOutput': MemoryReviewOutput}[prepared.output_model_name]
                result = requests.execute(
                    role=ProviderRole(prepared.role), unit_id=prepared.unit_reference,
                    fingerprint=prepared.canonical_input_fingerprint,
                    messages=[message.model_dump() for message in prepared.messages],
                    output_model=model, max_tokens=prepared.max_tokens,
                    invoke=lambda context: runner.run(prepared, context))
                return AppliedMemoryDecision(**result.__dict__)
        return Executor()


class LiveOfflineDurability:
    def __init__(self, ledger):
        self.ledger = ledger

    def checkpoint(self, event_kind, payload=None, *, unit_id=None):
        payload = {} if payload is None else payload
        body = {'unit_id': unit_id, 'payload': payload}
        cid = 'offline-checkpoint-' + canonical_sha256([event_kind, body])[7:]
        return self.ledger.commit_checkpoint(cid, event_kind, body).checkpoint_id

    def load_unit(self, unit_reference):
        event = self.ledger.get_checkpoint('offline-memory-unit-' + unit_reference)
        return None if event is None else MemoryUpdateUnitResult.model_validate_json(canonical_json_bytes(event.payload))

    def commit_unit(self, result):
        self.ledger.commit_checkpoint('offline-memory-unit-' + result.unit_id, 'offline_memory_unit_completed', result.model_dump(mode='json'))
        return result

    def high_water_marks(self):
        return self.ledger.store.high_water_marks()

    def effect(self, kind, payload, applications):
        digest = canonical_sha256(payload)
        cid = 'offline-effect-' + digest[7:]
        effect = self.ledger.commit_checkpoint_and_effect(
            checkpoint_id=cid, event_kind=kind.value, checkpoint_payload=payload,
            effect_kind=kind, effect_artifact_id=cid, effect_artifact_sha256=digest,
            ordered_application_ids=applications)
        return effect.effect_id, cid

    def commit_mutation(self, mutation, ordered_application_ids):
        kind = QwenEffectKind.POLICY_MEMORY_MUTATION if mutation.role == 'policy' else QwenEffectKind.WORLD_MEMORY_MUTATION
        return self.effect(kind, mutation.model_dump(mode='json'), ordered_application_ids)

    def record_non_substantive(self, outcome='rejected_candidate', application_ids=(), *, unit_id=None):
        self.ledger.record_non_substantive_outcome(outcome=NonSubstantiveOutcome(outcome), ordered_application_ids=application_ids)

    def commit_failure_mutation(self, *, unit_id, application_id, mutation_artifact_sha256):
        return self.effect(QwenEffectKind.FAILURE_MODE_MUTATION,
                           dict(unit_id=unit_id, mutation_artifact_sha256=mutation_artifact_sha256), (application_id,))[0]

    def commit_accepted_skill(self, *, unit_id, ordered_application_ids, candidate, mini_bench):
        if not mini_bench.accepted:
            raise ValueError('accepted Skill effect requires accepted native Mini-Bench')
        return self.effect(QwenEffectKind.ACCEPTED_SKILL_MUTATION,
                           dict(unit_id=unit_id, candidate=candidate.model_dump(mode='json'),
                                mini_bench=mini_bench.model_dump(mode='json')), ordered_application_ids)[0]


class LiveMemoryUpdater:
    def __init__(self, *, requests, snapshot, prompts, limits, limits_sha256, retriever,
                 policy_input_representation='v1', world_input_representation='v1'):
        if world_input_representation != 'v1':
            raise ValueError('World packed-v2 is not implemented')
        if snapshot.manifest.generation_id != f'g{requests.scope.round_index:03d}':
            raise ValueError('memory source generation mismatch')
        self.requests, self.prompts = requests, prompts
        self.orchestrator = MemoryUpdateOrchestrator(
            prompts=prompts, limits=limits, limits_sha256=limits_sha256,
            structured_output_wire_mode=requests.gateway.config.structured_output_wire_mode,
            role_executor=requests.memory_executor(), retriever=retriever,
            durability=LiveOfflineDurability(requests.ledger),
            source_generation_sha256=canonical_sha256(snapshot.manifest.model_dump(mode='json')),
            current_policy_memory=snapshot.policy_memory, current_world_memory=snapshot.world_memory,
            input_representation=policy_input_representation, world_input_representation=world_input_representation)
        self.input_representation_manifest = dict(policy=policy_input_representation, world=world_input_representation)

    def run(self, sealed_buffer):
        self.requests.validate_buffer(sealed_buffer.identity)
        if sealed_buffer.identity.prompt_manifest_sha256 != self.prompts.manifest_sha256:
            raise ValueError('memory prompt manifest mismatch')
        cid = 'offline-memory-format-' + canonical_sha256([sealed_buffer.identity.run_id, sealed_buffer.identity.round_index])[7:]
        self.requests.ledger.commit_checkpoint(cid, 'offline_memory_input_format',
            dict(config_manifest_sha256=sealed_buffer.identity.config_manifest_sha256,
                 input_representation=self.input_representation_manifest))
        return self.orchestrator.run(sealed_buffer)


class LiveSkillUpdater:
    def __init__(self, *, requests, current_skills, public_tool_inventory, public_tool_schemas,
                 prompts, prompt_hashes, prompt_manifest_sha256, token_limits_sha256,
                 max_tokens, mini_bench_executor, token_limit_configs=None, verify_token_limit_evidence=None):
        prompts, max_tokens, prompt_hashes = dict(prompts), dict(max_tokens), dict(prompt_hashes)
        selections = {} if token_limit_configs is None else dict(token_limit_configs)
        if set(selections) - {'failure_mode_update', 'skill_candidate'}:
            raise ValueError('unknown Skill token-limit role')
        for role in ('failure_mode_update', 'skill_candidate'):
            if 'sha256:' + sha256(prompts[role].encode()).hexdigest() != prompt_hashes[role]:
                raise ValueError('Skill prompt hash mismatch')
            selected = selections.get(role)
            if selected is None:
                if max_tokens[role] != ROLE_BOOTSTRAP_MAX_TOKENS[ProviderRole(role)]:
                    raise ValueError('non-bootstrap Skill ceiling requires explicit verified selection')
            else:
                from toolsandbox_pipeline.schemas.runtime import RoleTokenLimitConfig
                if type(selected) is not RoleTokenLimitConfig:
                    raise TypeError('validated Skill token-limit config required')
                selected = RoleTokenLimitConfig.model_validate_json(selected.model_dump_json())
                if selected.role != role or selected.max_tokens != max_tokens[role]:
                    raise ValueError('Skill token-limit role/ceiling mismatch')
                if selected.stage != 'bootstrap':
                    if (not callable(verify_token_limit_evidence)
                            or verify_token_limit_evidence(selected, token_limits_sha256) is not True):
                        raise ValueError('verified train token-limit evidence required')
                selections[role] = selected
        self.requests, self.current_skills = requests, tuple(skill for skill in current_skills if skill.status == "active")
        self.prompt_manifest_sha256, self.token_limits_sha256 = prompt_manifest_sha256, token_limits_sha256
        def invoke(role, unit_id, projection):
            runner = OfflineSkillRoleRunner(requests.gateway, role=ProviderRole(role), max_tokens=max_tokens[role], token_limit_config=selections.get(role))
            prepared = runner.prepare(unit_id=unit_id, messages=[{'role': 'system', 'content': prompts[role]},
                                                                {'role': 'user', 'content': canonical_json_bytes(projection).decode()}])
            return requests.execute(role=prepared.role, unit_id=prepared.unit_id, fingerprint=prepared.input_fingerprint,
                                    messages=prepared.messages, output_model=prepared.output_model,
                                    max_tokens=prepared.max_tokens, invoke=lambda context: runner.run(prepared, context))
        class FailureExecutor:
            def decide(self, *, evidence, current_buffer, unit_id):
                result = invoke('failure_mode_update', unit_id, failure_update_projection(evidence, current_buffer))
                return AppliedFailureDecision(result.output.as_decision(), result.application_id)
        class CandidateExecutor:
            def rewrite(self, *, skill, evidence, unit_id):
                generalized = tuple(failure_update_projection(item, ())['failure_evidence'] for item in evidence)
                projection = skill_rewrite_projection(skill=skill, statistics=skill.online_statistics,
                    failure_modes=skill.failure_mode_buffer, public_tool_schemas=public_tool_schemas,
                    generalized_train_trajectories=generalized)
                result = invoke('skill_candidate', unit_id, projection)
                return AppliedSkillCandidate(result.output.candidate, result.application_id)
        self.orchestrator = SkillUpdateOrchestrator(
            failure_executor=FailureExecutor(), candidate_executor=CandidateExecutor(),
            mini_bench_executor=mini_bench_executor, effect_sink=LiveOfflineDurability(requests.ledger),
            public_tool_inventory=public_tool_inventory)

    def run(self, sealed_buffer):
        self.requests.validate_buffer(sealed_buffer.identity)
        if (sealed_buffer.identity.prompt_manifest_sha256 != self.prompt_manifest_sha256
                or sealed_buffer.identity.token_limit_config_sha256 != self.token_limits_sha256):
            raise ValueError('Skill prompt/token-limit manifest mismatch')
        return self.orchestrator.run(buffer=sealed_buffer, current_skills=self.current_skills)


__all__ = ["LiveOfflineRequests", "LiveOfflineDurability", "LiveMemoryUpdater", "LiveSkillUpdater"]
