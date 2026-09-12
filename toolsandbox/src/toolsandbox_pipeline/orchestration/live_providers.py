"""Concrete lazy gateways with explicit accounting and durable response reuse."""
import math
from types import MethodType
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletion

from toolsandbox_pipeline.checkpointing import LogicalLLMRequestIdentity
from toolsandbox_pipeline.providers.contracts import GatewayResponse, ProviderRole, ProviderRequestError
from toolsandbox_pipeline.providers.embedding import EmbeddingGateway
from toolsandbox_pipeline.providers.qwen import QwenGateway
from toolsandbox_pipeline.providers.user_simulator import InstrumentedGPT4oMiniUser
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope
from toolsandbox_pipeline.schemas.runtime import QwenConfig, EmbeddingConfig, UserSimulatorConfig


class _RecoverableEmbeddingGateway(EmbeddingGateway):
    def __init__(self, services, **kwargs):
        super().__init__(services.embedding_config, **kwargs)
        self.services = services

    def embed(self, context, inputs):
        if (context.role is not ProviderRole.EMBEDDING or context.phase != self.services.phase
                or context.manifest_identity != self.services.manifest_identity):
            raise ValueError('embedding context role/phase/manifest mismatch')
        texts = [inputs] if type(inputs) is str else list(inputs)
        if context.input_fingerprint != self.services._embedding_fingerprint(texts):
            raise ValueError('embedding inputs differ from prepared request')
        plan = self.services.ledger.plan_recovery(context.logical_request_id)
        if plan.action.value in ('apply_stored_llm_response', 'restore_applied_llm_checkpoint'):
            material = self.services.ledger.load_completed_response_material(context.logical_request_id)
            vectors = material.validated_output
            if (type(vectors) is not list or len(vectors) != len(texts)
                    or any(type(v) is not list or not v or any(type(x) not in (float,int) or not math.isfinite(x) for x in v) for v in vectors)
                    or len({len(v) for v in vectors}) != 1
                    or (self.dimension is not None and len(vectors[0]) != self.dimension)):
                raise ValueError('invalid stored embedding vectors')
            self._dimension = len(vectors[0])
            return GatewayResponse(tuple(tuple(float(x) for x in vector) for vector in vectors), material.attempt, material.raw_response_body)
        try:
            return super().embed(context, inputs)
        except ProviderRequestError as error:
            self.services.ledger.record_failure(error)
            raise


class LiveProviderServices:
    """No network/client construction or credential lookup until explicit dispatch.

    Use a separate instance for each embedding phase. Scope is supplied per call,
    so setup, train, calibration and dev_minibench usage never share a guessed phase.
    User creation uses the EpisodeIdentity's exact phase and accounting fields.
    """
    def __init__(self, *, ledger, qwen_config, embedding_config, user_config,
                 manifest_identity, phase, qwen_transport=None, embedding_transport=None,
                 user_transport=None):
        if type(qwen_config) is not QwenConfig or type(embedding_config) is not EmbeddingConfig or type(user_config) is not UserSimulatorConfig:
            raise TypeError('strict provider configurations required')
        if manifest_identity != ledger.store.identity.config_manifest_sha256:
            raise ValueError('provider service manifest mismatch')
        if type(phase) is not str or not phase or any(c.isspace() for c in phase):
            raise ValueError('explicit provider phase required')
        self.ledger, self.manifest_identity, self.phase = ledger, manifest_identity, phase
        self.qwen_config, self.embedding_config, self.user_config = qwen_config, embedding_config, user_config
        self.user_transport = user_transport
        self.gateways = {role: QwenGateway(qwen_config, transport=qwen_transport) for role in ('policy','critic','revision')}
        self.qwen_gateway = QwenGateway(qwen_config, transport=qwen_transport)
        self.embedding_gateway = _RecoverableEmbeddingGateway(self, transport=embedding_transport)

    def _prepare(self, *, role, payload, scope, phase, unit, config, schema):
        if scope.run_id != self.ledger.store.identity.run_id:
            raise ValueError('provider accounting run mismatch')
        request = self.ledger.prepare_request(LogicalLLMRequestIdentity(
            run_id=scope.run_id, role=role, phase=phase, unit_reference=unit,
            input_fingerprint=canonical_sha256(payload), model=config.model,
            decoding_configuration_sha256=canonical_sha256(config.model_dump(mode='json')),
            output_schema_sha256=canonical_sha256(schema)))
        self.ledger.bind_accounting_scope(request.logical_request_id,scope)
        plan = self.ledger.plan_recovery(request.logical_request_id)
        if plan.action.value in ('apply_stored_llm_response','restore_applied_llm_checkpoint'):
            return self.ledger.load_completed_response_material(request.logical_request_id).attempt.context
        if plan.action.value == 'terminal_failure':
            raise RuntimeError('terminal provider request cannot be replayed')
        context = self.ledger.reconcile_and_allocate_attempt(request.logical_request_id, manifest_identity=self.manifest_identity)
        self.ledger.mark_in_flight(context)
        return context

    def _embedding_fingerprint(self, texts):
        return canonical_sha256({'inputs':list(texts),'config':self.embedding_config.model_dump(mode='json')})

    def embedding_context_factory(self, accounting_scope, ordinal, texts):
        if type(ordinal) is not int or ordinal < 0 or not texts:
            raise ValueError('embedding batch ordinal and nonempty inputs required')
        payload = {'inputs':list(texts),'config':self.embedding_config.model_dump(mode='json')}
        unit = 'embedding-' + canonical_sha256([accounting_scope.model_dump(mode='json'), ordinal, payload])[7:]
        return self._prepare(role=ProviderRole.EMBEDDING, payload=payload, scope=accounting_scope,
            phase=self.phase, unit=unit, config=self.embedding_config, schema={'type':'EmbeddingVectors','version':1})

    def embedding_record_durable(self, response):
        context = response.attempt.context
        if context.role is not ProviderRole.EMBEDDING or context.manifest_identity != self.manifest_identity:
            raise ValueError('embedding durable response identity mismatch')
        if self.ledger.request_status(context.logical_request_id).value not in ('response_completed','applied'):
            self.ledger.complete_response(response, validated_output=[list(v) for v in response.value],
                output_schema_name='EmbeddingVectors',output_schema_version=1)
        else:
            stored = self.ledger.load_completed_response_material(context.logical_request_id)
            if (stored.attempt != response.attempt or stored.raw_response_body != response.raw_response_body
                    or stored.validated_output != [list(v) for v in response.value]):
                raise ValueError('completed embedding response conflict')
        return True

    def user_factory(self, episode_identity, trajectory_store, ledger=None):
        if ledger is not None and ledger is not self.ledger:
            raise ValueError('User factory ledger mismatch')
        if episode_identity.run_id != self.ledger.store.identity.run_id or episode_identity.runtime_config_sha256 != self.manifest_identity:
            raise ValueError('User episode/run configuration mismatch')
        services = self
        scope = AccountingScope(run_id=episode_identity.run_id, round_index=episode_identity.round_index,
            task_id=episode_identity.episode_id, scenario_family_id=episode_identity.family_id,
            scenario_id=episode_identity.scenario_id,system_variant=episode_identity.system_variant)
        pending = {}
        def context_provider():
            payload = pending['payload']
            return services._prepare(role=ProviderRole.USER_SIMULATOR,payload=payload,scope=scope,
                phase=episode_identity.phase,unit='user-' + canonical_sha256([episode_identity.episode_id,payload])[7:],
                config=services.user_config,schema={'type':'ChatCompletion','version':1})
        class Seam:
            def load_completed_response(self, context):
                plan = services.ledger.plan_recovery(context.logical_request_id)
                if plan.action.value not in ('apply_stored_llm_response','restore_applied_llm_checkpoint'):
                    return None
                material = services.ledger.load_completed_response_material(context.logical_request_id)
                return GatewayResponse(ChatCompletion.model_validate(material.validated_output), material.attempt, material.raw_response_body)
            def persist_completed_response(self, response):
                services.ledger.complete_response(response,validated_output=response.value.model_dump(mode='json'),
                    output_schema_name='ChatCompletion',output_schema_version=1)
        # Preserve the exact native delegate type required by the official-live
        # transaction boundary. Wrap only this instance's inference entry; the
        # provider and upstream respond/prompt logic remain unchanged.
        user = InstrumentedGPT4oMiniUser(context_provider=context_provider,
            config=self.user_config, transport=self.user_transport, durability_seam=Seam())
        native_inference = user.model_inference
        def durable_inference(delegate, openai_messages, openai_tools):
            pending['payload'] = {'messages':openai_messages,
                'tools':{'omitted':True} if openai_tools is NOT_GIVEN else openai_tools,
                'temperature':'omitted','top_p':'omitted','seed':'omitted'}
            try:
                return native_inference(openai_messages,openai_tools)
            except ProviderRequestError as error:
                services.ledger.record_failure(error)
                raise
        user.model_inference = MethodType(durable_inference, user)
        # Trajectory persistence remains the TransactionalUserRole's responsibility.
        return user


__all__ = ["LiveProviderServices"]
