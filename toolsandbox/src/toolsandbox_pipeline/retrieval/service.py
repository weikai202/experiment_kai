"""Pinned-generation retrieval; Revision receives bundles, never this service."""
from toolsandbox_pipeline.online.controller_inputs import RetrievedSkillControllerView
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.generation import GenerationSnapshot
from toolsandbox_pipeline.schemas.memory import Digest, FrozenRecord, GenerationId, PolicyMemory, WorldMemory
from toolsandbox_pipeline.skills.views import RetrievedSkillPolicyView, skill_views
from .contracts import ResolvedEmbedding, RetrievalError, RetrievalHit
from .index import rank, validate_indexes
from .queries import policy_query, world_query


class PolicySkillRetrievalBundle(FrozenRecord):
    generation_id: GenerationId
    state_id: str
    query: ResolvedEmbedding
    policy_hits: tuple[RetrievalHit, ...]
    policy_memory: tuple[PolicyMemory, ...]
    skill_hits: tuple[RetrievalHit, ...]
    skills: tuple[RetrievedSkillPolicyView, ...]


class WorldRetrievalBundle(FrozenRecord):
    generation_id: GenerationId
    state_id: str
    query: ResolvedEmbedding
    hits: tuple[RetrievalHit, ...]
    world_memory: tuple[WorldMemory, ...]


def _hits(entries, query, generation):
    return tuple(RetrievalHit(
        record_id=entry.record_id, record_version=entry.record_version, generation_id=generation,
        rank=i + 1, score=score, document_sha256=entry.document_sha256,
        query_sha256=query.key.input_sha256, cache_hit=query.cache_hit, source_attempt_id=query.source_attempt_id,
    ) for i, (entry, score) in enumerate(rank(entries, query.vector)))


class RetrievalService:
    def __init__(self, snapshot: GenerationSnapshot, *, cache, gateway, context_factory, record_durable):
        if type(snapshot) is not GenerationSnapshot:
            raise TypeError("pinned GenerationSnapshot required")
        validate_indexes(snapshot)
        if cache.identity != snapshot.manifest.embedding or cache.dimension != snapshot.manifest.vector_dimension or cache.batches != snapshot.index_manifest.batching:
            raise RetrievalError("cache/index configuration mismatch")
        self.snapshot = snapshot
        self.cache = cache
        self.gateway = gateway
        self.context_factory = context_factory
        self.record_durable = record_durable

    def _query(self, text):
        return self.cache.resolve([text], gateway=self.gateway, context_factory=self.context_factory, record_durable=self.record_durable)[0]

    def retrieve_policy_skills(self, state, *, canonical_to_agent):
        snapshot = self.snapshot
        generation = snapshot.manifest.generation_id
        views = {}
        for record in snapshot.skills:
            if record.status == "active":
                value = skill_views(record, generation, state, canonical_to_agent)
                if value is not None:
                    views[(record.skill_id, record.version)] = value
        query = self._query(policy_query(state))
        policy_hits = _hits((e for e in snapshot.indexes if e.record_kind == "policy"), query, generation)
        skill_hits = _hits((e for e in snapshot.indexes if e.record_kind == "skill" and (e.record_id, e.record_version) in views), query, generation)
        memories = {r.memory_id: r for r in snapshot.policy_memory}
        return PolicySkillRetrievalBundle(
            generation_id=generation, state_id=state.state_id, query=query,
            policy_hits=policy_hits, policy_memory=tuple(memories[h.record_id] for h in policy_hits),
            skill_hits=skill_hits, skills=tuple(views[(h.record_id, h.record_version)][0].model_copy(update={"rank": h.rank}) for h in skill_hits),
        )

    def retrieve_world(self, state, action, *, controller_decision: ControllerDecision):
        if type(controller_decision) is not ControllerDecision or not (controller_decision.blocking_codes or controller_decision.critic_trigger_codes):
            raise RetrievalError("World retrieval requires a Controller Critic trigger")
        query = self._query(world_query(state, action))
        generation = self.snapshot.manifest.generation_id
        hits = _hits((e for e in self.snapshot.indexes if e.record_kind == "world"), query, generation)
        records = {r.memory_id: r for r in self.snapshot.world_memory}
        return WorldRetrievalBundle(generation_id=generation, state_id=state.state_id, query=query,
                                    hits=hits, world_memory=tuple(records[h.record_id] for h in hits))
