from pathlib import Path

from toolsandbox_pipeline.offline.memory_orchestrator import (
    AppliedMemoryDecision,
    MemoryTrajectoryReference,
    MemoryUpdateOrchestrator,
    SealedMemoryTrajectoryBuffer,
    selected_policy_entries,
)
from toolsandbox_pipeline.offline.memory_prompts import load_memory_prompts
from toolsandbox_pipeline.offline.memory_retrieval import MemoryCandidateRetriever
from toolsandbox_pipeline.offline.memory_roles import load_token_limits
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.retrieval.contracts import EmbeddingCacheKey, ResolvedEmbedding
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryReviewOutput,
    MemoryUpdateIdentity,
    PolicyMemoryCandidateDecision,
    PolicyTrajectoryProjection,
)


ROOT = Path(__file__).parents[2]


def digest(value):
    return "sha256:" + value * 64


def projection(position=0, value="a"):
    return PolicyTrajectoryProjection(
        trajectory_id=value if value.startswith("sha256:") else digest(value),
        manifest_position=position,
        visible_states=(), retrieved_policy_memory=(), retrieved_skills=(),
        proposed_actions=(ActionEnvelope(action={"type": "assistant_message", "content": "draft"}),),
        final_actions=(ActionEnvelope(action={"type": "assistant_message", "content": "final"}),),
        controller_codes=(), visible_tool_outcomes=(), native_similarity=1.0,
        fully_successful=True, host_attribution="successful",
    )


def reference(position=0, value="a"):
    projected = projection(position, value)
    return MemoryTrajectoryReference(
        trajectory_id=projected.trajectory_id, trajectory_sha256=digest("f"),
        run_id="run", round_index=0, shard_id="shard-0", generation_id="g000",
        dataset_manifest_sha256=digest("1"), config_manifest_sha256=digest("2"),
        split="train", status="completed_evaluated",
        eligible_for_train_offline_consumption=True, manifest_position=position,
        policy_projection=projected,
    )


def buffer(entries):
    refs = [
        {"trajectory_id": entry.trajectory_id, "trajectory_sha256": entry.trajectory_sha256,
         "manifest_position": entry.manifest_position}
        for entry in entries
    ]
    sealed = canonical_sha256({
        "protocol": "sealed-memory-buffer-v1", "run_id": "run", "round_index": 0,
        "shard_id": "shard-0", "generation_id": "g000", "entries": refs,
    })
    identity = MemoryUpdateIdentity(
        run_id="run", round_index=0, shard_id="shard-0",
        current_generation_id="g000", next_generation_id="g001",
        dataset_manifest_sha256=digest("1"), config_manifest_sha256=digest("2"),
        prompt_manifest_sha256=digest("3"), sealed_input_buffer_sha256=sealed,
    )
    return SealedMemoryTrajectoryBuffer(identity=identity, entries=tuple(entries))


class Resolver:
    def resolve_one(self, text):
        return ResolvedEmbedding(
            key=EmbeddingCacheKey(input_sha256=canonical_sha256(__import__("json").loads(text))),
            vector=(1.0, 0.0), cache_hit=False, source_attempt_id="embedding-attempt",
        )


class Executor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def execute_and_apply(self, prepared):
        self.calls.append(prepared)
        output = self.outputs.pop(0)
        suffix = str(len(self.calls))
        return AppliedMemoryDecision(
            output=output, logical_request_id="logical-" + suffix,
            source_attempt_id="attempt-" + suffix, application_id="application-" + suffix,
            checkpoint_id="application-checkpoint-" + suffix,
        )


class Durability:
    def __init__(self):
        self.units = {}
        self.effects = []
        self.no_effects = []
        self.checkpoints = []

    def load_unit(self, unit_reference):
        return self.units.get(unit_reference)

    def checkpoint(self, event_kind, payload):
        self.checkpoints.append((event_kind, payload))
        return "checkpoint-" + str(len(self.checkpoints))

    def commit_mutation(self, mutation, ordered_application_ids):
        self.effects.append((mutation, ordered_application_ids))
        return "effect-1", "checkpoint-effect"

    def record_non_substantive(self, outcome, application_ids):
        self.no_effects.append((outcome, application_ids))

    def commit_unit(self, result):
        self.units[result.unit_id] = result
        return result

    def high_water_marks(self):
        return {"offline_unit_transactions": len(self.units)}


def subject(executor, durability, **options):
    prompts = load_memory_prompts(ROOT.resolve(), (ROOT / "prompts/offline/memory_manifest.json").resolve())
    limits, limit_sha = load_token_limits((ROOT / "configs/offline_memory_token_limits.provisional.json").resolve())
    retriever = MemoryCandidateRetriever(
        generation_id="g000", policy_records=(), world_records=(), indexes=(), embedding_resolver=Resolver()
    )
    return MemoryUpdateOrchestrator(
        prompts=prompts, limits=limits, limits_sha256=limit_sha,
        structured_output_wire_mode="guided_json", role_executor=executor,
        retriever=retriever, durability=durability, source_generation_sha256=digest("4"),
        current_policy_memory=(), current_world_memory=(), **options,
    )


def test_policy_selection_is_last_50_then_ascending():
    entries = tuple(
        reference(index, "sha256:" + f"{index:064x}") for index in range(55)
    )
    assert [item.manifest_position for item in selected_policy_entries(entries)] == list(
        range(5, 55)
    )


def test_none_is_applied_and_audited_but_creates_no_effect():
    executor = Executor((PolicyMemoryCandidateDecision(result="NONE"),))
    durability = Durability()
    result = subject(executor, durability).run(buffer((reference(),)))
    assert result.policy_counts == {"ADD": 0, "MERGE": 0, "SKIP": 0, "NONE": 1}
    assert not result.staged_policy_memory and not durability.effects
    assert durability.no_effects == [("none", ("application-1",))]
    assert result.units[0].candidate_application_id == "application-1"


def test_add_commits_candidate_and_reviewer_chain_as_one_substantive_effect():
    candidate = PolicyMemoryCandidateDecision(
        result="CANDIDATE", role="policy",
        candidate={"scope": "General prerequisites", "applicability": [],
                   "action_guidance": "Validate required arguments before execution", "avoid": []},
    )
    executor = Executor((candidate, MemoryReviewOutput(decision="ADD", reason="Reusable new rule")))
    durability = Durability()
    result = subject(executor, durability).run(buffer((reference(),)))
    assert result.policy_counts["ADD"] == 1 and len(result.staged_policy_memory) == 1
    assert durability.effects[0][1] == ("application-1", "application-2")
    assert result.units[0].substantive_effect_id == "effect-1"


def test_packed_world_buffer_fails_before_any_policy_dispatch():
    import pytest
    from toolsandbox_pipeline.offline.memory_orchestrator import MemoryOrchestrationError
    from toolsandbox_pipeline.schemas.offline_memory import WorldTrajectoryProjection
    entry = reference()
    world = WorldTrajectoryProjection(
        trajectory_id=entry.trajectory_id, manifest_position=0, visible_states=(),
        draft_action=entry.policy_projection.proposed_actions[0], controller_codes=(), visible_tool_outcomes=(),
        critic_output={'verdict': 'accept', 'error_codes': [], 'predicted_outcome': 'success', 'predicted_effect': 'Answer delivered', 'correction': ''},
        revision_occurred=False, attributable_failure=False, attribution_kind='verified_success')
    executor = Executor(())
    durability = Durability()
    runner = subject(executor, durability)
    runner.input_representation = 'packed-v2'
    with pytest.raises(MemoryOrchestrationError, match='Policy-only'):
        runner.run(buffer((entry.model_copy(update={'world_projection': world}),)))
    assert not executor.calls and not durability.checkpoints



def test_explicit_policy_v2_world_v1_preserves_order_and_role_inputs():
    from toolsandbox_pipeline.schemas.offline_memory import WorldTrajectoryProjection, WorldMemoryCandidateDecision
    entry = reference()
    world = WorldTrajectoryProjection(
        trajectory_id=entry.trajectory_id, manifest_position=0, visible_states=(),
        draft_action=entry.policy_projection.proposed_actions[0], controller_codes=(), visible_tool_outcomes=(),
        critic_output={'verdict':'accept','error_codes':[],'predicted_outcome':'success','predicted_effect':'A visible effect','correction':''},
        revision_occurred=False, attributable_failure=False, attribution_kind='verified_success')
    executor = Executor((PolicyMemoryCandidateDecision(result='NONE'), WorldMemoryCandidateDecision(result='NONE')))
    runner = subject(executor, Durability(), input_representation='packed-v2', world_input_representation='v1')
    result = runner.run(buffer((entry.model_copy(update={'world_projection':world}),)))
    assert [(p.memory_role,p.prompt_version) for p in executor.calls] == [('policy','v2'),('world','v1')]
    assert result.policy_counts['NONE'] == result.world_counts['NONE'] == 1
