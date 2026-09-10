from tests.offline_memory.test_orchestrator import (
    Durability,
    Executor,
    buffer,
    reference,
    subject,
)
from toolsandbox_pipeline.schemas.offline_memory import PolicyMemoryCandidateDecision
from toolsandbox_pipeline.schemas.offline_memory import MemoryReviewOutput


def test_committed_none_unit_is_reused_without_second_model_call():
    durability = Durability()
    first_executor = Executor((PolicyMemoryCandidateDecision(result="NONE"),))
    sealed = buffer((reference(),))
    first = subject(first_executor, durability).run(sealed)
    second_executor = Executor(())
    second = subject(second_executor, durability).run(sealed)
    assert second == first
    assert len(first_executor.calls) == 1 and second_executor.calls == []
    assert durability.effects == []


def test_committed_add_restores_staged_record_without_repeating_model_calls():
    durability = Durability()
    candidate = PolicyMemoryCandidateDecision(
        result="CANDIDATE",
        role="policy",
        candidate={
            "scope": "General prerequisites",
            "applicability": [],
            "action_guidance": "Validate required arguments before execution",
            "avoid": [],
        },
    )
    first_executor = Executor(
        (candidate, MemoryReviewOutput(decision="ADD", reason="Reusable new rule"))
    )
    sealed = buffer((reference(),))
    first = subject(first_executor, durability).run(sealed)

    second_executor = Executor(())
    second = subject(second_executor, durability).run(sealed)

    assert second == first
    assert len(first_executor.calls) == 2 and second_executor.calls == []
    assert second.staged_policy_memory == first.staged_policy_memory
    assert len(durability.effects) == 1
