from pathlib import Path

import pytest

from toolsandbox_pipeline.offline.memory_prompts import (
    MemoryPromptError,
    candidate_envelope,
    load_memory_prompts,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.offline_memory import PolicyTrajectoryProjection


ROOT = Path(__file__).parents[2]


def test_exact_manifest_prompt_bytes_and_untrusted_framing():
    prompts = load_memory_prompts(
        ROOT.resolve(), (ROOT / "prompts/offline/memory_manifest.json").resolve()
    )
    for text in (prompts.candidate, prompts.review):
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert "untrusted JSON data envelope" in text
        assert "QWEN_API_KEY" not in text and "OPENAI_API_KEY" not in text


def test_candidate_envelope_is_one_canonical_trajectory():
    projection = PolicyTrajectoryProjection(
        trajectory_id="sha256:" + "a" * 64,
        manifest_position=0,
        visible_states=(),
        retrieved_policy_memory=(),
        retrieved_skills=(),
        proposed_actions=(ActionEnvelope(action={"type": "assistant_message", "content": "draft"}),),
        final_actions=(ActionEnvelope(action={"type": "assistant_message", "content": "final"}),),
        controller_codes=(),
        visible_tool_outcomes=(),
        native_similarity=1.0,
        fully_successful=True,
        host_attribution="successful",
    )
    text = candidate_envelope(projection)
    assert text.count(projection.trajectory_id) == 1
    assert text.startswith('{"memory_role":"policy","trajectory":')


def test_relative_manifest_path_is_rejected():
    with pytest.raises(MemoryPromptError):
        load_memory_prompts(ROOT.resolve(), Path("prompts/offline/memory_manifest.json"))
