from pathlib import Path

from toolsandbox_pipeline.offline.skill_prompts import load_skill_prompts


def test_committed_skill_prompts_match_manifest():
    root = Path(__file__).parents[2]
    prompts = load_skill_prompts(
        root / "prompts" / "offline",
        root / "prompts" / "offline" / "skill_manifest.json",
    )
    assert set(prompts) == {"failure_mode_update", "skill_candidate"}
    assert all(value.endswith("\n") for value in prompts.values())
