from pathlib import Path

from toolsandbox_pipeline.offline.memory_prompts import load_memory_prompts


ROOT = Path(__file__).parents[2]


def test_prompts_and_owned_source_contain_no_secret_or_hidden_payload_sentinels():
    prompts = load_memory_prompts(ROOT.resolve(), (ROOT / "prompts/offline/memory_manifest.json").resolve())
    combined = prompts.candidate + prompts.review
    for sentinel in ("sk-proj-", "Bearer ", "target_dataframe", "milestone_mapping", "minefield_mapping"):
        assert sentinel not in combined
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src/toolsandbox_pipeline/offline").glob("*.py")
    )
    assert "os.environ" not in source and "requests." not in source


def test_import_and_construction_paths_do_not_access_external_services():
    import toolsandbox_pipeline.offline as offline

    assert offline.MemoryUpdateOrchestrator.__name__ == "MemoryUpdateOrchestrator"
