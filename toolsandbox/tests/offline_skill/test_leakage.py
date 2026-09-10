from pathlib import Path


def test_task016_sources_contain_no_secret_or_live_access_patterns():
    root = Path(__file__).parents[2]
    files = [
        *sorted((root / "src" / "toolsandbox_pipeline" / "offline").glob("skill_*.py")),
        root / "src" / "toolsandbox_pipeline" / "offline" / "failure_modes.py",
        root / "src" / "toolsandbox_pipeline" / "offline" / "failure_lineage.py",
        root / "src" / "toolsandbox_pipeline" / "offline" / "dev_selector.py",
        root / "src" / "toolsandbox_pipeline" / "offline" / "dev_minibench.py",
    ]
    text = "\n".join(path.read_text() for path in files)
    assert "OPENAI_API_KEY" not in text
    assert "QWEN_API_KEY" not in text
    assert "requests.get" not in text
    assert "test_manifest" not in text
