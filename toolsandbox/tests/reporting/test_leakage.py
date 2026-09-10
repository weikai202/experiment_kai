from pathlib import Path
import re


def test_reporting_source_contains_no_credentials_or_network_clients():
    root = Path(__file__).parents[2] / "src" / "toolsandbox_pipeline" / "reporting"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "sk-proj-" not in text
    assert re.search(r"(?:from|import)\s+requests\b", text) is None
    assert re.search(r"(?:from|import)\s+httpx\b", text) is None
    assert "OPENAI_API_KEY" not in text
