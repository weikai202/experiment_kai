from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_task017_owned_text_contains_no_secret_or_test_access_sentinels():
    paths = [
        ROOT / "configs/run/run_manifest.schema.json",
        ROOT / "configs/run/train_smoke_v1.json",
        ROOT / "scripts/build_seed_skill_library.py",
        *(ROOT / "src/toolsandbox_pipeline/orchestration").glob("*.py"),
    ]
    forbidden = ("Bearer sk-", "OPENAI_API_KEY=", "QWEN_API_KEY=", '"split":"test"')
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path


def test_orchestration_imports_do_not_open_external_dependencies(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network access")),
    )
    __import__("toolsandbox_pipeline.orchestration.cli")
    __import__("toolsandbox_pipeline.orchestration.training_runner")
