from pathlib import Path
import shutil
import pytest
from toolsandbox_pipeline.online.prompt_loader import load_prompts

ROOT = Path(__file__).resolve().parents[2]


def test_prompt_manifest():
    prompts = load_prompts(ROOT)
    assert [p.entry.role for p in prompts] == ["policy", "critic", "revision"]
    assert all(p.text.endswith("prompt. Use only fields permitted for this role.\n") for p in prompts)


@pytest.mark.parametrize(
    "damage",
    ["extra", "offline_extra", "offline_symlink", "missing", "symlink", "hash", "crlf", "bom"],
)
def test_prompt_corruption(tmp_path, damage):
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    path = tmp_path / "prompts/critic_v1.txt"
    if damage == "extra":
        (tmp_path / "prompts/extra").touch()
    elif damage == "offline_extra":
        (tmp_path / "prompts/offline/unreviewed.txt").touch()
    elif damage == "offline_symlink":
        target = tmp_path / "prompts/offline/memory_candidate_v1.txt"
        target.unlink()
        target.symlink_to(ROOT / "prompts/offline/memory_candidate_v1.txt")
    elif damage == "missing":
        path.unlink()
    elif damage == "symlink":
        path.unlink()
        path.symlink_to(ROOT / "prompts/critic_v1.txt")
    else:
        raw = path.read_bytes()
        path.write_bytes(raw + b"x" if damage == "hash" else raw.replace(b"\n", b"\r\n") if damage == "crlf" else b"\xef\xbb\xbf" + raw)
    with pytest.raises(ValueError):
        load_prompts(tmp_path)


def test_reviewed_offline_subtree_is_allowed_but_not_loaded_online(tmp_path):
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    (tmp_path / "prompts/offline/failure_mode_update_v1.txt").write_text("owned by Task016\n")
    prompts = load_prompts(tmp_path)
    assert [prompt.entry.role for prompt in prompts] == ["policy", "critic", "revision"]


def test_reviewed_vanilla_files_are_allowed_but_not_loaded_online():
    prompts = load_prompts(ROOT)
    assert [prompt.entry.role for prompt in prompts] == ["policy", "critic", "revision"]
    assert (ROOT / "prompts/vanilla_agent_v1.txt").is_file()
    assert (ROOT / "prompts/vanilla_manifest.json").is_file()
