"""Explicit byte-verified loading of the three versioned system prompts."""
from pathlib import Path
from typing import Literal
from pydantic import model_validator
from toolsandbox_pipeline.schemas.memory import Digest, FrozenRecord
from toolsandbox_pipeline.retrieval.index import file_hash

ROLES = ("policy", "critic", "revision")
PATHS = ("prompts/initial_policy_v1.txt", "prompts/critic_v1.txt", "prompts/revision_v1.txt")
OUTPUTS = ("ActionEnvelope", "CriticOutput", "ActionEnvelope")
OFFLINE_ALLOWED = frozenset(
    {
        "memory_candidate_v1.txt",
        "memory_review_v1.txt",
        "memory_manifest.json",
        "failure_mode_update_v1.txt",
        "skill_candidate_v1.txt",
        "skill_manifest.json",
    }
)
VANILLA_ALLOWED = frozenset({"vanilla_agent_v1.txt", "vanilla_manifest.json"})


class PromptEntry(FrozenRecord):
    role: Literal["policy", "critic", "revision"]
    path: str
    sha256: Digest
    prompt_version: Literal["v1", "v2", "v3", "v4"]
    output_model_name: Literal["ActionEnvelope", "CriticOutput"]


    @model_validator(mode="after")
    def version_role(self):
        if self.prompt_version == "v2" and self.role != "critic":
            raise ValueError("prompt v2 is restricted to Critic")
        return self


class PromptManifest(FrozenRecord):
    schema_version: Literal[1]
    entries: tuple[PromptEntry, ...]

    @model_validator(mode="after")
    def layout(self):
        if tuple((e.role, e.path, e.output_model_name) for e in self.entries) != tuple(zip(ROLES, PATHS, OUTPUTS)):
            raise ValueError("exact prompt role/path/schema order required")
        return self


class LoadedPrompt(FrozenRecord):
    entry: PromptEntry
    text: str

    @model_validator(mode="after")
    def identity(self):
        raw = self.text.encode("utf-8")
        if file_hash(raw) != self.entry.sha256:
            raise ValueError("prompt hash mismatch")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("prompt encoding/newline mismatch")
        return self


def load_prompts(project_root):
    root = Path(project_root)
    directory = root / "prompts"
    if not root.is_absolute() or any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("explicit non-symlink project root required")
    expected = {Path(p).name for p in PATHS} | {"manifest.json"}
    top_level = {p.name for p in directory.iterdir()}
    if not expected <= top_level or top_level - expected - {"offline"} - VANILLA_ALLOWED:
        raise ValueError("unexpected or missing prompt file")
    if any(
        p.is_symlink() or (p.name != "offline" and not p.is_file())
        for p in directory.iterdir()
    ):
        raise ValueError("invalid prompt file")
    offline = directory / "offline"
    if offline.exists():
        if offline.is_symlink() or not offline.is_dir():
            raise ValueError("invalid offline prompt directory")
        children = tuple(offline.iterdir())
        if {p.name for p in children} - OFFLINE_ALLOWED or any(
            p.is_symlink() or not p.is_file() for p in children
        ):
            raise ValueError("unexpected or invalid offline prompt file")
    manifest = PromptManifest.model_validate_json((directory / "manifest.json").read_bytes())
    return tuple(LoadedPrompt(entry=e, text=(root / e.path).read_bytes().decode("utf-8")) for e in manifest.entries)
