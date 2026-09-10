"""Exact, manifest-verified Task 016 prompt loading and request construction."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from toolsandbox_pipeline.reproducibility import canonical_json_bytes


class SkillPromptError(ValueError):
    pass


def load_skill_prompts(prompt_dir: Path, manifest_path: Path) -> dict[str, str]:
    if not isinstance(prompt_dir, Path) or not isinstance(manifest_path, Path):
        raise TypeError("explicit prompt paths required")
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest_raw.count(b"\n") != 1 or not manifest_raw.endswith(b"\n"):
        raise SkillPromptError("prompt manifest requires one terminal LF")
    if canonical_json_bytes(manifest) != manifest_raw[:-1]:
        raise SkillPromptError("prompt manifest must be canonical JSON")
    if manifest.get("schema_version") != 1:
        raise SkillPromptError("unsupported prompt manifest")
    expected_names = (
        "failure_mode_update_v1.txt",
        "skill_candidate_v1.txt",
    )
    entries = manifest.get("prompts")
    if not isinstance(entries, list) or tuple(item.get("path") for item in entries) != expected_names:
        raise SkillPromptError("exact ordered Skill prompt files required")
    allowed_names = set(expected_names) | {
        "memory_candidate_v1.txt",
        "memory_manifest.json",
        "memory_review_v1.txt",
        "skill_manifest.json",
    }
    present_names = set(path.name for path in prompt_dir.iterdir())
    if not set(expected_names) <= present_names or not present_names <= allowed_names:
        raise SkillPromptError("unexpected Skill prompt file")
    loaded: dict[str, str] = {}
    for entry in entries:
        path = prompt_dir / entry["path"]
        if path.is_symlink() or not path.is_file():
            raise SkillPromptError("regular prompt file required")
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
            raise SkillPromptError("prompt must be UTF-8 LF with terminal newline")
        if "sha256:" + hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise SkillPromptError("prompt hash mismatch")
        loaded[entry["role"]] = raw.decode("utf-8")
    if set(loaded) != {"failure_mode_update", "skill_candidate"}:
        raise SkillPromptError("prompt roles mismatch")
    return loaded


def two_message_request(system_prompt: str, payload: dict[str, object]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": canonical_json_bytes(payload).decode("utf-8")},
    ]


__all__ = ["SkillPromptError", "load_skill_prompts", "two_message_request"]
