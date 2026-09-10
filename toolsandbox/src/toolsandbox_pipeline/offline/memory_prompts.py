"""Versioned prompt loading and canonical one-trajectory envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from pydantic import ConfigDict, Field
from typing import Annotated, Literal

from toolsandbox_pipeline.offline.memory_projection import projection_json
from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.base import StrictModel
from toolsandbox_pipeline.schemas.memory import PolicyMemory, WorldMemory
from toolsandbox_pipeline.schemas.offline_memory import (
    MemoryCandidateNone,
    PolicyMemoryCandidate,
    PolicyTrajectoryProjection,
    WorldMemoryCandidate,
    WorldTrajectoryProjection,
)


class MemoryPromptError(ValueError):
    pass


class _PromptEntry(StrictModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["memory_candidate", "memory_review"]
    path: Literal[
        "prompts/offline/memory_candidate_v1.txt",
        "prompts/offline/memory_review_v1.txt",
    ]
    sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    prompt_version: Literal["v1"]
    output_model_name: Literal[
        "MemoryCandidateDecision", "MemoryReviewDecision"
    ] = Field(alias="output_schema")


class _PromptManifest(StrictModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1]
    entries: tuple[_PromptEntry, _PromptEntry]


@dataclass(frozen=True)
class MemoryPromptSet:
    candidate: str
    review: str
    candidate_sha256: str
    review_sha256: str
    manifest_sha256: str


def _safe_file(project_root: Path, relative: str) -> Path:
    target = project_root / relative
    if (
        not project_root.is_absolute()
        or project_root.is_symlink()
        or target.is_symlink()
        or not target.is_file()
        or target.resolve().parent != (project_root.resolve() / "prompts/offline")
    ):
        raise MemoryPromptError("unsafe memory prompt path")
    return target


def load_memory_prompts(project_root: Path, manifest_path: Path) -> MemoryPromptSet:
    if (
        not project_root.is_absolute()
        or not manifest_path.is_absolute()
        or manifest_path.is_symlink()
        or manifest_path != project_root / "prompts/offline/memory_manifest.json"
    ):
        raise MemoryPromptError("explicit canonical prompt manifest required")
    raw_manifest = manifest_path.read_bytes()
    if raw_manifest.startswith(b"\xef\xbb\xbf") or b"\r" in raw_manifest:
        raise MemoryPromptError("invalid manifest encoding")
    try:
        manifest = _PromptManifest.model_validate_json(raw_manifest)
    except Exception as error:
        raise MemoryPromptError("invalid memory prompt manifest") from error
    expected_roles = ("memory_candidate", "memory_review")
    expected_paths = (
        "prompts/offline/memory_candidate_v1.txt",
        "prompts/offline/memory_review_v1.txt",
    )
    if tuple(entry.role for entry in manifest.entries) != expected_roles or tuple(
        entry.path for entry in manifest.entries
    ) != expected_paths:
        raise MemoryPromptError("memory prompt manifest order mismatch")
    texts: list[str] = []
    for entry in manifest.entries:
        raw = _safe_file(project_root, entry.path).read_bytes()
        if (
            raw.startswith(b"\xef\xbb\xbf")
            or b"\r" in raw
            or not raw.endswith(b"\n")
            or raw.endswith(b"\n\n")
            or "sha256:" + sha256(raw).hexdigest() != entry.sha256
        ):
            raise MemoryPromptError("memory prompt bytes mismatch")
        try:
            texts.append(raw.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise MemoryPromptError("memory prompt must be UTF-8") from error
    return MemoryPromptSet(
        candidate=texts[0],
        review=texts[1],
        candidate_sha256=manifest.entries[0].sha256,
        review_sha256=manifest.entries[1].sha256,
        manifest_sha256="sha256:" + sha256(raw_manifest).hexdigest(),
    )


def candidate_envelope(
    projection: PolicyTrajectoryProjection | WorldTrajectoryProjection,
) -> str:
    payload = {
        "memory_role": "policy" if type(projection) is PolicyTrajectoryProjection else "world",
        "trajectory": json.loads(projection_json(projection)),
    }
    return canonical_json_bytes(payload).decode("utf-8")


def _review_view(record: PolicyMemory | WorldMemory) -> dict[str, object]:
    data = record.model_dump(mode="json")
    if type(record) is PolicyMemory:
        keys = ("memory_id", "scope", "applicability", "action_guidance", "avoid")
    elif type(record) is WorldMemory:
        keys = (
            "memory_id",
            "action_pattern",
            "state_conditions",
            "schema_conditions",
            "likely_error_codes",
            "outcome_calibration",
            "correction_principle",
        )
    else:
        raise TypeError("active Policy or World memory required")
    if record.status != "active":
        raise MemoryPromptError("review candidate requires active memory")
    return {key: data[key] for key in keys}


def review_envelope(
    candidate: PolicyMemoryCandidate | WorldMemoryCandidate,
    matches: tuple[PolicyMemory | WorldMemory, ...],
) -> str:
    if type(candidate) not in (PolicyMemoryCandidate, WorldMemoryCandidate):
        raise TypeError("concrete memory candidate required")
    role_type = PolicyMemory if candidate.role == "policy" else WorldMemory
    if len(matches) > 3 or any(type(record) is not role_type for record in matches):
        raise MemoryPromptError("review matches must be role-local top three")
    ids = tuple(record.memory_id for record in matches)
    if len(ids) != len(set(ids)):
        raise MemoryPromptError("duplicate review target")
    payload = {
        "candidate": candidate.candidate.model_dump(mode="json"),
        "memory_role": candidate.role,
        "similar_memories": [_review_view(record) for record in matches],
    }
    return canonical_json_bytes(payload).decode("utf-8")


def prompt_input_fingerprint(
    *, role: str, prompt_sha256: str, user_envelope: str, output_schema_sha256: str,
    max_tokens: int, structured_output_wire_mode: str,
) -> str:
    return canonical_sha256(
        {
            "provider": "vllm_openai_compatible",
            "model": "Qwen/Qwen3-32B",
            "role": role,
            "prompt_sha256": prompt_sha256,
            "user_envelope": json.loads(user_envelope),
            "output_schema_sha256": output_schema_sha256,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "top_p": "omitted",
            "enable_thinking": False,
            "structured_output_wire_mode": structured_output_wire_mode,
        }
    )


__all__ = [
    "MemoryPromptError",
    "MemoryPromptSet",
    "candidate_envelope",
    "load_memory_prompts",
    "prompt_input_fingerprint",
    "review_envelope",
]
