"""Immutable, host-only inputs for the deterministic Controller."""

from __future__ import annotations

from enum import Enum

from pydantic import ConfigDict, Field

from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.base import JsonObject, StrictModel
from toolsandbox_pipeline.schemas.state import CompactVerifiedState, ControllerProvenanceSidecar
from toolsandbox_pipeline.schemas.tool_metadata import ControllerToolMetadata, MetadataPredicate


class _FrozenModel(StrictModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)


class SkillStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class RetrievedSkillControllerView(_FrozenModel):
    skill_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: SkillStatus
    generation_id: str = Field(min_length=1)
    tool_dependencies: tuple[str, ...] = ()
    applicability_required: tuple[MetadataPredicate, ...] = ()
    applicability_forbidden: tuple[MetadataPredicate, ...] = ()
    required_inputs: tuple[MetadataPredicate, ...] = ()


class ActionHistoryEntry(_FrozenModel):
    action_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    failed: bool
    visible: bool = True
    source_ref: str = Field(min_length=1)


class ReproducibilityProfile(str, Enum):
    STRICT_REPLAY = "strict_replay"
    OFFICIAL_LIVE = "official_live"


class ControllerInput(_FrozenModel):
    state: CompactVerifiedState
    provenance_sidecar: ControllerProvenanceSidecar
    action: ActionEnvelope
    tool_metadata: tuple[ControllerToolMetadata, ...]
    agent_to_canonical_name: dict[str, str]
    mapping_manifest_hash: str = Field(min_length=1)
    retrieved_skills: tuple[RetrievedSkillControllerView, ...] = ()
    action_history: tuple[ActionHistoryEntry, ...] = ()
    generation_id: str = Field(min_length=1)
    reproducibility_profile: ReproducibilityProfile
    structured_constraint_tension: bool = False


__all__ = ["ActionHistoryEntry", "ControllerInput", "ReproducibilityProfile", "RetrievedSkillControllerView", "SkillStatus"]
