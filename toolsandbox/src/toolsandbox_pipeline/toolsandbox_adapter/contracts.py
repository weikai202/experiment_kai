"""Trusted-host contracts for the pinned ToolSandbox adapter."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.state import AgentFacingToolInput, VisibleMessageInput


@dataclass(frozen=True)
class AgentTurnView:
    """The complete prompt-facing view for one Agent turn."""

    visible_messages: tuple[VisibleMessageInput, ...]
    available_tools: tuple[AgentFacingToolInput, ...]


@dataclass(frozen=True)
class ControllerToolContext:
    """Host-only tool identity and callable context.

    This type deliberately has no model-dump helper and is never nested in the
    prompt-facing :class:`AgentTurnView`.
    """

    agent_to_execution_name: Mapping[str, str]
    mapping_manifest_hash: str
    tool_objects: Mapping[str, Callable[..., object]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_to_execution_name",
            MappingProxyType(dict(self.agent_to_execution_name)),
        )
        object.__setattr__(self, "tool_objects", MappingProxyType(dict(self.tool_objects)))


@dataclass(frozen=True)
class AdapterTurn:
    """Trusted orchestration bundle; not a prompt serialization model."""

    agent_view: AgentTurnView
    controller_context: ControllerToolContext


class PipelineResponder(Protocol):
    """Synchronous project-owned response boundary."""

    def respond(self, turn: AdapterTurn) -> ActionEnvelope: ...
