"""Thin upstream Agent-role shell for project-owned orchestration."""

from __future__ import annotations

from typing import Any

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.roles.base_role import BaseRole
from toolsandbox_pipeline.schemas.action import ActionEnvelope

from .contracts import PipelineResponder
from .messages import action_to_messages, extract_visible_messages
from .tools import build_adapter_turn


class PipelineAgent(BaseRole):
    """Replace only ToolSandbox's Agent role and delegate policy behavior."""

    role_type = RoleType.AGENT

    def __init__(self, responder: PipelineResponder) -> None:
        if responder is None:
            raise TypeError("responder is required")
        self._responder = responder

    def respond(self, ending_index: int | None = None) -> None:
        visible_messages = extract_visible_messages(type(self), ending_index)
        if not visible_messages:
            raise ValueError("cannot respond without an Agent-visible message")

        # Validate against the full upstream message representation before exposing
        # a sanitized turn. Extraction has already applied the identical truncation.
        raw_messages = self.get_messages(ending_index)
        self.messages_validation(raw_messages)
        if raw_messages[-1].sender == RoleType.SYSTEM:
            return

        turn = build_adapter_turn(type(self), visible_messages)
        raw_action: Any = self._responder.respond(turn)
        action = ActionEnvelope.model_validate(raw_action)
        messages = action_to_messages(action, turn.controller_context)
        if not messages:
            raise ValueError("an Agent action must produce at least one message")
        self.add_messages(messages)

    def reset(self) -> None:
        lifecycle = getattr(self._responder, "reset", None)
        if lifecycle is not None:
            lifecycle()
    def teardown(self) -> None:
        lifecycle = getattr(self._responder, "teardown", None)
        if lifecycle is not None:
            lifecycle()
