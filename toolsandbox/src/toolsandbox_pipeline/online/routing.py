"""Pure implementation of the approved one-Revision routing table."""

from __future__ import annotations

from enum import Enum

from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.critic import CriticOutput, CriticVerdict


class RouteSelection(str, Enum):
    ORIGINAL = "original"
    REVISION = "revision"
    SAFE_CLARIFICATION = "safe_clarification"


def requires_critic(decision: ControllerDecision) -> bool:
    if type(decision) is not ControllerDecision:
        raise TypeError("validated ControllerDecision required")
    return bool(decision.blocking_codes or decision.critic_trigger_codes)


def route_after_critic(
    decision: ControllerDecision,
    critic: CriticOutput,
) -> RouteSelection:
    if not requires_critic(decision):
        raise ValueError("a clean Controller decision must skip Critic")
    if type(critic) is not CriticOutput:
        raise TypeError("validated CriticOutput required")
    if not decision.blocking_codes and critic.verdict is CriticVerdict.ACCEPT:
        return RouteSelection.ORIGINAL
    return RouteSelection.REVISION


def route_after_revision(decision: ControllerDecision) -> RouteSelection:
    if type(decision) is not ControllerDecision:
        raise TypeError("validated ControllerDecision required")
    # Post-Revision trigger codes are retained for audit but never route again.
    if decision.blocking_codes:
        return RouteSelection.SAFE_CLARIFICATION
    return RouteSelection.REVISION


__all__ = [
    "RouteSelection",
    "requires_critic",
    "route_after_critic",
    "route_after_revision",
]
