import pytest

from toolsandbox_pipeline.online.routing import (
    RouteSelection,
    requires_critic,
    route_after_critic,
    route_after_revision,
)
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.critic import CriticOutput


def decision(*, blocking=(), triggers=()):
    evidence = [
        {"code": code, "source_kind": "state", "source_ref": f"ref:{code}"}
        for code in (*blocking, *triggers)
    ]
    return ControllerDecision(
        blocking_codes=list(blocking),
        critic_trigger_codes=list(triggers),
        evidence=evidence,
    )


def critic(verdict):
    if verdict == "accept":
        return CriticOutput(
            verdict="accept",
            predicted_outcome="success",
            predicted_effect="A visible result",
            error_codes=[],
            correction="",
        )
    return CriticOutput(
        verdict=verdict,
        predicted_outcome="uncertain" if verdict == "uncertain" else "failure",
        predicted_effect="No safe result",
        error_codes=["INSUFFICIENT_CONTEXT"],
        correction="Obtain the missing information",
    )


def test_clean_decision_skips_critic():
    clean = decision()
    assert not requires_critic(clean)
    with pytest.raises(ValueError, match="skip Critic"):
        route_after_critic(clean, critic("accept"))


def test_only_unblocked_accept_selects_original():
    triggered = decision(triggers=("ASSISTANT_MESSAGE_REVIEW",))
    assert requires_critic(triggered)
    assert route_after_critic(triggered, critic("accept")) is RouteSelection.ORIGINAL
    assert route_after_critic(triggered, critic("revise")) is RouteSelection.REVISION
    assert route_after_critic(triggered, critic("uncertain")) is RouteSelection.REVISION


def test_blocking_code_forces_revision_even_if_critic_accepts():
    blocked = decision(blocking=("UNGROUNDED_ARGUMENT",))
    assert route_after_critic(blocked, critic("accept")) is RouteSelection.REVISION


def test_post_revision_ignores_triggers_but_not_blocks():
    assert (
        route_after_revision(decision(triggers=("ASSISTANT_MESSAGE_REVIEW",)))
        is RouteSelection.REVISION
    )
    assert (
        route_after_revision(decision(blocking=("MISSING_DEPENDENCY",)))
        is RouteSelection.SAFE_CLARIFICATION
    )
