"""Stable host execution IDs, distinct from model-local call labels."""
import json

from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope, AssistantMessageAction
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnDecision


def execution_action(decision: OnlineTurnDecision) -> ActionEnvelope:
    """Create a dispatch-only copy; the original decision and audit stay untouched.

    The identity is derived from persisted turn identity and the original action,
    so recreating a turn after a checkpoint yields exactly the same native IDs.
    """
    if type(decision) is not OnlineTurnDecision:
        raise TypeError("OnlineTurnDecision required")
    envelope = ActionEnvelope.model_validate_json(decision.final_action.model_dump_json())
    if isinstance(envelope.action, AssistantMessageAction):
        return envelope
    payload = envelope.model_dump(mode="json")
    action_hash = canonical_sha256(payload)
    calls = payload["action"].get("calls", [payload["action"]])
    for slot, call in enumerate(calls):
        call["call_id"] = "exec_" + canonical_sha256({
            "schema": "toolsandbox-execution-call-v1",
            "run_id": decision.identity.run_id,
            "episode_id": decision.identity.episode_id,
            "agent_turn_index": decision.identity.agent_turn_index,
            "action_sha256": action_hash,
            "slot": slot,
            "model_call_id": call["call_id"],
        }).removeprefix("sha256:")
    return ActionEnvelope.model_validate_json(json.dumps(payload))


def execution_call_ids(decision: OnlineTurnDecision) -> tuple[str, ...]:
    action = execution_action(decision).action
    if isinstance(action, AssistantMessageAction):
        return ()
    return tuple(call.call_id for call in getattr(action, "calls", (action,)))
