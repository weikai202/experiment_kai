"""Auditable data construction, isolated transitions, and split validation."""

import json
from copy import deepcopy

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
from tau2.user.user_simulator_base import is_valid_user_history_message
from tau2.utils.llm_utils import to_litellm_messages

SUPPORTED_DOMAINS = {"mock", "retail", "airline", "telecom"}
REFLECTION_INSTRUCTION = (
    "You may reason privately inside <think>...</think> before your response or "
    "tool call. Text outside these tags is sent to the customer."
)


def dumps(value):
    """Canonical serialization used for deduplication and prediction records."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def action_key(action):
    """Ignore generated call IDs while preserving call order and arguments."""
    if action.is_tool_call():
        return dumps([(tc.name, tc.arguments) for tc in action.tool_calls])
    return dumps(action.content)


def api_messages(messages):
    """Use native evaluator serialization, omitting provider-specific null fields."""
    result = to_litellm_messages(messages)
    for msg in result:
        if not msg.get("tool_calls"):
            msg.pop("tool_calls", None)
        for tc in msg.get("tool_calls", []):
            tc.pop("name", None)
    return result


def policy_prompt(policy):
    """Shared IL/SR training and evaluation instruction."""
    return SYSTEM_PROMPT.format(
        domain_policy=policy,
        agent_instruction=AGENT_INSTRUCTION + "\n" + REFLECTION_INSTRUCTION,
    )


def visible_context(policy, history):
    """Exclude private user tools, scenario instructions, rewards, and database."""
    return [{"role": "system", "content": policy_prompt(policy)}] + api_messages(
        [m for m in history if is_valid_agent_history_message(m)]
    )


def parse_action(value, index=0):
    """Parse proposer JSON; forbid spoofed user-tool requestors."""
    calls = value.get("tool_calls") or []
    content = value.get("content")
    if bool(calls) == bool(content):
        raise ValueError("Each candidate must contain either content or tool_calls")
    action = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=[
            ToolCall(id=f"ee_{index}_{j}", name=tc["name"], arguments=tc["arguments"])
            for j, tc in enumerate(calls)
        ]
        or None,
    )
    action.validate()
    return action


def probe(environment, action, user=None, history=(), max_user_steps=20):
    """Execute one action on an isolated env, through the next agent observation.

    Tool batches execute sequentially on the SAME clone. A spoken action invokes
    a freshly reconstructed simulator state; user tools run privately until the
    next spoken user response. Private user tool results never become targets.
    Only pure in-memory domains are supported by the caller.
    """
    env = deepcopy(environment)
    if action.is_tool_call():
        observations = [env.get_response(tc) for tc in action.tool_calls]
        return {"messages": api_messages(observations), "terminal": False}
    if user is None:
        raise ValueError("A user simulator is required for conversational actions")
    state = user.get_init_state(
        deepcopy([m for m in history if is_valid_user_history_message(m)])
    )
    message = deepcopy(action)
    for _ in range(max_user_steps):
        response, state = user.generate_next_message(message, state)
        response.validate()
        if not response.is_tool_call():
            return {
                "messages": api_messages([response]),
                "terminal": user.is_stop(response),
            }
        if any(tc.requestor != "user" for tc in response.tool_calls):
            raise ValueError("User simulator returned an assistant tool requestor")
        results = [env.get_response(tc) for tc in response.tool_calls]
        message = (
            results[0]
            if len(results) == 1
            else MultiToolMessage(role="tool", tool_messages=results)
        )
    raise RuntimeError("User branch exceeded max_user_steps; no fabricated next state")


def make_expert(context, tools, action):
    """A single final-assistant target, preserving native structured tool calls."""
    return {"messages": deepcopy(context) + api_messages([action]), "tools": tools}


def make_iwm(context, tools, action, outcome):
    """World-model instruction and delimited target distinct from action SFT."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "Predict the next agent-visible observation after the supplied action. Do not act. Return Observation: followed by JSON.",
            },
            {
                "role": "user",
                "content": dumps(
                    {
                        "state": context,
                        "tools": tools,
                        "action": api_messages([action])[0],
                    }
                ),
            },
            {"role": "assistant", "content": "Observation:\n" + dumps(outcome)},
        ]
    }


def make_reflection(expert, reflection):
    """Prepend a private reflection, leaving the committed action unchanged."""
    if not reflection.strip() or "<think>" in reflection or "</think>" in reflection:
        raise ValueError("Reflection must be nonempty plain text without think tags")
    result = deepcopy(expert)
    target = result["messages"][-1]
    target["content"] = (
        "<think>\n"
        + reflection.strip()
        + "\n</think>\n"
        + (target.get("content") or "")
    )
    return result


def validate_split(train_ids, eval_ids):
    """Reject empty, duplicate, or overlapping explicitly selected task sets."""
    if not train_ids or not eval_ids:
        raise ValueError("Training and evaluation task sets must both be nonempty")
    if len(set(train_ids)) != len(train_ids) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("Duplicate task IDs")
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise ValueError(f"Training/evaluation overlap: {sorted(overlap)}")
