"""Native tool execution and evaluator adapter. Global contexts are process-local."""

import copy
import random
import uuid

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from tool_sandbox.common.execution_context import (
    RoleType,
    get_current_context,
    new_context,
)
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.execution_environment import ExecutionEnvironment

from .protocol import validate_action


class AgentView(BaseRole):
    role_type = RoleType.AGENT


def visible_state():
    messages = AgentView.filter_messages(AgentView.get_messages())
    converted, _ = to_openai_messages(messages)
    # IDs are transport metadata, not observations. Normalize them for reproducible state hashes.
    ids = {}
    for message in converted:
        for call in message.get("tool_calls", []):
            old = call["id"]
            ids.setdefault(old, "call_" + str(len(ids)))
            call["id"] = ids[old]
        if "tool_call_id" in message:
            message["tool_call_id"] = ids[message["tool_call_id"]]
    return {
        "conversation": converted,
        "tools": convert_to_openai_tools(AgentView.get_available_tools()),
    }


def apply_action(action):
    action = validate_action(action)
    if "response" in action:
        AgentView.add_messages(
            [
                Message(
                    sender=RoleType.AGENT,
                    recipient=RoleType.USER,
                    content=action["response"],
                )
            ]
        )
        return
    context = get_current_context()
    available = set(AgentView.get_available_tools())
    # Construct all messages first, so malformed batches cannot partially mutate state.
    messages = []
    for call in action["tool_calls"]:
        if call["name"] not in available:
            raise ValueError("Action refers to an unavailable tool: " + call["name"])
        import json

        tool_call = ChatCompletionMessageToolCall(
            id="call_" + uuid.uuid4().hex,
            type="function",
            function={"name": call["name"], "arguments": json.dumps(call["arguments"])},
        )
        messages.append(
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=openai_tool_call_to_python_code(
                    tool_call,
                    available,
                    context.get_execution_facing_tool_name(call["name"]),
                ),
                openai_tool_call_id=tool_call.id,
                openai_function_name=call["name"],
            )
        )
    AgentView.add_messages(messages)


def advance(user):
    """Advance through the real tool/user roles until next agent decision or termination."""
    for _ in range(12):
        last = BaseRole.get_messages()[-1]
        if not last.conversation_active or last.recipient == RoleType.AGENT:
            return
        if last.recipient == RoleType.EXECUTION_ENVIRONMENT:
            ExecutionEnvironment().respond()
        elif last.recipient == RoleType.USER:
            user.respond()
        else:
            raise RuntimeError("Unexpected recipient")
    raise RuntimeError("Environment did not return to agent within 12 role turns")


def observation_since(index):
    # Only new tool/user messages visible to the agent, never database snapshots or labels.
    messages = AgentView.filter_messages(BaseRole.get_messages()[index:])
    result = [
        {"sender": str(m.sender), "content": m.content, "name": m.openai_function_name}
        for m in messages
        if m.sender != RoleType.AGENT
    ]
    return {
        "messages": result,
        "conversation_active": bool(BaseRole.get_messages()[-1].conversation_active),
    }


def probe(context, action, user):
    """Each branch starts from an independent full snapshot; always restore global context."""
    rng = random.getstate()
    try:
        with new_context(copy.deepcopy(context)):
            before = len(BaseRole.get_messages())
            apply_action(action)
            advance(user)
            return observation_since(before)
    finally:
        random.setstate(rng)


class PolicyAgent(AgentView):
    def __init__(self, policy, on_decision=None):
        self.policy, self.on_decision = policy, on_decision

    def respond(self, ending_index=None):
        messages = self.get_messages(ending_index)
        self.messages_validation(messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return
        state = visible_state()
        action = self.policy.action(state)
        if self.on_decision:
            self.on_decision(state, action, get_current_context())
        apply_action(action)
