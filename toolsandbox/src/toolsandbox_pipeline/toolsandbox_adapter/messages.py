"""Agent-visible message extraction and native message construction."""

from __future__ import annotations

import json
from typing import Iterable

import polars as pl
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    get_current_context,
)
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
)
from toolsandbox_pipeline.schemas.action import (
    ActionEnvelope,
    AssistantMessageAction,
    FunctionCall,
    FunctionCallAction,
    ParallelBatchAction,
)
from toolsandbox_pipeline.schemas.state import VisibleMessageInput, VisibleRole

from .contracts import ControllerToolContext


def extract_visible_messages(
    agent_class: type,
    ending_index: int | None = None,
) -> tuple[VisibleMessageInput, ...]:
    """Read SANDBOX history once, retain indices, then apply upstream filtering."""

    database = get_current_context().get_database(
        namespace=DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    )
    if ending_index is not None:
        database = database.filter(pl.col("sandbox_message_index") <= ending_index)
    rows = database.to_dicts()
    indices = [row["sandbox_message_index"] for row in rows]
    if any(not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError("sandbox message indices must be non-negative integers")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("sandbox message indices must be unique and increasing")

    indexed_messages: list[tuple[int, Message]] = []
    for row in rows:
        message_fields = dict(row)
        source_index = message_fields.pop("sandbox_message_index")
        indexed_messages.append((source_index, Message(**message_fields)))

    filtered_ids = {id(message) for message in agent_class.filter_messages(
        [message for _, message in indexed_messages]
    )}
    return tuple(
        VisibleMessageInput(
            source_message_index=source_index,
            sender=VisibleRole(message.sender.value),
            recipient=VisibleRole(message.recipient.value),
            content=message.content,
            openai_tool_call_id=message.openai_tool_call_id,
            openai_function_name=message.openai_function_name,
            tool_call_exception=message.tool_call_exception,
        )
        for source_index, message in indexed_messages
        if id(message) in filtered_ids
    )


def _convert_call(
    call: FunctionCall | FunctionCallAction,
    controller_context: ControllerToolContext,
) -> Message:
    available_names = set(controller_context.tool_objects)
    if call.name not in available_names:
        raise KeyError(f"unavailable agent-facing tool name: {call.name}")
    try:
        execution_name = controller_context.agent_to_execution_name[call.name]
    except KeyError as exc:
        raise KeyError(f"missing tool-name mapping for: {call.name}") from exc
    tool_call = ChatCompletionMessageToolCall(
        id=call.call_id,
        type="function",
        function=Function(
            name=call.name,
            arguments=json.dumps(
                call.arguments,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
        ),
    )
    code = openai_tool_call_to_python_code(
        tool_call=tool_call,
        available_tool_names=available_names,
        execution_facing_tool_name=execution_name,
    )
    return Message(
        sender=RoleType.AGENT,
        recipient=RoleType.EXECUTION_ENVIRONMENT,
        content=code,
        openai_tool_call_id=call.call_id,
        openai_function_name=call.name,
    )


def action_to_messages(
    envelope: ActionEnvelope,
    controller_context: ControllerToolContext,
) -> list[Message]:
    """Convert a validated action to native messages without executing it."""

    action = envelope.action
    if isinstance(action, AssistantMessageAction):
        return [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content=action.content,
            )
        ]
    calls: Iterable[FunctionCall | FunctionCallAction]
    if isinstance(action, FunctionCallAction):
        calls = (action,)
    elif isinstance(action, ParallelBatchAction):
        calls = action.calls
    else:  # pragma: no cover - closed by ActionEnvelope validation
        raise TypeError(f"unsupported action type: {type(action)!r}")

    calls = tuple(calls)
    names = set(controller_context.tool_objects)
    mapping = controller_context.agent_to_execution_name
    for call in calls:
        if call.name not in names:
            raise KeyError(f"unavailable agent-facing tool name: {call.name}")
        if call.name not in mapping:
            raise KeyError(f"missing tool-name mapping for: {call.name}")
    if len(mapping.values()) != len(set(mapping.values())):
        raise ValueError("execution-facing tool names must be unique")
    return [_convert_call(call, controller_context) for call in calls]
