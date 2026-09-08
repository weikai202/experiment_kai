# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""CoMAP-style agent role for ToolSandbox.

This role keeps ToolSandbox's native execution loop, but changes the policy
inference pattern to match CoMAP: draft an action, optionally ask a world model
to predict the next state, then reflect and emit the final action.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union, cast

from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole


def _jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _message_to_record(message: Message) -> dict[str, str]:
    if message.sender == RoleType.SYSTEM:
        role = "system"
    elif message.sender == RoleType.USER:
        role = "user"
    elif message.sender == RoleType.AGENT:
        role = "assistant"
    elif message.sender == RoleType.EXECUTION_ENVIRONMENT:
        role = "tool"
    else:
        role = str(message.sender)
    return {"role": role, "content": message.content}


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def _summarize_tool_call(message: Message) -> str:
    return message.content.strip()


class CoMAPAgent(BaseRole):
    """CoMAP-style OpenAI-compatible ToolSandbox agent.

    Environment variables:
        COMAP_POLICY_MODEL: Required policy model name.
        COMAP_OPENAI_BASE_URL: Optional OpenAI-compatible policy endpoint.
        COMAP_API_KEY: Optional policy API key. Defaults to OPENAI_API_KEY or
            "EMPTY", which works for many local OpenAI-compatible servers.
        COMAP_WORLD_MODEL: Optional world-model name. If omitted, reflection
            still runs with a textual placeholder.
        COMAP_WM_OPENAI_BASE_URL: Optional world-model endpoint.
        COMAP_WM_API_KEY: Optional world-model API key.
        COMAP_ROLLOUT_JSONL: Optional CoMAP-compatible step log path.
    """

    role_type: RoleType = RoleType.AGENT
    model_name = "CoMAP"

    def __init__(self) -> None:
        policy_model = os.environ.get("COMAP_POLICY_MODEL")
        if not policy_model:
            raise ValueError("Set COMAP_POLICY_MODEL to run the CoMAP ToolSandbox baseline.")

        self.policy_model = policy_model
        self.world_model = os.environ.get("COMAP_WORLD_MODEL")
        self.model_name = f"CoMAP_{policy_model}"
        self.policy_client = OpenAI(
            base_url=os.environ.get("COMAP_OPENAI_BASE_URL"),
            api_key=os.environ.get("COMAP_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        )
        self.world_model_client = OpenAI(
            base_url=os.environ.get("COMAP_WM_OPENAI_BASE_URL")
            or os.environ.get("COMAP_OPENAI_BASE_URL"),
            api_key=os.environ.get("COMAP_WM_API_KEY")
            or os.environ.get("COMAP_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY",
        )
        self.rollout_jsonl = Path(
            os.environ.get("COMAP_ROLLOUT_JSONL", "data/comap_toolsandbox_rollouts.jsonl")
        )
        self.round_id = int(os.environ.get("COMAP_ROUND_ID", "0"))
        self.policy_ckpt = os.environ.get("COMAP_POLICY_CKPT", policy_model)
        self.student_world_model_path = self.world_model
        self.teacher_world_model_path = os.environ.get("COMAP_TEACHER_WORLD_MODEL")
        self.logged_tool_call_ids: set[str] = set()
        self.pending_meta: dict[str, dict[str, Any]] = {}

    def respond(self, ending_index: Optional[int] = None) -> None:
        messages = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return

        self._log_completed_tool_steps(messages)

        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven],
            convert_to_openai_tools(available_tools)
            if messages[-1].sender in {RoleType.USER, RoleType.EXECUTION_ENVIRONMENT}
            else NOT_GIVEN,
        )
        openai_messages, _ = to_openai_messages(messages)

        draft = self._chat_completion(
            client=self.policy_client,
            model=self.policy_model,
            messages=openai_messages,
            tools=openai_tools,
        )
        imagined_next_state = self._predict_next_state(openai_messages, draft)
        reflected = self._chat_completion(
            client=self.policy_client,
            model=self.policy_model,
            messages=self._build_reflection_messages(
                openai_messages=openai_messages,
                draft=draft,
                imagined_next_state=imagined_next_state,
            ),
            tools=openai_tools,
        )

        response_messages = self._completion_to_messages(
            completion=reflected,
            available_tool_names=available_tool_names,
        )
        self._remember_pending_meta(
            response_messages=response_messages,
            draft=draft,
            reflected=reflected,
            imagined_next_state=imagined_next_state,
        )
        self.add_messages(response_messages)

    def _chat_completion(
        self,
        *,
        client: OpenAI,
        model: str,
        messages: list[
            dict[Literal["role", "content", "tool_call_id", "name", "tool_calls"], Any]
        ],
        tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        with all_logging_disabled():
            return client.chat.completions.create(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                tools=tools,
            )

    def _predict_next_state(
        self,
        openai_messages: list[
            dict[Literal["role", "content", "tool_call_id", "name", "tool_calls"], Any]
        ],
        draft: ChatCompletion,
    ) -> str:
        if not self.world_model:
            return "World model unavailable; reflect using the real conversation and tool schemas."
        prompt = (
            "Predict the likely next ToolSandbox state after the assistant draft. "
            "Focus on tool result, state change, safety risk, and whether the action helps the user.\n\n"
            f"Conversation:\n{json.dumps(openai_messages, ensure_ascii=False)}\n\n"
            f"Assistant draft:\n{draft.choices[0].message.model_dump_json()}"
        )
        completion = self._chat_completion(
            client=self.world_model_client,
            model=self.world_model,
            messages=[{"role": "user", "content": prompt}],
            tools=NOT_GIVEN,
        )
        return completion.choices[0].message.content or ""

    @staticmethod
    def _build_reflection_messages(
        *,
        openai_messages: list[
            dict[Literal["role", "content", "tool_call_id", "name", "tool_calls"], Any]
        ],
        draft: ChatCompletion,
        imagined_next_state: str,
    ) -> list[dict[str, Any]]:
        reflection_prompt = (
            "You are applying CoMAP future-aware reflection for ToolSandbox.\n"
            "Review the draft action and the predicted next state. Emit the final response. "
            "If a tool call is appropriate, call exactly the needed tool with valid arguments. "
            "Avoid unsafe minefield actions and unnecessary distractor tools.\n\n"
            f"Draft assistant message:\n{draft.choices[0].message.model_dump_json()}\n\n"
            f"Predicted next state:\n{imagined_next_state}"
        )
        return [*openai_messages, {"role": "user", "content": reflection_prompt}]

    def _completion_to_messages(
        self,
        *,
        completion: ChatCompletion,
        available_tool_names: set[str],
    ) -> list[Message]:
        current_context = get_current_context()
        response = completion.choices[0].message
        if response.tool_calls is None:
            return [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=response.content or "",
                )
            ]

        messages: list[Message] = []
        for tool_call in response.tool_calls:
            execution_facing_tool_name = current_context.get_execution_facing_tool_name(
                tool_call.function.name
            )
            messages.append(
                Message(
                    sender=self.role_type,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content=openai_tool_call_to_python_code(
                        tool_call,
                        available_tool_names,
                        execution_facing_tool_name=execution_facing_tool_name,
                    ),
                    openai_tool_call_id=tool_call.id,
                    openai_function_name=tool_call.function.name,
                )
            )
        return messages

    def _remember_pending_meta(
        self,
        *,
        response_messages: list[Message],
        draft: ChatCompletion,
        reflected: ChatCompletion,
        imagined_next_state: str,
    ) -> None:
        draft_message = draft.choices[0].message
        reflect_message = reflected.choices[0].message
        for message in response_messages:
            if message.openai_tool_call_id is None:
                continue
            self.pending_meta[message.openai_tool_call_id] = {
                "draft_response": draft_message.model_dump(mode="json", exclude_none=True),
                "reflect_response": reflect_message.model_dump(mode="json", exclude_none=True),
                "imagined_next_state": imagined_next_state,
                "used_revised_action": True,
            }

    def _log_completed_tool_steps(self, messages: list[Message]) -> None:
        for index, message in enumerate(messages):
            if message.sender != RoleType.EXECUTION_ENVIRONMENT:
                continue
            if message.recipient != RoleType.AGENT or message.openai_tool_call_id is None:
                continue
            tool_call_id = message.openai_tool_call_id
            if tool_call_id in self.logged_tool_call_ids:
                continue
            agent_message = self._find_agent_tool_message(messages[:index], tool_call_id)
            if agent_message is None:
                continue

            history_messages = [_message_to_record(m) for m in messages[: index + 1]]
            meta = self.pending_meta.get(tool_call_id, {})
            row = {
                "task_name": "toolsandbox",
                "task_id": os.environ.get("COMAP_SCENARIO_NAME", "unknown"),
                "episode_id": os.environ.get("COMAP_SCENARIO_NAME", "unknown"),
                "round_id": self.round_id,
                "step_id": len(self.logged_tool_call_ids),
                "goal": self._infer_goal(messages),
                "history_messages": history_messages,
                "history_text": _messages_to_text(history_messages),
                "observation": self._previous_observation(messages[:index]),
                "action": _summarize_tool_call(agent_message),
                "action_text": _summarize_tool_call(agent_message),
                "reward": 0.0,
                "done": False,
                "success": False,
                "next_observation": message.content,
                "feedback_text": self._format_feedback(message.content),
                "policy_ckpt": self.policy_ckpt,
                "trajectory_source": "toolsandbox_on_policy",
                "flow_mode": "comap_toolsandbox",
                "draft_response": json.dumps(meta.get("draft_response"), ensure_ascii=False),
                "draft_action": None,
                "pred_next_state_student": meta.get("imagined_next_state"),
                "reflect_response": json.dumps(meta.get("reflect_response"), ensure_ascii=False),
                "revised_action": _summarize_tool_call(agent_message),
                "revise_triggered": True,
                "used_revised_action": meta.get("used_revised_action"),
                "student_world_model_path": self.student_world_model_path,
                "teacher_world_model_path": self.teacher_world_model_path,
                "imagined_next_state": meta.get("imagined_next_state"),
                "real_next_state": message.content,
            }
            _jsonl_append(self.rollout_jsonl, row)
            self.logged_tool_call_ids.add(tool_call_id)

    @staticmethod
    def _find_agent_tool_message(
        messages: list[Message], tool_call_id: str
    ) -> Optional[Message]:
        for message in reversed(messages):
            if (
                message.sender == RoleType.AGENT
                and message.recipient == RoleType.EXECUTION_ENVIRONMENT
                and message.openai_tool_call_id == tool_call_id
            ):
                return message
        return None

    @staticmethod
    def _infer_goal(messages: list[Message]) -> str:
        for message in messages:
            if message.sender == RoleType.USER and message.recipient == RoleType.AGENT:
                return message.content
        return ""

    @staticmethod
    def _previous_observation(messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.sender in {RoleType.USER, RoleType.EXECUTION_ENVIRONMENT}:
                return message.content
        return ""

    @staticmethod
    def _format_feedback(next_observation: str) -> str:
        return "\n".join(
            [
                "Environment feedback summary:",
                "- benchmark: ToolSandbox",
                f"- next_observation: {next_observation}",
            ]
        )
