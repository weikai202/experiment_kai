"""Narrow tau3 native adapter with an agent-visible-only trust boundary."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class TauToolView:
    name: str
    description: str
    parameters: dict[str, Any]

    @classmethod
    def from_native(cls, tool: object) -> "TauToolView":
        schema = copy.deepcopy(getattr(tool, "openai_schema"))
        if set(schema) != {"type", "function"} or schema["type"] != "function":
            raise ValueError("invalid native tau tool schema")
        function = schema["function"]
        return cls(
            name=function["name"],
            description=function.get("description", ""),
            parameters=function["parameters"],
        )


@dataclass(frozen=True)
class AssistantAction:
    content: str | None = None
    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = ()

    def __post_init__(self) -> None:
        has_message = self.content is not None
        if has_message == bool(self.tool_calls):
            raise ValueError("action must contain exactly one of message or tool calls")
        if has_message and (type(self.content) is not str or not self.content.strip()):
            raise ValueError("message content must be non-empty")
        if any(
            type(name) is not str or not name or type(args) is not dict
            for name, args in self.tool_calls
        ):
            raise ValueError("invalid tool call")


class TurnPolicy(Protocol):
    def respond(
        self,
        *,
        domain_policy: str,
        tools: tuple[TauToolView, ...],
        visible_history: tuple[dict[str, Any], ...],
    ) -> AssistantAction: ...


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def visible_message(message: object) -> dict[str, Any]:
    """Project a native message without evaluator or user-private information."""

    role = _field(message, "role")
    if role not in {"assistant", "user", "tool"}:
        raise ValueError("unsupported or hidden message role")
    calls = _field(message, "tool_calls") or ()
    requestor = _field(message, "requestor")
    if role == "user" and calls:
        raise ValueError("user-private tool calls cannot enter agent state")
    if role == "tool" and requestor != "assistant":
        raise ValueError("user-private tool results cannot enter agent state")
    result: dict[str, Any] = {"role": role, "content": _field(message, "content")}
    if calls:
        result["tool_calls"] = [
            {
                "id": _field(call, "id"),
                "name": _field(call, "name"),
                "arguments": copy.deepcopy(_field(call, "arguments")),
            }
            for call in calls
        ]
    if role == "tool":
        result.update(id=_field(message, "id"), error=_field(message, "error"))
    return result


def _validate_json_type(value: Any, expected: str) -> bool:
    return {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
        "number": type(value) in {int, float},
        "boolean": type(value) is bool,
        "null": value is None,
    }.get(expected, False)


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ValueError("tool parameter root must be an object")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not set(required) <= set(arguments):
        raise ValueError("missing required tool argument")
    if set(arguments) - set(properties):
        raise ValueError("unknown tool argument")
    for name, value in arguments.items():
        expected = properties[name].get("type")
        if isinstance(expected, list):
            valid = any(_validate_json_type(value, item) for item in expected)
        else:
            valid = _validate_json_type(value, expected)
        if expected is not None and not valid:
            raise ValueError(f"invalid type for tool argument {name}")
        if "enum" in properties[name] and value not in properties[name]["enum"]:
            raise ValueError(f"invalid enum for tool argument {name}")


def validate_action(action: AssistantAction, tools: tuple[TauToolView, ...]) -> None:
    schemas = {tool.name: tool.parameters for tool in tools}
    for name, arguments in action.tool_calls:
        if name not in schemas:
            raise ValueError("unknown native tau tool")
        _validate_arguments(arguments, schemas[name])


def _incoming_messages(message: object) -> Sequence[object]:
    messages = _field(message, "tool_messages")
    return messages if messages is not None else (message,)


class TauAgentAdapter:
    """Implements the native HalfDuplexAgent method shape without importing tau2 eagerly."""

    def __init__(self, tools: Sequence[object], domain_policy: str, policy: TurnPolicy):
        self.tools = tuple(TauToolView.from_native(tool) for tool in tools)
        self.domain_policy = domain_policy
        self.policy = policy
        self._turn = 0
        self._seed = 0
        self._stopped = False

    def set_seed(self, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("native orchestrator seed must be an integer")
        self._seed = seed
        if hasattr(self.policy, "set_seed"):
            self.policy.set_seed(seed)

    @classmethod
    def is_stop(cls, message: object) -> bool:
        return False

    def stop(self, message: object | None = None, state: object | None = None) -> None:
        self._stopped = True

    def get_init_state(
        self, message_history: Sequence[object] | None = None
    ) -> list[dict[str, Any]]:
        self._turn = 0
        self._stopped = False
        return [visible_message(message) for message in (message_history or ())]

    def generate_action(
        self, message: object, state: Sequence[dict[str, Any]]
    ) -> tuple[AssistantAction, list[dict[str, Any]]]:
        history = list(copy.deepcopy(state))
        history.extend(visible_message(item) for item in _incoming_messages(message))
        action = self.policy.respond(
            domain_policy=self.domain_policy,
            tools=self.tools,
            visible_history=tuple(copy.deepcopy(history)),
        )
        validate_action(action, self.tools)
        assistant = {"role": "assistant", "content": action.content}
        if action.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": f"tau3_evolution_{self._turn}_{index}",
                    "name": name,
                    "arguments": copy.deepcopy(arguments),
                }
                for index, (name, arguments) in enumerate(action.tool_calls)
            ]
        history.append(assistant)
        self._turn += 1
        return action, history

    def generate_next_message(self, message: object, state: Sequence[dict[str, Any]]):
        """Return a native AssistantMessage when executed in the pinned tau runtime."""

        action, updated = self.generate_action(message, state)
        try:
            from tau2.data_model.message import AssistantMessage, ToolCall
        except ImportError as error:  # pragma: no cover - live Python 3.12 runtime only
            raise RuntimeError("native tau2 runtime is not installed") from error
        if action.content is not None:
            native = AssistantMessage(role="assistant", content=action.content)
        else:
            native = AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=f"tau3_evolution_{self._turn - 1}_{index}",
                        name=name,
                        arguments=arguments,
                    )
                    for index, (name, arguments) in enumerate(action.tool_calls)
                ],
            )
        return native, updated


@dataclass(frozen=True)
class NativeEpisodeOutcome:
    episode_id: str
    domain: str
    task_id: str
    system_id: str
    generation_id: str | None
    reward: float
    trajectory_sha256: str
    evaluator_record_sha256: str
    tool_names: tuple[str, ...]
    simulator_seed: int
    committed_decision_chain_sha256: str

    @property
    def fully_successful(self) -> bool:
        return self.reward == 1.0
