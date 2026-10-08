from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol


@dataclass(frozen=True)
class LiteralCall:
    name: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]

    def to_source(self) -> str:
        values = [repr(x) for x in self.args] + [f"{k}={v!r}" for k, v in self.kwargs]
        return f"{self.name}({', '.join(values)})"


def _decode_call(node: ast.AST) -> LiteralCall:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("Only direct function calls are allowed")
    if node.func.id.startswith("_"):
        raise ValueError("Private functions are forbidden")
    args = tuple(ast.literal_eval(value) for value in node.args)
    kwargs: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in seen:
            raise ValueError("Expanded or duplicate keyword arguments are forbidden")
        seen.add(keyword.arg)
        kwargs.append((keyword.arg, ast.literal_eval(keyword.value)))
    return LiteralCall(node.func.id, args, tuple(kwargs))


def parse_literal_calls(text: str) -> tuple[LiteralCall, ...]:
    """Parse BFCL text calls without eval, imports, attributes, or expressions."""
    text = text.strip()
    if text.startswith("```python\n") and text.endswith("```"):
        text = text[len("```python\n"):-3].strip()
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        if "(" in text or text.startswith("[") or "<tool_call>" in text:
            raise ValueError("Malformed BFCL call output") from exc
        return ()
    if isinstance(node, ast.Call):
        nodes = (node,)
    elif isinstance(node, (ast.List, ast.Tuple)):
        nodes = tuple(node.elts)
    else:
        if text.startswith(("[", "(")) or "<tool_call>" in text:
            raise ValueError("Expected a literal direct call or list of calls")
        return ()
    return tuple(_decode_call(x) for x in nodes)


def stringify_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    return str(value)


class NativeEnvironment:
    """Isolated adapter over BFCL's official executable backend classes."""

    def __init__(self, case: dict[str, Any]):
        from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES

        self.case = copy.deepcopy(case)
        self.instances: dict[str, Any] = {}
        self.method_owner: dict[str, str] = {}
        for class_name in case["involved_classes"]:
            cls = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name]), class_name)
            instance = cls()
            if class_name not in STATELESS_CLASSES:
                instance._load_scenario(
                    copy.deepcopy(case.get("initial_config", {}).get(class_name, {})),
                    long_context=case["id"].startswith("multi_turn_long_context_"),
                )
            self.instances[class_name] = instance
            for method, _ in inspect.getmembers(instance, predicate=inspect.ismethod):
                if method.startswith("_"):
                    continue
                if method in self.method_owner:
                    raise ValueError(f"Ambiguous BFCL method: {method}")
                self.method_owner[method] = class_name

    def clone(self) -> "NativeEnvironment":
        return copy.deepcopy(self)

    def execute(self, calls: Iterable[LiteralCall]) -> tuple[str, ...]:
        outputs: list[str] = []
        for call in calls:
            try:
                owner = self.method_owner[call.name]
                method = getattr(self.instances[owner], call.name)
                outputs.append(stringify_result(method(*call.args, **dict(call.kwargs))))
            except Exception as exc:  # BFCL returns tool errors as observations.
                outputs.append(f"Error during execution: {type(exc).__name__}: {exc}")
        return tuple(outputs)

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps({k: vars(v) for k, v in self.instances.items()}, ensure_ascii=False, default=str))


class TurnPolicy(Protocol):
    def respond(self, messages: tuple[dict[str, Any], ...], functions: tuple[dict[str, Any], ...]) -> str: ...


@dataclass(frozen=True)
class EpisodeResult:
    case_id: str
    predictions: tuple[tuple[tuple[str, ...], ...], ...]
    observation_batches: tuple[tuple[tuple[str, ...], ...], ...]
    terminal_state: dict[str, Any]


def _validate_exposed(calls: tuple[LiteralCall, ...], functions: list[dict[str, Any]]) -> None:
    exposed = {schema.get("name") for schema in functions if isinstance(schema, dict)}
    unknown = [call.name for call in calls if call.name not in exposed]
    if unknown:
        raise ValueError(f"BFCL action calls functions not currently exposed: {unknown}")


def run_episode(case: dict[str, Any], policy: TurnPolicy, environment: NativeEnvironment, max_steps: int = 20) -> EpisodeResult:
    """Run all BFCL turns in order while preserving one stateful environment."""
    messages: list[dict[str, Any]] = []
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    predictions: list[tuple[tuple[str, ...], ...]] = []
    observations: list[tuple[tuple[str, ...], ...]] = []
    functions = list(copy.deepcopy(case["function"]))
    missed = case.get("missed_function", {})
    for turn_index, user_batch in enumerate(case["question"]):
        if str(turn_index) in missed:
            newly_exposed = copy.deepcopy(missed[str(turn_index)])
            functions.extend(newly_exposed)
            from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING
            messages.append({"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(functions=newly_exposed)})
        else:
            messages.extend(copy.deepcopy(user_batch))
        turn_steps: list[tuple[str, ...]] = []
        turn_observations: list[tuple[str, ...]] = []
        for _ in range(max_steps):
            raw = policy.respond(tuple(messages), tuple(functions))
            calls = parse_literal_calls(raw)
            _validate_exposed(calls, functions)
            messages.append({"role": "assistant", "content": raw})
            if not calls:
                break
            call_text = tuple(call.to_source() for call in calls)
            outputs = environment.execute(calls)
            turn_steps.append(call_text)
            turn_observations.append(outputs)
            for call, output in zip(call_text, outputs):
                messages.append({"role": "tool", "name": call, "content": output})
        else:
            raise RuntimeError("BFCL per-turn step limit reached")
        predictions.append(tuple(turn_steps))
        observations.append(tuple(turn_observations))
    return EpisodeResult(case["id"], tuple(predictions), tuple(observations), environment.snapshot())


_OFFICIAL_CHECK_LOCK = threading.Lock()


def score_with_official_checker(case: dict[str, Any], predictions: list[list[list[str]]], ground_truth: list[list[list[str]]]) -> dict[str, Any]:
    """Use official BFCL checkers with the same literal-only executor on both sides.

    BFCL's checker calls a module-level execution seam. The scoped patch is
    serialized and restored in a finally block; experiment configuration fixes
    process_count=1 as an additional invariant.
    """
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_checker as module

    if len(predictions) != len(ground_truth):
        return {"valid": False, "error_type": "pipeline:turn_count_mismatch"}
    sessions: dict[tuple[Any, ...], NativeEnvironment] = {}

    def execute(call_list: list[str], initial_config: Any, involved_classes: Any, model_name: str, test_entry_id: str, long_context: bool = False, is_evaL_run: bool = False):
        key = (model_name, test_entry_id, is_evaL_run)
        env = sessions.setdefault(key, NativeEnvironment(case))
        calls = tuple(call for text in call_list for call in parse_literal_calls(text))
        return list(env.execute(calls)), env.instances

    with _OFFICIAL_CHECK_LOCK:
        original = module.execute_multi_turn_func_call
        module.execute_multi_turn_func_call = execute
        try:
            result = module.multi_turn_checker(predictions, ground_truth, case, case["id"].rsplit("_", 1)[0], uuid.uuid4().hex)
            if result.get("valid"):
                result = module.multi_turn_irrelevance_checker(predictions, ground_truth)
            return result
        finally:
            module.execute_multi_turn_func_call = original
