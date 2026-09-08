"""External adapter: no changes to the installed BFCL package or its globals."""
import ast
import copy
import importlib
import inspect
import json
import uuid


def parse_calls(text):
    """Accept only a Python call or list of calls with literal arguments.

    SR uses a separate <reflection> block. Never evaluate generated Python.
    Non-call prose ends a turn; malformed call-like output is an error.
    """
    text = text.strip()
    if text.startswith("<reflection>"):
        if text.count("</reflection>") != 1:
            raise ValueError("Unclosed reflection")
        text = text.split("</reflection>", 1)[1].strip()
    if text.startswith("```python\n") and text.endswith("```"):
        text = text[len("```python\n"):-3].strip()
    try:
        root = ast.parse(text, mode="eval").body
    except SyntaxError:
        if "(" in text or text.startswith("[") or "<tool_call>" in text:
            raise ValueError("Malformed function call output")
        return []
    if isinstance(root, ast.Call):
        nodes = [root]
    elif isinstance(root, ast.List):
        nodes = root.elts
    else:
        if text.startswith("["):
            raise ValueError("Expected a literal list of function calls")
        return []
    for node in nodes:
        validate_call(node)
    return [ast.unparse(node) for node in nodes]


def validate_call(node):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("Only direct function calls are supported")
    if node.func.id.startswith("_"):
        raise ValueError("Private functions are forbidden")
    for value in node.args:
        ast.literal_eval(value)
    seen = set()
    for kw in node.keywords:
        if kw.arg is None or kw.arg in seen:
            raise ValueError("Expanded or duplicate keyword arguments are forbidden")
        seen.add(kw.arg)
        ast.literal_eval(kw.value)


def action_text(calls):
    return "[" + ", ".join(calls) + "]"


def stringify(value):
    # Match official execute_multi_turn_func_call's output conversion.
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            pass
    return str(value)


class Environment:
    def __init__(self, case):
        from bfcl_eval.constants.executable_backend_config import (
            CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES,
        )
        self.instances = {}
        self.method_owner = {}
        for name in case["involved_classes"]:
            cls = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[name]), name)
            instance = cls()
            if name not in STATELESS_CLASSES:
                instance._load_scenario(copy.deepcopy(case["initial_config"].get(name, {})),
                                        long_context="long_context" in case["id"])
            self.instances[name] = instance
            for method, _ in inspect.getmembers(instance, predicate=inspect.ismethod):
                if not method.startswith("_"):
                    if method in self.method_owner:
                        raise ValueError(f"Ambiguous simulator method: {method}")
                    self.method_owner[method] = name

    def clone(self):
        return copy.deepcopy(self)

    def execute(self, calls):
        outputs = []
        for call in calls:
            try:
                node = ast.parse(call, mode="eval").body
                validate_call(node)
                name = node.func.id
                if name not in self.method_owner:
                    raise ValueError(f"Unknown simulator function: {name}")
                method = getattr(self.instances[self.method_owner[name]], name)
                value = method(*[ast.literal_eval(x) for x in node.args],
                               **{kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
                outputs.append(stringify(value))
            except Exception as error:
                outputs.append(f"Error during execution: {error}")
        return outputs

    def snapshot(self):
        # BFCL object reprs preserve its filesystem tree, including parent cycles.
        return json.loads(json.dumps({name: vars(obj) for name, obj in self.instances.items()},
                                     default=str, ensure_ascii=False))

    def probe(self, calls):
        branch = self.clone()
        return {"calls": calls, "outputs": branch.execute(calls), "post_state": branch.snapshot()}


def load_cases(category):
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
    cases = load_dataset_entry(category)
    ground_truth = {row["id"]: row["ground_truth"] for row in load_ground_truth_entry(category)}
    return cases, ground_truth


def initial_messages(case):
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
    return system_prompt_pre_processing_chat_model(copy.deepcopy(case["question"][0]),
                                                  case["function"], case["id"])


def turn_messages(case, index):
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING
    if str(index) in case.get("missed_function", {}):
        return [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
            functions=case["missed_function"][str(index)])}]
    return copy.deepcopy(case["question"][index])


def add_observations(messages, calls, outputs):
    for call, output in zip(calls, outputs):
        messages.append({"role": "tool", "name": call, "content": output})


def score_case(case, predictions, ground_truth):
    """Official state/response and missing-function/parameter irrelevance checks.

    Inject our literal-only executor for BOTH policy and reference execution.
    This avoids the upstream unrestricted eval while retaining checker semantics.
    The monkeypatch is local and restored; this function is deliberately serial.
    """
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_checker as module
    if len(predictions) != len(ground_truth):
        return {"valid": False, "error_type": "ee:turn_count_mismatch"}
    original = module.execute_multi_turn_func_call
    sessions = {}

    def execute(func_call_list, initial_config, involved_classes, model_name,
                test_entry_id, long_context=False, is_evaL_run=False):
        key = (model_name, test_entry_id, is_evaL_run)
        if key not in sessions:
            sessions[key] = Environment(case)
        env = sessions[key]
        return env.execute(func_call_list), env.instances

    module.execute_multi_turn_func_call = execute
    try:
        result = module.multi_turn_checker(predictions, ground_truth, case,
                                          case["id"].rsplit("_", 1)[0], uuid.uuid4().hex)
        if result["valid"]:
            result = module.multi_turn_irrelevance_checker(predictions, ground_truth)
        return result
    finally:
        module.execute_multi_turn_func_call = original
