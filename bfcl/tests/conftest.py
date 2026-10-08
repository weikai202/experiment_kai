from __future__ import annotations

import pytest

from bfcl_pipeline.dataset import BFCL_VARIANTS, build_families


def record(variant: str, index: int) -> dict:
    functions = [{"name": "read", "parameters": {"required": ["key"]}}, {"name": "write", "parameters": {"required": ["key", "value"]}}]
    missed = {}
    if variant == "miss_func":
        functions = functions[:1]
        missed = {"0": [{"name": "write", "parameters": {"required": ["key", "value"]}}]}
    elif variant == "miss_param":
        functions[1] = {"name": "write", "parameters": {"required": ["key"]}}
    elif variant == "long_context":
        functions[0] = {**functions[0], "description": "long context"}
    return {
        "id": f"multi_turn_{variant}_{index}",
        "question": [[{"role": "user", "content": f"turn {index}"}]],
        "initial_config": {"State": {"variant": variant}},
        "path": ["State.read", "State.write"],
        "involved_classes": ["State"],
        "excluded_function": ["delete"],
        "function": functions,
        "missed_function": missed,
    }


@pytest.fixture
def records_by_variant():
    return {variant: [record(variant, i) for i in range(200)] for variant in BFCL_VARIANTS}


@pytest.fixture
def families(records_by_variant):
    return build_families(records_by_variant)
