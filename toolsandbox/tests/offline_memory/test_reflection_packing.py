"""Lossless representation tests; all fixtures are synthetic and train-independent."""

import copy
import json
import random

import pytest

from toolsandbox_pipeline.offline.reflection_packing import (
    apply,
    canonical,
    delta,
    intern,
    pack,
    unintern,
    unpack,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.offline_memory import PolicyTrajectoryProjection


def synthetic_projection():
    """Include every projection field and a changing, cumulative visible history."""
    message = {"role": "user", "content": "查找最新消息 — café 🧪"}
    tool_result = {
        "content": "Synthetic exact result with Unicode: 北京 / café / 🧪",
        "call_id": "synthetic-call-1",
        "value": {"$ref": 2, "$literal": [["reserved", None]]},
    }
    tool = {"name": "synthetic_search", "parameters": {"query": "string"}}
    first = {
        "state_id": "sha256:" + "1" * 64,
        "agent_turn_index": 0,
        "available_tools": [tool],
        "visible_messages": [message],
        "completed_tool_calls": [],
        "verified_facts": {},
        "failed_actions": [],
        "pending_dependencies": ["synthetic-query"],
        "conversation_status": "active",
        "current_observation": None,
    }
    second = copy.deepcopy(first)
    second.update({
        "state_id": "sha256:" + "2" * 64,
        "agent_turn_index": 1,
        "current_observation": tool_result,
        "completed_tool_calls": [tool_result],
        "verified_facts": {"result": tool_result, "numeric_type": 1},
        "pending_dependencies": [],
    })
    second["visible_messages"].append({"role": "tool", "content": tool_result["content"]})
    third = copy.deepcopy(second)
    third.update({
        "state_id": "sha256:" + "3" * 64,
        "agent_turn_index": 2,
        "conversation_status": "completed",
        "failed_actions": [{"call_id": "synthetic-call-2", "error": "synthetic failure"}],
    })
    third["verified_facts"]["numeric_type"] = 1.0
    third["visible_messages"].append({"role": "assistant", "content": "原样保留结果"})
    skill = {"skill_id": "synthetic-skill", "instruction": "Preserve exact tool results", "nested": tool_result}
    action = ActionEnvelope(action={"type": "assistant_message", "content": "Synthetic response"})
    projection = PolicyTrajectoryProjection(
        trajectory_id="sha256:" + "a" * 64,
        manifest_position=0,
        visible_states=(first, second, third),
        retrieved_policy_memory=({"scope": "synthetic", "guidance": "Use exact results"},),
        retrieved_skills=(skill, skill, {"skill_id": "other", "instruction": "Other synthetic guidance"}, skill),
        proposed_actions=(action, action, action),
        final_actions=(action, action, action),
        controller_codes=("INSUFFICIENT_CONTEXT",),
        visible_tool_outcomes=(tool_result, tool_result),
        native_similarity=0.5,
        fully_successful=False,
        host_attribution="unsuccessful",
    )
    return projection.model_dump(mode="json")


@pytest.mark.parametrize("threshold", [16, 32, 48, 80])
def test_full_projection_roundtrip_preserves_every_field(threshold):
    original = synthetic_projection()
    snapshot = canonical(original)
    encoded = json.loads(canonical(intern(pack(original), minimum=threshold)))
    restored = unpack(unintern(encoded))
    assert canonical(restored) == snapshot
    assert canonical(original) == snapshot
    assert PolicyTrajectoryProjection.model_validate_json(canonical(restored)) == (
        PolicyTrajectoryProjection.model_validate_json(snapshot)
    )


def test_skill_pool_preserves_order_and_duplicate_occurrences():
    original = synthetic_projection()
    encoded = pack(original)
    assert len(encoded["retrieved_skills"]["pool"]) == 2
    assert encoded["retrieved_skills"]["indices"] == [0, 0, 1, 0]
    assert unpack(encoded)["retrieved_skills"] == original["retrieved_skills"]


def test_empty_timeline_and_retrieval_roundtrip():
    original = synthetic_projection()
    original["visible_states"] = []
    original["retrieved_skills"] = []
    assert canonical(unpack(unintern(intern(pack(original))))) == canonical(original)


@pytest.mark.parametrize("before,after", [
    (1, True),
    (1, 1.0),
    ({"a": 1}, {"b": 2}),
    ([1, 2], [1]),
    ({"x": [1]}, {"x": [1, 2]}),
    ({"x": 1}, {"x": None}),
    ({"x": {"a": 1, "b": 2}}, {"x": {"b": 3}}),
    ([1], [True, 2]),
])
def test_delta_preserves_types_removals_and_list_replacements(before, after):
    snapshot = canonical(before)
    assert canonical(apply(before, delta(before, after))) == canonical(after)
    assert canonical(before) == snapshot


def test_reserved_reference_markers_are_literal_data():
    original = {
        "$ref": 1,
        "$literal": [["a", 2]],
        "first": ["重复" * 50, {"$ref": 1}],
        "second": ["重复" * 50, {"$ref": 1}],
    }
    encoded = json.loads(canonical(intern(original, minimum=1)))
    assert canonical(unintern(encoded)) == canonical(original)


def test_trajectory_reference_pools_are_independent():
    first = synthetic_projection()
    second = synthetic_projection()
    second["visible_states"][0]["visible_messages"][0]["content"] = "Different exact message"
    first_encoded = intern(pack(first), minimum=32)
    second_encoded = intern(pack(second), minimum=32)
    assert canonical(unpack(unintern(second_encoded))) == canonical(second)
    assert canonical(unpack(unintern(first_encoded))) == canonical(first)
    restored = unpack(unintern(first_encoded))
    restored["visible_states"][0]["visible_messages"].clear()
    assert canonical(unpack(unintern(first_encoded))) == canonical(first)


def test_deterministic_random_json_roundtrips():
    rng = random.Random(31)

    def tree(depth):
        if not depth:
            return rng.choice([None, True, False, 0, 1, 1.0, "é", "重复", "long" * 20])
        return rng.choice([
            lambda: [tree(depth - 1) for _ in range(3)],
            lambda: {key: tree(depth - 1) for key in ["$ref", "x", "y"]},
            lambda: tree(0),
        ])()

    for _ in range(100):
        before, after = tree(3), tree(3)
        assert canonical(apply(before, delta(before, after))) == canonical(after)
        assert canonical(unintern(intern(before, minimum=1))) == canonical(before)
