"""Shared training/inference serialization; no privileged environment fields."""

import copy
import hashlib
import json

POLICY_SYSTEM = """You are a ToolSandbox assistant. Use only the conversation and available tools below.
Return exactly one JSON object. To call tools use {"tool_calls":[{"name":"tool_name","arguments":{}}]}.
To speak to the user use {"response":"text"}. Never combine response and tool_calls.
You may precede the JSON with <reflection>your reasoning</reflection>. Do not report an action as completed before observing its result."""
WORLD_SYSTEM = "Predict the next agent-visible observation after an action in ToolSandbox. Output Observation: followed by JSON. Do not choose an action."


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_action(action):
    if not isinstance(action, dict):
        raise ValueError("Action must be an object")
    if (
        set(action) == {"response"}
        and isinstance(action["response"], str)
        and action["response"].strip()
    ):
        return copy.deepcopy(action)
    if (
        set(action) != {"tool_calls"}
        or not isinstance(action["tool_calls"], list)
        or not action["tool_calls"]
    ):
        raise ValueError("Expected response or a nonempty tool_calls list")
    for call in action["tool_calls"]:
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("Each call needs name and arguments")
        if (
            not isinstance(call["name"], str)
            or not call["name"].isidentifier()
            or not isinstance(call["arguments"], dict)
        ):
            raise ValueError("Invalid tool call")
    canonical(action)
    return copy.deepcopy(action)


def parse_action(text):
    text = text.strip()
    if text.startswith("<reflection>"):
        _, marker, text = text.partition("</reflection>")
        if not marker:
            raise ValueError("Unclosed reflection")
    return validate_action(json.loads(text.strip()))


def policy_messages(state):
    return [
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user", "content": canonical(state)},
    ]


def sft_records(record):
    state, expert = record["state"], record["expert_action"]
    expert_record = {
        "messages": policy_messages(state)
        + [{"role": "assistant", "content": canonical(expert)}]
    }
    iwm, sr = [], []
    # Both expert and non-expert transitions are retained; no reward filtering.
    transitions = [
        {"action": expert, "observation": record["expert_observation"]}
    ] + record["alternatives"]
    for transition in transitions:
        iwm.append(
            {
                "messages": [
                    {"role": "system", "content": WORLD_SYSTEM},
                    {
                        "role": "user",
                        "content": canonical(
                            {"state": state, "action": transition["action"]}
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "Observation:\n"
                        + canonical(transition["observation"]),
                    },
                ]
            }
        )
    for transition in record["alternatives"]:
        if "reflection" in transition:
            sr.append(
                {
                    "messages": policy_messages(state)
                    + [
                        {
                            "role": "assistant",
                            "content": "<reflection>"
                            + transition["reflection"]
                            + "</reflection>\n"
                            + canonical(expert),
                        }
                    ]
                }
            )
    return expert_record, iwm, sr


def validate_manifest(manifest):
    if set(manifest) != {"train", "dev", "test"}:
        raise ValueError("Manifest must contain train, dev, test")
    seen_ids, seen_families, seen_bases = set(), set(), set()
    suffixes = [
        "_3_distraction_tools_" + kind + "_scrambled"
        for kind in ("tool_description", "arg_type", "arg_description", "tool_name")
    ]
    suffixes += ["_3_distraction_tools", "_10_distraction_tools", "_all_tools"]
    for split, entries in manifest.items():
        families, bases = set(), set()
        for entry in entries:
            if set(entry) != {"id", "family"} or not all(
                isinstance(v, str) and v for v in entry.values()
            ):
                raise ValueError("Manifest entries require nonempty id and family")
            if entry["id"] in seen_ids:
                raise ValueError("Duplicate scenario ID across manifest")
            if entry["family"] in seen_families:
                raise ValueError("Scenario family leakage across splits")
            base = entry["id"]
            for suffix in suffixes:
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            if base in seen_bases:
                raise ValueError("Upstream augmented scenario leakage across splits")
            bases.add(base)
            seen_ids.add(entry["id"])
            families.add(entry["family"])
        seen_families.update(families)
        seen_bases.update(bases)
    return manifest
