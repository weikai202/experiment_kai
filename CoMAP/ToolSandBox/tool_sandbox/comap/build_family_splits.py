# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Build CoMAP family-level ToolSandbox splits.

The audited CoMAP setup uses 129 scenario families with eight augmentation
variants per family. With seed 0, families are split into 79/25/25 train/dev/test
families, and train is further split into three evolution rounds of 27/26/26
families, corresponding to 216/208/208 scenarios.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios import named_scenarios

AUGMENTATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"_(?:0|3|10)_distraction_tools"),
    re.compile(r"_all_tools"),
    re.compile(r"_tool_name_scrambled"),
    re.compile(r"_tool_description_scrambled"),
    re.compile(r"_arg_description_scrambled"),
    re.compile(r"_arg_name_scrambled"),
    re.compile(r"_arg_type_scrambled"),
)


def scenario_family_name(scenario_name: str) -> str:
    """Strip ToolSandbox augmentation suffixes to recover the scenario family."""
    previous = None
    family = scenario_name
    while previous != family:
        previous = family
        for pattern in AUGMENTATION_PATTERNS:
            family = pattern.sub("", family)
    return family


def build_splits(seed: int) -> dict[str, Any]:
    scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
    family_to_scenarios: dict[str, list[str]] = {}
    for scenario_name in sorted(scenarios):
        family_to_scenarios.setdefault(scenario_family_name(scenario_name), []).append(
            scenario_name
        )

    families = sorted(family_to_scenarios)
    rng = random.Random(seed)
    rng.shuffle(families)

    train_families = families[:79]
    dev_families = families[79:104]
    test_families = families[104:129]
    train_round_families = {
        "round_0": train_families[:27],
        "round_1": train_families[27:53],
        "round_2": train_families[53:79],
    }

    def expand(selected_families: list[str]) -> list[str]:
        scenario_names: list[str] = []
        for family in selected_families:
            scenario_names.extend(sorted(family_to_scenarios[family]))
        return scenario_names

    return {
        "seed": seed,
        "family_to_scenarios": family_to_scenarios,
        "splits": {
            "train": {"families": train_families, "scenarios": expand(train_families)},
            "dev": {"families": dev_families, "scenarios": expand(dev_families)},
            "test": {"families": test_families, "scenarios": expand(test_families)},
        },
        "train_rounds": {
            name: {"families": round_families, "scenarios": expand(round_families)}
            for name, round_families in train_round_families.items()
        },
    }


def validate_expected_counts(split_data: dict[str, Any]) -> None:
    family_count = len(split_data["family_to_scenarios"])
    scenario_count = sum(
        len(scenarios) for scenarios in split_data["family_to_scenarios"].values()
    )
    expected = {
        "families": 129,
        "scenarios": 1032,
        "train_families": 79,
        "dev_families": 25,
        "test_families": 25,
        "train_scenarios": 632,
        "dev_scenarios": 200,
        "test_scenarios": 200,
        "round_0_scenarios": 216,
        "round_1_scenarios": 208,
        "round_2_scenarios": 208,
    }
    actual = {
        "families": family_count,
        "scenarios": scenario_count,
        "train_families": len(split_data["splits"]["train"]["families"]),
        "dev_families": len(split_data["splits"]["dev"]["families"]),
        "test_families": len(split_data["splits"]["test"]["families"]),
        "train_scenarios": len(split_data["splits"]["train"]["scenarios"]),
        "dev_scenarios": len(split_data["splits"]["dev"]["scenarios"]),
        "test_scenarios": len(split_data["splits"]["test"]["scenarios"]),
        "round_0_scenarios": len(split_data["train_rounds"]["round_0"]["scenarios"]),
        "round_1_scenarios": len(split_data["train_rounds"]["round_1"]["scenarios"]),
        "round_2_scenarios": len(split_data["train_rounds"]["round_2"]["scenarios"]),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatches:
        raise ValueError(f"Split counts do not match audited CoMAP setup: {mismatches}")


def write_name_lists(split_data: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in split_data["splits"].items():
        (output_dir / f"{split_name}.txt").write_text(
            "\n".join(split["scenarios"]) + "\n", encoding="utf-8"
        )
    for round_name, split in split_data["train_rounds"].items():
        (output_dir / f"train_{round_name}.txt").write_text(
            "\n".join(split["scenarios"]) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/comap_splits"))
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()

    split_data = build_splits(seed=args.seed)
    if not args.allow_count_mismatch:
        validate_expected_counts(split_data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits.json").write_text(
        json.dumps(split_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_name_lists(split_data, args.output_dir)
    print(f"Wrote CoMAP ToolSandbox splits to {args.output_dir}")


if __name__ == "__main__":
    main()
