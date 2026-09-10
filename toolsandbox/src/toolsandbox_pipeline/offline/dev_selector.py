"""The only Task 016 boundary allowed to consume restricted dev metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.offline_skill import DevScenarioSelection


@dataclass(frozen=True)
class DevFamilySelectorView:
    family_id: str
    split: str
    necessary_canonical_tools: tuple[str, ...]
    expanded_scenario_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split != "dev":
            raise ValueError("selector accepts dev metadata only")
        for values in (self.necessary_canonical_tools, self.expanded_scenario_ids):
            if not values or any(type(value) is not str or not value for value in values):
                raise ValueError("non-empty selector metadata required")
            if len(set(values)) != len(values):
                raise ValueError("duplicate selector metadata")


def _selection_key(data_seed: int, skill_id: str, scenario_id: str) -> tuple[str, bytes]:
    raw = f"{data_seed}\0{skill_id}\0{scenario_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), scenario_id.encode("utf-8")


def select_dev_scenarios(
    *,
    data_seed: int,
    skill_id: str,
    tool_dependencies: tuple[str, ...],
    dataset_manifest_sha256: str,
    families: tuple[DevFamilySelectorView, ...],
) -> DevScenarioSelection:
    if type(data_seed) is not int or data_seed != 0:
        raise ValueError("frozen data seed required")
    dependencies = set(tool_dependencies) - {"end_conversation"}
    eligible = tuple(
        family
        for family in families
        if (set(family.necessary_canonical_tools) - {"end_conversation"}) & dependencies
    )
    scenario_ids = tuple(
        scenario_id
        for family in eligible
        for scenario_id in family.expanded_scenario_ids
    )
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("dev scenario appears in multiple family views")
    selected = tuple(
        sorted(
            scenario_ids,
            key=lambda scenario_id: _selection_key(data_seed, skill_id, scenario_id),
        )[:20]
    )
    selector_payload = [
        {
            "family_id": family.family_id,
            "necessary_canonical_tools": list(family.necessary_canonical_tools),
            "expanded_scenario_ids": list(family.expanded_scenario_ids),
        }
        for family in families
    ]
    return DevScenarioSelection(
        skill_id=skill_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        selector_input_sha256=canonical_sha256(selector_payload),
        selected_scenario_ids=selected,
        selected_scenario_ids_sha256=canonical_sha256(list(selected)),
        eligible_family_count=len(eligible),
    )


__all__ = ["DevFamilySelectorView", "select_dev_scenarios"]
