from toolsandbox_pipeline.offline.dev_selector import (
    DevFamilySelectorView,
    select_dev_scenarios,
)


DIGEST = "sha256:" + "b" * 64


def test_dev_selector_is_relevant_deterministic_and_bounded():
    families = tuple(
        DevFamilySelectorView(
            family_id=f"family-{index}",
            split="dev",
            necessary_canonical_tools=("search_stock",),
            expanded_scenario_ids=tuple(
                f"scenario-{index}-{variant}" for variant in range(8)
            ),
        )
        for index in range(4)
    )
    selected = select_dev_scenarios(
        data_seed=0,
        skill_id="skill",
        tool_dependencies=("search_stock",),
        dataset_manifest_sha256=DIGEST,
        families=families,
    )
    assert len(selected.selected_scenario_ids) == 20
    assert selected == select_dev_scenarios(
        data_seed=0,
        skill_id="skill",
        tool_dependencies=("search_stock",),
        dataset_manifest_sha256=DIGEST,
        families=families,
    )
