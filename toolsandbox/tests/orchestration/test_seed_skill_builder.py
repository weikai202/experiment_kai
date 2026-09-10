import json
import stat

import pytest

from toolsandbox_pipeline.online.tool_metadata import public_tool_inventory
from toolsandbox_pipeline.orchestration.seed_skill_builder import (
    GENERATOR_VERSION,
    build_seed_skills,
    extract_public_tool_schemas,
    publish_seed_skill_library,
    seed_skill_bytes,
)


def test_one_zero_stat_v1_skill_per_actionable_public_schema():
    skills, provenance = build_seed_skills()
    inventory = set(public_tool_inventory())
    assert len(skills) == 33
    assert {skill.tool_dependencies[0] for skill in skills} == inventory - {"end_conversation"}
    assert [skill.skill_id for skill in skills] == sorted(
        (skill.skill_id for skill in skills), key=lambda value: value.encode("utf-8")
    )
    for skill in skills:
        assert skill.version == "v1.0" and skill.status == "active"
        assert skill.online_statistics.evaluated_uses == 0
        assert skill.failure_mode_buffer == () and skill.validation is None
        prose = " ".join((
            skill.name, skill.description, skill.instruction,
            *skill.applicability.best_used_when, *skill.expected_outputs,
            *skill.success_criteria,
        ))
        assert not any(name in prose for name in inventory)
    assert provenance["generator_version"] == GENERATOR_VERSION
    assert provenance["source_types"] == ["toolsandbox_public_tool_schema"]
    assert provenance["dev_test_or_scenario_artifacts_used"] is False


def test_schema_and_output_are_byte_deterministic():
    schemas = extract_public_tool_schemas()
    assert len(schemas) == 34
    assert [item["function"]["name"] for item in schemas] == list(public_tool_inventory())
    first, first_provenance = build_seed_skills()
    second, second_provenance = build_seed_skills()
    assert first == second
    assert seed_skill_bytes(first) == seed_skill_bytes(second)
    assert first_provenance == second_provenance


def test_publish_to_two_new_roots_is_byte_identical_and_private(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = publish_seed_skill_library(first)
    second_report = publish_seed_skill_library(second)
    assert first_report == second_report
    for name in ("seed_skills.jsonl", "seed_skill_provenance.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert stat.S_IMODE((first / name).stat().st_mode) == 0o600
    assert json.loads((first / "seed_skill_provenance.json").read_bytes())["skill_count"] == 33


def test_output_must_be_new_absolute_and_metadata_is_hash_verified(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(ValueError, match="new absolute"):
        publish_seed_skill_library(target)
    with pytest.raises(ValueError, match="absolute"):
        publish_seed_skill_library("relative")

    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception):
        build_seed_skills(metadata_path=metadata)


def test_generic_semantics_do_not_depend_on_description_or_parameter_types(monkeypatch):
    original = extract_public_tool_schemas()
    baseline_skills, _ = build_seed_skills()
    changed = []
    for schema in original:
        item = json.loads(json.dumps(schema))
        item["function"].pop("description", None)
        properties = item["function"].get("parameters", {}).get("properties", {})
        for value in properties.values():
            value.pop("description", None)
            value.pop("type", None)
        changed.append(item)
    monkeypatch.setattr(
        "toolsandbox_pipeline.orchestration.seed_skill_builder.extract_public_tool_schemas",
        lambda: tuple(changed),
    )
    changed_skills, _ = build_seed_skills()
    assert len(changed_skills) == 33
    baseline_by_dependency = {skill.tool_dependencies: skill for skill in baseline_skills}
    for changed_skill in changed_skills:
        baseline = baseline_by_dependency[changed_skill.tool_dependencies]
        assert changed_skill.model_dump(exclude={"skill_id"}) == baseline.model_dump(
            exclude={"skill_id"}
        )
