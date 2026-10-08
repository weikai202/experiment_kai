from __future__ import annotations

import json
import sys
import types
from copy import deepcopy

import pytest

from bfcl_pipeline.canonical import read_hashed_json
from bfcl_pipeline.evolution import seed_generation
from bfcl_pipeline.manifest_cli import build_from_directory
from bfcl_pipeline.seed_library import BFCL_PINNED_REVISION, compile_seed_skill_library, generation_from_seed_library, seed_library_from_payload


def test_seed_library_is_deterministic_schema_only_and_variant_aware(records_by_variant):
    first_records = deepcopy(records_by_variant)
    for rows in first_records.values():
        for row in rows:
            row["ground_truth"] = {"private": "must-not-leak"}
            row["question"] = [[{"role": "user", "content": "private scenario text"}]]
    first = compile_seed_skill_library(first_records, BFCL_PINNED_REVISION)
    reversed_records = {variant: tuple(reversed(rows)) for variant, rows in first_records.items()}
    second = compile_seed_skill_library(reversed_records, BFCL_PINNED_REVISION)

    assert first.library_sha256 == second.library_sha256
    assert first.resource_manifest.manifest_sha256 == second.resource_manifest.manifest_sha256
    assert {row.variant for row in first.schema_provenance} == {"base", "miss_func", "miss_param", "long_context"}
    assert any(row.exposure == "delayed" for row in first.schema_provenance)
    assert all("private" not in resource.content and "scenario" not in resource.content for resource in first.resources)
    assert all(set(resource.canonical_tool_dependencies) for resource in first.resources)


def test_seed_library_binds_g000_generation_and_rejects_arbitrary_hash(records_by_variant):
    library = compile_seed_skill_library(records_by_variant, BFCL_PINNED_REVISION)
    generated = generation_from_seed_library("memory", library)
    direct = seed_generation("memory", library.resource_manifest)
    assert generated.skill_library_sha256 == library.resource_manifest.manifest_sha256
    assert direct.skill_library_sha256 == library.resource_manifest.manifest_sha256
    assert direct.accepted_skill_versions
    with pytest.raises((AttributeError, TypeError)):
        seed_generation("memory", "fabricated")  # type: ignore[arg-type]


def test_seed_compiler_requires_pinned_revision(records_by_variant):
    with pytest.raises(ValueError, match="pinned"):
        compile_seed_skill_library(records_by_variant, "unreviewed")



def test_manifest_cli_writes_validated_seed_library_artifact(records_by_variant, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for variant, rows in records_by_variant.items():
        path = data_dir / f"BFCL_v4_multi_turn_{variant}.json"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    utils = types.ModuleType("bfcl_eval.utils")
    utils.load_dataset_entry = lambda category: deepcopy(records_by_variant[category.removeprefix("multi_turn_")])
    package = types.ModuleType("bfcl_eval")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "bfcl_eval", package)
    monkeypatch.setitem(sys.modules, "bfcl_eval.utils", utils)

    manifest_path = tmp_path / "split.json"
    seed_path = tmp_path / "seed.json"
    build_from_directory(data_dir, manifest_path, BFCL_PINNED_REVISION, seed_path)
    payload = read_hashed_json(seed_path)
    library = seed_library_from_payload(payload)
    generation = generation_from_seed_library("memory", library)
    assert generation.skill_library_sha256 == library.resource_manifest.manifest_sha256
    assert manifest_path.exists() and seed_path.exists()
