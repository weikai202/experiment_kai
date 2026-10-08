from __future__ import annotations

import pytest
import random

from bfcl_pipeline.dataset import build_families
from bfcl_pipeline.seed_library import BFCL_PINNED_REVISION, compile_seed_skill_library
from bfcl_pipeline.splits import build_split_manifest


def test_family_construction_uses_structure_and_allows_variant_state(records_by_variant):
    families = build_families(records_by_variant)
    assert len(families) == 200
    assert [x.variant for x in families[0].variants] == ["base", "miss_func", "miss_param", "long_context"]
    assert len(families[0].provenance_sha256) == 64


@pytest.mark.parametrize("field,value", [
    ("path", ["Other.call"]),
    ("involved_classes", ["Other"]),
    ("excluded_function", ["other"]),
    ("function", [{"name": "other"}]),
])
def test_family_rejects_suffix_only_match(records_by_variant, field, value):
    records_by_variant["miss_param"][7][field] = value
    with pytest.raises(ValueError, match="inconsistent"):
        build_families(records_by_variant)


def test_split_is_deterministic_disjoint_and_family_preserving(families):
    one = build_split_manifest(families, "revision")
    two = build_split_manifest(reversed(families), "revision")
    assert one == two
    assert (len(one.train_family_ids), len(one.dev_family_ids), len(one.sealed_test_family_ids)) == (120, 40, 40)
    assert len(set(one.train_family_ids) | set(one.dev_family_ids) | set(one.sealed_test_family_ids)) == 200
    assert [len(x.family_ids) for x in one.rounds] == [40, 40, 40]
    assert [len(x.case_ids) for x in one.rounds] == [160, 160, 160]
    expected = sorted((f.family_id for f in families), key=lambda x: x.encode("utf-8"))
    random.Random(0).shuffle(expected)
    assert one.sealed_test_family_ids == tuple(expected[:40])
    assert one.dev_family_ids == tuple(expected[40:80])
    assert one.train_family_ids == tuple(expected[80:])
    for shard in one.rounds:
        for family_id in shard.family_ids:
            index = int(family_id.rsplit("_", 1)[1])
            assert sum(case_id.endswith(f"_{index}") for case_id in shard.case_ids) == 4


def test_pinned_official_loader_builds_real_200_family_setup():
    utils = pytest.importorskip("bfcl_eval.utils")
    records = {
        variant: utils.load_dataset_entry(f"multi_turn_{variant}")
        for variant in ("base", "miss_func", "miss_param", "long_context")
    }
    assert all(len(rows) == 200 for rows in records.values())
    families = build_families(records)
    manifest = build_split_manifest(families, BFCL_PINNED_REVISION)
    library = compile_seed_skill_library(records, BFCL_PINNED_REVISION)
    assert len(families) == 200
    assert (len(manifest.sealed_test_family_ids), len(manifest.dev_family_ids), len(manifest.train_family_ids)) == (40, 40, 120)
    assert library.resources and library.resource_manifest.generation_id == "g000"
