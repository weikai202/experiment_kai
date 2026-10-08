import copy
import json
from pathlib import Path

import pytest

from tau3_evolution.manifests import (
    COMPATIBILITY_PROTOCOL,
    EVOLUTION_PROTOCOL,
    assert_comparable,
    build_manifests,
    validate_manifest,
)


def official_fixture():
    return {
        "airline": {"train": [str(i) for i in range(30)], "test": [f"t{i}" for i in range(20)]},
        "retail": {"train": [str(i) for i in range(74)], "test": [f"t{i}" for i in range(40)]},
        "telecom": {
            "train": [f"task-{i}" for i in range(74)],
            "test": [f"test-{i}" for i in range(40)],
        },
    }


def test_seed0_split_counts_and_round_composition():
    evolution, compatibility, sealed = build_manifests(official_fixture())
    assert evolution["protocol"] == EVOLUTION_PROTOCOL
    assert compatibility["protocol"] == COMPATIBILITY_PROTOCOL
    assert [len(evolution["domains"][name]["dev"]) for name in evolution["domains"]] == [6, 14, 14]
    assert [
        [len(shard) for shard in evolution["domains"][name]["rounds"]]
        for name in evolution["domains"]
    ] == [[8, 8, 8], [20, 20, 20], [20, 20, 20]]
    assert sum(len(values) for values in sealed["domains"].values()) == 100


def test_seed0_manifest_is_deterministic():
    assert build_manifests(official_fixture()) == build_manifests(copy.deepcopy(official_fixture()))


def test_test_membership_and_order_are_unchanged():
    official = official_fixture()
    _, _, sealed = build_manifests(official)
    assert sealed["domains"] == {domain: values["test"] for domain, values in official.items()}


@pytest.mark.parametrize("seed", [1, 42, -1])
def test_nonzero_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        build_manifests(official_fixture(), seed)


def test_hash_tampering_rejected():
    official = official_fixture()
    evolution, _, sealed = build_manifests(official)
    bad = copy.deepcopy(evolution)
    bad["domains"]["airline"]["dev"].pop()
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_manifest(bad, sealed, official)


def test_cross_split_leak_rejected_even_with_rehashed_document():
    from tau3_evolution.canonical import content_hash

    official = official_fixture()
    evolution, _, sealed = build_manifests(official)
    bad = copy.deepcopy(evolution)
    bad["domains"]["airline"]["dev"][0] = sealed["domains"]["airline"][0]
    bad["manifest_sha256"] = content_hash(bad)
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(bad, sealed)


def test_unlike_protocols_cannot_be_compared():
    evolution, compatibility, _ = build_manifests(official_fixture())
    with pytest.raises(ValueError, match="unlike"):
        assert_comparable(evolution, compatibility)
    assert_comparable(evolution, evolution)


def test_published_manifest_has_expected_exact_counts():
    root = Path(__file__).parents[1]
    evolution = json.loads((root / "configs/splits/evolution_seed0.json").read_text())
    sealed = json.loads((root / "configs/sealed/official_test.json").read_text())
    validate_manifest(evolution, sealed)
    assert evolution["evolution_train_task_count"] == 144
    assert evolution["dev_task_count"] == 34
    assert sealed["task_count"] == 100
