import copy

import pytest

from comap_tau3 import TAU_COMMIT
from comap_tau3.splits import COUNTS, digest, validate_manifest


def fixture_manifest():
    official, domains = {}, {}
    for name, (train, test, shard) in COUNTS.items():
        ids = [str(i) for i in range(train)]
        heldout = [str(train + i) for i in range(test)]
        official[name] = {"train": ids, "test": heldout}
        domains[name] = {
            "rounds": [ids[i * shard : (i + 1) * shard] for i in range(3)],
            "dev": ids[3 * shard :],
            "test": heldout,
        }
    return {
        "schema_version": 1,
        "tau_commit": TAU_COMMIT,
        "official_sha256": digest(official),
        "domains": domains,
    }, official


def test_counts_and_no_overlap():
    manifest, official = fixture_manifest()
    validate_manifest(manifest, official)
    assert [len(v["dev"]) for v in manifest["domains"].values()] == [6, 14, 14]
    for mutation in ("overlap", "test_leak", "missing"):
        bad = copy.deepcopy(manifest)
        split = bad["domains"]["retail"]
        if mutation == "overlap":
            split["rounds"][1][0] = split["rounds"][0][0]
        elif mutation == "test_leak":
            split["rounds"][0][0] = split["test"][0]
        else:
            split["dev"].pop()
        with pytest.raises(ValueError):
            validate_manifest(bad, official)
