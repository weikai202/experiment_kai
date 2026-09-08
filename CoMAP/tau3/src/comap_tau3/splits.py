"""ID-only split preparation. Never uses task content to construct a split."""

import hashlib
import json
import subprocess
from pathlib import Path

from . import TAU_COMMIT

COUNTS = {"airline": (30, 20, 8), "retail": (74, 40, 20), "telecom": (74, 40, 20)}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_repo(root):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != TAU_COMMIT:
        raise ValueError(f"Expected tau commit {TAU_COMMIT}; got {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty.strip():
        raise ValueError("Pinned tau source has tracked modifications")
    return root


def official_splits(root):
    root = verify_repo(root)
    result = {}
    for domain, (train_n, test_n, _) in COUNTS.items():
        path = root / "data/tau2/domains" / domain / "split_tasks.json"
        split = json.loads(path.read_text())
        train, test = split["train"], split["test"]
        if len(train) != train_n or len(test) != test_n:
            raise ValueError(f"Unexpected official counts for {domain}")
        if len(set(train)) != train_n or len(set(test)) != test_n or set(train) & set(test):
            raise ValueError(f"Invalid official partition for {domain}")
        result[domain] = {"train": train, "test": test}
    return result


def make_manifest(root, seed, protocol="official_full_train"):
    if protocol not in {"official_full_train", "heldout_dev"}:
        raise ValueError("Unknown split protocol")
    official = official_splits(root)
    domains = {}
    for domain, (_, _, shard_n) in COUNTS.items():
        if protocol == "official_full_train":
            ordered = official[domain]["train"]
            sizes = [len(ordered) // 3 + (i < len(ordered) % 3) for i in range(3)]
            offsets = [0, sizes[0], sizes[0] + sizes[1]]
            domains[domain] = {
                "rounds": [ordered[start : start + size] for start, size in zip(offsets, sizes)],
                "dev": [],
                "test": official[domain]["test"],
            }
            continue
        # Hash ordering avoids dependency on Python's shuffle implementation.
        ordered = sorted(official[domain]["train"], key=lambda task: digest([seed, domain, task]))
        domains[domain] = {
            "rounds": [ordered[i * shard_n : (i + 1) * shard_n] for i in range(3)],
            "dev": ordered[3 * shard_n :],
            "test": official[domain]["test"],
        }
    manifest = {
        "protocol": protocol,
        "schema_version": 1,
        "tau_commit": TAU_COMMIT,
        "provenance": "candidate_generated_not_existing_pipeline_split",
        "seed": seed,
        "ordering": "sha256-json-seed-domain-task",
        "rounding": "prioritize three equal shards: dev counts 6/14/14",
        "official_sha256": digest(official),
        "domains": domains,
    }
    if protocol == "official_full_train":
        manifest.update(
            provenance="official_train_test_with_baseline_round_partition",
            seed=None,
            ordering="official-train-list-order-contiguous-balanced-rounds",
            rounding="all 178 training tasks; no dev; round sizes 10/10/10 and 25/25/24",
        )
    validate_manifest(manifest, official)
    return manifest


def validate_manifest(manifest, official):
    if manifest.get("schema_version") != 1 or manifest.get("tau_commit") != TAU_COMMIT:
        raise ValueError("Manifest schema/commit mismatch")
    if manifest.get("official_sha256") != digest(official):
        raise ValueError("Official split fingerprint mismatch")
    if set(manifest["domains"]) != set(COUNTS):
        raise ValueError("Manifest must contain exactly airline, retail, telecom")
    protocol = manifest.get("protocol", "heldout_dev")
    if protocol not in {"official_full_train", "heldout_dev"}:
        raise ValueError("Unknown split protocol")
    for domain, (train_n, test_n, shard_n) in COUNTS.items():
        split = manifest["domains"][domain]
        rounds, dev, test = split["rounds"], split["dev"], split["test"]
        expected = (
            [train_n // 3 + (i < train_n % 3) for i in range(3)]
            if protocol == "official_full_train"
            else [shard_n] * 3
        )
        if protocol == "official_full_train" and dev:
            raise ValueError("Full-training protocol cannot reserve dev tasks")
        if [len(shard) for shard in rounds] != expected:
            raise ValueError(f"Wrong round sizes: {domain}")
        train = [task for shard in rounds for task in shard] + dev
        if any(not isinstance(task, str) for task in train + test):
            raise ValueError("Task IDs must be strings")
        if len(train) != len(set(train)) or set(train) != set(official[domain]["train"]):
            raise ValueError(f"Overlapping or incomplete training partition: {domain}")
        if len(test) != test_n or set(test) != set(official[domain]["test"]):
            raise ValueError(f"Wrong test partition: {domain}")
        if set(train) & set(test):
            raise ValueError(f"Train/test leakage: {domain}")


def save_new_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
