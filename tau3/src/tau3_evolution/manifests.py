"""Deterministic ID-only tau3 split manifests and protocol validation."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any

from . import DOMAINS, TAU_COMMIT
from .canonical import canonical_sha256, content_hash, read_json, verify_content_hash

OFFICIAL_COUNTS = {
    "airline": {"train": 30, "dev": 6, "round": 8, "test": 20},
    "retail": {"train": 74, "dev": 14, "round": 20, "test": 40},
    "telecom": {"train": 74, "dev": 14, "round": 20, "test": 40},
}
EVOLUTION_PROTOCOL = "evolution144_dev34_seed0_v1"
COMPATIBILITY_PROTOCOL = "official178_no_dev_v1"


def verify_tau_checkout(root: str | Path) -> Path:
    resolved = Path(root).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != TAU_COMMIT:
        raise ValueError(f"expected tau commit {TAU_COMMIT}; got {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(resolved), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty.strip():
        raise ValueError("pinned tau source has tracked modifications")
    return resolved


def read_official_ids(
    root: str | Path, *, verify_git: bool = True
) -> dict[str, dict[str, list[str]]]:
    source = verify_tau_checkout(root) if verify_git else Path(root).resolve()
    result: dict[str, dict[str, list[str]]] = {}
    for domain in DOMAINS:
        document = read_json(source / "data" / "tau2" / "domains" / domain / "split_tasks.json")
        if not {"train", "test"} <= set(document):
            raise ValueError(f"missing official train/test fields for {domain}")
        train = document["train"]
        test = document["test"]
        expected = OFFICIAL_COUNTS[domain]
        if len(train) != expected["train"] or len(test) != expected["test"]:
            raise ValueError(f"unexpected official task counts for {domain}")
        if any(type(task_id) is not str or not task_id for task_id in train + test):
            raise ValueError("official task IDs must be non-empty strings")
        if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
            raise ValueError(f"invalid official split for {domain}")
        result[domain] = {"train": list(train), "test": list(test)}
    return result


def _finalize(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result["manifest_sha256"] = content_hash(result)
    return result


def build_manifests(
    official: dict[str, dict[str, list[str]]], seed: int = 0
) -> tuple[dict, dict, dict]:
    if type(seed) is not int or seed != 0:
        raise ValueError("the pipeline-aligned protocol fixes data seed to 0")
    _validate_official_structure(official)
    official_sha = canonical_sha256(official)
    test_domains = {domain: list(official[domain]["test"]) for domain in DOMAINS}
    sealed = _finalize(
        {
            "schema_version": 1,
            "kind": "sealed_official_test",
            "tau_commit": TAU_COMMIT,
            "official_split_sha256": official_sha,
            "domains": test_domains,
            "task_count": 100,
        }
    )

    evolution_domains: dict[str, dict[str, list]] = {}
    compatibility_domains: dict[str, dict[str, list]] = {}
    for domain in DOMAINS:
        counts = OFFICIAL_COUNTS[domain]
        ordered = sorted(official[domain]["train"], key=lambda value: value.encode("utf-8"))
        random.Random(seed).shuffle(ordered)
        evolution_count = 3 * counts["round"]
        evolution_domains[domain] = {
            "rounds": [
                ordered[index * counts["round"] : (index + 1) * counts["round"]]
                for index in range(3)
            ],
            "dev": ordered[evolution_count:],
        }
        official_order = official[domain]["train"]
        base, remainder = divmod(len(official_order), 3)
        sizes = [base + (index < remainder) for index in range(3)]
        starts = [0, sizes[0], sizes[0] + sizes[1]]
        compatibility_domains[domain] = {
            "rounds": [
                official_order[start : start + size]
                for start, size in zip(starts, sizes, strict=True)
            ],
            "dev": [],
        }

    common = {
        "schema_version": 1,
        "tau_commit": TAU_COMMIT,
        "official_split_sha256": official_sha,
        "sealed_test_manifest_sha256": sealed["manifest_sha256"],
        "test_task_count": 100,
        "formal_domains": list(DOMAINS),
    }
    evolution = _finalize(
        {
            **common,
            "protocol": EVOLUTION_PROTOCOL,
            "seed": 0,
            "ordering": "utf8-sort-then-python310-random-Random(0)-shuffle-per-domain",
            "domains": evolution_domains,
            "evolution_train_task_count": 144,
            "dev_task_count": 34,
        }
    )
    compatibility = _finalize(
        {
            **common,
            "protocol": COMPATIBILITY_PROTOCOL,
            "seed": None,
            "ordering": "official-train-list-order-contiguous-balanced-rounds",
            "domains": compatibility_domains,
            "evolution_train_task_count": 178,
            "dev_task_count": 0,
        }
    )
    validate_manifest(evolution, sealed, official)
    validate_manifest(compatibility, sealed, official)
    return evolution, compatibility, sealed


def _validate_official_structure(official: dict[str, dict[str, list[str]]]) -> None:
    if tuple(official) != DOMAINS:
        raise ValueError("official split must contain ordered airline, retail, telecom domains")
    for domain in DOMAINS:
        values = official[domain]
        if set(values) != {"train", "test"}:
            raise ValueError("official domain split requires train and test")
        counts = OFFICIAL_COUNTS[domain]
        if len(values["train"]) != counts["train"] or len(values["test"]) != counts["test"]:
            raise ValueError("official count mismatch")
        combined = values["train"] + values["test"]
        if any(type(value) is not str or not value for value in combined):
            raise ValueError("invalid task ID")
        if len(set(combined)) != len(combined):
            raise ValueError("official train/test overlap")


def validate_manifest(manifest: dict, sealed: dict, official: dict | None = None) -> None:
    verify_content_hash(manifest)
    verify_content_hash(sealed)
    if manifest.get("schema_version") != 1 or manifest.get("tau_commit") != TAU_COMMIT:
        raise ValueError("manifest schema or tau commit mismatch")
    if sealed.get("kind") != "sealed_official_test" or sealed.get("task_count") != 100:
        raise ValueError("invalid sealed test manifest")
    if manifest.get("sealed_test_manifest_sha256") != sealed["manifest_sha256"]:
        raise ValueError("main/sealed manifest binding mismatch")
    if manifest.get("official_split_sha256") != sealed.get("official_split_sha256"):
        raise ValueError("main/sealed official split fingerprint mismatch")
    if tuple(manifest.get("domains", {})) != DOMAINS:
        raise ValueError("manifest domains must be ordered airline, retail, telecom")
    protocol = manifest.get("protocol")
    if protocol not in {EVOLUTION_PROTOCOL, COMPATIBILITY_PROTOCOL}:
        raise ValueError("unknown experiment protocol")
    seen_train: set[tuple[str, str]] = set()
    for domain in DOMAINS:
        split = manifest["domains"][domain]
        if set(split) != {"rounds", "dev"} or len(split["rounds"]) != 3:
            raise ValueError("each domain requires three rounds and dev")
        expected = OFFICIAL_COUNTS[domain]
        round_size = expected["round"] if protocol == EVOLUTION_PROTOCOL else None
        if protocol == EVOLUTION_PROTOCOL:
            if [len(value) for value in split["rounds"]] != [round_size] * 3:
                raise ValueError("wrong evolution round size")
            if len(split["dev"]) != expected["dev"]:
                raise ValueError("wrong dev size")
        else:
            target = expected["train"]
            base, remainder = divmod(target, 3)
            if [len(value) for value in split["rounds"]] != [
                base + (index < remainder) for index in range(3)
            ]:
                raise ValueError("wrong official178 compatibility round size")
            if split["dev"]:
                raise ValueError("official178 compatibility protocol has no dev")
        values = [task for shard in split["rounds"] for task in shard] + split["dev"]
        if any(type(task) is not str or not task for task in values):
            raise ValueError("invalid task ID")
        if len(values) != len(set(values)):
            raise ValueError("task duplicated across train rounds/dev")
        seen_train.update((domain, task) for task in values)
    if tuple(sealed.get("domains", {})) != DOMAINS:
        raise ValueError("sealed test domains must be ordered airline, retail, telecom")
    if {domain: len(sealed["domains"][domain]) for domain in DOMAINS} != {
        "airline": 20,
        "retail": 40,
        "telecom": 40,
    }:
        raise ValueError("sealed official test domain counts must be 20/40/40")
    test_values = [(domain, task) for domain in DOMAINS for task in sealed["domains"][domain]]
    if len(test_values) != 100 or len(set(test_values)) != 100:
        raise ValueError("sealed test must contain exactly 100 unique domain/task pairs")
    if seen_train & set(test_values):
        raise ValueError("train/dev/test leakage")
    if official is not None:
        _validate_official_structure(official)
        if manifest["official_split_sha256"] != canonical_sha256(official):
            raise ValueError("official split fingerprint mismatch")
        for domain in DOMAINS:
            selected = [
                task for shard in manifest["domains"][domain]["rounds"] for task in shard
            ] + manifest["domains"][domain]["dev"]
            if set(selected) != set(official[domain]["train"]):
                raise ValueError("manifest does not partition official training tasks")
            if sealed["domains"][domain] != official[domain]["test"]:
                raise ValueError("sealed manifest changed official test membership/order")


def assert_comparable(left: dict, right: dict) -> None:
    fields = (
        "protocol",
        "manifest_sha256",
        "official_split_sha256",
        "sealed_test_manifest_sha256",
    )
    if any(left.get(field) != right.get(field) for field in fields):
        raise ValueError("results use unlike manifests/protocols and must not be compared")
