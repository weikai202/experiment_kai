from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

from .canonical import sha256_json
from .dataset import BFCL_VARIANTS, Family


@dataclass(frozen=True)
class RoundShard:
    round_index: int
    family_ids: tuple[str, ...]
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class SplitManifest:
    protocol: str
    seed: int
    source_revision: str
    train_family_ids: tuple[str, ...]
    dev_family_ids: tuple[str, ...]
    sealed_test_family_ids: tuple[str, ...]
    rounds: tuple[RoundShard, ...]
    family_provenance: tuple[tuple[str, str], ...]
    case_record_sha256: tuple[tuple[str, str], ...]
    manifest_sha256: str


def _expand(ids: Iterable[str], lookup: dict[str, Family]) -> tuple[str, ...]:
    return tuple(case.case_id for family_id in ids for case in lookup[family_id].variants)


def build_split_manifest(families: Iterable[Family], source_revision: str, seed: int = 0) -> SplitManifest:
    rows = sorted(families, key=lambda f: f.family_id.encode("utf-8"))
    if len(rows) != 200 or len({f.family_id for f in rows}) != 200:
        raise ValueError("Split input must contain exactly 200 unique families")
    if any(tuple(c.variant for c in f.variants) != BFCL_VARIANTS for f in rows):
        raise ValueError("Every family must contain all four variants in canonical order")
    random.Random(seed).shuffle(rows)
    test, dev, train = rows[:40], rows[40:80], rows[80:]
    lookup = {f.family_id: f for f in rows}
    rounds = tuple(
        RoundShard(i, tuple(f.family_id for f in train[i * 40:(i + 1) * 40]), _expand((f.family_id for f in train[i * 40:(i + 1) * 40]), lookup))
        for i in range(3)
    )
    family_provenance = tuple((f.family_id, f.provenance_sha256) for f in sorted(rows, key=lambda x: x.family_id))
    case_record_sha256 = tuple((case.case_id, case.record_sha256) for f in sorted(rows, key=lambda x: x.family_id) for case in f.variants)
    hash_core = {
        "protocol": "bfcl_evolution_family_split_v1",
        "seed": seed,
        "source_revision": source_revision,
        "train_family_ids": [f.family_id for f in train],
        "dev_family_ids": [f.family_id for f in dev],
        "sealed_test_family_ids": [f.family_id for f in test],
        "rounds": [asdict(r) for r in rounds],
        "family_provenance": family_provenance,
        "case_record_sha256": case_record_sha256,
    }
    return SplitManifest(
        protocol="bfcl_evolution_family_split_v1",
        seed=seed,
        source_revision=source_revision,
        train_family_ids=tuple(f.family_id for f in train),
        dev_family_ids=tuple(f.family_id for f in dev),
        sealed_test_family_ids=tuple(f.family_id for f in test),
        rounds=rounds,
        family_provenance=family_provenance,
        case_record_sha256=case_record_sha256,
        manifest_sha256=sha256_json(hash_core),
    )
