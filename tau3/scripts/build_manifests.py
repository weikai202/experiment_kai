#!/usr/bin/env python3
"""Build exact ID-only seed-0 and sealed official-test manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from tau3_evolution.canonical import atomic_write_json
from tau3_evolution.manifests import build_manifests, read_official_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("configs"))
    args = parser.parse_args()
    official = read_official_ids(args.tau_root)
    evolution, compatibility, sealed = build_manifests(official)
    atomic_write_json(args.output_root / "splits" / "evolution_seed0.json", evolution)
    atomic_write_json(args.output_root / "splits" / "official178_compatibility.json", compatibility)
    atomic_write_json(args.output_root / "sealed" / "official_test.json", sealed)


if __name__ == "__main__":
    main()
