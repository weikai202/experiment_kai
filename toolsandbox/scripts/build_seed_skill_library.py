#!/usr/bin/env python3
"""Build one deterministic seed Skill per actionable public ToolSandbox schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toolsandbox_pipeline.orchestration.seed_skill_builder import publish_seed_skill_library


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--metadata-manifest", type=Path)
    args = parser.parse_args()
    report = publish_seed_skill_library(
        args.output_dir,
        metadata_path=args.metadata,
        metadata_manifest_path=args.metadata_manifest,
    )
    print(json.dumps({
        "status": "complete",
        "skill_count": report["skill_count"],
        "seed_skills_sha256": report["seed_skills_sha256"],
        "provenance_sha256": report["public_schema_inventory_sha256"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
