from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .canonical import sha256_json, write_hashed_json
from .dataset import BFCL_VARIANTS, build_families
from .seed_library import compile_seed_skill_library, seed_library_to_payload
from .splits import build_split_manifest

DEFAULT_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_from_directory(data_dir: Path, output: Path, revision: str = DEFAULT_REVISION, seed_library_output: Path | None = None) -> str:
    from bfcl_eval.utils import load_dataset_entry

    records = {}
    for variant in BFCL_VARIANTS:
        raw = _load(data_dir / f"BFCL_v4_multi_turn_{variant}.json")
        loaded = load_dataset_entry(f"multi_turn_{variant}")
        if [x.get("id") for x in raw] != [x.get("id") for x in loaded]:
            raise ValueError("Installed BFCL data disagrees with the selected data directory")
        if any("function" not in x for x in loaded):
            raise ValueError("Official BFCL loader did not compile function schemas")
        records[variant] = loaded
    manifest = build_split_manifest(build_families(records), revision)
    payload = asdict(manifest)
    payload.pop("manifest_sha256")
    if manifest.manifest_sha256 != sha256_json(payload):
        raise ValueError("Internal manifest hash disagreement")
    if seed_library_output is not None:
        library = compile_seed_skill_library(records, revision)
        seed_library_output.parent.mkdir(parents=True, exist_ok=True)
        write_hashed_json(seed_library_output, seed_library_to_payload(library))
    return write_hashed_json(output, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the pipeline-aligned BFCL family split manifest")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-library-output", type=Path)
    parser.add_argument("--bfcl-revision", default=DEFAULT_REVISION)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = build_from_directory(args.data_dir, args.output, args.bfcl_revision, args.seed_library_output)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
