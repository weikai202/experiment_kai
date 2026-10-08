#!/usr/bin/env python3
"""Compile a schema export produced in the pinned native tau environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from tau3_evolution.canonical import atomic_write_json, read_json
from tau3_evolution.seed_library import compile_seed_library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exported = read_json(args.schema_export)
    if set(exported) != {"source_commit", "domain_sources"}:
        raise ValueError("native schema export has missing or extra fields")
    document = compile_seed_library(
        exported["domain_sources"], source_commit=exported["source_commit"]
    )
    atomic_write_json(args.output, document)
    print(document["library_sha256"])


if __name__ == "__main__":
    main()
