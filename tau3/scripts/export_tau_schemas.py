#!/usr/bin/env python3
"""Run under pinned tau Python >=3.12 to export only public Agent tool schemas."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "17e07b1da2bbc0cadfddeea36412686e0604127b"
DOMAINS = ("airline", "retail", "telecom")
TOOL_CLASSES = {
    "airline": ("tau2.domains.airline.tools", "AirlineTools"),
    "retail": ("tau2.domains.retail.tools", "RetailTools"),
    "telecom": ("tau2.domains.telecom.tools", "TelecomTools"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.tau_root.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("tau checkout is not at the pinned commit")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty.strip():
        raise ValueError("pinned tau checkout has tracked modifications")
    sys.path.insert(0, str(root / "src"))
    sources = {}
    for domain in DOMAINS:
        module_name, class_name = TOOL_CLASSES[domain]
        toolkit = getattr(importlib.import_module(module_name), class_name)(None)
        source_path = Path("src") / "tau2" / "domains" / domain / "tools.py"
        raw = (root / source_path).read_bytes()
        sources[domain] = {
            "source_path": source_path.as_posix(),
            "source_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "schemas": [tool.openai_schema for tool in toolkit.get_tools().values()],
        }
    payload = {"source_commit": commit, "domain_sources": sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
