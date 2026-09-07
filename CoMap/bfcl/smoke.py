"""Run a bounded real-model sample, saved in official BFCL result format."""

import argparse
import os
from pathlib import Path

from bfcl_eval.utils import load_dataset_entry
from .handler import CoMAPHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    registry = os.environ["COMAP_REGISTRY_NAME"]
    if (args.result_dir / registry.replace("/", "_")).exists():
        parser.error("Use a fresh result directory for each smoke run")
    handler = CoMAPHandler(os.environ["COMAP_POLICY_MODEL"], 0, registry)
    try:
        for category in ("simple_python", "multi_turn_base"):
            for entry in load_dataset_entry(category)[: args.count]:
                print(f"Running {entry['id']}", flush=True)
                result, metadata = handler.inference(entry, True, False)
                handler.write(
                    {"id": entry["id"], "result": result, **metadata},
                    result_dir=args.result_dir,
                )
                print(f"Saved {entry['id']}", flush=True)
    finally:
        handler.client.close()
        handler.wm_client.close()


if __name__ == "__main__":
    main()
