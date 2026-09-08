"""Download the public expert corpus, validating the exact known content hash."""
import argparse
import urllib.request
from pathlib import Path

from .io import digest

PREFIX = "https://raw.githubusercontent.com/HuanzhiMao/BFCL-Result/main/2025-12-16"
FILES = [
    ("result", "result", "opus_base_result.jsonl", "5f9571d51a2d228ea39fc231edc8fcefb3592b7db12f676c89c1ea98c77ead86"),
    ("score", "score", "opus_base_score.jsonl", "5589fd0dcc93ebfe15187bdeff934af59dc6c91e0726eaeeeae3336b07ff74eb"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/source")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for directory, suffix, filename, expected in FILES:
        path = output / filename
        if not path.exists():
            url = f"{PREFIX}/{directory}/claude-opus-4-5-20251101-FC/multi_turn/BFCL_v4_multi_turn_base_{suffix}.json"
            temporary = path.with_suffix(".download")
            with urllib.request.urlopen(url, timeout=60) as response:
                temporary.write_bytes(response.read())
            if digest(temporary) != expected:
                raise ValueError(f"Upstream content changed: {url}")
            temporary.replace(path)
        if digest(path) != expected:
            raise ValueError(f"Unexpected expert file hash: {path}")
        print(f"Verified {path}")


if __name__ == "__main__":
    main()
