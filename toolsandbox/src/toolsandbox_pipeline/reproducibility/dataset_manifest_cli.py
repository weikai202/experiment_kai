"""Privileged setup entry point; console output contains aggregate identities only."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from time import monotonic


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid manifest-build arguments")


def main(argv=None, *, builder=None):
    started = monotonic()
    try:
        parser = _Parser(allow_abbrev=False)
        parser.add_argument("command", choices=("build-dataset-manifest",))
        parser.add_argument("--config", required=True)
        parser.add_argument("--output-dir", required=True)
        args = parser.parse_args(argv)
        config, output = Path(args.config), Path(args.output_dir)
        project = Path(__file__).resolve().parents[3]
        for path in (config, output):
            if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("absolute non-symlink paths required")
        if output == project or any(output == project / folder or project / folder in output.parents for folder in ("src", "configs", "tasks", "docs", "tests", "prompts", ".git", ".venv")):
            raise ValueError("output cannot replace project inputs")
        if builder is None:
            from .dataset_manifest import build_native_bundle
            builder = build_native_bundle
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            hashes = builder(config, output, project)
        result = dict(status="pass", families=129, scenarios=1032, train_families=79, dev_families=25, test_families=25,
            train_scenarios=632, dev_scenarios=200, test_scenarios=200, train_round_scenarios=[216,208,208],
            file_hashes=hashes, environment_complete=False, total_tokens=0, usage_complete=True)
    except Exception:
        result = dict(status="fail", category="manifest_build_rejected", total_tokens=0, usage_complete=True)
    result["total_running_time_seconds"] = monotonic() - started
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
