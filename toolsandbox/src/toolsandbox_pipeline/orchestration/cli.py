"""Fixed Task017 command allowlist with sanitized JSON-only results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from toolsandbox_pipeline.orchestration.checkpoint_registry import (
    query_checkpoint_registry,
)
from toolsandbox_pipeline.orchestration.run_manifest import (
    load_run_manifest,
    load_train_smoke_config,
    validate_component_files,
    validate_train_smoke,
    verify_schema_file,
)
from toolsandbox_pipeline.orchestration.seed_skill_builder import (
    publish_seed_skill_library,
)


COMMANDS = (
    "validate-config",
    "validate-seeds",
    "build-seed-skills",
    "build-generation-zero",
    "query-checkpoints",
    "calibrate-online",
    "calibrate-offline-memory",
    "calibrate-offline-skill",
    "train-smoke",
    "dev-mini-bench",
    "train",
    "resume",
)

EXTERNAL_COMMANDS = (
    "build-generation-zero",
    "calibrate-online",
    "calibrate-offline-memory",
    "calibrate-offline-skill",
    "train-smoke",
    "dev-mini-bench",
    "train",
    "resume",
)


class CLIError(RuntimeError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIError("invalid_arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m toolsandbox_pipeline.orchestration.cli")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate-config")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--schema", type=Path, required=True)

    seeds = subcommands.add_parser("validate-seeds")
    seeds.add_argument("--manifest", type=Path, required=True)

    build_seed = subcommands.add_parser("build-seed-skills")
    build_seed.add_argument("--output-dir", type=Path, required=True)
    build_seed.add_argument("--metadata", type=Path)
    build_seed.add_argument("--metadata-manifest", type=Path)

    query = subcommands.add_parser("query-checkpoints")
    query.add_argument("--registry", type=Path, required=True)

    for name in (
        "build-generation-zero",
        "calibrate-online",
        "calibrate-offline-memory",
        "calibrate-offline-skill",
        "dev-mini-bench",
        "train",
    ):
        command = subcommands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)

    smoke = subcommands.add_parser("train-smoke")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--smoke-config", type=Path, required=True)

    resume = subcommands.add_parser("resume")
    resume.add_argument("--manifest", type=Path, required=True)
    resume.add_argument("--run-root", type=Path, required=True)
    return parser


def _safe(value: object) -> object:
    denied = {
        "secret",
        "api_key",
        "authorization",
        "headers",
        "prompt_text",
        "raw_response",
        "sk-",
        "bearer ",
        "http://",
        "https://",
    }
    if isinstance(value, dict):
        if any(any(token in str(key).casefold() for token in denied) for key in value):
            raise CLIError("unsafe_result")
        return {str(key): _safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    if isinstance(value, str):
        folded = value.casefold()
        if any(token in folded for token in denied):
            raise CLIError("unsafe_result")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise CLIError("unsafe_result")


def dispatch(
    args: argparse.Namespace,
    *,
    operations: Mapping[str, Callable[[argparse.Namespace], dict[str, object]]] | None = None,
) -> dict[str, object]:
    if operations is not None:
        unknown = set(operations).difference(EXTERNAL_COMMANDS)
        if unknown or any(not callable(operation) for operation in operations.values()):
            raise CLIError("invalid_coordinator_runtime")
    if args.command == "validate-config":
        manifest = load_run_manifest(args.manifest)
        verify_schema_file(args.schema)
        validate_component_files(manifest)
        return {"status": "complete", "manifest_sha256": manifest.manifest_sha256}
    if args.command == "validate-seeds":
        manifest = load_run_manifest(args.manifest)
        validate_component_files(manifest)
        return {"status": "complete", "manifest_sha256": manifest.manifest_sha256}
    if args.command == "build-seed-skills":
        report = publish_seed_skill_library(
            args.output_dir,
            metadata_path=args.metadata,
            metadata_manifest_path=args.metadata_manifest,
        )
        return {
            "status": "complete",
            "skill_count": report["skill_count"],
            "seed_skills_sha256": report["seed_skills_sha256"],
            "provenance_sha256": report["public_schema_inventory_sha256"],
        }
    if args.command == "query-checkpoints":
        registry = query_checkpoint_registry(args.registry)
        return {"status": "complete", "registry": registry.model_dump(mode="json")}

    manifest = load_run_manifest(
        args.manifest,
        run_root_state="existing" if args.command == "resume" else "new",
    )
    if args.command == "resume" and args.run_root != Path(manifest.run_root):
        raise CLIError("resume_run_root_mismatch")
    if args.command == "train-smoke":
        smoke, smoke_hash = load_train_smoke_config(args.smoke_config)
        validate_train_smoke(manifest, smoke)
    else:
        smoke_hash = None
    operation = (operations or {}).get(args.command)
    if operation is None:
        raise CLIError("command_requires_coordinator_runtime")
    result = operation(args)
    if type(result) is not dict:
        raise CLIError("invalid_operation_result")
    result = dict(result)
    result.setdefault("manifest_sha256", manifest.manifest_sha256)
    if smoke_hash is not None:
        result.setdefault("smoke_config_sha256", smoke_hash)
    return _safe(result)  # type: ignore[return-value]


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    operations: Mapping[str, Callable[[argparse.Namespace], dict[str, object]]] | None = None,
) -> int:
    try:
        args = build_parser().parse_args(argv)
        result = _safe(dispatch(args, operations=operations))
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except CLIError as error:
        print(json.dumps({"status": "error", "error_class": str(error)}, separators=(",", ":")))
        return 2
    except Exception as error:
        print(json.dumps({"status": "error", "error_class": type(error).__name__}, separators=(",", ":")))
        return 2


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLIError", "COMMANDS", "build_parser", "dispatch", "run_cli"]
