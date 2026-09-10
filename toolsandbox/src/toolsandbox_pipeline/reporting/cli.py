"""Sanitized, allowlisted command line for final evaluation reporting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

from .artifacts import verify_report
from .evaluation_plan import load_final_plan


class CLIError(RuntimeError):
    """A command cannot proceed within its approved access class."""


class ReportingCommandServices(Protocol):
    """Launcher-owned capabilities; implementations enforce Task 009 access."""

    def calibrate_vanilla(self, *, train_manifest: Path, output_reference: str) -> dict: ...
    def dry_run_report(self, *, synthetic_input: Path) -> dict: ...
    def run_final_test(self, *, plan, capability_reference: str, resume: bool) -> dict: ...


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIError("invalid command arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(prog="toolsandbox-reporting")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-plan")
    validate.add_argument("--plan", type=Path, required=True)
    calibrate = subcommands.add_parser("calibrate-vanilla")
    calibrate.add_argument("--train-manifest", type=Path, required=True)
    calibrate.add_argument("--output-reference", required=True)
    dry_run = subcommands.add_parser("dry-run-report")
    dry_run.add_argument("--synthetic-input", type=Path, required=True)
    final_test = subcommands.add_parser("final-test")
    final_test.add_argument("--plan", type=Path, required=True)
    final_test.add_argument("--capability-reference", required=True)
    resume = subcommands.add_parser("resume-final-test")
    resume.add_argument("--plan", type=Path, required=True)
    resume.add_argument("--capability-reference", required=True)
    verify = subcommands.add_parser("verify-report")
    verify.add_argument("--root", type=Path, required=True)
    return parser


def _safe_output(value: object) -> object:
    """Reject launcher output that is not a sanitized reporting reference."""

    if isinstance(value, dict):
        denied_keys = (
            "secret", "api_key", "authorization", "header", "prompt",
            "raw_response", "capability",
        )
        if any(
            any(token in str(key).casefold() for token in denied_keys)
            for key in value
        ):
            raise CLIError("unsafe launcher result")
        return {str(key): _safe_output(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_output(child) for child in value]
    if isinstance(value, str):
        lowered = value.casefold()
        denied_fragments = (
            "sk-" + "proj-", "bearer ", "http://", "https://",
            "api_key", "authorization", "traceback", "/home/", "/tmp/",
            "../", "\\",
        )
        if any(fragment in lowered for fragment in denied_fragments):
            raise CLIError("unsafe launcher result")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise CLIError("unsafe launcher result")


def _emit(payload: dict) -> None:
    from toolsandbox_pipeline.reproducibility import canonical_json_bytes

    print(canonical_json_bytes(_safe_output(payload)).decode("utf-8"))


def main(
    argv: list[str] | None = None,
    *,
    services: ReportingCommandServices | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "validate-plan":
        plan = load_final_plan(arguments.plan)
        _emit({"status": "valid", "plan_sha256": plan.plan_sha256})
        return 0
    if arguments.command == "verify-report":
        manifest = verify_report(arguments.root)
        _emit({"status": "verified", "manifest_sha256": manifest.manifest_sha256})
        return 0
    if arguments.command == "calibrate-vanilla":
        if services is None:
            raise CLIError("authorized calibration launcher required")
        result = services.calibrate_vanilla(
            train_manifest=arguments.train_manifest,
            output_reference=arguments.output_reference,
        )
        _emit(result)
        return 0
    if arguments.command == "dry-run-report":
        if services is None:
            raise CLIError("synthetic report service required")
        _emit(services.dry_run_report(synthetic_input=arguments.synthetic_input))
        return 0
    if arguments.command in {"final-test", "resume-final-test"}:
        if services is None:
            raise CLIError("authorized final-test launcher required")
        plan = load_final_plan(arguments.plan)
        result = services.run_final_test(
            plan=plan,
            capability_reference=arguments.capability_reference,
            resume=arguments.command == "resume-final-test",
        )
        _emit(result)
        return 0
    raise CLIError("unsupported reporting command")


def entrypoint() -> int:
    try:
        return main()
    except CLIError:
        _emit({"status": "error", "error_code": "reporting_command_rejected"})
        return 2
    except Exception:
        _emit({"status": "error", "error_code": "reporting_command_failed"})
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
