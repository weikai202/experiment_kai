import json

import pytest

from toolsandbox_pipeline.reporting.cli import CLIError, build_parser, main


class Services:
    def calibrate_vanilla(self, *, train_manifest, output_reference):
        if train_manifest.name != "train_manifest.json":
            raise CLIError("train-only access gate rejected")
        return {"status": "calibrated", "output_reference": output_reference}

    def dry_run_report(self, *, synthetic_input):
        payload = json.loads(synthetic_input.read_bytes())
        if payload.get("access_class") != "synthetic":
            raise CLIError("synthetic input required")
        return {"status": "synthetic_report_verified"}

    def run_final_test(self, **kwargs):
        return {"status": "completed"}


def test_exact_command_allowlist_and_no_arbitrary_scenario_override():
    parser = build_parser()
    assert set(parser._subparsers._group_actions[0].choices) == {
        "validate-plan", "calibrate-vanilla", "dry-run-report",
        "final-test", "resume-final-test", "verify-report",
    }
    with pytest.raises(CLIError):
        parser.parse_args(["final-test", "--scenario-id", "one"])
    with pytest.raises(CLIError):
        parser.parse_args([
            "final-test", "--plan", "/plan", "--capability-reference", "ref",
            "--registry-root", "/caller-controlled",
        ])


def test_dry_run_accepts_only_explicit_synthetic_input(tmp_path, capsys):
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(json.dumps({"access_class": "synthetic"}), encoding="utf-8")
    assert main(["dry-run-report", "--synthetic-input", str(synthetic)], services=Services()) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "synthetic_report_verified"
    test_input = tmp_path / "test.json"
    test_input.write_text(json.dumps({"access_class": "test"}), encoding="utf-8")
    with pytest.raises(CLIError, match="synthetic"):
        main(["dry-run-report", "--synthetic-input", str(test_input)], services=Services())


def test_calibration_rejects_non_train_manifest(tmp_path):
    manifest = tmp_path / "test_manifest.json"
    manifest.write_text("not opened by cli", encoding="utf-8")
    with pytest.raises(CLIError, match="train"):
        main(["calibrate-vanilla", "--train-manifest", str(manifest), "--output-reference", "ref"], services=Services())


def test_capability_is_not_echoed_when_launcher_is_absent(tmp_path, capsys):
    with pytest.raises(CLIError, match="launcher"):
        main([
            "final-test", "--plan", str(tmp_path / "missing"),
            "--capability-reference", "sensitive-reference",
        ])
    assert "sensitive-reference" not in capsys.readouterr().out


def test_launcher_result_is_recursively_sanitized(tmp_path):
    class UnsafeServices(Services):
        def dry_run_report(self, *, synthetic_input):
            return {"status": "complete", "authorization_header": "redacted"}

    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text('{"access_class":"synthetic"}', encoding="utf-8")
    with pytest.raises(CLIError, match="unsafe"):
        main(
            ["dry-run-report", "--synthetic-input", str(synthetic)],
            services=UnsafeServices(),
        )

    class EmbeddedUnsafeServices(Services):
        def dry_run_report(self, *, synthetic_input):
            return {"status": "failed at /home/private/report"}

    with pytest.raises(CLIError, match="unsafe"):
        main(
            ["dry-run-report", "--synthetic-input", str(synthetic)],
            services=EmbeddedUnsafeServices(),
        )
