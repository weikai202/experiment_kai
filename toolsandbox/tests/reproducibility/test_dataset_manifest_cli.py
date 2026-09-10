import json
from toolsandbox_pipeline.reproducibility.dataset_manifest_cli import main


def test_cli_sanitizes_builder_output(tmp_path, capsys):
    def builder(*args):
        print("synthetic hidden content")
        raise RuntimeError("synthetic scenario-id and hidden content")
    assert main(["build-dataset-manifest", "--config", str(tmp_path / "config"), "--output-dir", str(tmp_path / "out")], builder=builder) == 1
    output = capsys.readouterr()
    assert "synthetic" not in output.out + output.err
    assert json.loads(output.out)["category"] == "manifest_build_rejected"


def test_invalid_cli_paths_and_extra_flags(capsys):
    called = []
    assert main(["build-dataset-manifest", "--config", "relative", "--output-dir", "/tmp/private"], builder=lambda *args: called.append(1)) == 1
    assert main(["build-dataset-manifest", "--config", "/tmp/config", "--output-dir", "/tmp/private", "--test"], builder=lambda *args: called.append(1)) == 1
    assert not called
