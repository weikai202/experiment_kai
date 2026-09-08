import subprocess
from pathlib import Path

from .io import digest


def source_hashes():
    from bfcl_eval import __file__ as bfcl_file
    package = Path(bfcl_file).resolve().parent
    files = list((package / "eval_checker/multi_turn_eval").rglob("*.py"))
    files += [package / "utils.py", package / "model_handler/utils.py"]
    files += list((package / "constants").glob("*.py"))
    return {str(p.relative_to(package)): digest(p) for p in sorted(set(files))}


def verify_manifest(manifest):
    from bfcl_eval import __file__ as bfcl_file
    package = Path(bfcl_file).resolve().parent
    for name, expected in manifest["bfcl_data_sha256"].items():
        if digest(package / "data" / name) != expected:
            raise ValueError(f"BFCL data drift: {name}")
    if "bfcl_source_sha256" in manifest and source_hashes() != manifest["bfcl_source_sha256"]:
        raise ValueError("BFCL simulator/evaluator source changed since preparation")
    commit = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"], capture_output=True, text=True)
    if manifest["bfcl_commit"] != "unknown" and commit.stdout.strip() != manifest["bfcl_commit"]:
        raise ValueError("BFCL commit changed since preparation")
