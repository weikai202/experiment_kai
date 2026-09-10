"""Acceptance tests for the standalone project scaffold."""

from __future__ import annotations

import builtins
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "toolsandbox_pipeline"
COMMIT = "165848b9a78cead7ca7fe7c89c688b58e6501219"


def test_project_package_imports_from_src_tree() -> None:
    package = importlib.import_module("toolsandbox_pipeline")
    assert Path(package.__file__).resolve().is_relative_to(PACKAGE_ROOT)


def test_upstream_package_is_an_installed_dependency() -> None:
    package = importlib.import_module("tool_sandbox")
    path = Path(package.__file__).resolve()
    assert path.is_relative_to(PROJECT_ROOT / ".venv")
    assert not (PROJECT_ROOT / "tool_sandbox").exists()


def test_upstream_distribution_records_exact_git_commit() -> None:
    text = importlib.metadata.distribution("tool-sandbox").read_text("direct_url.json")
    assert text is not None
    direct_url = json.loads(text)
    assert direct_url["url"] == "https://github.com/apple/ToolSandbox.git"
    assert direct_url["vcs_info"] == {
        "commit_id": COMMIT,
        "requested_revision": COMMIT,
        "vcs": "git",
    }


def test_project_import_has_no_external_or_write_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("toolsandbox_pipeline", None)
    effects: list[str] = []
    real_open = builtins.open
    real_os_open = os.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            effects.append(f"open:{mode}")
            raise AssertionError("filesystem write during import")
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & write_flags:
            effects.append(f"os.open:{flags}")
            raise AssertionError("filesystem write during import")
        return real_os_open(path, flags, *args, **kwargs)

    def blocked_network(*args: Any, **kwargs: Any) -> None:
        effects.append("network")
        raise AssertionError("network access during import")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(socket.socket, "connect", blocked_network)
    monkeypatch.setattr(socket, "create_connection", blocked_network)
    importlib.import_module("toolsandbox_pipeline")
    assert effects == []


def test_project_version_is_nonempty_static_string() -> None:
    package = importlib.import_module("toolsandbox_pipeline")
    assert isinstance(package.__version__, str)
    assert package.__version__.strip()
