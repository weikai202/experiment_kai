"""Scoped upstream module clocks, separate from real telemetry clocks."""
import ast
import datetime
import importlib
import locale
import os
from pathlib import Path
import threading
import time
from types import ModuleType
from toolsandbox_pipeline.schemas.dataset import DatasetBuildConfig

CLOCK_SITES = {
    "common/utils.py": (431, 443),
    "scenarios/base_scenarios.py": (120, 132, 144, 156, 168, 180, 183, 192, 195, 204, 207),
    "scenarios/multiple_tool_call_scenarios.py": (1781, 1965, 2165, 2368),
    "scenarios/multiple_user_turn_scenarios.py": (1250, 1307, 1434, 1507),
    "tools/messaging.py": (82,),
    "tools/reminder.py": (71, 130),
    "tools/utilities.py": (33, 254),
}


def audit_clock_sites(root):
    actual = {}
    for folder in ("common", "scenarios", "tools"):
        for path in sorted((Path(root) / folder).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            sites = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in ("now", "utcnow", "today"):
                    if ast.unparse(func) != "datetime.datetime.now":
                        raise ValueError("unaccounted dynamic clock reference")
                    sites.append(node.lineno)
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "time" and func.attr in ("time", "localtime", "gmtime"):
                    raise ValueError("unaccounted time reference")
            if sites:
                actual[path.relative_to(root).as_posix()] = tuple(sorted(sites))
    if actual != CLOCK_SITES:
        raise ValueError("pinned upstream clock-site audit mismatch")
    return tuple(actual)


class FixedWorldClock:
    _lock = threading.Lock()

    def __init__(self, config: DatasetBuildConfig, *, modules=None):
        self.config = DatasetBuildConfig.model_validate(config)
        self._modules = modules
        self._saved = []

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            raise ValueError("nested or concurrent world clock use")
        self._owner = threading.get_ident()
        try:
            if os.environ.get("TZ") != "UTC" or time.timezone != 0 or locale.setlocale(locale.LC_ALL) != "C.UTF-8":
                raise ValueError("coordinator must select TZ=UTC and locale=C.UTF-8")
            names = tuple("tool_sandbox." + path[:-3].replace("/", ".") for path in CLOCK_SITES)
            if self._modules is None:
                from importlib.metadata import distribution
                import json
                dist = distribution("tool-sandbox")
                if json.loads(dist.read_text("direct_url.json"))["vcs_info"]["commit_id"] != self.config.upstream_commit:
                    raise ValueError("wrong upstream commit")
                audit_clock_sites(Path(dist.locate_file("tool_sandbox")))
                modules = {name: importlib.import_module(name) for name in names}
            else:
                modules = self._modules
            if tuple(modules) != names:
                raise ValueError("clock module allowlist mismatch")
            for name, module in modules.items():
                if not isinstance(module, ModuleType) or module.__name__ != name or getattr(module, "datetime", None) is not datetime:
                    raise ValueError("wrong upstream datetime module identity")
            epoch = self.config.world_epoch_unix
            class FixedDateTime(datetime.datetime):
                @classmethod
                def now(cls, tz=None):
                    if threading.get_ident() != self._owner:
                        raise ValueError("world clock cannot cross threads")
                    value = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc)
                    return value.replace(tzinfo=None) if tz is None else value.astimezone(tz)
            proxy = ModuleType("datetime")
            proxy.__dict__.update(datetime.__dict__)
            proxy.datetime = FixedDateTime
            for module in modules.values():
                self._saved.append((module, module.datetime))
                module.datetime = proxy
            return self
        except BaseException:
            for module, original in reversed(self._saved):
                module.datetime = original
            self._saved.clear()
            self._lock.release()
            raise

    def __exit__(self, *args):
        for module, original in reversed(self._saved):
            module.datetime = original
        self._saved.clear()
        self._lock.release()
