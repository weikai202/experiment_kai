import datetime
import threading
import time
from pathlib import Path
from types import ModuleType
import pytest
from toolsandbox_pipeline.schemas.dataset import DatasetBuildConfig
from toolsandbox_pipeline.reproducibility.clock import FixedWorldClock, CLOCK_SITES, audit_clock_sites

ROOT = Path(__file__).resolve().parents[2]


def config():
    return DatasetBuildConfig.model_validate_json((ROOT / "configs/reproducibility/dataset_build_v1.json").read_bytes())


def modules():
    result = {}
    for path in CLOCK_SITES:
        name = "tool_sandbox." + path[:-3].replace("/", ".")
        module = ModuleType(name)
        module.datetime = datetime
        result[name] = module
    return result


def environment(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr("toolsandbox_pipeline.reproducibility.clock.locale.setlocale", lambda *args: "C.UTF-8")
    monkeypatch.setattr("toolsandbox_pipeline.reproducibility.clock.time.timezone", 0)


def test_scoped_fixed_time_and_restoration(monkeypatch):
    environment(monkeypatch)
    targets = modules()
    start = time.monotonic()
    real = datetime.datetime.now(datetime.timezone.utc)
    with FixedWorldClock(config(), modules=targets):
        for module in targets.values():
            assert module.datetime.datetime.now() == datetime.datetime(2024, 5, 1, 12)
            assert module.datetime.datetime.now().year == 2024
            assert module.datetime.datetime.fromtimestamp(0, datetime.timezone.utc).year == 1970
            assert module.datetime.timedelta(days=1) == datetime.timedelta(days=1)
        assert datetime.datetime.now(datetime.timezone.utc) >= real
    assert time.monotonic() >= start
    assert all(module.datetime is datetime for module in targets.values())
    with pytest.raises(RuntimeError):
        with FixedWorldClock(config(), modules=targets):
            raise RuntimeError("synthetic failure")
    assert all(module.datetime is datetime for module in targets.values())


def test_nested_and_cross_thread_rejected(monkeypatch):
    environment(monkeypatch)
    targets = modules()
    failures = []
    with FixedWorldClock(config(), modules=targets):
        with pytest.raises(ValueError):
            with FixedWorldClock(config(), modules=modules()):
                pass
        def other_thread():
            try:
                next(iter(targets.values())).datetime.datetime.now()
            except ValueError:
                failures.append(True)
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
    assert failures == [True]


def test_pinned_source_audit():
    from importlib.metadata import distribution
    root = Path(distribution("tool-sandbox").locate_file("tool_sandbox"))
    assert len(audit_clock_sites(root)) == 7


@pytest.mark.parametrize("key,value", [("data_seed", True), ("data_seed", 1), ("world_epoch_unix", 1714564800), ("timezone", "EST"), ("locale", "C"), ("test_fraction", "0.2")])
def test_config_strict_constants(key, value):
    import json
    data = config().model_dump(mode="json")
    data[key] = value
    with pytest.raises(ValueError):
        DatasetBuildConfig.model_validate_json(json.dumps(data))
