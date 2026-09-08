import copy

import pytest
from test_splits import fixture_manifest

from comap_tau3 import splits
from comap_tau3.users import PipelineUserSimulator


def test_full_train_protocol(monkeypatch):
    _, official = fixture_manifest()
    monkeypatch.setattr(splits, "official_splits", lambda _: official)
    manifest = splits.make_manifest("unused", 0)
    splits.validate_manifest(manifest, official)
    assert (
        sum(len(shard) for domain in manifest["domains"].values() for shard in domain["rounds"])
        == 178
    )
    assert all(not domain["dev"] for domain in manifest["domains"].values())
    assert [len(s) for s in manifest["domains"]["airline"]["rounds"]] == [10, 10, 10]
    assert [len(s) for s in manifest["domains"]["retail"]["rounds"]] == [25, 25, 24]
    for name, partition in manifest["domains"].items():
        assert sum(partition["rounds"], []) == official[name]["train"]
        assert partition["test"] == official[name]["test"]
    bad = copy.deepcopy(manifest)
    bad["domains"]["airline"]["dev"] = ["0"]
    with pytest.raises(ValueError):
        splits.validate_manifest(bad, official)


def test_pipeline_user_omits_sampling_fields():
    user = PipelineUserSimulator(instructions="test", tools=[], llm="scripted", llm_args={})
    user.set_seed(42)
    assert user.llm_args == {}
    for field in ("seed", "temperature", "top_p"):
        with pytest.raises(ValueError):
            PipelineUserSimulator(
                instructions="test", tools=[], llm="scripted", llm_args={field: 0}
            )
