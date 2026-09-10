from contextlib import nullcontext
from types import SimpleNamespace
import json
import pytest
from toolsandbox_pipeline.schemas.dataset import DatasetEnvironment, REGISTRIES, SUFFIXES, REQUIRED_CATEGORIES, VARIANTS
from toolsandbox_pipeline.reproducibility.dataset_manifest import construct_manifests, publish_bundle, BUNDLE_FILES
from toolsandbox_pipeline.retrieval.index import file_hash
from tests.reproducibility.test_clock import config, ROOT
from tests.reproducibility.test_splits import families

HASH = "sha256:" + "a" * 64


def synthetic_hashes(scenario):
    return dict(starting_context_sha256=HASH, evaluation_definition_sha256=HASH,
                agent_facing_tool_names_sha256=HASH, agent_facing_tool_schema_sha256=HASH)


def synthetic_scenario(variant):
    index = VARIANTS.index(variant)
    categories = (["THREE_DISTRACTION_TOOLS"] if index >= 4 else []) + [REQUIRED_CATEGORIES[index]]
    return SimpleNamespace(categories=categories, max_messages=30, mutable=[])


def blobs():
    build = config()
    env = DatasetEnvironment(upstream_commit=build.upstream_commit, upstream_source_sha256=HASH, dependency_lock_sha256=HASH,
        python_patch_version="3.10.20", platform="synthetic-linux", timezone="UTC", locale="C.UTF-8", build_config_sha256=HASH,
        clock_adapter_version=build.clock_adapter_version, preferred_tool_backend="DEFAULT", container_image_digest=None)
    factories = {label: (lambda label=label, **kw: {name: None for name, source in families() if source == label}) for label in REGISTRIES}
    registry = tuple(name for label in REGISTRIES for name, source in families() if source == label)
    def augmented(**kwargs):
        mapping = {name: synthetic_scenario(VARIANTS[0]) for name in registry}
        for name in registry:
            for suffix, variant in zip(SUFFIXES[1:], VARIANTS[1:]):
                mapping[name + suffix] = synthetic_scenario(variant)
        return mapping
    return construct_manifests(build, env, config_path="/synthetic/config.json", factories=factories,
        augmented_factory=augmented, backend="DEFAULT", hash_scenario=synthetic_hashes, clock=nullcontext())


def test_manifest_bundle_determinism_permissions_and_visibility(tmp_path):
    first = blobs()
    assert first == blobs()
    result = publish_bundle(tmp_path / "private", first)
    assert result == publish_bundle(tmp_path / "private", first)
    assert set(result) == set(BUNDLE_FILES)
    assert "family-" not in first["dataset_index.json"].decode()
    for name in BUNDLE_FILES:
        assert (tmp_path / "private" / name).stat().st_mode & 0o777 == 0o600
    second = dict(first)
    second["dataset_index.json"] += b" "
    with pytest.raises(ValueError):
        publish_bundle(tmp_path / "private", second)
    assert (tmp_path / "private/dataset_index.json").read_bytes() == first["dataset_index.json"]
