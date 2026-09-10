"""Privileged manifest construction and immutable private bundle publication."""
import os
from pathlib import Path
import shutil
import tempfile

from toolsandbox_pipeline.reproducibility.canonical import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.schemas.dataset import DatasetBuildConfig, DatasetEnvironment, DatasetIndex, ScenarioRecord, SplitManifest, VARIANTS, REGISTRIES, REQUIRED_CATEGORIES
from .splits import build_family_registry, build_augmented_registry, assign_splits

BUNDLE_FILES = ("dataset_index.json", "train_manifest.json", "dev_manifest.json", "sealed/test_manifest.json")


def load_build_config(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("absolute non-symlink build config required")
    raw = path.read_bytes()
    return DatasetBuildConfig.model_validate_json(raw), file_hash(raw)


def construct_manifests(config, environment, *, config_path, factories, augmented_factory, backend, hash_scenario, clock):
    if type(config) is not DatasetBuildConfig or type(environment) is not DatasetEnvironment:
        raise TypeError("validated build config and environment required")
    with clock:
        families = build_family_registry(factories, backend)
        expanded = build_augmented_registry(families, augmented_factory, backend)
        assigned = assign_splits(families)
        env_hash = canonical_sha256(environment.model_dump(mode="json"))
        manifests = {}
        for split in ("train", "dev", "test"):
            records = []
            for family in assigned[split]:
                native_categories = set(str(v) for v in expanded[family.family_id].categories) - set(REQUIRED_CATEGORIES)
                for variant, scenario_id in zip(VARIANTS, family.ordered_variant_scenario_ids):
                    scenario = expanded[scenario_id]
                    if not native_categories <= set(str(v) for v in scenario.categories):
                        raise ValueError("variant lost native task categories")
                    records.append(ScenarioRecord(scenario_id=scenario_id, scenario_family_id=family.family_id,
                        variant=variant, categories=tuple(str(v) for v in scenario.categories), max_messages=scenario.max_messages,
                        **hash_scenario(scenario)))
            manifests[split] = SplitManifest(schema_version=1, split=split, build_config_sha256=environment.build_config_sha256,
                environment_sha256=env_hash, families=assigned[split], scenarios=tuple(records))
    blobs = {f"{'sealed/' if split == 'test' else ''}{split}_manifest.json": canonical_json_bytes(manifest.model_dump(mode="json")) for split, manifest in manifests.items()}
    index = DatasetIndex(schema_version=1, config_path=str(config_path), config_sha256=environment.build_config_sha256,
        environment=environment, environment_sha256=env_hash, status="complete" if environment.container_image_digest else "setup_only",
        family_count=129, scenario_count=1032, train_manifest_sha256=file_hash(blobs["train_manifest.json"]),
        dev_manifest_sha256=file_hash(blobs["dev_manifest.json"]), test_manifest_sha256=file_hash(blobs["sealed/test_manifest.json"]))
    blobs["dataset_index.json"] = canonical_json_bytes(index.model_dump(mode="json"))
    return blobs


def publish_bundle(output_dir, blobs):
    root = Path(output_dir)
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)) or set(blobs) != set(BUNDLE_FILES):
        raise ValueError("invalid explicit private bundle path or layout")
    if root.exists():
        paths = {p.relative_to(root).as_posix() for p in root.rglob("*")}
        if paths != set(BUNDLE_FILES) | {"sealed"}:
            raise ValueError("existing bundle layout mismatch")
        if root.stat().st_mode & 0o777 != 0o700 or (root / "sealed").is_symlink() or (root / "sealed").stat().st_mode & 0o777 != 0o700:
            raise ValueError("private directory permissions mismatch")
        for name, raw in blobs.items():
            path = root / name
            if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600 or path.read_bytes() != raw:
                raise ValueError("existing bundle content/permissions mismatch")
    else:
        stage = Path(tempfile.mkdtemp(prefix=".dataset-build-", dir=root.parent))
        try:
            (stage / "sealed").mkdir(mode=0o700)
            for name in BUNDLE_FILES:
                descriptor = os.open(stage / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(blobs[name])
                    stream.flush()
                    os.fsync(stream.fileno())
            for directory in (stage / "sealed", stage):
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            os.rename(stage, root)
            descriptor = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return {name: file_hash(blobs[name]) for name in BUNDLE_FILES}


def native_environment(config, config_hash, project_root, *, container_image_digest=None):
    import platform
    from importlib.metadata import distribution
    import json
    dist = distribution("tool-sandbox")
    if json.loads(dist.read_text("direct_url.json"))["vcs_info"]["commit_id"] != config.upstream_commit:
        raise ValueError("upstream commit mismatch")
    source_root = Path(dist.locate_file("tool_sandbox"))
    source = [[p.relative_to(source_root).as_posix(), file_hash(p.read_bytes())] for p in sorted(source_root.rglob("*.py"))]
    return DatasetEnvironment(upstream_commit=config.upstream_commit, upstream_source_sha256=canonical_sha256(source),
        dependency_lock_sha256=file_hash((Path(project_root) / "uv.lock").read_bytes()), python_patch_version=platform.python_version(),
        platform=platform.platform(), timezone=config.timezone, locale=config.locale, build_config_sha256=config_hash,
        clock_adapter_version=config.clock_adapter_version, preferred_tool_backend=config.preferred_tool_backend,
        container_image_digest=container_image_digest)


def build_native_bundle(config_path, output_dir, project_root):
    from tool_sandbox.common.tool_discovery import ToolBackend
    from .clock import FixedWorldClock
    from .scenario_hashes import scenario_hashes
    config, digest = load_build_config(config_path)
    environment = native_environment(config, digest, project_root)
    # Registry functions are imported only for this explicit privileged operation.
    from tool_sandbox.scenarios import (named_scenarios, named_single_tool_call_scenarios,
        named_multiple_tool_call_scenarios, named_multiple_user_turn_scenarios, named_insufficient_information_scenarios)
    factories = dict(zip(REGISTRIES, (named_single_tool_call_scenarios, named_multiple_tool_call_scenarios,
                                    named_multiple_user_turn_scenarios, named_insufficient_information_scenarios)))
    blobs = construct_manifests(config, environment, config_path=config_path, factories=factories,
        augmented_factory=named_scenarios, backend=ToolBackend.DEFAULT, hash_scenario=scenario_hashes, clock=FixedWorldClock(config))
    return publish_bundle(output_dir, blobs)
