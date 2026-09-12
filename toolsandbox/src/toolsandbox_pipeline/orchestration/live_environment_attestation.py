"""Separate current execution evidence from immutable dataset-build provenance.

Reads only dataset_index.json, build configuration, lock/distribution metadata,
plus upstream Python files for byte hashing using the original environment rule.
Never imports a scenario registry or opens any split manifest/scenario content.
"""
from importlib import metadata
from pathlib import Path
import json

import tomli
from packaging.markers import Marker, default_environment
from packaging.utils import canonicalize_name

from toolsandbox_pipeline.reproducibility.dataset_manifest import native_environment, load_build_config
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.retrieval.index import file_hash
from toolsandbox_pipeline.schemas.dataset import DatasetIndex

VERSION = 'dataset-build-execution-environment-v2'


def validate_distribution_inventory(lock, installed, *, marker_environment=None):
    """Compare exact installed versions and current-platform locked dependencies.

    Installed metadata is not a proof that every third-party file matches a
    wheel. Upstream ToolSandbox source bytes are separately verified in full.
    """
    packages = {}
    for package in lock['package']:
        name = canonicalize_name(package['name'])
        if name in packages:
            raise ValueError('ambiguous multiple locked package versions')
        packages[name] = package
    versions = {}
    for name, version in installed:
        name = canonicalize_name(name)
        if name in versions:
            raise ValueError('duplicate installed distribution metadata')
        versions[name] = version
        if name not in packages or packages[name]['version'] != version:
            raise ValueError('installed distribution differs from current dependency lock: ' + name)
    environment = default_environment() if marker_environment is None else dict(marker_environment)
    environment.setdefault('extra','')
    for name in versions:
        for dependency in packages[name].get('dependencies',()):
            if dependency.get('marker') and not Marker(dependency['marker']).evaluate(environment):
                continue
            if canonicalize_name(dependency['name']) not in versions:
                raise ValueError('missing active locked dependency: ' + dependency['name'])
    return dict(locked_package_count=len(packages), installed_package_count=len(versions),
        installed_distributions=[dict(name=name,version=versions[name],locked_source=packages[name]['source'])
                                 for name in sorted(versions)],
        uninstalled_locked_packages=sorted(set(packages)-set(versions)),
        marker_environment=environment, version_mismatches=[], unlocked_installed_packages=[],
        active_locked_dependencies_complete=True)


def attest_execution_environment(*, project_root, dataset_index_path, expected_dataset_index_sha256):
    project, index_path = Path(project_root), Path(dataset_index_path)
    for path in (project,index_path):
        if not path.is_absolute() or any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('absolute non-symlink attestation paths required')
    raw = index_path.read_bytes()
    if file_hash(raw) != expected_dataset_index_sha256:
        raise ValueError('immutable dataset index hash mismatch')
    index = DatasetIndex.model_validate_json(raw)
    if index.environment is None:
        raise ValueError('original dataset environment evidence required')
    config, config_sha = load_build_config(Path(index.config_path))
    if config_sha != index.config_sha256:
        raise ValueError('dataset build configuration changed')
    current = native_environment(config,config_sha,project,
        container_image_digest=index.environment.container_image_digest)
    build = index.environment.model_dump(mode='json')
    execution = current.model_dump(mode='json')
    if canonical_sha256(build) != index.environment_sha256:
        raise ValueError('dataset environment hash mismatch')
    differences = [key for key in build if build[key] != execution[key]]
    if any(key != 'dependency_lock_sha256' for key in differences):
        raise ValueError('dataset source/Python/clock environment changed: ' + ','.join(differences))
    lock_path = project/'uv.lock'
    lock_raw = lock_path.read_bytes()
    if file_hash(lock_raw) != current.dependency_lock_sha256:
        raise ValueError('lock changed during attestation')
    lock = tomli.loads(lock_raw.decode())
    distributions = tuple(metadata.distributions())
    inventory = validate_distribution_inventory(lock,tuple((d.metadata['Name'],d.version) for d in distributions))
    project_dist = next((d for d in distributions if canonicalize_name(d.metadata['Name'])=='toolsandbox-pipeline'),None)
    if project_dist is None:
        raise ValueError('installed project distribution required')
    direct = json.loads(project_dist.read_text('direct_url.json') or '{}')
    from urllib.parse import urlparse,unquote
    source = urlparse(direct.get('url',''))
    if source.scheme != 'file' or Path(unquote(source.path)).resolve() != project.resolve() or direct.get('dir_info',{}).get('editable') is not True:
        raise ValueError('installed editable project does not match current source root')
    return dict(version=VERSION,dataset_index_sha256=expected_dataset_index_sha256,
        dataset_build_status=index.status,
        dataset_build_environment=build,dataset_build_environment_sha256=index.environment_sha256,
        execution_environment=execution,execution_environment_sha256=canonical_sha256(execution),
        dependency_lock_sha256=current.dependency_lock_sha256,
        changed_environment_fields=differences,installed_inventory=inventory,
        installed_inventory_sha256=canonical_sha256(inventory),project_editable_source_matches=True,
        upstream_commit_and_source_identical=True,
        old_and_current_lock_semantic_equivalence='not_claimed',
        split_manifest_contents_accessed=False)


def validate_execution_attestation(attestation, *, project_root, dataset_index_path, expected_dataset_index_sha256):
    current = attest_execution_environment(project_root=project_root,dataset_index_path=dataset_index_path,
        expected_dataset_index_sha256=expected_dataset_index_sha256)
    if type(attestation) is not dict or attestation != current:
        raise ValueError('execution environment attestation is stale or mismatched')
    return True
