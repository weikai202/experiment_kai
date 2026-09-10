"""Atomic, immutable, sanitized final-report artifact publication."""

from __future__ import annotations

import hashlib
import os
import tempfile
from math import fsum
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal

from pydantic import ConfigDict

from toolsandbox_pipeline.reproducibility import canonical_json_bytes, canonical_sha256
from toolsandbox_pipeline.schemas.reporting import (
    ArtifactManifestEntry,
    CategoryResultRecord,
    ClusterBootstrapReport,
    FailureModeAttributionRow,
    FailureModeRepairSummary,
    FamilyStabilityRecord,
    FinalEvaluationPlan,
    MainTable,
    ReportManifest,
    ReproducibilityChecklist,
    ScenarioEvaluationRecord,
    SystemAggregate,
    TrainingRoundHeadline,
)
from toolsandbox_pipeline.schemas.base import StrictModel

from .aggregates import aggregate_system
from .cluster_statistics import paired_cluster_bootstrap
from .failure_mode_analysis import compute_family_stability

_FORBIDDEN_FRAGMENTS = (
    "sk-" + "proj-", "authorization:", "api_key", "https://", "http://",
    "traceback (most recent call last)", "raw_trajectory", "tool_arguments",
    "tool_results", "raw_evaluator_definition", "model_response", "prompt_text",
)

_REQUIRED_FILES = {
    "plan.json", "plan.sha256", "scenario_results.jsonl",
    "family_results.jsonl", "category_results.jsonl",
    "failure_mode_case_attribution.jsonl", "failure_mode_repair_summary.json",
    "pairwise_cluster_bootstrap.json", "reproducibility_checklist.json",
    "training_round_headlines.json", "main_table.json", "report.md",
}


class ArtifactError(RuntimeError):
    """Artifact violates immutability, completeness, or sanitization rules."""


def _safe_relative(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ArtifactError("artifact path must be normalized and relative")
    return value


def _scan(raw: bytes) -> None:
    lowered = raw.decode("utf-8", errors="ignore").lower()
    if any(item in lowered for item in _FORBIDDEN_FRAGMENTS):
        raise ArtifactError("sensitive or restricted report content")


class ArtifactPublisher:
    """Explicit publisher; construction performs no filesystem access."""

    def __init__(self, root: Path, plan_sha256: str) -> None:
        if not root.is_absolute():
            raise ArtifactError("artifact root must be absolute")
        self.root = root
        self.plan_sha256 = plan_sha256
        self._entries: dict[str, ArtifactManifestEntry] = {}

    def publish_json(self, relative_path: str, payload: object) -> ArtifactManifestEntry:
        return self._publish(relative_path, canonical_json_bytes(payload))

    def publish_jsonl(self, relative_path: str, rows: Iterable[object]) -> ArtifactManifestEntry:
        raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        return self._publish(relative_path, raw)

    def publish_text(self, relative_path: str, text: str) -> ArtifactManifestEntry:
        if not text.endswith("\n"):
            text += "\n"
        return self._publish(relative_path, text.encode("utf-8"))

    def _publish(self, relative_path: str, raw: bytes) -> ArtifactManifestEntry:
        relative = _safe_relative(relative_path)
        if relative.name == "manifest.json":
            raise ArtifactError("manifest is finalized separately")
        _scan(raw)
        target = self.root.joinpath(*relative.parts)
        _validate_target(self.root, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_create(target, raw)
        entry = ArtifactManifestEntry(
            path=str(relative), sha256=_raw_sha256(raw),
            byte_count=len(raw), mode="0600",
        )
        self._entries[str(relative)] = entry
        return entry

    def finalize(self) -> ReportManifest:
        paths = set(self._entries)
        missing = _REQUIRED_FILES - paths
        missing_metrics = [
            system_id
            for system_id in ("vanilla", "generation_0", "updated")
            if not any(path.startswith(f"systems/{system_id}/metrics/") for path in paths)
        ]
        if missing or missing_metrics:
            raise ArtifactError("required report artifact structure is incomplete")
        _validate_semantic_tree(self.root, self.plan_sha256)
        entries = tuple(
            self._entries[key]
            for key in sorted(self._entries, key=lambda item: item.encode("utf-8"))
        )
        payload = {
            "schema_version": 1,
            "plan_sha256": self.plan_sha256,
            "entries": [item.model_dump(mode="json") for item in entries],
        }
        manifest = ReportManifest(
            schema_version=1,
            plan_sha256=self.plan_sha256,
            entries=entries,
            manifest_sha256=canonical_sha256(payload),
        )
        target = self.root / "manifest.json"
        raw = canonical_json_bytes(manifest.model_dump(mode="json"))
        _scan(raw)
        _validate_target(self.root, target)
        _atomic_create(target, raw)
        return manifest


def verify_report(root: Path) -> ReportManifest:
    """Verify canonical manifest, every listed hash/size/mode, and sanitization."""

    manifest_path = root / "manifest.json"
    _validate_target(root, manifest_path)
    raw = manifest_path.read_bytes()
    _scan(raw)
    try:
        manifest = ReportManifest.model_validate_json(raw, strict=True)
    except Exception as error:
        raise ArtifactError("invalid report manifest") from error
    if raw != canonical_json_bytes(manifest.model_dump(mode="json")):
        raise ArtifactError("report manifest is not canonical")
    payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if canonical_sha256(payload) != manifest.manifest_sha256:
        raise ArtifactError("report manifest identity mismatch")
    expected = {item.path for item in manifest.entries} | {"manifest.json"}
    actual = {
        str(path.relative_to(root).as_posix())
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ArtifactError("unlisted or missing report artifact")
    for entry in manifest.entries:
        path = root.joinpath(*_safe_relative(entry.path).parts)
        _validate_target(root, path)
        content = path.read_bytes()
        _scan(content)
        if len(content) != entry.byte_count or _raw_sha256(content) != entry.sha256:
            raise ArtifactError("artifact hash or size mismatch")
        if path.stat().st_mode & 0o777 != 0o600:
            raise ArtifactError("artifact permission mismatch")
    if manifest_path.stat().st_mode & 0o777 != 0o600:
        raise ArtifactError("manifest permission mismatch")
    _validate_semantic_tree(root, manifest.plan_sha256)
    return manifest


def _raw_sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _validate_target(root: Path, target: Path) -> None:
    if not root.is_absolute() or not target.is_relative_to(root):
        raise ArtifactError("artifact target escapes absolute report root")
    current = Path(root.anchor)
    for part in target.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ArtifactError("artifact path ancestry cannot contain symlinks")


def _atomic_create(target: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            if target.is_symlink() or target.read_bytes() != raw:
                raise ArtifactError("refusing to overwrite immutable artifact") from error
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_model(path: Path, model):
    try:
        return model.model_validate_json(path.read_bytes(), strict=True)
    except Exception as error:
        raise ArtifactError(f"invalid typed artifact: {path.name}") from error


def _load_jsonl(path: Path, model):
    raw = path.read_bytes()
    rows = []
    try:
        for line in raw.splitlines():
            if not line:
                raise ValueError("empty JSONL line")
            rows.append(model.model_validate_json(line, strict=True))
    except Exception as error:
        raise ArtifactError(f"invalid typed JSONL artifact: {path.name}") from error
    return tuple(rows)


def _validate_semantic_tree(root: Path, plan_sha256: str) -> None:
    plan = _load_model(root / "plan.json", FinalEvaluationPlan)
    if plan.plan_sha256 != plan_sha256:
        raise ArtifactError("manifest and plan identity mismatch")
    if (root / "plan.sha256").read_text(encoding="utf-8") != plan_sha256 + "\n":
        raise ArtifactError("plan SHA artifact mismatch")
    scenarios = _load_jsonl(root / "scenario_results.jsonl", ScenarioEvaluationRecord)
    expected_scenarios = tuple(
        (system_id, item.manifest_position, item.scenario_id, item.scenario_family_id, item.variant, item.categories)
        for system_id in plan.ordered_system_ids
        for item in plan.scenarios
    )
    actual_scenarios = tuple(
        (item.system_id, item.manifest_position, item.scenario_id, item.scenario_family_id, item.variant, item.categories)
        for item in scenarios
    )
    if actual_scenarios != expected_scenarios:
        raise ArtifactError("scenario results do not match frozen plan/order")
    families = _load_jsonl(root / "family_results.jsonl", FamilyStabilityRecord)
    ordered_family_ids = tuple(sorted(
        {item.scenario_family_id for item in plan.scenarios},
        key=lambda item: item.encode("utf-8"),
    ))
    expected_family_order = tuple(
        (system_id, family_id)
        for system_id in plan.ordered_system_ids
        for family_id in ordered_family_ids
    )
    if tuple(
        (item.system_id, item.scenario_family_id) for item in families
    ) != expected_family_order:
        raise ArtifactError("family results require exact system/family order")
    categories = _load_jsonl(root / "category_results.jsonl", CategoryResultRecord)
    if not categories:
        raise ArtifactError("category results cannot be empty")
    attributions = _load_jsonl(
        root / "failure_mode_case_attribution.jsonl", FailureModeAttributionRow
    )
    expected_attribution_order = tuple(
        (system_id, item.scenario_id)
        for system_id in plan.ordered_system_ids
        for item in plan.scenarios
    )
    if tuple((item.system_id, item.scenario_id) for item in attributions) != expected_attribution_order:
        raise ArtifactError("failure attribution requires exact system/scenario order")
    for position in range(200):
        shared = tuple(attributions[offset * 200 + position] for offset in range(3))
        reference = shared[0].model_dump(mode="json", exclude={"system_id"})
        if any(
            item.model_dump(mode="json", exclude={"system_id"}) != reference
            for item in shared[1:]
        ):
            raise ArtifactError("failure attribution differs across system projections")
    failure_summary = _load_model(
        root / "failure_mode_repair_summary.json", FailureModeRepairSummary
    )
    if failure_summary != _failure_summary_from_rows(attributions):
        raise ArtifactError("failure summary does not match attribution rows")
    bootstrap = _load_model(
        root / "pairwise_cluster_bootstrap.json", ClusterBootstrapReport
    )
    if bootstrap != paired_cluster_bootstrap(scenarios):
        raise ArtifactError("cluster bootstrap does not match scenario results")
    checklist = _load_model(
        root / "reproducibility_checklist.json", ReproducibilityChecklist
    )
    if checklist.profile != plan.profile:
        raise ArtifactError("reproducibility profile differs from frozen plan")
    rounds = _load_model(root / "training_round_headlines.json", _RoundArtifact)
    main = _load_model(root / "main_table.json", MainTable)
    if (
        main.plan_sha256 != plan.plan_sha256
        or main.failure_mode_repair_summary != failure_summary
        or main.training_round_headlines != rounds.rounds
    ):
        raise ArtifactError("main table cross-binding mismatch")
    metric_rows = tuple(
        _load_model(root / f"systems/{system_id}/metrics/headline.json", SystemAggregate)
        for system_id in plan.ordered_system_ids
    )
    if metric_rows != main.system_aggregates:
        raise ArtifactError("system metric/main-table cross-binding mismatch")
    if any(
        item.scenario_count != 200
        or item.family_count != 25
        or item.total_running_time_seconds is None
        for item in metric_rows
    ):
        raise ArtifactError("system accounting headline is incomplete")
    recomputed_metrics = tuple(
        aggregate_system(
            tuple(item for item in scenarios if item.system_id == aggregate.system_id),
            system_id=aggregate.system_id,
            total_running_time_seconds=aggregate.total_running_time_seconds,
            total_tokens=aggregate.total_tokens,
            usage_complete=aggregate.usage_complete,
            total_cost=aggregate.total_cost,
            cost_complete=aggregate.cost_complete,
            failure_count=aggregate.failure_count,
            timeout_count=aggregate.timeout_count,
            reconciliation_count=aggregate.reconciliation_count,
        )
        for aggregate in metric_rows
    )
    if metric_rows != recomputed_metrics:
        raise ArtifactError("system metrics do not match scenario results")
    recomputed_families, recomputed_family_summaries = compute_family_stability(
        scenarios, attributions
    )
    if families != recomputed_families:
        raise ArtifactError("family stability does not match scenario attribution")
    if main.family_stability_summaries != recomputed_family_summaries:
        raise ArtifactError("family summaries do not match family stability")
    expected_categories = tuple(
        CategoryResultRecord(system_id=aggregate.system_id, metric=metric)
        for aggregate in metric_rows
        for metric in aggregate.category_metrics
    )
    if categories != expected_categories:
        raise ArtifactError("category/system metric cross-binding mismatch")
    report = (root / "report.md").read_text(encoding="utf-8")
    required_statements = (
        "Qwen weights were never trained or modified",
        "observational", "not causal", "total_running_time_seconds",
        "qwen_effective_output_tokens",
    )
    if any(statement not in report for statement in required_statements):
        raise ArtifactError("report is missing required evidence qualification")
    if plan.profile == "official_live" and (
        "gpt-4o-mini" not in report or "upstream GPT-4o deviation" not in report
    ):
        raise ArtifactError("official-live report is missing User-model deviation")


def _failure_summary_from_rows(
    rows: tuple[FailureModeAttributionRow, ...],
) -> FailureModeRepairSummary:
    generation_0 = tuple(item for item in rows if item.system_id == "generation_0")
    related = tuple(
        item for item in generation_0
        if item.classification in {"related_repaired", "related_unrepaired"}
    )
    repaired = sum(item.classification == "related_repaired" for item in generation_0)
    return FailureModeRepairSummary(
        generation_0_failure_case_count=sum(
            not item.generation_0_fully_successful for item in generation_0
        ),
        failure_mode_related_case_count=len(related),
        failure_mode_repaired_case_count=repaired,
        failure_mode_repair_rate=(None if not related else repaired / len(related)),
        repair_rate_complete=bool(related),
        failure_mode_unmatched_case_count=sum(
            item.classification == "unmatched" for item in generation_0
        ),
        failure_mode_ambiguous_case_count=sum(
            item.classification == "ambiguous" for item in generation_0
        ),
        incomplete_evidence_case_count=sum(
            item.classification == "incomplete_evidence" for item in generation_0
        ),
        updated_minus_generation_0_similarity_points_overall=fsum(
            item.updated_similarity - item.generation_0_similarity
            for item in generation_0
        ) / len(generation_0),
        updated_minus_generation_0_similarity_points_related_subset=(
            None if not related else fsum(
                item.updated_similarity - item.generation_0_similarity
                for item in related
            ) / len(related)
        ),
        updated_minus_vanilla_similarity_points_overall=fsum(
            item.updated_similarity - item.vanilla_similarity
            for item in generation_0
        ) / len(generation_0),
    )


class _RoundArtifact(StrictModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    schema_version: Literal[1]
    rounds: tuple[TrainingRoundHeadline, TrainingRoundHeadline, TrainingRoundHeadline]


__all__ = ["ArtifactError", "ArtifactPublisher", "verify_report"]
