from pathlib import Path

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.orchestration.checkpoint_registry import (
    build_checkpoint_registry,
    query_checkpoint_registry,
    registry_bytes,
)
from toolsandbox_pipeline.schemas.run import GenerationObservation


HASH = "sha256:" + "c" * 64


def observation(generation, similarity=None, success=None, *, tokens=10, cost=3):
    number = int(generation[1:])
    common = dict(
        generation_id=generation,
        generation_manifest_sha256=HASH,
        generation_content_sha256=HASH,
    )
    if number == 0:
        return GenerationObservation(**common)
    return GenerationObservation(
        **common,
        producing_round_index=number - 1,
        train_shard_id=f"shard-{number - 1}",
        mean_native_similarity=similarity,
        fully_successful_rate=success,
        round_metric_record_id=HASH,
        total_running_time_seconds=float(number),
        total_tokens=tokens,
        usage_complete=True,
        total_cost=cost,
        cost_unit="qwen_effective_output_tokens",
        cost_complete=True,
        completion_status="complete",
    )


def entries():
    return (
        observation("g000"),
        observation("g001", 0.5, 0.4),
        observation("g002", 0.7, 0.6),
        observation("g003", 0.7, 0.5),
    )


def test_registry_keeps_g003_updated_but_diagnostic_ties_choose_lower_generation():
    registry = build_checkpoint_registry("run", entries())
    assert registry.updated_generation_id == "g003"
    assert registry.highest_observed_train_mean_similarity.generation_id == "g002"
    assert registry.highest_observed_train_fully_successful_rate.generation_id == "g002"
    for pointer in (
        registry.highest_observed_train_mean_similarity,
        registry.highest_observed_train_fully_successful_rate,
    ):
        assert pointer.selection_allowed is False
        assert pointer.comparability == "different_train_shards_not_directly_comparable"


def test_registry_round_rows_keep_direct_time_tokens_cost_and_completeness():
    registry = build_checkpoint_registry("run", entries())
    assert [row.total_running_time_seconds for row in registry.entries[1:]] == [1.0, 2.0, 3.0]
    assert [row.total_tokens for row in registry.entries[1:]] == [10, 10, 10]
    assert [row.total_cost for row in registry.entries[1:]] == [3, 3, 3]
    assert all(row.usage_complete and row.cost_complete for row in registry.entries[1:])


def test_incomplete_usage_and_cost_are_independent():
    rows = list(entries())
    rows[1] = rows[1].model_copy(update={
        "total_tokens": None, "usage_complete": False,
        "total_cost": None, "cost_complete": False,
    })
    registry = build_checkpoint_registry("run", tuple(rows))
    assert registry.entries[1].total_tokens is None
    assert registry.entries[1].total_cost is None


def test_order_mapping_hash_tampering_and_extra_fields_fail_closed():
    registry = build_checkpoint_registry("run", entries())
    with pytest.raises(ValidationError, match="ordered G000-G003"):
        registry.__class__.model_validate(
            {**registry.model_dump(), "entries": tuple(reversed(registry.entries))},
            strict=True,
        )
    with pytest.raises(ValidationError, match="identity"):
        registry.__class__.model_validate(
            {**registry.model_dump(), "registry_sha256": "sha256:" + "0" * 64},
            strict=True,
        )
    with pytest.raises(ValidationError):
        GenerationObservation.model_validate(
            {**entries()[0].model_dump(), "test_score": 1.0}, strict=True,
        )


def test_canonical_query_is_read_only(tmp_path):
    registry = build_checkpoint_registry("run", entries())
    path = tmp_path / "registry.json"
    path.write_bytes(registry_bytes(registry))
    before = path.read_bytes()
    assert query_checkpoint_registry(path) == registry
    assert path.read_bytes() == before
    path.write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        query_checkpoint_registry(path)
    with pytest.raises(ValueError, match="absolute"):
        query_checkpoint_registry(Path("relative.json"))
