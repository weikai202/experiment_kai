from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from toolsandbox_pipeline.checkpointing import effective_effect_id
from toolsandbox_pipeline.metrics import MetricsAggregator
from toolsandbox_pipeline.schemas.accounting import (
    LogicalRequestAccountingInput,
    PhysicalAttemptAccountingInput,
)
from toolsandbox_pipeline.schemas.usage import TokenUsage
from toolsandbox_pipeline.schemas.checkpoint import QwenEffectiveEffect, QwenEffectKind


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
HASH = "sha256:" + "a" * 64
QWEN = "Qwen/Qwen3-32B"


def usage(input_tokens=10, output_tokens=4):
    return TokenUsage(
        input_tokens=input_tokens,
        uncached_input_tokens=input_tokens,
        cache_read_input_tokens=0,
        cache_write_input_tokens=0,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        usage_complete=True,
    )


def attempt(
    attempt_id="a", logical_id="l", *, role="policy", provider="vllm",
    model=QWEN, token_usage=None, status="completed", dispatched=True,
    recovery=False,
):
    evidence = dispatched
    return PhysicalAttemptAccountingInput(
        run_id="run", round_index=0, logical_request_id=logical_id,
        attempt_id=attempt_id, attempt_ordinal=0, role=role, phase="online",
        provider=provider, model=model,
        endpoint_kind="embedding" if role == "embedding" else "chat",
        dispatched=dispatched, replayed_after_unknown_outcome=recovery,
        status=status,
        unknown_outcome_kind="other" if status == "unknown_outcome" else None,
        started_at_utc=NOW if evidence else None,
        completed_at_utc=NOW if status == "completed" else None,
        latency_seconds=Decimal("0.25") if status == "completed" else None,
        usage=(token_usage if token_usage is not None else usage()) if evidence else None,
        response_sha256=HASH if status == "completed" else None,
    )


def canonical_application_id(label):
    return "application-" + canonical_label(label)


def canonical_label(label):
    from hashlib import sha256

    return sha256(label.encode("utf-8")).hexdigest()


def logical(logical_id="l", attempt_id="a", application_id="app", *, role="policy", provider="vllm", model=QWEN):
    return LogicalRequestAccountingInput(
        run_id="run", round_index=0, logical_request_id=logical_id,
        role=role, phase="online", provider=provider, model=model,
        status="applied", source_attempt_id=attempt_id,
        application_id=canonical_application_id(application_id),
    )


def effect(label="effect", *application_labels):
    applications = tuple(
        canonical_application_id(item) for item in (application_labels or ("app",))
    )
    values = {
        "effect_kind": QwenEffectKind.COMMITTED_ONLINE_ACTION,
        "effect_artifact_id": f"artifact-{label}",
        "effect_artifact_sha256": HASH,
        "ordered_application_ids": applications,
        "committed_checkpoint_id": f"checkpoint-{label}",
    }
    return QwenEffectiveEffect(
        effect_id=effective_effect_id(**values),
        **values,
    )


def aggregator():
    return MetricsAggregator(qwen_provider="vllm", qwen_model=QWEN)


def test_all_physical_tokens_but_only_effective_qwen_output_cost():
    rows = (
        attempt("policy", "lp", token_usage=usage(10, 4)),
        attempt("embedding", "le", role="embedding", provider="openai", model="text-embedding-3-small", token_usage=usage(3, 0)),
        attempt("user", "lu", role="user_simulator", provider="openai", model="gpt-4o-mini-2024-07-18", token_usage=usage(5, 2)),
        attempt("retry", "lp", token_usage=usage(10, 6), recovery=True),
    )
    requests = (logical("lp", "policy", "app-policy"),)
    totals = aggregator().aggregate(rows, requests, (effect("effect", "app-policy"),))
    assert totals.total_tokens == 40
    assert totals.total_cost == 4
    assert totals.cost_unit == "qwen_effective_output_tokens"
    assert totals.physical_dispatch_count == 4
    assert totals.dispatched_logical_request_count == 3
    assert totals.recovery_dispatch_count == 1


def test_revision_chain_cost_deduplicates_applications_across_effects():
    rows = (
        attempt("policy", "lp", token_usage=usage(1, 2)),
        attempt("critic", "lc", role="critic", token_usage=usage(1, 3)),
        attempt("revision", "lr", role="revision", token_usage=usage(1, 5)),
    )
    requests = (
        logical("lp", "policy", "ap"),
        logical("lc", "critic", "ac", role="critic"),
        logical("lr", "revision", "ar", role="revision"),
    )
    effects = (effect("e1", "ap", "ac", "ar"), effect("e2", "ar"))
    totals = aggregator().aggregate(rows, requests, effects)
    assert totals.total_cost == 10
    assert totals.total_tokens == 13


def test_no_substantive_effect_means_zero_cost_even_when_applied():
    totals = aggregator().aggregate((attempt(),), (logical(),), ())
    assert totals.total_cost == 0 and totals.cost_complete
    assert totals.total_tokens == 14


def test_linked_missing_output_usage_is_incomplete_independently():
    incomplete = TokenUsage(
        input_tokens=10, uncached_input_tokens=10,
        cache_read_input_tokens=0, cache_write_input_tokens=0,
        output_tokens=None, total_tokens=None, usage_complete=False,
    )
    totals = aggregator().aggregate(
        (attempt(token_usage=incomplete),), (logical(),), (effect(),),
    )
    assert totals.total_tokens is None and not totals.usage_complete
    assert totals.total_cost is None and not totals.cost_complete
    records = aggregator().request_records(
        (attempt(token_usage=incomplete),),
        (logical(),),
        (effect(),),
        recorded_at_utc=NOW,
    )
    assert records[0].effective_output_tokens is None


def test_unknown_attempt_poisons_usage_but_not_known_effect_cost():
    unknown = attempt("unknown", "other", status="unknown_outcome", token_usage=TokenUsage())
    totals = aggregator().aggregate(
        (attempt(), unknown), (logical(),), (effect(),),
    )
    assert totals.total_tokens is None and not totals.usage_complete
    assert totals.total_cost == 4 and totals.cost_complete
    assert totals.unknown_outcome_attempt_count == 1


def test_timeout_is_an_unknown_outcome_subtype_not_a_remapped_status():
    row = attempt(
        "timeout", "logical-timeout", status="unknown_outcome",
        token_usage=TokenUsage(),
    ).model_copy(update={"unknown_outcome_kind": "timeout"})
    totals = aggregator().aggregate((row,))
    assert totals.timeout_attempt_count == 1
    assert totals.unknown_outcome_attempt_count == 1


def test_undispatched_lifecycle_does_not_poison_usage():
    abandoned = attempt(
        "abandoned", "never", status="abandoned_before_dispatch", dispatched=False,
    )
    totals = aggregator().aggregate((abandoned,))
    assert totals.total_tokens == 0 and totals.usage_complete
    assert totals.abandoned_before_dispatch_count == 1


def test_idempotence_conflict_and_missing_source_fail_closed():
    row = attempt()
    assert aggregator().aggregate((row, row)).physical_dispatch_count == 1
    with pytest.raises(ValueError, match="conflicting attempt_id"):
        aggregator().aggregate((row, row.model_copy(update={"model": "other"})))
    with pytest.raises(ValueError, match="missing completed source"):
        aggregator().aggregate((row,), (logical(attempt_id="missing"),))


def test_effect_must_reference_applied_pinned_qwen():
    embedding = attempt(
        role="embedding", provider="openai", model="text-embedding-3-small",
        token_usage=usage(3, 0),
    )
    request = logical(
        role="embedding", provider="openai", model="text-embedding-3-small",
    )
    with pytest.raises(ValueError, match="pinned Qwen"):
        aggregator().aggregate((embedding,), (request,), (effect(),))


def test_effects_must_be_exact_and_canonically_identity_valid():
    valid = effect()
    with pytest.raises(TypeError, match="exact QwenEffectiveEffect"):
        aggregator().aggregate((attempt(),), (logical(),), (valid.model_dump(),))

    class FakeEffect:
        effect_id = valid.effect_id
        ordered_application_ids = valid.ordered_application_ids

    with pytest.raises(TypeError, match="exact QwenEffectiveEffect"):
        aggregator().aggregate((attempt(),), (logical(),), (FakeEffect(),))

    tampered = valid.model_copy(update={"effect_artifact_id": "tampered"})
    with pytest.raises(ValidationError, match="identity mismatch"):
        aggregator().aggregate((attempt(),), (logical(),), (tampered,))


def test_breakdown_filters_only_after_validating_original_effect():
    rows = (
        attempt("policy", "lp", token_usage=usage(1, 2)),
        attempt("critic", "lc", role="critic", token_usage=usage(1, 3)),
    )
    requests = (
        logical("lp", "policy", "ap"),
        logical("lc", "critic", "ac", role="critic"),
    )
    linked = effect("chain", "ap", "ac")
    by_role = {
        item.key: item.totals.total_cost
        for item in aggregator().breakdowns(rows, requests, (linked,))
        if item.dimension == "role"
    }
    assert by_role == {("critic",): 3, ("policy",): 2}
    tampered = linked.model_copy(update={"committed_checkpoint_id": "other"})
    with pytest.raises(ValidationError, match="identity mismatch"):
        aggregator().breakdowns(rows, requests, (tampered,))
    missing = effect("missing", "not-present")
    with pytest.raises(ValueError, match="missing applied response"):
        aggregator().breakdowns(rows, requests, (missing,))


def test_breakdowns_and_request_records_are_deterministic():
    rows = (attempt("b", "lb"), attempt("a", "la"))
    requests = (logical("lb", "b", "ab"), logical("la", "a", "aa"))
    effects = (effect("e", "aa"),)
    breakdowns = aggregator().breakdowns(rows, requests, effects)
    assert {row.dimension for row in breakdowns} == {
        "provider", "model", "role", "phase", "system_variant",
        "provider_model", "role_phase",
    }
    records = aggregator().request_records(rows, requests, effects, recorded_at_utc=NOW)
    assert [record.attempt.attempt_id for record in records] == ["a", "b"]
    assert [record.effective_output_tokens for record in records] == [4, 0]
    assert records == aggregator().request_records(rows, requests, effects, recorded_at_utc=NOW)


def test_strict_schema_rejects_coercion_and_extra_fields():
    with pytest.raises(ValidationError):
        attempt().model_copy(update={"attempt_ordinal": "0"}).__class__.model_validate(
            {**attempt().model_dump(), "attempt_ordinal": "0"}, strict=True,
        )
    with pytest.raises(ValidationError):
        PhysicalAttemptAccountingInput.model_validate(
            {**attempt().model_dump(), "raw_response": "forbidden"}, strict=True,
        )
