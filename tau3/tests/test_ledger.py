import pytest

from tau3_evolution.ledger import (
    DurableAccountingLedger,
    LogicalRequest,
    OutputApplication,
    PhysicalAttempt,
    SubstantiveEffect,
)


def request(identity, scope="scope", role="qwen"):
    return LogicalRequest(
        identity,
        scope,
        role,
        "purpose",
        f"sha256:request-{identity}",
        "Qwen/Qwen3-32B" if role == "qwen" else "provider-model",
        "sha256:config",
    )


def attempt(identity, logical, input_tokens=1, output_tokens=2, completed=True):
    return PhysicalAttempt(
        identity, logical, input_tokens, output_tokens, completed, f"sha256:response-{identity}"
    )


def test_durable_exact_attempt_application_effect_snapshot(tmp_path):
    ledger = DurableAccountingLedger(tmp_path / "ledger.json")
    ledger.record_request(request("l1"))
    ledger.record_attempt(attempt("a1", "l1", 10, 6))
    ledger.record_effect(SubstantiveEffect("e1", "scope", "action", "sha256:x", True, True))
    ledger.record_application(OutputApplication("p1", "l1", "a1", "e1", "action", True, 0))
    ledger.record_request(request("l2", role="user_simulator"))
    ledger.record_attempt(attempt("a2", "l2", 4, 2))
    snapshot = ledger.snapshot(scope_id="scope", checkpoint_material_sha256="sha256:checkpoint")
    assert snapshot.total_tokens == 22 and snapshot.total_cost == 6


def test_ordered_multi_output_chain_counts_each_physical_output_once(tmp_path):
    ledger = DurableAccountingLedger(tmp_path / "ledger.json")
    ledger.record_effect(SubstantiveEffect("effect", "scope", "action", "sha256:x", True, True))
    for order, identity, tokens in ((0, "policy", 5), (1, "critic", 7), (2, "revision", 11)):
        ledger.record_request(request(identity))
        ledger.record_attempt(attempt(f"a-{identity}", identity, 1, tokens))
        ledger.record_application(
            OutputApplication(
                f"p-{identity}", identity, f"a-{identity}", "effect", identity, True, order
            )
        )
    snapshot = ledger.snapshot(scope_id="scope", checkpoint_material_sha256="sha256:c")
    assert snapshot.total_cost == 23
    assert snapshot.committed_application_count == 3


def test_rejected_and_noop_outputs_cost_zero(tmp_path):
    ledger = DurableAccountingLedger(tmp_path / "ledger.json")
    ledger.record_request(request("l"))
    ledger.record_attempt(attempt("a", "l", 10, 99))
    ledger.record_application(OutputApplication("p", "l", "a", None, "rejected", False, 0))
    snapshot = ledger.snapshot(scope_id="scope", checkpoint_material_sha256="sha256:c")
    assert snapshot.total_cost == 0 and snapshot.cost_complete


def test_missing_usage_or_attempt_is_incomplete(tmp_path):
    ledger = DurableAccountingLedger(tmp_path / "ledger.json")
    ledger.record_request(request("l"))
    assert not ledger.snapshot(
        scope_id="scope", checkpoint_material_sha256="sha256:c"
    ).usage_complete
    ledger.record_attempt(attempt("a", "l", None, None, False))
    assert not ledger.snapshot(
        scope_id="scope", checkpoint_material_sha256="sha256:c"
    ).usage_complete


def test_application_requires_exact_attempt_effect_and_contiguous_chain(tmp_path):
    ledger = DurableAccountingLedger(tmp_path / "ledger.json")
    ledger.record_request(request("l"))
    ledger.record_attempt(attempt("a", "l"))
    ledger.record_effect(SubstantiveEffect("e", "scope", "action", "sha256:x", True, True))
    ledger.record_application(OutputApplication("p", "l", "a", "e", "action", True, 1))
    with pytest.raises(ValueError, match="contiguous"):
        ledger.snapshot(scope_id="scope", checkpoint_material_sha256="sha256:c")
