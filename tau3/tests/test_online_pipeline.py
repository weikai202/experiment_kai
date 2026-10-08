import pytest

from tau3_evolution.ledger import DurableAccountingLedger
from tau3_evolution.model_boundary import CalibrationSample, OutputLimitCalibration
from tau3_evolution.native_adapter import TauToolView
from tau3_evolution.online_pipeline import (
    RetrievedContext,
    TauOnlineTurnPipeline,
    committed_action_ledger_events,
)


class ScriptedTransport:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def complete(self, request, *, logical_request_id):
        self.requests.append(request)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return {
            "attempt_id": f"attempt-{len(self.requests)}",
            "logical_request_id": logical_request_id,
            "input_tokens": 10,
            "output_tokens": 5,
            "content": output,
        }


TOOLS = (
    TauToolView(
        "lookup",
        "Look up a record.",
        {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
    ),
)
CONTEXT = RetrievedContext(
    ("policy memory",),
    ("world memory",),
    ("skill",),
    ("policy-id",),
    ("world-id",),
    (("policy-id", 1),),
    (("world-id", 1),),
    (("skill-id", 1),),
    "sha256:retrieval",
)


def respond(transport):
    calibration = OutputLimitCalibration(
        source_kind="real_train_smoke",
        evolution_manifest_sha256="sha256:evolution-manifest",
        train_task_refs=(("airline", "train-task"),),
        samples=(CalibrationSample("airline", "train-task", 96, "stop"),),
        chosen_max_tokens=192,
        model="Qwen/Qwen3-32B",
        runtime_sha256="sha256:runtime",
    )
    return TauOnlineTurnPipeline(transport, calibration=calibration).respond(
        turn_id="turn", domain_policy="policy", tools=TOOLS, visible_history=(), retrieved=CONTEXT
    )


def test_keep_path_uses_policy_and_critic_only():
    transport = ScriptedTransport(
        [
            {"type": "tool_calls", "calls": [{"name": "lookup", "arguments": {"record_id": "1"}}]},
            {"decision": "KEEP", "reason": "grounded"},
        ]
    )
    result = respond(transport)
    assert result.controller_code == "critic_keep"
    assert not result.used_revision and len(result.usage_attempts) == 2
    assert result.applied_request_ids == result.request_ids
    assert all(
        request["chat_template_kwargs"] == {"enable_thinking": False}
        for request in transport.requests
    )


def test_revision_is_called_at_most_once_and_must_be_valid():
    transport = ScriptedTransport(
        [
            {"type": "message", "content": "initial"},
            {"decision": "REVISE", "reason": "use tool"},
            {"type": "tool_calls", "calls": [{"name": "lookup", "arguments": {"record_id": "1"}}]},
        ]
    )
    result = respond(transport)
    assert result.used_revision and result.controller_code == "revision_accepted"
    assert len(transport.requests) == 3


def test_invalid_initial_action_yields_fixed_clarification_without_execution():
    transport = ScriptedTransport(
        [{"type": "tool_calls", "calls": [{"name": "unknown", "arguments": {}}]}]
    )
    result = respond(transport)
    assert result.action.content and "clarify" in result.action.content
    assert result.controller_code == "invalid_initial_action_clarify"
    assert len(transport.requests) == 1
    assert result.applied_request_ids == ()


def test_malformed_critic_attempt_keeps_usage_and_only_policy_is_effective(tmp_path):
    transport = ScriptedTransport([{"type": "message", "content": "initial"}, "{malformed"])
    result = respond(transport)
    assert result.controller_code == "critic_incomplete_keep_initial"
    assert len(result.usage_attempts) == 2
    assert all(row.completed for row in result.attempt_evidence)
    events = committed_action_ledger_events(
        result, scope_id="scope", effect_id="effect", artifact_sha256="sha256:action"
    )
    ledger = DurableAccountingLedger(tmp_path / "ledger.json")
    for event in events:
        method = {
            "LogicalRequest": ledger.record_request,
            "PhysicalAttempt": ledger.record_attempt,
            "OutputApplication": ledger.record_application,
            "SubstantiveEffect": ledger.record_effect,
        }[type(event).__name__]
        method(event)
    snapshot = ledger.snapshot(scope_id="scope", checkpoint_material_sha256="sha256:checkpoint")
    assert snapshot.total_tokens == 30
    assert snapshot.total_cost == 5


def test_transport_failure_is_an_explicit_incomplete_attempt():
    result = respond(ScriptedTransport([RuntimeError("transport down")]))
    assert len(result.usage_attempts) == 1
    assert result.usage_attempts[0].input_tokens is None
    assert not result.attempt_evidence[0].completed
    assert result.attempt_evidence[0].response_sha256.startswith("sha256:")


def test_formal_pipeline_rejects_synthetic_calibration():
    synthetic = OutputLimitCalibration(
        source_kind="synthetic",
        evolution_manifest_sha256="sha256:evolution-manifest",
        train_task_refs=(("airline", "train-task"),),
        samples=(CalibrationSample("airline", "train-task", 96, "stop"),),
        chosen_max_tokens=192,
        model="Qwen/Qwen3-32B",
        runtime_sha256="sha256:runtime",
    )
    with pytest.raises(ValueError, match="real train-only smoke"):
        TauOnlineTurnPipeline(ScriptedTransport([]), calibration=synthetic)
