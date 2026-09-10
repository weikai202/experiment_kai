from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from toolsandbox_pipeline.online.durable_roles import (
    DurableRoleApplication,
    DurableRoleExecution,
    DurableRoleExecutor,
    DurableOnlineActionCommit,
    LedgerQwenResponseSeam,
    Task011DurableRoleBackend,
)
from toolsandbox_pipeline.checkpointing import (
    CheckpointStore,
    LLMLedger,
    LogicalLLMRequestIdentity,
    RunIdentity,
)
from toolsandbox_pipeline.online.prompt_contracts import PreparedRoleRequest
from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    PhysicalAttemptResult,
    PhysicalAttemptStatus,
    ProviderRole,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnStage
from toolsandbox_pipeline.schemas.usage import PhysicalAttemptMetrics, TokenUsage
from toolsandbox_pipeline.schemas.ledger_accounting import AccountingScope


def prepared(role="policy"):
    return PreparedRoleRequest.model_construct(role=role, state_id="state:1")


def action(content="ok"):
    return ActionEnvelope(action={"type": "assistant_message", "content": content})


class Backend:
    def __init__(self):
        self.calls = []

    def execute(self, request):
        self.calls.append(("execute", request.role))
        return DurableRoleExecution(request, action(), "logical-1", "attempt-1")

    def apply(self, execution, *, stage, state_id, application_payload):
        self.calls.append(("apply", stage, state_id, application_payload))
        return DurableRoleApplication(execution, "application-1", "checkpoint-1")

    def commit_online_action(self, **values):
        self.calls.append(("effect", values))
        return DurableOnlineActionCommit("effect-1", "checkpoint-final")


def test_executor_validates_and_preserves_one_backend_transition():
    backend = Backend()
    executor = DurableRoleExecutor(backend)
    request = prepared()
    execution = executor.execute(request)
    application = executor.apply(
        execution,
        stage=OnlineTurnStage.INITIAL_POLICY_APPLIED,
        state_id="state:1",
        application_payload={"selected": "original"},
    )
    effect = executor.commit_online_action(
        checkpoint_id="checkpoint-final",
        action=action(),
        application_ids=(application.application_id,),
        checkpoint_payload={"decision": "final"},
    )
    assert effect.effect_id == "effect-1"
    assert [call[0] for call in backend.calls] == ["execute", "apply", "effect"]


def test_executor_rejects_wrong_application_stage_and_duplicate_effect_links():
    executor = DurableRoleExecutor(Backend())
    execution = executor.execute(prepared())
    with pytest.raises(ValueError, match="application stage"):
        executor.apply(
            execution,
            stage=OnlineTurnStage.FINAL_ACTION_COMMITTED,
            state_id="state:1",
            application_payload={},
        )
    with pytest.raises(ValueError, match="unique"):
        executor.commit_online_action(
            checkpoint_id="checkpoint-final",
            action=action(),
            application_ids=("same", "same"),
            checkpoint_payload={},
        )


def test_execution_output_must_match_role():
    with pytest.raises(TypeError, match="output type"):
        DurableRoleExecution(prepared("critic"), action(), "logical", "attempt")


def digest(character):
    return "sha256:" + character * 64


def real_prepared():
    return PreparedRoleRequest.model_construct(
        role="policy",
        state_id="state-1",
        generation_id="g000",
        canonical_input_fingerprint=digest("a"),
        output_schema_sha256=digest("c"),
    )


@pytest.fixture
def real_ledger(tmp_path):
    config = (
        Path(__file__).parents[2]
        / "configs/reproducibility/checkpointing_v1.json"
    )
    store = CheckpointStore.create(
        tmp_path / "run",
        RunIdentity(
            run_id="run-1",
            profile="offline",
            environment_identity=digest("1"),
            dataset_manifest_sha256=digest("2"),
            config_manifest_sha256=digest("3"),
            prompt_manifest_sha256=digest("4"),
            generation_manifest_sha256=digest("5"),
            fixture_manifest_sha256=digest("6"),
        ),
        config.resolve(),
    )
    try:
        yield LLMLedger(store)
    finally:
        store.close()


class SeamRunner:
    def __init__(self, seam):
        self.seam = seam
        self.dispatches = 0
        self.contexts = []

    def run(self, request, context):
        self.contexts.append(context)
        response = self.seam.load_completed_response(context)
        if response is None:
            self.dispatches += 1
            output = action("durable")
            raw = b'{"synthetic":"response"}'
            now = datetime.now(timezone.utc)
            usage = TokenUsage(
                input_tokens=3,
                uncached_input_tokens=3,
                cache_read_input_tokens=0,
                cache_write_input_tokens=0,
                output_tokens=5,
                total_tokens=8,
                usage_complete=True,
            )
            attempt = PhysicalAttemptResult(
                context=context,
                status=PhysicalAttemptStatus.COMPLETED,
                model="Qwen/Qwen3-32B",
                returned_model="Qwen/Qwen3-32B",
                finish_reason="stop",
                response_hash="sha256:" + sha256(raw).hexdigest(),
                metrics=PhysicalAttemptMetrics(
                    started_at=now,
                    completed_at=now,
                    latency_seconds=0.01,
                    usage=usage,
                ),
            )
            response = GatewayResponse(output, attempt, raw)
            self.seam.persist_completed_response(response)
        return SimpleNamespace(
            prepared_request=request,
            truncated=False,
            output=response.value,
            attempt=response.attempt,
        )


def real_backend(ledger):
    seam = LedgerQwenResponseSeam(ledger)
    policy = SeamRunner(seam)
    inert = SeamRunner(seam)
    backend = Task011DurableRoleBackend(
        ledger=ledger,
        runners={"policy": policy, "critic": inert, "revision": inert},
        run_id="run-1",
        phase="online",
        qwen_model="Qwen/Qwen3-32B",
        decoding_configuration_sha256=digest("b"),
        manifest_identity=digest("d"),
        accounting_scope=AccountingScope(
            run_id="run-1",
            task_id="episode-1",
            scenario_family_id="family-1",
            scenario_id="scenario-1",
            system_variant="generation_0",
        ),
    )
    return backend, policy


def test_task011_backend_reuses_completed_attempt_and_costs_one_effect(real_ledger):
    backend, runner = real_backend(real_ledger)
    request = real_prepared()
    first = backend.execute(request)
    applied = backend.apply(
        first,
        stage=OnlineTurnStage.INITIAL_POLICY_APPLIED,
        state_id="state-1",
        application_payload={"selected": "original"},
    )
    second = backend.execute(request)
    assert second.source_attempt_id == first.source_attempt_id
    assert second.reused_completed_response is True
    assert runner.dispatches == 1
    same_application = backend.apply(
        second,
        stage=OnlineTurnStage.INITIAL_POLICY_APPLIED,
        state_id="state-1",
        application_payload={"selected": "original"},
    )
    assert same_application.application_id == applied.application_id
    committed = backend.commit_online_action(
        checkpoint_id="checkpoint-final-state-1",
        action=second.output,
        application_ids=(applied.application_id,),
        checkpoint_payload={"state_id": "state-1", "final_action": "durable"},
    )
    assert committed.effect_id.startswith("effect-")
    assert real_ledger.effective_output_cost() == (5, True)


def test_task011_backend_reconciles_unknown_attempt_with_new_id(real_ledger):
    backend, runner = real_backend(real_ledger)
    request = real_prepared()
    identity = LogicalLLMRequestIdentity(
        run_id="run-1",
        role=ProviderRole.POLICY,
        phase="online",
        unit_reference="state-1",
        input_fingerprint=digest("a"),
        model="Qwen/Qwen3-32B",
        decoding_configuration_sha256=digest("b"),
        output_schema_sha256=digest("c"),
    )
    record = real_ledger.prepare_request(identity)
    abandoned = real_ledger.allocate_attempt(
        record.logical_request_id,
        manifest_identity=digest("d"),
        replayed_after_unknown_outcome=False,
    )
    real_ledger.mark_in_flight(abandoned)
    recovered = backend.execute(request)
    assert recovered.source_attempt_id != abandoned.attempt_id
    assert runner.contexts[-1].replayed_after_unknown_outcome is True
    assert real_ledger.attempt_status(abandoned.attempt_id).value == "unknown_outcome"
