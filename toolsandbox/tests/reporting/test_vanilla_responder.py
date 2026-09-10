from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest
from tool_sandbox.common.execution_context import (
    ExecutionContext,
    RoleType,
    set_current_context,
)
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.checkpointing import CheckpointStore, LLMLedger, RunIdentity
from toolsandbox_pipeline.providers import QwenGateway
from toolsandbox_pipeline.providers.contracts import (
    GatewayResponse,
    PhysicalAttemptResult,
    PhysicalAttemptStatus,
    ProviderRole,
    TransportResponse,
)
from toolsandbox_pipeline.reporting.vanilla_responder import (
    TransactionalVanillaAgentRole,
    VanillaRequestNamespace,
    VanillaResponder,
    load_vanilla_assets,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.runtime import QwenConfig
from toolsandbox_pipeline.schemas.trajectory import EpisodeIdentity
from toolsandbox_pipeline.schemas.state import (
    AgentFacingToolInput,
    VisibleMessageInput,
    VisibleRole,
)
from toolsandbox_pipeline.schemas.usage import PhysicalAttemptMetrics, TokenUsage
from toolsandbox_pipeline.toolsandbox_adapter.contracts import (
    AdapterTurn,
    AgentTurnView,
    ControllerToolContext,
)
from toolsandbox_pipeline.toolsandbox_adapter.episode_runner import EpisodeRunner
from toolsandbox_pipeline.toolsandbox_adapter.trajectory_store import TrajectoryStore
from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import EpisodeAgentRole


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_CONFIG = ROOT / "configs/reproducibility/checkpointing_v1.json"
DIGEST = "sha256:" + "1" * 64


def namespace(*, episode_id="episode", scenario_id="scenario", turn_index=0):
    return VanillaRequestNamespace(
        run_id="vanilla-run",
        episode_id=episode_id,
        scenario_family_id="family",
        scenario_id=scenario_id,
        starting_turn_index=turn_index,
    )


class FakeGateway:
    def __init__(self) -> None:
        self.config = QwenConfig(structured_output_wire_mode="guided_json")
        self.calls: list[tuple[object, list[dict], object, dict]] = []

    def generate(self, context, messages, output_model, **kwargs):
        self.calls.append((context, messages, output_model, kwargs))
        action = ActionEnvelope.model_validate(
            {"action": {"type": "assistant_message", "content": "done"}}
        )
        raw = b'{"synthetic":"restricted"}'
        now = datetime.now(timezone.utc)
        attempt = PhysicalAttemptResult(
            context=context,
            status=PhysicalAttemptStatus.COMPLETED,
            model=self.config.model,
            returned_model=self.config.model,
            finish_reason="stop",
            response_hash="sha256:" + sha256(raw).hexdigest(),
            metrics=PhysicalAttemptMetrics(
                started_at=now,
                completed_at=now,
                latency_seconds=0.01,
                usage=TokenUsage(
                    input_tokens=5,
                    uncached_input_tokens=5,
                    cache_read_input_tokens=0,
                    cache_write_input_tokens=0,
                    output_tokens=3,
                    total_tokens=8,
                    usage_complete=True,
                ),
            ),
        )
        return GatewayResponse(action, attempt, raw)


class CaptureTransport:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        payload = {
            "model": "Qwen/Qwen3-32B",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"action":{"type":"assistant_message","content":"ok"}}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        return TransportResponse(json.dumps(payload).encode(), payload)


@pytest.fixture
def assets():
    return load_vanilla_assets(ROOT)


@pytest.fixture
def ledger(tmp_path, assets):
    identity = RunIdentity(
        run_id="vanilla-run",
        profile="offline",
        environment_identity=DIGEST,
        dataset_manifest_sha256=DIGEST,
        config_manifest_sha256=DIGEST,
        prompt_manifest_sha256=assets.prompt_manifest_sha256,
        generation_manifest_sha256=DIGEST,
        fixture_manifest_sha256=DIGEST,
    )
    store = CheckpointStore.create(
        tmp_path / "checkpoint", identity, CHECKPOINT_CONFIG.resolve()
    )
    try:
        yield LLMLedger(store)
    finally:
        store.close()


def adapter_turn() -> AdapterTurn:
    return AdapterTurn(
        agent_view=AgentTurnView(
            visible_messages=(
                VisibleMessageInput(
                    source_message_index=1,
                    sender=VisibleRole.USER,
                    recipient=VisibleRole.AGENT,
                    content="visible request",
                ),
            ),
            available_tools=(
                AgentFacingToolInput(
                    name="public_tool",
                    schema={
                        "type": "function",
                        "function": {
                            "name": "public_tool",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ),
            ),
        ),
        controller_context=ControllerToolContext(
            agent_to_execution_name={"public_tool": "canonical-secret-sentinel"},
            mapping_manifest_hash=DIGEST,
            tool_objects={"canonical-secret-sentinel": lambda: None},
        ),
    )


def test_assets_are_exact_and_provisional_is_not_formal(assets, ledger):
    assert assets.prompt_manifest.role == "vanilla"
    assert assets.prompt_manifest.output_model_name == "ActionEnvelope"
    assert assets.token_limit.status == "provisional"
    assert assets.token_limit.max_tokens == 256
    with pytest.raises(ValueError, match="calibrated"):
        VanillaResponder(
            ledger=ledger,
            gateway=FakeGateway(),
            assets=assets,
            namespace=namespace(),
            phase="final_test",
            manifest_identity=DIGEST,
            mode="formal",
        )


def test_vanilla_uses_only_agent_view_and_durable_effect_cost(assets, ledger):
    gateway = FakeGateway()
    responder = VanillaResponder(
        ledger=ledger,
        gateway=gateway,
        assets=assets,
        namespace=namespace(),
        phase="synthetic_test",
        manifest_identity=DIGEST,
    )
    turn = adapter_turn()
    first = responder.respond(turn)
    evidence = responder.last_evidence
    recovered = VanillaResponder(
        ledger=ledger,
        gateway=gateway,
        assets=assets,
        namespace=namespace(),
        phase="synthetic_test",
        manifest_identity=DIGEST,
    )
    assert recovered.respond(turn) == first
    assert len(gateway.calls) == 1
    context, messages, output_model, kwargs = gateway.calls[0]
    assert context.role is ProviderRole.VANILLA
    assert output_model is ActionEnvelope
    assert kwargs["max_tokens"] == 256
    assert kwargs["token_limit_config"].role == "vanilla"
    assert messages[0] == {"role": "system", "content": assets.prompt}
    assert "visible request" in messages[1]["content"]
    assert "public_tool" in messages[1]["content"]
    assert "canonical-secret-sentinel" not in str(messages)
    assert evidence is not None
    assert ledger.effective_output_cost() == (3, True)
    assert len(ledger.applications()) == 1
    assert len(ledger.effects()) == 1


def test_vanilla_qwen_wire_is_non_thinking_and_schema_bound():
    transport = CaptureTransport()
    gateway = QwenGateway(
        QwenConfig(structured_output_wire_mode="guided_json"), transport=transport
    )
    from toolsandbox_pipeline.providers.contracts import RequestContext

    context = RequestContext(
        logical_request_id="logical-vanilla",
        attempt_id="attempt-vanilla",
        role=ProviderRole.VANILLA,
        phase="synthetic",
        unit_reference="visible-state",
        input_fingerprint=DIGEST,
        replayed_after_unknown_outcome=False,
        manifest_identity=DIGEST,
    )
    gateway.generate(
        context,
        [{"role": "user", "content": "visible"}],
        ActionEnvelope,
        max_tokens=256,
    )
    request = transport.calls[0]
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert request["extra_body"]["guided_json"] == ActionEnvelope.model_json_schema()


def test_no_substantive_effect_means_zero_effective_cost(ledger):
    assert ledger.applications() == ()
    assert ledger.effects() == ()
    assert ledger.effective_output_cost() == (0, True)


def test_identical_visible_payloads_in_distinct_scenarios_never_reuse(assets, ledger):
    gateway = FakeGateway()
    first = VanillaResponder(
        ledger=ledger,
        gateway=gateway,
        assets=assets,
        namespace=namespace(episode_id="episode-1", scenario_id="scenario-1"),
        phase="synthetic_test",
        manifest_identity=DIGEST,
    )
    second = VanillaResponder(
        ledger=ledger,
        gateway=gateway,
        assets=assets,
        namespace=namespace(episode_id="episode-2", scenario_id="scenario-2"),
        phase="synthetic_test",
        manifest_identity=DIGEST,
    )
    first.respond(adapter_turn())
    second.respond(adapter_turn())
    assert first.last_evidence.logical_request_id != second.last_evidence.logical_request_id
    assert len(gateway.calls) == 2


def test_vanilla_role_runs_as_explicit_marker_and_lookalike_is_rejected(
    assets, ledger
):
    responder = VanillaResponder(
        ledger=ledger,
        gateway=FakeGateway(),
        assets=assets,
        namespace=namespace(),
        phase="synthetic_test",
        manifest_identity=DIGEST,
    )
    trajectory_store = TrajectoryStore(ledger.store, environment_identity=DIGEST)
    identity = EpisodeIdentity(
        run_id="vanilla-run",
        profile="offline",
        phase="synthetic_test",
        family_id="family",
        scenario_id="scenario",
        episode_id="episode",
        manifest_position=0,
        system_variant="vanilla",
        generation_id="g000",
        starting_context_sha256=DIGEST,
        evaluation_definition_sha256=DIGEST,
        agent_tool_schema_sha256=DIGEST,
        dataset_manifest_sha256=DIGEST,
        runtime_config_sha256=DIGEST,
        prompt_manifest_sha256=assets.prompt_manifest_sha256,
        token_limit_config_sha256=assets.token_limit_sha256,
        fixture_manifest_sha256=DIGEST,
        environment_sha256=DIGEST,
        max_messages=4,
    )
    role = TransactionalVanillaAgentRole(
        responder, identity=identity, trajectory_store=trajectory_store
    )
    context = ExecutionContext(tool_allow_list=[])
    set_current_context(context)
    BaseRole.add_messages(
        [Message(RoleType.USER, RoleType.AGENT, "visible", conversation_active=True)]
    )
    role.respond()
    assert isinstance(role, EpisodeAgentRole)
    assert len(role.records) == 1

    class LookalikeAgent(BaseRole):
        role_type = RoleType.AGENT
        records = ()

        def respond(self, ending_index=None):
            return None

    with pytest.raises(TypeError, match="EpisodeAgentRole required"):
        EpisodeRunner._validate_roles({RoleType.AGENT: LookalikeAgent()})
