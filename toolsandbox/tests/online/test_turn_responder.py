from types import SimpleNamespace

import pytest

from toolsandbox_pipeline.online.durable_roles import (
    DurableRoleApplication,
    DurableRoleExecution,
    DurableRoleExecutor,
    DurableOnlineActionCommit,
)
from toolsandbox_pipeline.online.controller_inputs import ReproducibilityProfile
from toolsandbox_pipeline.online.prompt_contracts import (
    CriticContext,
    InitialPolicyContext,
    PreparedRoleRequest,
)
from toolsandbox_pipeline.online.state_builder import StateBuilder
from toolsandbox_pipeline.online.turn_context import (
    PreparedCritic,
    PreparedInitialPolicy,
    SAFE_CLARIFICATION_TEXT,
    TurnContext,
    TurnRoleRequestBuilders,
)
from toolsandbox_pipeline.online.turn_responder import DurableTurnResponder
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.controller import ControllerDecision
from toolsandbox_pipeline.schemas.critic import CriticOutput
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnIdentity, OnlineTurnStage
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnFailure, OnlineTurnFailureCode
from toolsandbox_pipeline.schemas.state import VisibleMessageInput, VisibleRole
from toolsandbox_pipeline.toolsandbox_adapter.contracts import (
    AdapterTurn,
    AgentTurnView,
    ControllerToolContext,
)


DIGEST = "sha256:" + "2" * 64


def identity():
    return OnlineTurnIdentity(
        run_id="run",
        profile=ReproducibilityProfile.STRICT_REPLAY,
        phase="train",
        round_index=0,
        shard_id="shard-0",
        family_id="family",
        scenario_id="scenario",
        episode_id="episode",
        agent_turn_index=0,
        expected_generation_id="g000",
        dataset_manifest_sha256=DIGEST,
        runtime_config_sha256=DIGEST,
        prompt_manifest_sha256=DIGEST,
        token_limit_config_sha256=DIGEST,
        fixture_manifest_sha256=DIGEST,
        environment_identity="environment",
    )


def adapter_turn(content="Please continue"):
    mapping = {}
    return AdapterTurn(
        agent_view=AgentTurnView(
            visible_messages=(
                VisibleMessageInput(
                    source_message_index=0,
                    sender=VisibleRole.USER,
                    recipient=VisibleRole.AGENT,
                    content=content,
                ),
            ),
            available_tools=(),
        ),
        controller_context=ControllerToolContext(
            agent_to_execution_name=mapping,
            mapping_manifest_hash=canonical_sha256(mapping),
            tool_objects={},
        ),
    )


def controller_decision(*, blocking=(), triggers=()):
    return ControllerDecision(
        blocking_codes=list(blocking),
        critic_trigger_codes=list(triggers),
        evidence=[
            {"code": code, "source_kind": "state", "source_ref": f"private:{code}"}
            for code in (*blocking, *triggers)
        ],
    )


def critic_output(verdict="accept"):
    if verdict == "accept":
        return CriticOutput(
            verdict="accept",
            predicted_outcome="success",
            predicted_effect="A visible result",
            error_codes=[],
            correction="",
        )
    return CriticOutput(
        verdict=verdict,
        predicted_outcome="uncertain" if verdict == "uncertain" else "failure",
        predicted_effect="No safe result",
        error_codes=["INSUFFICIENT_CONTEXT"],
        correction="Ask the missing information",
    )


def action(content):
    return ActionEnvelope(action={"type": "assistant_message", "content": content})


class FakeRetrieval:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.policy_calls = 0
        self.world_calls = 0

    @staticmethod
    def _query():
        return SimpleNamespace(key=SimpleNamespace(input_sha256=DIGEST))

    def retrieve_policy_skills(self, state, *, canonical_to_agent):
        self.policy_calls += 1
        assert canonical_to_agent == {}
        return SimpleNamespace(
            generation_id="g000",
            state_id=state.state_id,
            query=self._query(),
            policy_hits=(),
            policy_memory=(),
            skill_hits=(),
            skills=(),
        )

    def retrieve_world(self, state, proposed, *, controller_decision):
        self.world_calls += 1
        return SimpleNamespace(
            generation_id="g000",
            state_id=state.state_id,
            query=self._query(),
            hits=(),
            world_memory=(),
        )


class FakeController:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.inputs = []

    def __call__(self, value):
        self.inputs.append(value)
        return self.decisions.pop(0)


class RoleBackend:
    def __init__(self, *, policy, critic=None, revision=None):
        self.outputs = {"policy": policy, "critic": critic, "revision": revision}
        self.executed = []
        self.applied = []
        self.effects = []

    def execute(self, prepared):
        self.executed.append(prepared.role)
        return DurableRoleExecution(
            prepared_request=prepared,
            output=self.outputs[prepared.role],
            logical_request_id=f"logical-{prepared.role}",
            source_attempt_id=f"attempt-{prepared.role}",
        )

    def apply(self, execution, *, stage, state_id, application_payload):
        self.applied.append((execution.prepared_request.role, stage, state_id))
        return DurableRoleApplication(
            execution=execution,
            application_id=f"application-{execution.prepared_request.role}",
            committed_checkpoint_id=f"checkpoint-{stage.value}",
        )

    def commit_online_action(self, **values):
        self.effects.append(values)
        self.checkpoints[values["checkpoint_id"]] = (
            OnlineTurnStage.FINAL_ACTION_COMMITTED.value,
            values["checkpoint_payload"],
        )
        return DurableOnlineActionCommit(
            "effect-online", values["checkpoint_id"]
        )


class Checkpoints:
    def __init__(self, stored=None):
        self.events = []
        self.stored = {} if stored is None else stored

    def load_checkpoint(self, checkpoint_id):
        return self.stored.get(checkpoint_id)

    def commit_stage(self, *, identity, state_id, stage, component_payload):
        self.events.append((stage, state_id, component_payload))
        return f"checkpoint-{len(self.events)}-{stage.value}"


class Builders:
    def initial(self, state, retrieval):
        initial = InitialPolicyContext.model_construct(
            state_id=state.state_id,
            generation_id="g000",
        )
        request = PreparedRoleRequest.model_construct(
            role="policy", state_id=state.state_id, generation_id="g000"
        )
        return PreparedInitialPolicy(initial, request)

    def critic(self, initial, proposed, decision, retrieval):
        context = CriticContext.model_construct(initial=initial)
        request = PreparedRoleRequest.model_construct(
            role="critic", state_id=initial.state_id, generation_id="g000"
        )
        return PreparedCritic(context, request)

    def revision(self, initial, critic, output):
        return PreparedRoleRequest.model_construct(
            role="revision", state_id=initial.state_id, generation_id="g000"
        )


def responder(*, decisions, policy, critic=None, revision=None):
    generation = SimpleNamespace(
        manifest=SimpleNamespace(generation_id="g000"), skills=()
    )
    retrieval = FakeRetrieval(generation)
    backend = RoleBackend(policy=policy, critic=critic, revision=revision)
    checkpoints = Checkpoints()
    backend.checkpoints = checkpoints.stored
    builders = Builders()
    context = TurnContext(
        identity=identity(),
        generation=generation,
        retrieval=retrieval,
        state_builder=StateBuilder({}),
        controller=FakeController(*decisions),
        controller_tool_metadata=(),
        role_request_builders=TurnRoleRequestBuilders(
            initial_policy=builders.initial,
            critic=builders.critic,
            revision=builders.revision,
        ),
        durable_roles=DurableRoleExecutor(backend),
        checkpoint_sink=checkpoints,
    )
    return DurableTurnResponder(context), context, backend, checkpoints


def test_clean_path_calls_policy_only_and_commits_before_return():
    draft = action("original")
    subject, context, backend, checkpoints = responder(
        decisions=(controller_decision(),), policy=draft
    )
    result = subject.respond(adapter_turn())
    assert result == draft
    assert backend.executed == ["policy"]
    assert context.retrieval.policy_calls == 1
    assert context.retrieval.world_calls == 0
    assert all(
        event[0] is not OnlineTurnStage.FINAL_ACTION_COMMITTED
        for event in checkpoints.events
    )
    assert backend.effects[0]["application_ids"] == ("application-policy",)


def test_trigger_and_critic_accept_keep_original_without_revision():
    draft = action("original")
    subject, context, backend, _ = responder(
        decisions=(controller_decision(triggers=("ASSISTANT_MESSAGE_REVIEW",)),),
        policy=draft,
        critic=critic_output("accept"),
    )
    decision = subject.respond_decision(adapter_turn())
    assert decision.final_action == draft
    assert decision.revision_count == 0
    assert backend.executed == ["policy", "critic"]
    assert context.retrieval.world_calls == 1


def test_trigger_revision_runs_once_and_ignores_post_revision_triggers():
    revised = action("revised")
    subject, context, backend, _ = responder(
        decisions=(
            controller_decision(triggers=("ASSISTANT_MESSAGE_REVIEW",)),
            controller_decision(triggers=("ASSISTANT_MESSAGE_REVIEW",)),
        ),
        policy=action("original"),
        critic=critic_output("revise"),
        revision=revised,
    )
    decision = subject.respond_decision(adapter_turn())
    assert decision.final_action == revised
    assert decision.revision_count == 1
    assert backend.executed == ["policy", "critic", "revision"]
    assert context.retrieval.policy_calls == 1
    assert context.retrieval.world_calls == 1


def test_initial_block_cannot_be_overridden_and_revised_block_clarifies():
    subject, _, backend, _ = responder(
        decisions=(
            controller_decision(blocking=("UNGROUNDED_ARGUMENT",)),
            controller_decision(blocking=("MISSING_DEPENDENCY",)),
        ),
        policy=action("original"),
        critic=critic_output("accept"),
        revision=action("still blocked"),
    )
    decision = subject.respond_decision(adapter_turn())
    assert decision.final_action.action.content == SAFE_CLARIFICATION_TEXT
    assert decision.revision_count == 1
    assert backend.executed == ["policy", "critic", "revision"]
    assert backend.effects[0]["application_ids"] == (
        "application-policy",
        "application-critic",
        "application-revision",
    )


def test_exact_reentry_reuses_final_decision_without_reexecution():
    subject, _, backend, _ = responder(
        decisions=(controller_decision(),), policy=action("original")
    )
    turn = adapter_turn()
    first = subject.respond_decision(turn)
    second = subject.respond_decision(turn)
    assert second is first
    assert backend.executed == ["policy"]


def test_new_responder_restores_final_decision_without_rerunning_dependencies():
    subject, context, backend, checkpoints = responder(
        decisions=(controller_decision(),), policy=action("original")
    )
    turn = adapter_turn()
    first = subject.respond_decision(turn)
    replacement = DurableTurnResponder(context)
    second = replacement.respond_decision(turn)
    assert second == first
    assert backend.executed == ["policy"]
    assert context.retrieval.policy_calls == 1
    assert len(context.controller.inputs) == 1
    assert len(checkpoints.stored) == 1


def test_conflicting_reentry_stops_without_second_model_execution():
    subject, _, backend, checkpoints = responder(
        decisions=(controller_decision(),), policy=action("original")
    )
    subject.respond(adapter_turn("first"))
    with pytest.raises(OnlineTurnFailure) as caught:
        subject.respond(adapter_turn("changed"))
    assert caught.value.code is OnlineTurnFailureCode.IDENTITY_FAILURE
    assert backend.executed == ["policy"]
    assert checkpoints.events[-1][0] is OnlineTurnStage.TERMINAL_FAILURE
