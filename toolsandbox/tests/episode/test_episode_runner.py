from collections import OrderedDict

from tool_sandbox.common.evaluation import EvaluationResult
from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.checkpointing import ToolLedger
from toolsandbox_pipeline.schemas.checkpoint import BlobReference
from toolsandbox_pipeline.schemas.trajectory import OnlineTurnRecord
from toolsandbox_pipeline.toolsandbox_adapter.episode_runner import (
    EpisodeRunInput,
    EpisodeRunner,
    build_skill_attributions,
)
from toolsandbox_pipeline.schemas.trajectory import TrustedEvaluatorRecord
from toolsandbox_pipeline.toolsandbox_adapter.native_evaluator import NativeEvaluator
from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import (
    EpisodeAgentRole,
    ToolActionBinding,
    TransactionalExecutionEnvironment,
    TransactionalUserRole,
)


def test_skill_attribution_requires_executed_nonrolledback_calls():
    evaluator = TrustedEvaluatorRecord(
        milestone_similarity=1.0,
        minefield_similarity=0.0,
        similarity=1.0,
        turn_count=1,
        milestone_mapping=(),
        minefield_mapping=(),
        fully_successful=True,
        evaluation_definition_sha256="sha256:" + "1" * 64,
        ending_context_sha256="sha256:" + "2" * 64,
    )
    assert build_skill_attributions(
        (),
        evaluator_record=evaluator,
        evaluator_record_sha256="sha256:" + "3" * 64,
        generation_id="g000",
        skill_versions={},
    ) == ()
    assert EpisodeRunner is not None and EpisodeRunInput is not None


def test_preinstall_validation_failure_does_not_persist_unrelated_context(
    trajectory_store,
    start_context,
    episode_identity,
    scenario_record,
):
    class Scenario:
        starting_context = start_context
        max_messages = 4

    runner = EpisodeRunner(
        trajectory_store=trajectory_store,
        native_evaluator=NativeEvaluator(trajectory_store),
        physical_attempt_provider=lambda _: (),
        boot_id="boot-1",
    )
    before = trajectory_store.store.high_water_marks()["checkpoint_event_ordinal"]
    import pytest

    with pytest.raises(TypeError, match="EpisodeAgentRole required"):
        runner.run(
            EpisodeRunInput(
                scenario=Scenario(),
                manifest_record=scenario_record,
                identity=episode_identity,
                roles={},
            )
        )
    assert trajectory_store.store.high_water_marks()["checkpoint_event_ordinal"] == before


def test_real_episode_flow_binds_executed_calls_to_originating_turn(
    trajectory_store,
    start_context,
    episode_identity,
    scenario_record,
):
    action_sha = "sha256:" + "b" * 64

    def reference(character):
        return BlobReference(
            sha256="sha256:" + character * 64,
            byte_count=1,
            media_type="application/vnd.toolsandbox.canonical+json",
            schema_name="SyntheticDecision",
            schema_version=1,
            content_visibility="restricted",
        )

    class Agent(EpisodeAgentRole):
        def __init__(self):
            self._records = []

        @property
        def records(self):
            return tuple(self._records)

        def respond(self, ending_index=None):
            index = len(self._records)
            if index == 0:
                self.add_messages(
                    [
                        Message(
                            RoleType.AGENT,
                            RoleType.EXECUTION_ENVIRONMENT,
                            "alpha()",
                            openai_tool_call_id="call-1",
                            openai_function_name="alpha",
                        )
                    ]
                )
                final_sha = action_sha
                logical = "logical-tool-turn"
                attempt = "attempt-tool-turn"
                application = "application-tool-turn"
            else:
                self.add_messages(
                    [
                        Message(
                            RoleType.AGENT,
                            RoleType.USER,
                            "done",
                            conversation_active=False,
                        )
                    ]
                )
                final_sha = "sha256:" + "c" * 64
                logical = "logical-final-turn"
                attempt = "attempt-final-turn"
                application = "application-final-turn"
            self._records.append(
                OnlineTurnRecord(
                    agent_turn_index=index,
                    state_id=f"state-{index}",
                    decision_reference=reference("d" if index == 0 else "e"),
                    decision_sha256="sha256:" + ("d" if index == 0 else "e") * 64,
                    final_action_sha256=final_sha,
                    logical_request_ids=(logical,),
                    source_attempt_ids=(attempt,),
                    application_ids=(application,),
                )
            )

    class Environment(BaseRole):
        role_type = RoleType.EXECUTION_ENVIRONMENT

        def respond(self, ending_index=None):
            self.add_messages(
                [
                    Message(
                        RoleType.EXECUTION_ENVIRONMENT,
                        RoleType.AGENT,
                        "ok",
                        openai_tool_call_id="call-1",
                        openai_function_name="alpha",
                    )
                ]
            )

    class User(BaseRole):
        role_type = RoleType.USER

        def respond(self, ending_index=None):
            raise AssertionError("synthetic User should not be called")

    class Evaluation:
        def evaluate(self, **kwargs):
            return EvaluationResult(
                milestone_mapping=OrderedDict([(0, (0, 1.0))]),
                minefield_mapping=OrderedDict(),
                milestone_similarity=1.0,
                minefield_similarity=0.0,
                turn_count=2,
            )

    class Scenario:
        starting_context = start_context
        max_messages = 4
        evaluation = Evaluation()

    agent = Agent()
    environment = TransactionalExecutionEnvironment(
        Environment(),
        identity=episode_identity,
        tool_ledger=ToolLedger(trajectory_store.store),
        trajectory_store=trajectory_store,
        binding_provider=lambda _: ToolActionBinding(
            action_sha256=action_sha,
            call_ids=("call-1",),
            selected_skill_ids=("skill-1",),
            canonical_tool_ids=("alpha",),
            effect_classes=("sandbox_read",),
            action_ordinal=1,
        ),
        backend_manifest_sha256="sha256:" + "f" * 64,
    )
    user = TransactionalUserRole(
        User(), identity=episode_identity, trajectory_store=trajectory_store
    )
    result = EpisodeRunner(
        trajectory_store=trajectory_store,
        native_evaluator=NativeEvaluator(trajectory_store),
        physical_attempt_provider=lambda logical_ids: tuple(
            "physical-" + item for item in logical_ids
        ),
        boot_id="boot-1",
    ).run(
        EpisodeRunInput(
            scenario=Scenario(),
            manifest_record=scenario_record,
            identity=episode_identity,
            roles={
                RoleType.AGENT: agent,
                RoleType.USER: user,
                RoleType.EXECUTION_ENVIRONMENT: environment,
            },
            skill_versions={"skill-1": "v1"},
            eligible_for_train_offline_consumption=True,
        )
    )
    trajectory = trajectory_store.load_trajectory(
        result.trusted_trajectory_reference
    )
    assert trajectory.online_turns[0].executed_call_ids == ("call-1",)
    assert trajectory.online_turns[1].executed_call_ids == ()
    assert trajectory.skill_attributions[0].executed_call_ids == ("call-1",)
    assert trajectory.skill_attributions[0].fully_successful is True
