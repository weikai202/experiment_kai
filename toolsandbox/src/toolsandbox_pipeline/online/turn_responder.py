"""Durable orchestration for exactly one Agent-visible ToolSandbox state."""

from __future__ import annotations

from toolsandbox_pipeline.online.controller_inputs import ControllerInput
from toolsandbox_pipeline.reproducibility.canonical import (
    canonical_json_bytes,
    canonical_sha256,
)
from toolsandbox_pipeline.schemas.action import ActionEnvelope
from toolsandbox_pipeline.schemas.online_turn import (
    OnlineTurnAuditRecord,
    OnlineTurnDecision,
    OnlineTurnFailure,
    OnlineTurnFailureCode,
    OnlineTurnInput,
    OnlineTurnStage,
    OnlineTurnStageRecord,
    RetrievalReference,
)
from toolsandbox_pipeline.schemas.state import StateBuildInput
from toolsandbox_pipeline.skills.views import controller_view
from toolsandbox_pipeline.toolsandbox_adapter.contracts import AdapterTurn

from .durable_roles import DurableRoleApplication, DurableRoleExecution
from .routing import (
    RouteSelection,
    requires_critic,
    route_after_critic,
    route_after_revision,
)
from .turn_context import TurnContext


def _json(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)  # type: ignore[union-attr]
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    return value


def _retrieval_references(policy_bundle, world_bundle=None) -> tuple[RetrievalReference, ...]:
    references: list[RetrievalReference] = []
    for kind, hits in (
        ("policy", policy_bundle.policy_hits),
        ("skill", policy_bundle.skill_hits),
        ("world", () if world_bundle is None else world_bundle.hits),
    ):
        references.extend(
            RetrievalReference(
                retrieval_kind=kind,
                record_id=hit.record_id,
                record_version=hit.record_version,
                generation_id=hit.generation_id,
                rank=hit.rank,
                score=hit.score,
                document_sha256=hit.document_sha256,
                query_sha256=hit.query_sha256,
                cache_hit=hit.cache_hit,
                source_attempt_id=hit.source_attempt_id,
            )
            for hit in hits
        )
    return tuple(references)


class DurableTurnResponder:
    """Episode-local responder implementing the approved fixed routing table."""

    def __init__(self, context: TurnContext):
        if type(context) is not TurnContext:
            raise TypeError("TurnContext required")
        self._context = context
        self._input_fingerprint: str | None = None
        self._completed: OnlineTurnDecision | None = None
        self._stages: list[OnlineTurnStageRecord] = []
        self._current_stage: OnlineTurnStage | None = None
        self._active_state_id: str | None = None

    def respond(self, turn: AdapterTurn) -> ActionEnvelope:
        return self.respond_decision(turn).final_action

    def respond_decision(self, turn: AdapterTurn) -> OnlineTurnDecision:
        try:
            validated_input = OnlineTurnInput(
                identity=self._context.identity,
                adapter_turn=turn,
                committed_tool_outcomes=self._context.committed_tool_outcomes,
                pending_dependencies=self._context.pending_dependencies,
                prior_visible_failed_action_history=(
                    self._context.prior_visible_failed_action_history
                ),
                structured_constraint_tension=(
                    self._context.structured_constraint_tension
                ),
            )
            turn = validated_input.adapter_turn
            fingerprint = self._validate_and_fingerprint(turn)
            if self._input_fingerprint is not None and fingerprint != self._input_fingerprint:
                raise OnlineTurnFailure(
                    OnlineTurnFailureCode.IDENTITY_FAILURE,
                    stage=self._current_stage,
                    state_id=self._state_id(),
                )
            if self._completed is not None:
                return self._completed
            self._input_fingerprint = fingerprint
            final_checkpoint_id = self._final_checkpoint_id(fingerprint)
            restored = self._restore_final_decision(final_checkpoint_id, fingerprint)
            if restored is not None:
                self._completed = restored
                return restored
            decision = self._execute(turn, fingerprint)
        except OnlineTurnFailure as failure:
            self._write_terminal_failure(failure)
            raise
        except Exception:
            failure = OnlineTurnFailure(
                self._failure_code(),
                stage=self._current_stage,
                state_id=self._state_id(),
            )
            self._write_terminal_failure(failure)
            raise failure from None
        self._completed = decision
        return decision

    def _validate_and_fingerprint(self, turn: AdapterTurn) -> str:
        if type(turn) is not AdapterTurn:
            raise TypeError("AdapterTurn required")
        identity = self._context.identity
        names = tuple(tool.name for tool in turn.agent_view.available_tools)
        mapping = dict(turn.controller_context.agent_to_execution_name)
        if set(mapping) != set(names) or len(set(mapping.values())) != len(mapping):
            raise OnlineTurnFailure(
                OnlineTurnFailureCode.IDENTITY_FAILURE,
                stage=None,
                state_id=None,
            )
        if set(turn.controller_context.tool_objects) != set(names):
            raise OnlineTurnFailure(
                OnlineTurnFailureCode.IDENTITY_FAILURE,
                stage=None,
                state_id=None,
            )
        if canonical_sha256(mapping) != turn.controller_context.mapping_manifest_hash:
            raise OnlineTurnFailure(
                OnlineTurnFailureCode.IDENTITY_FAILURE,
                stage=None,
                state_id=None,
            )
        for tool in turn.agent_view.available_tools:
            function = tool.schema_.get("function")
            if not isinstance(function, dict) or function.get("name") != tool.name:
                raise OnlineTurnFailure(
                    OnlineTurnFailureCode.IDENTITY_FAILURE,
                    stage=None,
                    state_id=None,
                )
        payload = {
            "identity": identity.model_dump(mode="json"),
            "visible_messages": [item.model_dump(mode="json") for item in turn.agent_view.visible_messages],
            "available_tools": [item.model_dump(mode="json", by_alias=True) for item in turn.agent_view.available_tools],
            "agent_to_execution_name": mapping,
            "mapping_manifest_hash": turn.controller_context.mapping_manifest_hash,
            "committed_tool_outcomes": [_json(item) for item in self._context.committed_tool_outcomes],
            "pending_dependencies": [_json(item) for item in self._context.pending_dependencies],
            "action_history": [_json(item) for item in self._context.prior_visible_failed_action_history],
            "structured_constraint_tension": self._context.structured_constraint_tension,
        }
        return canonical_sha256(payload)

    @staticmethod
    def _final_checkpoint_id(input_fingerprint: str) -> str:
        return "checkpoint-" + canonical_sha256(
            {
                "protocol": "online-final-action-v1",
                "input_fingerprint": input_fingerprint,
            }
        )[7:]

    def _restore_final_decision(
        self,
        checkpoint_id: str,
        input_fingerprint: str,
    ) -> OnlineTurnDecision | None:
        loaded = self._context.checkpoint_sink.load_checkpoint(checkpoint_id)
        if loaded is None:
            return None
        event_kind, payload = loaded
        if event_kind != OnlineTurnStage.FINAL_ACTION_COMMITTED.value:
            raise ValueError("final checkpoint event kind mismatch")
        if not isinstance(payload, dict):
            raise TypeError("final checkpoint payload must be an object")
        if payload.get("input_fingerprint") != input_fingerprint:
            raise ValueError("final checkpoint input identity mismatch")
        decision = OnlineTurnDecision.model_validate_json(
            canonical_json_bytes(payload.get("decision")), strict=True
        )
        audit = OnlineTurnAuditRecord.model_validate_json(
            canonical_json_bytes(payload.get("audit_record")), strict=True
        )
        if decision.identity != self._context.identity or audit.identity != decision.identity:
            raise ValueError("restored turn identity mismatch")
        if decision.audit_record_sha256 != canonical_sha256(_json(audit)):
            raise ValueError("restored audit digest mismatch")
        if audit.final_action_sha256 != canonical_sha256(_json(decision.final_action)):
            raise ValueError("restored final action digest mismatch")
        if not audit.stages or audit.stages[-1].checkpoint_id != checkpoint_id:
            raise ValueError("restored final checkpoint reference mismatch")
        if audit.stages[-1].stage is not OnlineTurnStage.FINAL_ACTION_COMMITTED:
            raise ValueError("restored final stage mismatch")
        if payload.get("application_ids") != list(audit.application_ids):
            raise ValueError("restored application identity mismatch")
        self._stages = list(audit.stages)
        self._active_state_id = decision.state_id
        self._current_stage = OnlineTurnStage.FINAL_ACTION_COMMITTED
        return decision

    def _execute(self, turn: AdapterTurn, input_fingerprint: str) -> OnlineTurnDecision:
        context = self._context
        identity = context.identity
        mapping = dict(turn.controller_context.agent_to_execution_name)

        self._current_stage = OnlineTurnStage.STATE_BUILT
        build_result = context.state_builder.build(
            StateBuildInput(
                episode_id=identity.episode_id,
                scenario_id=identity.scenario_id,
                scenario_family_id=identity.family_id,
                visible_messages=turn.agent_view.visible_messages,
                committed_tool_outcomes=context.committed_tool_outcomes,
                available_tools=turn.agent_view.available_tools,
                pending_dependencies=context.pending_dependencies,
            )
        )
        state = build_result.state
        self._active_state_id = state.state_id
        if (
            state.agent_turn_index != identity.agent_turn_index
            or state.scenario_id != identity.scenario_id
            or state.scenario_family_id != identity.family_id
            or build_result.provenance_sidecar.state_id != state.state_id
        ):
            raise ValueError("built state identity mismatch")
        self._commit_stage(
            OnlineTurnStage.STATE_BUILT,
            {
                "state": _json(state),
                "provenance_sidecar": _json(build_result.provenance_sidecar),
                "fact_events": _json(build_result.fact_events),
                "input_fingerprint": input_fingerprint,
            },
        )

        self._current_stage = OnlineTurnStage.POLICY_RETRIEVAL_COMPLETED
        policy_bundle = context.retrieval.retrieve_policy_skills(
            state,
            canonical_to_agent={value: key for key, value in mapping.items()},
        )
        if (
            policy_bundle.state_id != state.state_id
            or policy_bundle.generation_id != identity.expected_generation_id
        ):
            raise ValueError("Policy retrieval identity mismatch")
        self._commit_stage(
            OnlineTurnStage.POLICY_RETRIEVAL_COMPLETED,
            {
                "generation_id": policy_bundle.generation_id,
                "state_id": policy_bundle.state_id,
                "query_sha256": policy_bundle.query.key.input_sha256,
                "references": _json(_retrieval_references(policy_bundle)),
            },
        )

        prepared_policy = context.role_request_builders.initial_policy(state, policy_bundle)
        policy_execution = self._execute_role(
            prepared_policy.request,
            OnlineTurnStage.INITIAL_POLICY_COMPLETED,
        )
        original_action = policy_execution.output
        assert isinstance(original_action, ActionEnvelope)
        policy_application = self._apply_role(
            policy_execution,
            OnlineTurnStage.INITIAL_POLICY_APPLIED,
            {"proposed_action": _json(original_action)},
        )

        controller_skills = self._controller_skill_views(policy_bundle)
        controller_input = ControllerInput(
            state=state,
            provenance_sidecar=build_result.provenance_sidecar,
            action=original_action,
            tool_metadata=context.controller_tool_metadata,
            agent_to_canonical_name=mapping,
            mapping_manifest_hash=turn.controller_context.mapping_manifest_hash,
            retrieved_skills=controller_skills,
            action_history=context.prior_visible_failed_action_history,
            generation_id=identity.expected_generation_id,
            reproducibility_profile=identity.profile,
            structured_constraint_tension=context.structured_constraint_tension,
        )
        self._current_stage = OnlineTurnStage.INITIAL_CONTROLLER_COMPLETED
        initial_controller = context.controller(controller_input)
        self._commit_stage(
            OnlineTurnStage.INITIAL_CONTROLLER_COMPLETED,
            {"controller_decision": _json(initial_controller)},
        )

        critic_execution = None
        critic_application = None
        critic_output = None
        revision_execution = None
        revision_application = None
        post_revision_controller = None
        world_bundle = None
        final_action = original_action

        if requires_critic(initial_controller):
            self._current_stage = OnlineTurnStage.WORLD_RETRIEVAL_COMPLETED
            world_bundle = context.retrieval.retrieve_world(
                state,
                original_action,
                controller_decision=initial_controller,
            )
            if (
                world_bundle.state_id != state.state_id
                or world_bundle.generation_id != identity.expected_generation_id
            ):
                raise ValueError("World retrieval identity mismatch")
            self._commit_stage(
                OnlineTurnStage.WORLD_RETRIEVAL_COMPLETED,
                {
                    "generation_id": world_bundle.generation_id,
                    "state_id": world_bundle.state_id,
                    "query_sha256": world_bundle.query.key.input_sha256,
                    "references": _json(_retrieval_references(policy_bundle, world_bundle)),
                },
            )
            prepared_critic = context.role_request_builders.critic(
                prepared_policy.context,
                original_action,
                initial_controller,
                world_bundle,
            )
            critic_execution = self._execute_role(
                prepared_critic.request,
                OnlineTurnStage.CRITIC_COMPLETED,
            )
            critic_output = critic_execution.output
            from toolsandbox_pipeline.schemas.critic import CriticOutput

            assert isinstance(critic_output, CriticOutput)
            critic_application = self._apply_role(
                critic_execution,
                OnlineTurnStage.CRITIC_APPLIED,
                {"critic_verdict": _json(critic_output)},
            )
            if route_after_critic(initial_controller, critic_output) is RouteSelection.REVISION:
                prepared_revision = context.role_request_builders.revision(
                    prepared_policy.context,
                    prepared_critic.context,
                    critic_output,
                )
                revision_execution = self._execute_role(
                    prepared_revision,
                    OnlineTurnStage.REVISION_COMPLETED,
                )
                revised_action = revision_execution.output
                assert isinstance(revised_action, ActionEnvelope)
                revision_application = self._apply_role(
                    revision_execution,
                    OnlineTurnStage.REVISION_APPLIED,
                    {"revised_action": _json(revised_action)},
                )
                self._current_stage = OnlineTurnStage.POST_REVISION_CONTROLLER_COMPLETED
                post_revision_controller = context.controller(
                    controller_input.model_copy(update={"action": revised_action})
                )
                self._commit_stage(
                    OnlineTurnStage.POST_REVISION_CONTROLLER_COMPLETED,
                    {"controller_decision": _json(post_revision_controller)},
                )
                if route_after_revision(post_revision_controller) is RouteSelection.REVISION:
                    final_action = revised_action
                else:
                    final_action = context.safe_failure_action_factory()
                    from toolsandbox_pipeline.online.turn_context import SAFE_CLARIFICATION_TEXT
                    from toolsandbox_pipeline.schemas.action import AssistantMessageAction

                    if (
                        not isinstance(final_action.action, AssistantMessageAction)
                        or final_action.action.content != SAFE_CLARIFICATION_TEXT
                    ):
                        raise ValueError("safe failure content/version mismatch")

        executions = tuple(
            item
            for item in (policy_execution, critic_execution, revision_execution)
            if item is not None
        )
        applications = tuple(
            item
            for item in (policy_application, critic_application, revision_application)
            if item is not None
        )
        retrieval_references = _retrieval_references(policy_bundle, world_bundle)
        application_ids = tuple(item.application_id for item in applications)
        final_action_sha256 = canonical_sha256(_json(final_action))
        final_checkpoint_id = self._final_checkpoint_id(input_fingerprint)
        final_stage_record = OnlineTurnStageRecord(
            stage=OnlineTurnStage.FINAL_ACTION_COMMITTED,
            ordinal=len(self._stages),
            component_sha256=final_action_sha256,
            checkpoint_id=final_checkpoint_id,
        )
        audit = OnlineTurnAuditRecord(
            identity=identity,
            state_id=state.state_id,
            generation_id=identity.expected_generation_id,
            stages=(*self._stages, final_stage_record),
            initial_controller_decision=initial_controller,
            post_revision_controller_decision=post_revision_controller,
            critic_verdict=critic_output,
            final_action_sha256=final_action_sha256,
            logical_request_ids=tuple(item.logical_request_id for item in executions),
            source_attempt_ids=tuple(item.source_attempt_id for item in executions),
            application_ids=application_ids,
            retrieval_references=retrieval_references,
        )
        audit_sha256 = canonical_sha256(_json(audit))
        decision = OnlineTurnDecision(
            identity=identity,
            state_id=state.state_id,
            generation_id=identity.expected_generation_id,
            final_action=final_action,
            initial_policy_logical_request_id=policy_execution.logical_request_id,
            critic_logical_request_id=None if critic_execution is None else critic_execution.logical_request_id,
            revision_logical_request_id=None if revision_execution is None else revision_execution.logical_request_id,
            initial_controller_decision=initial_controller,
            post_revision_controller_decision=post_revision_controller,
            critic_verdict=critic_output,
            revision_count=0 if revision_execution is None else 1,
            retrieval_references=retrieval_references,
            audit_record_sha256=audit_sha256,
        )
        self._current_stage = OnlineTurnStage.FINAL_ACTION_COMMITTED
        final_payload = {
            "input_fingerprint": input_fingerprint,
            "decision": _json(decision),
            "audit_record": _json(audit),
            "application_ids": list(application_ids),
        }
        hooks = context.lifecycle_hooks
        if hooks is not None:
            hooks.before_stage(OnlineTurnStage.FINAL_ACTION_COMMITTED)
        final_commit = context.durable_roles.commit_online_action(
            checkpoint_id=final_checkpoint_id,
            action=final_action,
            application_ids=application_ids,
            checkpoint_payload=final_payload,
        )
        if final_commit.committed_checkpoint_id != final_checkpoint_id:
            raise ValueError("final checkpoint identity mismatch")
        self._stages.append(final_stage_record)
        if hooks is not None:
            hooks.after_stage(OnlineTurnStage.FINAL_ACTION_COMMITTED)
        return decision

    def _controller_skill_views(self, bundle):
        by_identity = {
            (skill.skill_id, skill.version): skill for skill in self._context.generation.skills
        }
        views = []
        for hit in bundle.skill_hits:
            skill = by_identity.get((hit.record_id, hit.record_version))
            if skill is None:
                raise ValueError("retrieved Skill is absent from pinned generation")
            views.append(controller_view(skill, bundle.generation_id))
        return tuple(views)

    def _execute_role(
        self,
        prepared,
        stage: OnlineTurnStage,
    ) -> DurableRoleExecution:
        self._current_stage = stage
        execution = self._context.durable_roles.execute(prepared)
        self._commit_stage(
            stage,
            {
                "role": prepared.role,
                "logical_request_id": execution.logical_request_id,
                "source_attempt_id": execution.source_attempt_id,
                "output_sha256": canonical_sha256(_json(execution.output)),
                "reused_completed_response": execution.reused_completed_response,
            },
        )
        return execution

    def _apply_role(
        self,
        execution: DurableRoleExecution,
        stage: OnlineTurnStage,
        payload: object,
    ) -> DurableRoleApplication:
        self._current_stage = stage
        hooks = self._context.lifecycle_hooks
        if hooks is not None:
            hooks.before_stage(stage)
        application = self._context.durable_roles.apply(
            execution,
            stage=stage,
            state_id=self._state_id() or execution.prepared_request.state_id,
            application_payload=payload,
        )
        self._stages.append(
            OnlineTurnStageRecord(
                stage=stage,
                ordinal=len(self._stages),
                component_sha256=canonical_sha256(_json(payload)),
                checkpoint_id=application.committed_checkpoint_id,
            )
        )
        if hooks is not None:
            hooks.after_stage(stage)
        return application

    def _commit_stage(self, stage: OnlineTurnStage, payload: object) -> str:
        hooks = self._context.lifecycle_hooks
        try:
            if hooks is not None:
                hooks.before_stage(stage)
            checkpoint_id = self._context.checkpoint_sink.commit_stage(
                identity=self._context.identity,
                state_id=self._state_id() or "state-pending",
                stage=stage,
                component_payload=payload,
            )
        except OnlineTurnFailure:
            raise
        except Exception:
            raise OnlineTurnFailure(
                OnlineTurnFailureCode.CHECKPOINT_FAILURE,
                stage=stage,
                state_id=self._state_id(),
            ) from None
        if type(checkpoint_id) is not str or not checkpoint_id:
            raise ValueError("checkpoint sink returned an invalid identity")
        self._stages.append(
            OnlineTurnStageRecord(
                stage=stage,
                ordinal=len(self._stages),
                component_sha256=canonical_sha256(_json(payload)),
                checkpoint_id=checkpoint_id,
            )
        )
        if hooks is not None:
            hooks.after_stage(stage)
        return checkpoint_id

    def _state_id(self) -> str | None:
        return self._active_state_id

    def _failure_code(self) -> OnlineTurnFailureCode:
        stage = self._current_stage
        if stage in (OnlineTurnStage.STATE_BUILT,):
            return OnlineTurnFailureCode.STATE_FAILURE
        if stage in (
            OnlineTurnStage.POLICY_RETRIEVAL_COMPLETED,
            OnlineTurnStage.WORLD_RETRIEVAL_COMPLETED,
        ):
            return OnlineTurnFailureCode.RETRIEVAL_FAILURE
        if stage in (
            OnlineTurnStage.INITIAL_POLICY_COMPLETED,
            OnlineTurnStage.CRITIC_COMPLETED,
            OnlineTurnStage.REVISION_COMPLETED,
        ):
            return OnlineTurnFailureCode.ROLE_FAILURE
        if stage in (
            OnlineTurnStage.INITIAL_CONTROLLER_COMPLETED,
            OnlineTurnStage.POST_REVISION_CONTROLLER_COMPLETED,
        ):
            return OnlineTurnFailureCode.CONTROLLER_FAILURE
        if stage in (
            OnlineTurnStage.INITIAL_POLICY_APPLIED,
            OnlineTurnStage.CRITIC_APPLIED,
            OnlineTurnStage.REVISION_APPLIED,
            OnlineTurnStage.FINAL_ACTION_COMMITTED,
        ):
            return OnlineTurnFailureCode.APPLICATION_FAILURE
        return OnlineTurnFailureCode.CHECKPOINT_FAILURE

    def _write_terminal_failure(self, failure: OnlineTurnFailure) -> None:
        try:
            self._commit_stage(
                OnlineTurnStage.TERMINAL_FAILURE,
                {
                    "failure_code": failure.code.value,
                    "failed_stage": None if failure.stage is None else failure.stage.value,
                    "state_id": failure.state_id,
                },
            )
        except Exception:
            pass


__all__ = ["DurableTurnResponder"]
