from tool_sandbox.common.execution_context import RoleType, set_current_context
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.checkpointing import ToolLedger
from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import (
    ToolActionBinding,
    TransactionalExecutionEnvironment,
)


class SyntheticEnvironment(BaseRole):
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


def test_environment_prepares_executes_and_commits_one_batch(
    trajectory_store, start_context, episode_identity
):
    set_current_context(start_context)
    BaseRole.add_messages(
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
    binding = ToolActionBinding(
        action_sha256="sha256:" + "b" * 64,
        call_ids=("call-1",),
        selected_skill_ids=("skill-1",),
        canonical_tool_ids=("alpha",),
        effect_classes=("sandbox_read",),
        action_ordinal=1,
    )
    role = TransactionalExecutionEnvironment(
        SyntheticEnvironment(),
        identity=episode_identity,
        tool_ledger=ToolLedger(trajectory_store.store),
        trajectory_store=trajectory_store,
        binding_provider=lambda messages: binding,
        backend_manifest_sha256="sha256:" + "c" * 64,
    )
    role.respond()
    assert len(role.records) == 1
    assert role.records[0].committed and role.records[0].executed
    assert role.records[0].result_message_indices == (2,)
