import pytest
from tool_sandbox.common.execution_context import RoleType, set_current_context
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.toolsandbox_adapter.transactional_roles import (
    TransactionalRoleError,
    TransactionalUserRole,
)


class SyntheticUser(BaseRole):
    role_type = RoleType.USER

    def respond(self, ending_index=None):
        self.add_messages(
            [Message(RoleType.USER, RoleType.AGENT, "next", conversation_active=False)]
        )


def test_synthetic_user_commits_post_message_context(
    trajectory_store, start_context, episode_identity
):
    set_current_context(start_context)
    role = TransactionalUserRole(
        SyntheticUser(), identity=episode_identity, trajectory_store=trajectory_store
    )
    role.respond()
    assert start_context.max_sandbox_message_index == 1
    assert role.last_checkpoint_ordinal >= 1


def test_user_with_no_message_is_rejected(trajectory_store, start_context, episode_identity):
    class Empty(SyntheticUser):
        def respond(self, ending_index=None):
            return None

    set_current_context(start_context)
    role = TransactionalUserRole(
        Empty(), identity=episode_identity, trajectory_store=trajectory_store
    )
    with pytest.raises(TransactionalRoleError, match="no message"):
        role.respond()
