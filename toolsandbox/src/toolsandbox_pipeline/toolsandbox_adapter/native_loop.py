"""Audited, checkpoint-friendly equivalent of pinned ``Scenario.play``."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Mapping

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
    set_current_context,
)
from tool_sandbox.roles.base_role import BaseRole


class NativeLoopError(RuntimeError):
    """Sanitized native-loop contract failure."""


@dataclass(frozen=True)
class RoleBoundary:
    recipient: RoleType
    ending_index: int | None
    before_message_index: int
    after_message_index: int
    initial_system_setup: bool


@dataclass(frozen=True)
class NativeLoopResult:
    context: ExecutionContext
    initial_max_message_index: int
    final_message_index: int
    invocation_counts: tuple[tuple[RoleType, int], ...]
    termination_reason: str

    @property
    def normally_terminated(self) -> bool:
        return self.termination_reason in {"conversation_ended", "message_limit"}


BoundaryHook = Callable[[RoleBoundary, ExecutionContext], None]
ContextInstallHook = Callable[[ExecutionContext], None]


class NativeLoop:
    """Run one supplied native scenario without evaluator or file side effects."""

    def run_fresh(
        self,
        scenario: object,
        roles: Mapping[RoleType, BaseRole],
        *,
        boundary_hook: BoundaryHook | None = None,
        context_install_hook: ContextInstallHook | None = None,
    ) -> NativeLoopResult:
        starting_context = getattr(scenario, "starting_context", None)
        max_messages = getattr(scenario, "max_messages", None)
        if type(starting_context) is not ExecutionContext:
            raise TypeError("native scenario starting context required")
        if type(max_messages) is not int or max_messages <= 0:
            raise TypeError("native scenario max_messages required")
        context = copy.deepcopy(starting_context)
        set_current_context(context)
        if context_install_hook is not None:
            context_install_hook(context)
        initial_max = context.max_sandbox_message_index
        counts: dict[RoleType, int] = {}
        sandbox = self._sandbox(context, all_history=True)
        for message_index in range(initial_max + 1):
            if (
                sandbox["recipient"][message_index]
                == RoleType.EXECUTION_ENVIRONMENT
                and sandbox["sender"][message_index] == RoleType.SYSTEM
            ):
                self._dispatch(
                    roles,
                    RoleType.EXECUTION_ENVIRONMENT,
                    counts,
                    ending_index=message_index,
                    initial_system_setup=True,
                    boundary_hook=boundary_hook,
                )
        if get_current_context().max_sandbox_message_index != initial_max:
            raise NativeLoopError("initial system setup appended a message")
        return self._continue(
            roles,
            max_messages=max_messages,
            initial_max_message_index=initial_max,
            counts=counts,
            boundary_hook=boundary_hook,
        )

    def run_resumed(
        self,
        context: ExecutionContext,
        roles: Mapping[RoleType, BaseRole],
        *,
        max_messages: int,
        initial_max_message_index: int,
        boundary_hook: BoundaryHook | None = None,
        context_install_hook: ContextInstallHook | None = None,
    ) -> NativeLoopResult:
        if type(context) is not ExecutionContext:
            raise TypeError("validated native ExecutionContext required")
        if type(max_messages) is not int or max_messages <= 0:
            raise TypeError("positive max_messages required")
        if type(initial_max_message_index) is not int or initial_max_message_index < 0:
            raise TypeError("valid initial message index required")
        if context.max_sandbox_message_index < initial_max_message_index:
            raise NativeLoopError("resume context predates initial scenario state")
        set_current_context(context)
        if context_install_hook is not None:
            context_install_hook(context)
        return self._continue(
            roles,
            max_messages=max_messages,
            initial_max_message_index=initial_max_message_index,
            counts={},
            boundary_hook=boundary_hook,
        )

    def _continue(
        self,
        roles: Mapping[RoleType, BaseRole],
        *,
        max_messages: int,
        initial_max_message_index: int,
        counts: dict[RoleType, int],
        boundary_hook: BoundaryHook | None,
    ) -> NativeLoopResult:
        while True:
            sandbox = self._sandbox(get_current_context())
            active = sandbox["conversation_active"][-1]
            last_index = int(sandbox["sandbox_message_index"][-1])
            if not active:
                reason = "conversation_ended"
                break
            if last_index >= max_messages + initial_max_message_index:
                reason = "message_limit"
                break
            recipient = sandbox["recipient"][-1]
            if not isinstance(recipient, RoleType):
                recipient = RoleType(recipient)
            self._dispatch(
                roles,
                recipient,
                counts,
                ending_index=None,
                initial_system_setup=False,
                boundary_hook=boundary_hook,
            )
        context = get_current_context()
        return NativeLoopResult(
            context=context,
            initial_max_message_index=initial_max_message_index,
            final_message_index=context.max_sandbox_message_index,
            invocation_counts=tuple(
                (role, counts[role]) for role in RoleType if counts.get(role, 0)
            ),
            termination_reason=reason,
        )

    @staticmethod
    def _dispatch(
        roles: Mapping[RoleType, BaseRole],
        recipient: RoleType,
        counts: dict[RoleType, int],
        *,
        ending_index: int | None,
        initial_system_setup: bool,
        boundary_hook: BoundaryHook | None,
    ) -> None:
        try:
            role = roles[recipient]
        except KeyError as error:
            raise NativeLoopError("missing native recipient role") from error
        before = get_current_context().max_sandbox_message_index
        role.respond(ending_index=ending_index)
        context = get_current_context()
        after = context.max_sandbox_message_index
        counts[recipient] = counts.get(recipient, 0) + 1
        if boundary_hook is not None:
            boundary_hook(
                RoleBoundary(
                    recipient=recipient,
                    ending_index=ending_index,
                    before_message_index=before,
                    after_message_index=after,
                    initial_system_setup=initial_system_setup,
                ),
                context,
            )

    @staticmethod
    def _sandbox(context: ExecutionContext, *, all_history: bool = False):
        return context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=all_history,
        )


__all__ = [
    "BoundaryHook",
    "ContextInstallHook",
    "NativeLoop",
    "NativeLoopError",
    "NativeLoopResult",
    "RoleBoundary",
]
