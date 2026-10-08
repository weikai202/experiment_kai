"""Lazy bridge that creates a real tau2 HalfDuplexAgent subclass in the live runtime."""

from __future__ import annotations

from typing import Any, Sequence

from .native_adapter import TauAgentAdapter, TurnPolicy


def make_native_agent(tools: Sequence[object], domain_policy: str, policy: TurnPolicy) -> object:
    try:
        from tau2.agent.base_agent import HalfDuplexAgent
    except ImportError as error:  # pragma: no cover - live Python 3.12 runtime only
        raise RuntimeError("native tau2 runtime is not installed") from error

    class NativeTauEvolutionAgent(HalfDuplexAgent[list[dict[str, Any]]]):
        def __init__(self) -> None:
            super().__init__(tools=list(tools), domain_policy=domain_policy)
            self._delegate = TauAgentAdapter(tools, domain_policy, policy)

        def get_init_state(self, message_history=None):
            return self._delegate.get_init_state(message_history)

        def generate_next_message(self, message, state):
            return self._delegate.generate_next_message(message, state)

        def set_seed(self, seed: int) -> None:
            self._delegate.set_seed(seed)

        @classmethod
        def is_stop(cls, message) -> bool:
            return False

        def stop(self, message=None, state=None) -> None:
            self._delegate.stop(message, state)

    return NativeTauEvolutionAgent()
