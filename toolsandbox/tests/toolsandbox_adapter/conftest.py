from __future__ import annotations

import pytest

from tool_sandbox.common.execution_context import ExecutionContext, new_context


@pytest.fixture
def sandbox_context():
    context = ExecutionContext(tool_allow_list=[])
    with new_context(context):
        yield context


def alpha(value: int) -> dict[str, int]:
    """Return a synthetic value.

    Args:
        value: Synthetic input value.
    """

    return {"value": value}


def beta(text: str) -> str:
    """Return synthetic text.

    Args:
        text: Synthetic input text.
    """

    return text
