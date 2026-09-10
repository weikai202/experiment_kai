from __future__ import annotations

from types import MappingProxyType

import pytest

from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.toolsandbox_adapter.tools import build_adapter_turn

def alpha(value: int) -> dict[str, int]:
    """Return a value.

    Args:
        value: Input value.
    """
    return {"value": value}


def beta(text: str) -> str:
    """Return text.

    Args:
        text: Input text.
    """
    return text


class SyntheticAgent:
    @classmethod
    def get_available_tools(cls):
        return {"public_alpha": alpha, "public_beta": beta}


def test_schema_order_and_controller_only_mapping(monkeypatch, sandbox_context):
    sandbox_context.name_to_tool = {"alpha": alpha, "beta": beta}
    monkeypatch.setattr(
        sandbox_context,
        "get_agent_to_execution_facing_tool_name",
        lambda: {"public_alpha": "canonical_one", "public_beta": "canonical_two"},
    )
    turn = build_adapter_turn(SyntheticAgent, ())
    assert [tool.name for tool in turn.agent_view.available_tools] == [
        "public_alpha",
        "public_beta",
    ]
    assert [tool.schema_["function"]["name"] for tool in turn.agent_view.available_tools] == [
        "public_alpha",
        "public_beta",
    ]
    assert dict(turn.controller_context.agent_to_execution_name) == {
        "public_alpha": "canonical_one",
        "public_beta": "canonical_two",
    }
    assert turn.controller_context.mapping_manifest_hash == canonical_sha256(
        {"public_alpha": "canonical_one", "public_beta": "canonical_two"}
    )
    assert isinstance(turn.controller_context.agent_to_execution_name, MappingProxyType)
    assert "canonical_one" not in repr(turn.agent_view)
    assert not hasattr(turn.agent_view, "controller_context")
    assert not hasattr(turn, "model_dump")


def test_mapping_must_be_complete_and_one_to_one(monkeypatch, sandbox_context):
    monkeypatch.setattr(
        sandbox_context,
        "get_agent_to_execution_facing_tool_name",
        lambda: {"public_alpha": "same"},
    )
    with pytest.raises(KeyError):
        build_adapter_turn(SyntheticAgent, ())

    monkeypatch.setattr(
        sandbox_context,
        "get_agent_to_execution_facing_tool_name",
        lambda: {"public_alpha": "same", "public_beta": "same"},
    )
    with pytest.raises(ValueError, match="unique"):
        build_adapter_turn(SyntheticAgent, ())


def test_end_conversation_is_not_added_by_adapter(monkeypatch, sandbox_context):
    monkeypatch.setattr(
        sandbox_context,
        "get_agent_to_execution_facing_tool_name",
        lambda: {"public_alpha": "canonical_one", "public_beta": "canonical_two"},
    )
    turn = build_adapter_turn(SyntheticAgent, ())
    assert "end_conversation" not in [x.name for x in turn.agent_view.available_tools]
