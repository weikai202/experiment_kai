"""Extraction of augmented Agent-facing schemas and host-only name mappings."""

from __future__ import annotations

from tool_sandbox.common.execution_context import get_current_context
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.state import AgentFacingToolInput

from .contracts import AgentTurnView, AdapterTurn, ControllerToolContext


def build_adapter_turn(agent_class: type, visible_messages: tuple) -> AdapterTurn:
    """Build separated prompt-facing and Controller-only turn contexts."""

    available_tools = agent_class.get_available_tools()
    schemas = convert_to_openai_tools(available_tools)
    if len(schemas) != len(available_tools):
        raise ValueError("upstream schema conversion changed tool cardinality")

    prompt_tools = tuple(
        AgentFacingToolInput(name=name, schema=schema)
        for name, schema in zip(available_tools, schemas, strict=True)
    )
    schema_names = [tool.schema_["function"]["name"] for tool in prompt_tools]
    if schema_names != list(available_tools):
        raise ValueError("upstream schema conversion changed tool names or order")

    complete_mapping = get_current_context().get_agent_to_execution_facing_tool_name()
    missing = set(available_tools) - set(complete_mapping)
    if missing:
        raise KeyError(f"missing current tool-name mappings: {sorted(missing)}")
    mapping = {name: complete_mapping[name] for name in available_tools}
    if set(mapping) != set(available_tools):
        raise ValueError("tool-name mapping keys do not match available tools")
    if len(mapping.values()) != len(set(mapping.values())):
        raise ValueError("execution-facing tool names must be unique")

    controller_context = ControllerToolContext(
        agent_to_execution_name=mapping,
        mapping_manifest_hash=canonical_sha256(mapping),
        tool_objects=available_tools,
    )
    return AdapterTurn(
        agent_view=AgentTurnView(
            visible_messages=visible_messages,
            available_tools=prompt_tools,
        ),
        controller_context=controller_context,
    )
