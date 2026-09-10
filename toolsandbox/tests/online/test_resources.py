from toolsandbox_pipeline.online.resources import calls_are_independent, resolve_resource_template, validate_resource_template
from toolsandbox_pipeline.schemas.action import FunctionCall
from toolsandbox_pipeline.schemas.tool_metadata import ControllerToolMetadata, ResourceTemplate, ToolEffect, ToolRisk


def meta(*, reads=(), writes=(), safe=True):
    return ControllerToolMetadata(canonical_tool_name="x", effect=ToolEffect.SANDBOX_WRITE, risk=ToolRisk.MEDIUM, prerequisites=(), parallel_safe=safe, read_resources=reads, write_resources=writes, critic_required=False, agent_forbidden=False)


def call(value):
    return FunctionCall(call_id=str(value), selected_skill_id=None, name="x", arguments={"id":value,"a/b":value})


def test_template_rfc6901_scalar_and_invalid_values():
    assert resolve_resource_template(ResourceTemplate("row:{arg:/a~1b}"), {"a/b":3}) == "row:3"
    assert resolve_resource_template("row:{arg:/missing}", {}) is None
    assert resolve_resource_template("row:{arg:/x}", {"x":[]}) is None
    try:
        validate_resource_template("bad:{value}")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid placeholder accepted")


def test_conflict_unresolved_and_disjoint_independence():
    writes = (ResourceTemplate("row:{arg:/id}"),)
    assert calls_are_independent((call(1),call(2)), (meta(writes=writes),meta(writes=writes)))
    assert not calls_are_independent((call(1),call(1)), (meta(writes=writes),meta(writes=writes)))
    unresolved = (ResourceTemplate("row:{arg:/missing}"),)
    assert not calls_are_independent((call(1),call(2)), (meta(writes=unresolved),meta(writes=writes)))
