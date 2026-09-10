"""Hash-only projections of native context/evaluator definitions. Never log input."""
import copy
from enum import Enum
import importlib
import inspect
import math

from toolsandbox_pipeline.reproducibility.canonical import canonical_sha256


def _json(value):
    if isinstance(value, Enum):
        if not type(value).__module__.startswith("tool_sandbox."):
            raise ValueError("unsupported enum identity")
        return _json(value.value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_json(v) for v in value]
    if type(value) is dict and all(type(k) is str or isinstance(k, Enum) for k in value):
        return {_json(k): _json(v) for k, v in value.items()}
    raise ValueError("unsupported canonical scenario value")


def _dtype(dtype):
    import polars as pl
    base = dtype.base_type()
    name = base.__name__
    if name in ("Null", "Boolean", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "String", "Utf8", "Binary", "Date", "Time"):
        return name
    if isinstance(dtype, pl.List):
        return {"List": _dtype(dtype.inner)}
    if isinstance(dtype, pl.Struct):
        return {"Struct": [[f.name, _dtype(f.dtype)] for f in dtype.fields]}
    if isinstance(dtype, pl.Datetime):
        return {"Datetime": [dtype.time_unit, dtype.time_zone]}
    if isinstance(dtype, pl.Duration):
        return {"Duration": dtype.time_unit}
    if isinstance(dtype, pl.Enum):
        return {"Enum": dtype.categories.to_list()}
    raise ValueError("unsupported dataframe dtype")


def _table(table):
    import polars as pl
    if type(table) is not pl.DataFrame:
        raise ValueError("native Polars table required")
    return dict(columns=list(table.columns), dtypes=[_dtype(v) for v in table.dtypes], rows=_json(table.rows()))


def _context_projection(context):
    from tool_sandbox.common.execution_context import DatabaseNamespace, ExecutionContext
    if type(context) is not ExecutionContext:
        raise TypeError("native ExecutionContext required")
    serialized = context.to_dict(serialize_console=False)
    expected = {"_dbs", "interactive_console", "tool_allow_list", "tool_deny_list", "trace_tool", "tool_augmentation_list", "preferred_tool_backend"}
    if set(serialized) != expected or serialized["interactive_console"] is not None or set(serialized["_dbs"]) != set(DatabaseNamespace):
        raise ValueError("upstream context serialization shape drift")
    projected = {key: _json(value) for key, value in serialized.items() if key != "_dbs"}
    projected["_dbs"] = []
    for namespace in DatabaseNamespace:
        table = context._dbs[namespace]
        if _json(serialized["_dbs"][namespace]) != _json(table.to_dicts()):
            raise ValueError("context serialization/database mismatch")
        projected["_dbs"].append([namespace.value, _table(table)])
    return projected, serialized


def context_sha256(context):
    from tool_sandbox.common.execution_context import ExecutionContext
    original, wire = _context_projection(copy.deepcopy(context))
    restored, _ = _context_projection(ExecutionContext.from_dict(copy.deepcopy(wire)))
    digest = canonical_sha256(original)
    if canonical_sha256(restored) != digest:
        raise ValueError("context round-trip identity mismatch")
    return digest


def _callable_identity(function):
    if not inspect.isfunction(function) or function.__closure__ or "<" in function.__qualname__ or not function.__module__.startswith("tool_sandbox."):
        raise ValueError("unsafe evaluator callable")
    target = importlib.import_module(function.__module__)
    for part in function.__qualname__.split("."):
        target = getattr(target, part)
    if target is not function:
        raise ValueError("evaluator callable identity mismatch")
    return function.__module__ + ":" + function.__qualname__


def _fields(value, expected):
    import attrs
    if not attrs.has(type(value)) or {f.name for f in attrs.fields(type(value))} != set(expected):
        raise ValueError("upstream evaluator record shape drift")


def evaluator_sha256(evaluation):
    from tool_sandbox.common.evaluation import Evaluation, MilestoneMatcher, Milestone, Minefield, SnapshotConstraint
    if type(evaluation) is not Evaluation:
        raise TypeError("native Evaluation required")
    _fields(evaluation, ("milestone_matcher", "minefield_matcher"))
    output = {}
    for key, node_type in (("milestone_matcher", Milestone), ("minefield_matcher", Minefield)):
        matcher = getattr(evaluation, key)
        if type(matcher) is not MilestoneMatcher:
            raise ValueError("unknown matcher")
        _fields(matcher, ("milestones", "edge_list", "milestone_dag"))
        nodes = []
        for node in matcher.milestones:
            if type(node) is not node_type:
                raise ValueError("milestone/minefield type mismatch")
            _fields(node, ("snapshot_constraints", "guardrail_database_list", "guardrail_database_exclusion_list"))
            constraints = []
            for constraint in node.snapshot_constraints:
                if type(constraint) is not SnapshotConstraint:
                    raise ValueError("unknown snapshot constraint")
                _fields(constraint, ("database_namespace", "snapshot_constraint", "reference_milestone_node_index", "target_dataframe", "column_similarity_measure"))
                constraints.append(dict(database_namespace=_json(constraint.database_namespace),
                    snapshot_constraint=_callable_identity(constraint.snapshot_constraint),
                    reference_milestone_node_index=_json(constraint.reference_milestone_node_index),
                    target_dataframe=None if constraint.target_dataframe is None else _table(constraint.target_dataframe),
                    column_similarity_measure=None if constraint.column_similarity_measure is None else
                        [[name, _callable_identity(fn)] for name, fn in constraint.column_similarity_measure.items()]))
            nodes.append(dict(type=node_type.__name__, snapshot_constraints=constraints,
                guardrail_database_list=_json(node.guardrail_database_list),
                guardrail_database_exclusion_list=_json(node.guardrail_database_exclusion_list)))
        output[key] = dict(milestones=nodes, edge_list=_json(matcher.edge_list),
                           graph_nodes=_json(list(matcher.milestone_dag.nodes)), graph_edges=_json(list(matcher.milestone_dag.edges)))
    return canonical_sha256(output)


def agent_tools_sha256(context):
    from tool_sandbox.common.execution_context import new_context
    from toolsandbox_pipeline.toolsandbox_adapter.pipeline_agent import PipelineAgent
    from toolsandbox_pipeline.toolsandbox_adapter.tools import build_adapter_turn
    with new_context(copy.deepcopy(context)):
        view = build_adapter_turn(PipelineAgent, ()).agent_view
    schemas = [tool.model_dump(mode="json", by_alias=True)["schema"] for tool in view.available_tools]
    return canonical_sha256([tool.name for tool in view.available_tools]), canonical_sha256(schemas)


def scenario_hashes(scenario):
    from tool_sandbox.common.scenario import Scenario
    if type(scenario) is not Scenario:
        raise TypeError("native Scenario required")
    _fields(scenario, ("starting_context", "evaluation", "max_messages", "categories"))
    names, schemas = agent_tools_sha256(scenario.starting_context)
    return dict(starting_context_sha256=context_sha256(scenario.starting_context),
                evaluation_definition_sha256=evaluator_sha256(scenario.evaluation),
                agent_facing_tool_names_sha256=names, agent_facing_tool_schema_sha256=schemas)
