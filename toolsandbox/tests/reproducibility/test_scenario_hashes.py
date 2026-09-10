import copy
from functools import partial
import polars as pl
import pytest
from tool_sandbox.common.execution_context import ExecutionContext, DatabaseNamespace
from tool_sandbox.common.evaluation import Evaluation, MilestoneMatcher, Milestone, SnapshotConstraint, snapshot_similarity
from toolsandbox_pipeline.reproducibility.scenario_hashes import context_sha256, evaluator_sha256, agent_tools_sha256


def test_context_roundtrip_and_flags():
    context = ExecutionContext(tool_allow_list=[])
    digest = context_sha256(context)
    assert digest == context_sha256(copy.deepcopy(context))
    context.trace_tool = not context.trace_tool
    assert digest != context_sha256(context)


def test_evaluator_hash_and_unsafe_callable():
    def build(function, value=1):
        constraint = SnapshotConstraint(database_namespace=DatabaseNamespace.SANDBOX, snapshot_constraint=function,
                                        target_dataframe=pl.DataFrame({"synthetic": [value]}))
        return Evaluation(milestone_matcher=MilestoneMatcher(milestones=[Milestone(snapshot_constraints=[constraint], guardrail_database_list=[])]))
    assert evaluator_sha256(build(snapshot_similarity)) == evaluator_sha256(build(snapshot_similarity))
    assert evaluator_sha256(build(snapshot_similarity)) != evaluator_sha256(build(snapshot_similarity, 2))
    for function in (lambda **kw: 1, partial(snapshot_similarity), len):
        with pytest.raises(ValueError):
            evaluator_sha256(build(function))


def test_empty_agent_tools_are_hashed():
    context = ExecutionContext(tool_allow_list=[])
    names, schemas = agent_tools_sha256(context)
    assert names.startswith("sha256:") and schemas.startswith("sha256:")


def test_augmented_agent_schema_hashes_change_without_tool_execution():
    from tool_sandbox.common.execution_context import ScenarioCategories
    context = ExecutionContext(tool_allow_list=["search_contacts"])
    original = agent_tools_sha256(context)
    context.tool_augmentation_list = [ScenarioCategories.TOOL_NAME_SCRAMBLED]
    scrambled = agent_tools_sha256(context)
    assert original[0] != scrambled[0] and original[1] != scrambled[1]
    assert scrambled == agent_tools_sha256(copy.deepcopy(context))


def test_evaluator_row_order_dtype_and_nonfinite():
    def build(table):
        constraint = SnapshotConstraint(database_namespace=DatabaseNamespace.SANDBOX, snapshot_constraint=snapshot_similarity, target_dataframe=table)
        return Evaluation(milestone_matcher=MilestoneMatcher(milestones=[Milestone(snapshot_constraints=[constraint], guardrail_database_list=[])]))
    table = pl.DataFrame({"synthetic": [1, 2]})
    baseline = evaluator_sha256(build(table))
    assert baseline != evaluator_sha256(build(table.reverse()))
    assert baseline != evaluator_sha256(build(table.cast({"synthetic": pl.Float64})))
    with pytest.raises(ValueError):
        evaluator_sha256(build(pl.DataFrame({"synthetic": [float("nan")]})))
