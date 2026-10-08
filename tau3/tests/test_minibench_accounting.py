import pytest

from tau3_evolution.access import AccessAudit
from tau3_evolution.accounting import UsageAttempt, summarize_round
from tau3_evolution.canonical import canonical_sha256
from tau3_evolution.minibench import (
    BranchResult,
    DevTaskView,
    SkillCandidate,
    evaluate_paired_branches,
    select_relevant_dev_tasks,
)


def candidate():
    return SkillCandidate("retail.refund", 0, 1, ("retail",), ("get_order",))


def views():
    return tuple(
        [DevTaskView("retail", f"r{i}", ("get_order", "refund")) for i in range(25)]
        + [DevTaskView("airline", "a0", ("get_order",))]
        + [DevTaskView("retail", "other", ("cancel_trip",))]
    )


def audit(task_views):
    return AccessAudit(
        "dev",
        "skill_ab_validation",
        "skill",
        "sha256:m",
        canonical_sha256([{"domain": row.domain, "task_id": row.task_id} for row in task_views]),
        len(task_views),
    )


def make_selection(task_views):
    return select_relevant_dev_tasks(
        candidate=candidate(),
        manifest_sha256="sha256:m",
        task_views=task_views,
        access_audit=audit(task_views),
    )


def test_relevant_selector_is_deterministic_bounded_and_domain_aware():
    first = make_selection(views())
    second = make_selection(views())
    assert first == second
    assert len(first.selected) == 20
    assert all(domain == "retail" and task.startswith("r") for domain, task in first.selected)


def branch_results(selection, previous=(0.0, 1.0), candidate_rewards=(1.0, 1.0), *, complete=True):
    rows = []
    for index, (domain, task_id) in enumerate(selection.selected):
        for branch, rewards in (("previous", previous), ("candidate", candidate_rewards)):
            rows.append(
                BranchResult(
                    domain,
                    task_id,
                    branch,
                    selection.skill_id,
                    float(rewards[index]),
                    complete,
                    f"{domain}-{task_id}-{branch}",
                    selection.host_shared_configuration_sha256,
                    selection.manifest_sha256,
                    selection.access_receipt_sha256,
                    f"sha256:trajectory-{index}-{branch}",
                    f"sha256:evaluator-{index}-{branch}",
                )
            )
    return tuple(rows)


def test_minibench_accepts_more_full_successes():
    selection = make_selection(views()[:2])
    result = evaluate_paired_branches(selection, branch_results(selection))
    assert result.accepted and result.reason == "higher_full_success"


def test_minibench_accepts_higher_reward_at_tied_success():
    selection = make_selection(views()[:2])
    result = evaluate_paired_branches(
        selection,
        branch_results(selection, previous=(0.2, 1.0), candidate_rewards=(0.7, 1.0)),
    )
    assert result.accepted and result.reason.startswith("higher_native_reward")


def test_minibench_rejects_incomplete_or_mismatched_pairs():
    selection = make_selection(views()[:2])
    assert not evaluate_paired_branches(
        selection, branch_results(selection, complete=False)
    ).accepted
    with pytest.raises(ValueError, match="paired"):
        evaluate_paired_branches(selection, branch_results(selection)[:-1])


def test_selector_rejects_caller_views_outside_audited_manifest_dev():
    audited = audit(views()[:2])
    with pytest.raises(ValueError, match="exact audited"):
        select_relevant_dev_tasks(
            candidate=candidate(),
            manifest_sha256="sha256:m",
            task_views=views()[:3],
            access_audit=audited,
        )


def test_all_tokens_and_effective_qwen_cost_are_separate():
    attempts = (
        UsageAttempt("a1", "l1", "qwen", 10, 5, "effect-1"),
        UsageAttempt("a2", "l2", "qwen", 8, 7, None),
        UsageAttempt("a3", "l3", "embedding", 4, 0, None),
        UsageAttempt("a4", "l4", "user_simulator", 6, 3, None),
    )
    result = summarize_round(
        round_index=0,
        direct_latency_seconds=1.5,
        attempts=attempts,
        committed_substantive_effect_ids=frozenset({"effect-1"}),
    )
    assert result.total_tokens == 43
    assert result.total_cost == 5
    assert result.cost_unit == "qwen_effective_output_tokens"


def test_missing_usage_is_incomplete_not_zero():
    result = summarize_round(
        round_index=1,
        direct_latency_seconds=None,
        attempts=(UsageAttempt("a", "l", "qwen", None, None, "effect"),),
        committed_substantive_effect_ids=frozenset({"effect"}),
    )
    assert result.total_tokens is None and not result.usage_complete
    assert result.total_cost is None and not result.cost_complete


def test_rejected_qwen_output_costs_zero():
    result = summarize_round(
        round_index=2,
        direct_latency_seconds=0.2,
        attempts=(UsageAttempt("a", "l", "qwen", 10, 99, None),),
        committed_substantive_effect_ids=frozenset(),
    )
    assert result.total_cost == 0 and result.cost_complete


def test_multi_tool_skill_is_not_selected_with_partial_tool_visibility():
    task_views = (DevTaskView("retail", "partial", ("get_order",)),)
    multi = SkillCandidate("retail.refund", 0, 1, ("retail",), ("get_order", "refund"))
    selected = select_relevant_dev_tasks(
        candidate=multi,
        manifest_sha256="sha256:m",
        task_views=task_views,
        access_audit=audit(task_views),
    )
    assert selected.selected == ()
