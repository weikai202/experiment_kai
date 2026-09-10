from toolsandbox_pipeline.offline.dev_minibench import evaluate_dev_branches
from toolsandbox_pipeline.reproducibility import canonical_sha256
from toolsandbox_pipeline.schemas.offline_skill import DevBranchResult, DevScenarioSelection


DIGEST = "sha256:" + "c" * 64


def result(branch, *, success, similarity, minefield):
    return DevBranchResult(
        scenario_id="scenario",
        branch=branch,
        episode_id=f"episode-{branch}",
        evaluated_skill_id="skill",
        evaluated_skill_version="v1.0" if branch == "previous" else "v1.1",
        shared_configuration_sha256=DIGEST,
        complete=True,
        fully_successful=success,
        similarity=similarity,
        minefield_hit=minefield,
        evaluator_record_sha256=DIGEST,
    )


def test_acceptance_uses_only_native_success_similarity_and_minefields():
    selection = DevScenarioSelection(
        skill_id="skill",
        dataset_manifest_sha256=DIGEST,
        selector_input_sha256=DIGEST,
        selected_scenario_ids=("scenario",),
        selected_scenario_ids_sha256=canonical_sha256(["scenario"]),
        eligible_family_count=1,
    )
    accepted = evaluate_dev_branches(
        selection,
        (
            result("previous", success=False, similarity=0.2, minefield=False),
            result("candidate", success=True, similarity=1.0, minefield=False),
        ),
    )
    assert accepted.accepted
    tied = evaluate_dev_branches(
        selection,
        (
            result("previous", success=False, similarity=0.2, minefield=False),
            result("candidate", success=False, similarity=0.2, minefield=False),
        ),
    )
    assert not tied.accepted
