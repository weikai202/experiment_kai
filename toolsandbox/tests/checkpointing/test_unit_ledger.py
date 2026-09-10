from pathlib import Path

import pytest

from toolsandbox_pipeline.checkpointing import (
    CheckpointStore,
    GenerationPublicationIdentity,
    OfflineUnitIdentity,
    RunIdentity,
    UnitLedger,
    UnitLedgerConflictError,
)


CONFIG = Path(__file__).parents[2] / "configs/reproducibility/checkpointing_v1.json"


def digest(character):
    return "sha256:" + character * 64


@pytest.fixture
def units(tmp_path):
    identity = RunIdentity(
        run_id="run-1",
        profile="offline",
        environment_identity=digest("1"),
        dataset_manifest_sha256=digest("2"),
        config_manifest_sha256=digest("3"),
        prompt_manifest_sha256=digest("4"),
        generation_manifest_sha256=digest("5"),
        fixture_manifest_sha256=digest("6"),
    )
    store = CheckpointStore.create(tmp_path / "run", identity, CONFIG.resolve())
    try:
        yield UnitLedger(store), store
    finally:
        store.close()


def test_offline_unit_stages_and_commits_exact_outputs_idempotently(units):
    identity = OfflineUnitIdentity(
        run_id="run-1",
        round_index=0,
        input_generation_id="g000",
        unit_kind="policy_memory",
        unit_key="trajectory-1",
        input_fingerprint=digest("7"),
        expected_output_artifact_ids=("memory-next",),
    )
    prepared = units[0].prepare_unit(identity, request_ids=("llm-request",))
    assert prepared == units[0].prepare_unit(identity, request_ids=("llm-request",))
    outputs = {"memory-next": digest("8")}
    staged = units[0].stage_outputs(prepared.unit_id, outputs=outputs)
    assert staged.status.value == "outputs_staged"
    assert staged == units[0].stage_outputs(prepared.unit_id, outputs=outputs)
    committed = units[0].commit_unit(prepared.unit_id, outputs=outputs)
    assert committed.status.value == "committed"
    assert committed == units[0].commit_unit(prepared.unit_id, outputs=outputs)
    with pytest.raises(UnitLedgerConflictError, match="conflict"):
        units[0].commit_unit(
            prepared.unit_id, outputs={"memory-next": digest("9")}
        )


def test_publication_requires_existing_checkpoints_and_is_idempotent(units):
    publication = units[0].prepare_publication(
        GenerationPublicationIdentity(
            run_id="run-1",
            round_index=0,
            input_generation_id="g000",
            destination_generation_id="g001",
            staging_manifest_sha256=digest("a"),
            staging_content_sha256=digest("b"),
        )
    )
    with pytest.raises(UnitLedgerConflictError, match="checkpoint missing"):
        units[0].commit_publication(
            publication.publication_id,
            pre_checkpoint_id="before-publication",
            post_checkpoint_id="after-publication",
        )
    from toolsandbox_pipeline.checkpointing import LLMLedger

    ledger = LLMLedger(units[1])
    ledger.commit_checkpoint("before-publication", "before_publication", {})
    ledger.commit_checkpoint("after-publication", "after_publication", {})
    committed = units[0].commit_publication(
        publication.publication_id,
        pre_checkpoint_id="before-publication",
        post_checkpoint_id="after-publication",
    )
    assert committed.status.value == "committed"
    assert committed == units[0].commit_publication(
        publication.publication_id,
        pre_checkpoint_id="before-publication",
        post_checkpoint_id="after-publication",
    )


def test_terminal_failure_cannot_rewind(units):
    identity = OfflineUnitIdentity(
        run_id="run-1",
        round_index=1,
        input_generation_id="g001",
        unit_kind="world_memory",
        unit_key="trajectory-2",
        input_fingerprint=digest("c"),
        expected_output_artifact_ids=(),
    )
    prepared = units[0].prepare_unit(identity)
    failed = units[0].fail_unit(prepared.unit_id, failure_class="SchemaFailure")
    assert failed.status.value == "terminal_failure"
    with pytest.raises(UnitLedgerConflictError):
        units[0].stage_outputs(prepared.unit_id, outputs={})
