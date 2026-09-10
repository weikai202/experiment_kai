from types import SimpleNamespace

import pytest

from toolsandbox_pipeline.orchestration.generation_publisher import (
    GenerationPublicationError,
    publish_generation,
    publish_generation_zero,
)

from .test_generation_builder import build


class Ledger:
    def __init__(self, events):
        self.events = events

    def prepare_publication(self, identity):
        self.events.append(("prepare", identity))
        return SimpleNamespace(publication_id="publication-" + "a" * 64)

    def commit_publication(self, publication_id, *, pre_checkpoint_id, post_checkpoint_id):
        self.events.append(("commit", publication_id, pre_checkpoint_id, post_checkpoint_id))
        return SimpleNamespace(status="committed")


class Checkpoints:
    def __init__(self, events):
        self.events = events

    def commit_checkpoint(self, checkpoint_id, event_kind, payload):
        self.events.append(("checkpoint", event_kind, checkpoint_id))


def test_generation_zero_and_next_generation_publish_atomically(tmp_path):
    _, staged_zero = build(tmp_path, "g000")
    generations = tmp_path / "generations"
    generations.mkdir(mode=0o700)
    zero_identity = publish_generation_zero(staged_zero, generations)
    assert (generations / "g000").is_dir()
    assert zero_identity[0].startswith("sha256:")

    _, staged_one = build(tmp_path, "g001", "g000")
    events = []
    committed = publish_generation(
        staged_one,
        generations,
        run_id="run",
        round_index=0,
        input_generation_id="g000",
        ledger=Ledger(events),
        checkpoints=Checkpoints(events),
    )
    assert committed.status == "committed"
    assert [event[0] for event in events] == [
        "prepare", "checkpoint", "checkpoint", "commit",
    ]
    assert (generations / "latest.json").read_bytes().startswith(b"{")


def test_incomplete_or_conflicting_destination_fails_closed(tmp_path):
    _, staged = build(tmp_path, "g000")
    generations = tmp_path / "generations"
    generations.mkdir(mode=0o700)
    destination = generations / "g000"
    destination.mkdir(mode=0o700)
    (destination / "bad").write_bytes(b"x")
    with pytest.raises(GenerationPublicationError, match="conflicting"):
        publish_generation_zero(staged, generations)
