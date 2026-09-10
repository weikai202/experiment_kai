from types import SimpleNamespace

import pytest

from toolsandbox_pipeline.online.turn_context import (
    SAFE_CLARIFICATION_TEXT,
    TurnContext,
    TurnRoleRequestBuilders,
    default_safe_failure_action,
)
from toolsandbox_pipeline.online.controller_inputs import ReproducibilityProfile
from toolsandbox_pipeline.schemas.online_turn import OnlineTurnIdentity


DIGEST = "sha256:" + "1" * 64


def identity(**updates):
    values = dict(
        run_id="run",
        profile=ReproducibilityProfile.STRICT_REPLAY,
        phase="train",
        round_index=0,
        shard_id="shard-0",
        family_id="family",
        scenario_id="scenario",
        episode_id="episode",
        agent_turn_index=0,
        expected_generation_id="g000",
        dataset_manifest_sha256=DIGEST,
        runtime_config_sha256=DIGEST,
        prompt_manifest_sha256=DIGEST,
        token_limit_config_sha256=DIGEST,
        fixture_manifest_sha256=DIGEST,
        environment_identity="environment",
    )
    values.update(updates)
    return OnlineTurnIdentity(**values)


def context(**updates):
    generation = SimpleNamespace(
        manifest=SimpleNamespace(generation_id="g000"), skills=()
    )
    values = dict(
        identity=identity(),
        generation=generation,
        retrieval=SimpleNamespace(snapshot=generation),
        state_builder=object(),
        controller=object(),
        controller_tool_metadata=(),
        role_request_builders=TurnRoleRequestBuilders(
            initial_policy=lambda *_: None,
            critic=lambda *_: None,
            revision=lambda *_: None,
        ),
        durable_roles=object(),
        checkpoint_sink=object(),
    )
    values.update(updates)
    return TurnContext(**values)


def test_context_construction_is_side_effect_free_and_pins_generation():
    built = context()
    assert built.identity.expected_generation_id == "g000"
    assert default_safe_failure_action().action.content == SAFE_CLARIFICATION_TEXT
    with pytest.raises(ValueError, match="pin"):
        context(identity=identity(expected_generation_id="g001"))


def test_context_rejects_different_retrieval_snapshot():
    with pytest.raises(ValueError, match="exact supplied"):
        context(retrieval=SimpleNamespace(snapshot=object()))


def test_strict_replay_requires_fixture_identity():
    with pytest.raises(ValueError, match="fixture"):
        identity(fixture_manifest_sha256=None)
