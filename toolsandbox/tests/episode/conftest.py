from pathlib import Path

import pytest
from tool_sandbox.common.execution_context import ExecutionContext, RoleType, set_current_context
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole

from toolsandbox_pipeline.checkpointing import CheckpointStore, RunIdentity
from toolsandbox_pipeline.reproducibility.scenario_hashes import context_sha256
from toolsandbox_pipeline.schemas.dataset import ScenarioRecord
from toolsandbox_pipeline.schemas.trajectory import EpisodeIdentity
from toolsandbox_pipeline.toolsandbox_adapter.trajectory_store import TrajectoryStore


CONFIG = Path(__file__).parents[2] / "configs/reproducibility/checkpointing_v1.json"


def digest(character: str) -> str:
    return "sha256:" + character * 64


@pytest.fixture
def start_context():
    context = ExecutionContext(tool_allow_list=[])
    set_current_context(context)
    BaseRole.add_messages(
        [Message(RoleType.USER, RoleType.AGENT, "start", conversation_active=True)]
    )
    return context


@pytest.fixture
def episode_identity(start_context):
    return EpisodeIdentity(
        run_id="run-episode",
        profile="offline",
        phase="train_round",
        round_index=0,
        shard_id="shard-0",
        family_id="family",
        scenario_id="family",
        episode_id="episode-1",
        manifest_position=0,
        system_variant="generation_0",
        generation_id="g000",
        starting_context_sha256=context_sha256(start_context),
        evaluation_definition_sha256=digest("2"),
        agent_tool_schema_sha256=digest("3"),
        dataset_manifest_sha256=digest("4"),
        runtime_config_sha256=digest("5"),
        prompt_manifest_sha256=digest("6"),
        token_limit_config_sha256=digest("7"),
        fixture_manifest_sha256=digest("8"),
        environment_sha256=digest("1"),
        max_messages=4,
    )


@pytest.fixture
def scenario_record(episode_identity):
    return ScenarioRecord(
        scenario_id="family",
        scenario_family_id="family",
        variant="no_distraction",
        categories=("NO_DISTRACTION_TOOLS",),
        max_messages=4,
        starting_context_sha256=episode_identity.starting_context_sha256,
        evaluation_definition_sha256=episode_identity.evaluation_definition_sha256,
        agent_facing_tool_names_sha256=digest("9"),
        agent_facing_tool_schema_sha256=episode_identity.agent_tool_schema_sha256,
    )


@pytest.fixture
def trajectory_store(tmp_path, episode_identity):
    run_identity = RunIdentity(
        run_id=episode_identity.run_id,
        profile=episode_identity.profile,
        environment_identity=episode_identity.environment_sha256,
        dataset_manifest_sha256=episode_identity.dataset_manifest_sha256,
        config_manifest_sha256=episode_identity.runtime_config_sha256,
        prompt_manifest_sha256=episode_identity.prompt_manifest_sha256,
        generation_manifest_sha256=digest("a"),
        fixture_manifest_sha256=episode_identity.fixture_manifest_sha256,
    )
    store = CheckpointStore.create(tmp_path / "run", run_identity, CONFIG.resolve())
    try:
        yield TrajectoryStore(store, environment_identity=episode_identity.environment_sha256)
    finally:
        store.close()
