"""Native tau orchestrator and evaluator; training data uses visible traces only."""

import json
import os
import time
from pathlib import Path

from . import COMAP_COMMIT, TAU_COMMIT
from .splits import digest, official_splits, save_new_json, validate_manifest, verify_repo


def run(config, manifest, domain, split, output, *, round_index=None, limit=None):
    for role in ("policy", "world_model", "user_simulator"):
        if not config[role].get("model") or config[role]["model"].startswith("REQUIRED"):
            raise ValueError(f"Configure {role}.model before running")
        if any(
            key in config[role].get("args", {}) for key in ("api_key", "api_token", "authorization")
        ):
            raise ValueError("Use environment credentials, not secrets in snapshotted configs")
    if config.get("num_trials", 1) < 1:
        raise ValueError("num_trials must be positive")
    root = verify_repo(config["tau_root"])
    validate_manifest(manifest, official_splits(root))
    os.environ["TAU2_DATA_DIR"] = str(root / "data")
    # Import after data root is set. Refuse accidentally importing another tau checkout.
    import tau2
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner import build_environment, build_user, get_tasks, run_simulation

    if not Path(tau2.__file__).resolve().is_relative_to(root):
        raise ValueError(
            "Installed tau2 does not point at the verified checkout; use editable install"
        )
    from .agent import CoMAPAgent, visible_message
    from .backends import make_backend

    if split == "train":
        if round_index not in (0, 1, 2):
            raise ValueError("train requires round index 0, 1, or 2")
        ids = manifest["domains"][domain]["rounds"][round_index]
    elif split in ("dev", "test"):
        ids = manifest["domains"][domain][split]
    else:
        raise ValueError("Unknown split")
    if limit is not None:
        if split == "test" or limit <= 0:
            raise ValueError("Task limits are only allowed for train/dev development runs")
        ids = ids[:limit]
    if not ids:
        raise ValueError("Selected split is empty; the full-training protocol has no dev split")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Snapshot before any model calls. Never overwrite an earlier experiment.
    save_new_json(output / "manifest.json", manifest)
    save_new_json(
        output / "run.json",
        {
            "config": config,
            "manifest_sha256": digest(manifest),
            "tau_commit": TAU_COMMIT,
            "comap_commit": COMAP_COMMIT,
            "domain": domain,
            "split": split,
            "round_index": round_index,
            "task_ids": ids,
            "development_only": split != "test",
            "status": "started",
        },
    )
    policy = make_backend(config["policy"])
    world = make_backend(config["world_model"])
    task_split = "test" if split == "test" else "train"
    tasks = {task.id: task for task in get_tasks(domain, task_split_name=task_split, task_ids=ids)}
    results = []
    all_calls = []
    started = time.monotonic()
    try:
        for task_id in ids:
            for trial in range(config.get("num_trials", 1)):
                env = build_environment(domain)
                agent = CoMAPAgent(
                    env.get_tools(),
                    env.get_policy(),
                    policy,
                    world,
                    options=config.get("inference", {}),
                )
                user_cfg = config["user_simulator"]
                user = build_user(
                    "user_simulator",
                    env,
                    tasks[task_id],
                    llm=user_cfg["model"],
                    llm_args=user_cfg.get("args", {}),
                )
                if user_cfg.get("seed_mode", "native") == "omit":
                    from .users import PipelineUserSimulator

                    user = PipelineUserSimulator(
                        tools=user.tools,
                        instructions=user.instructions,
                        llm=user_cfg["model"],
                        llm_args=user_cfg.get("args", {}),
                    )
                elif user_cfg.get("seed_mode", "native") != "native":
                    raise ValueError("user_simulator.seed_mode must be native or omit")
                seed = config.get("seed", 42) + trial
                orchestrator = Orchestrator(
                    domain=domain,
                    agent=agent,
                    user=user,
                    environment=env,
                    task=tasks[task_id],
                    max_steps=config.get("max_steps", 100),
                    max_errors=config.get("max_errors", 10),
                    seed=seed,
                )
                try:
                    result = run_simulation(orchestrator)
                except Exception:
                    save_new_json(
                        output / f"{domain}_{task_id}_{trial}.partial.json",
                        {"steps": agent.traces, "calls": agent.calls},
                    )
                    raise
                reward = result.reward_info.reward
                # Capture a last visible observation if the episode terminates before another agent turn.
                if agent.traces and agent.traces[-1]["next_observation"] is None:
                    last = max(
                        (i for i, msg in enumerate(result.messages) if msg.role == "assistant"),
                        default=-1,
                    )
                    tail = [
                        visible_message(msg)
                        for msg in result.messages[last + 1 :]
                        if (msg.role == "user" and not getattr(msg, "tool_calls", None))
                        or (msg.role == "tool" and msg.requestor == "assistant")
                    ]
                    if tail:
                        agent.traces[-1]["next_observation"] = tail
                episode_id = f"{domain}_{task_id}_{trial}"
                metadata = {
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "domain": domain,
                    "split": split,
                    "round_index": round_index,
                    "trial": trial,
                    "reward": reward,
                    "success": reward == 1.0,
                    "manifest_sha256": digest(manifest),
                }
                save_new_json(
                    output / f"{episode_id}.simulation.json", result.model_dump(mode="json")
                )
                save_new_json(
                    output / f"{episode_id}.trace.json",
                    {
                        **metadata,
                        "steps": agent.traces,
                        "calls": agent.calls,
                    },
                )
                if split == "train":
                    with (output / "transitions.jsonl").open("a") as stream:
                        for step in agent.traces:
                            if step["next_observation"] is not None:
                                stream.write(
                                    json.dumps({**metadata, **step}, ensure_ascii=False) + "\n"
                                )
                all_calls.extend(agent.calls)
                all_calls.extend(
                    {
                        "phase": "user_simulator",
                        "input_tokens": (msg.usage or {}).get("prompt_tokens"),
                        "output_tokens": (msg.usage or {}).get("completion_tokens"),
                        "cost": msg.cost,
                    }
                    for msg in result.messages
                    if msg.role == "user"
                )
                results.append(metadata)
    except Exception as exc:
        save_new_json(
            output / "failure.json",
            {"type": type(exc).__name__, "completed_episodes": len(results)},
        )
        raise
    summary = {
        "episodes": results,
        "mean_reward": sum(r["reward"] for r in results) / len(results),
        "elapsed_seconds": time.monotonic() - started,
        "status": "complete",
        "usage": summarize_usage(all_calls),
    }
    save_new_json(output / "summary.json", summary)
    return summary


def summarize_usage(calls):
    """Never replace unreported usage with zero. Native provider retries may be opaque."""
    result = {}
    for phase in sorted({call["phase"] for call in calls}):
        selected = [call for call in calls if call["phase"] == phase]
        totals = {"calls": len(selected)}
        for key in ("input_tokens", "output_tokens", "cost"):
            values = [call.get(key) for call in selected]
            totals[key] = sum(values) if all(value is not None for value in values) else None
        totals["usage_complete"] = (
            totals["input_tokens"] is not None and totals["output_tokens"] is not None
        )
        result[phase] = totals
    return {
        "by_phase": result,
        "scope": "returned completions; provider-internal retries may be unreported",
    }
