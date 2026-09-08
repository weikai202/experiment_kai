"""Explicit train collection, SFT preparation and native evaluation entrypoints."""

import argparse
import json
import random
from pathlib import Path

from .pipeline import prepare, write_json
from .protocol import digest, validate_manifest

UPSTREAM = "165848b9a78cead7ca7fe7c89c688b58e6501219"


def load_json(path):
    return json.loads(Path(path).read_text())


def run(args):
    from attrs import asdict
    from tool_sandbox.common.execution_context import RoleType
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.roles.execution_environment import ExecutionEnvironment
    from tool_sandbox.scenarios import named_scenarios

    from .adapter import PolicyAgent
    from .clients import ModelClient, ReplayPolicy, UserSimulator
    from .pipeline import Recorder

    config, manifest = (
        load_json(args.config),
        validate_manifest(load_json(args.manifest)),
    )
    split = "train" if args.command == "collect" else args.split
    selected = manifest[split]
    if not selected:
        raise ValueError("Selected split is empty")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "upstream_commit": UPSTREAM,
        "split": split,
        "manifest_hash": digest(manifest),
        "scenario_ids": [e["id"] for e in selected],
        "seed": config["seed"],
        "config": config,
        "complete": False,
        "status": "running",
        "expert_source": "user-supplied actions"
        if args.experts
        else "model demonstrations (unverified)",
        "format": "ee-json-action-v1",
        "expert_rule": getattr(args, "expert_rule", "all_completed"),
        "development_only": split != "test",
    }
    write_json(output / "run.json", provenance)
    random.seed(config["seed"])
    usage, results = [], {}
    try:
        # Upstream registry constructs scenarios; only manifest-selected episodes are played/exported.
        scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
        user = UserSimulator(config["user"], usage)
        expert_data = load_json(args.experts) if args.experts else None
        proposer = (
            ModelClient(config["generator"], usage)
            if args.command == "collect"
            else None
        )
        with (output / "transitions.jsonl").open("w") as stream:

            def emit(record):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()

            for entry in selected:
                name = entry["id"]
                if name not in scenarios:
                    raise ValueError("Unknown scenario ID: " + name)
                policy = (
                    ReplayPolicy(expert_data[name])
                    if expert_data is not None
                    else ModelClient(
                        config["expert" if args.command == "collect" else "policy"],
                        usage,
                    )
                )
                recorder = (
                    Recorder(name, proposer, user, config["k"], emit)
                    if proposer
                    else None
                )
                agent = PolicyAgent(policy, recorder)
                result = scenarios[name].play_and_evaluate(
                    roles={
                        RoleType.AGENT: agent,
                        RoleType.USER: user,
                        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
                    },
                    output_directory=output,
                    scenario_name=name,
                )
                if expert_data is not None and policy.index != len(policy.entries):
                    raise ValueError("Unused expert actions at episode end")
                results[name] = {
                    "evaluation": asdict(result.evaluation_result),
                    "categories": [str(c) for c in scenarios[name].categories],
                }
                write_json(output / "results.json", results)
        if args.command == "collect":
            provenance["accepted_scenario_ids"] = [
                name
                for name, result in results.items()
                if provenance["expert_rule"] == "all_completed"
                or (
                    result["evaluation"]["milestone_similarity"] == 1
                    and result["evaluation"]["minefield_similarity"] == 0
                )
            ]
        scores = [v["evaluation"]["similarity"] for v in results.values()]
        categories = {}
        for result in results.values():
            for category in result["categories"]:
                categories.setdefault(category, []).append(
                    result["evaluation"]["similarity"]
                )
        write_json(
            output / "summary.json",
            {
                "mean_similarity": sum(scores) / len(scores),
                "scenario_count": len(scores),
                "by_category": {
                    key: {"count": len(vals), "mean_similarity": sum(vals) / len(vals)}
                    for key, vals in categories.items()
                },
            },
        )
        provenance.update(
            complete=True,
            status="complete",
            mean_similarity=sum(scores) / len(scores),
            scenario_count=len(scores),
        )
    except BaseException as error:
        provenance.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_json(output / "run.json", provenance)
        write_json(output / "usage.json", usage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument(
            "--output",
            required=True,
            help="New run directory; never overwrite previous results",
        )
        if name == "collect":
            p.add_argument(
                "--experts",
                help="JSON mapping scenario ID to [{action, optional state_hash}]",
            )
        else:
            p.set_defaults(experts=None)
            p.add_argument("--split", choices=["dev", "test"], default="dev")
    # Explicit collection selection rule, applied only after native evaluation.
    sub.choices["collect"].add_argument(
        "--expert-rule",
        choices=["all_completed", "fully_successful"],
        default="fully_successful",
    )
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--reflect-config",
        help="Enable SR using generator in this config; makes model requests",
    )
    args = parser.parse_args()
    if args.command != "prepare":
        run(args)
        return
    provenance = load_json(Path(args.run) / "run.json")
    if provenance["split"] != "train" or not provenance["complete"]:
        raise ValueError("Only completed TRAIN runs may produce SFT data")
    usage = []
    reflector = None
    if args.reflect_config:
        from .clients import ModelClient

        reflector = ModelClient(load_json(args.reflect_config)["generator"], usage)
    try:
        print(
            prepare(
                Path(args.run) / "transitions.jsonl", args.output, provenance, reflector
            )
        )
    finally:
        if args.reflect_config and Path(args.output).is_dir():
            write_json(Path(args.output) / "reflection_usage.json", usage)


if __name__ == "__main__":
    main()
