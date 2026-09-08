"""Command-line workflow for Early Experience on tau3 text domains."""

import argparse
import json
from pathlib import Path

from tau2.data_model.simulation import TextRunConfig
from tau2.registry import registry
from tau2.runner import get_tasks, run_domain
from tau2.utils.utils import get_commit_hash

from .agent import create_agent
from .core import SUPPORTED_DOMAINS, validate_split
from .pipeline import Generator, export_sft, generate_records, read_json


def main():
    """Validate explicit splits before any collection or evaluation calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    split = subs.add_parser("split")
    split.add_argument("--domain", choices=sorted(SUPPORTED_DOMAINS), required=True)
    split.add_argument("--train-split", default="train")
    split.add_argument("--eval-split", default="test")
    split.add_argument("--output", required=True)
    for name in ("collect", "evaluate"):
        sub = subs.add_parser(name)
        sub.add_argument(
            "--config",
            required=True,
            help="Native TextRunConfig JSON, with explicit agent/user models",
        )
        sub.add_argument("--split", required=True)
        sub.add_argument("--output", required=True)
        if name == "evaluate":
            sub.add_argument("--manifest", required=True, help="Training data manifest")
    generate = subs.add_parser("generate")
    generate.add_argument("--results", required=True)
    generate.add_argument("--split", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--generator-model", required=True)
    generate.add_argument(
        "--generator-args",
        default="{}",
        help="JSON LiteLLM kwargs; do not put API secrets here",
    )
    generate.add_argument("--k", type=int, default=3)
    generate.add_argument("--max-states", type=int)
    export = subs.add_parser("export")
    export.add_argument("--data", required=True)
    args = parser.parse_args()
    if args.command == "split":
        train = [t.id for t in get_tasks(args.domain, args.train_split)]
        evaluation = [t.id for t in get_tasks(args.domain, args.eval_split)]
        validate_split(train, evaluation)
        Path(args.output).write_text(
            json.dumps(
                {"domain": args.domain, "train_ids": train, "eval_ids": evaluation},
                indent=2,
            )
        )
    elif args.command == "generate":
        generate_records(
            args.results,
            args.split,
            args.output,
            Generator(args.generator_model, json.loads(args.generator_args)),
            args.k,
            args.max_states,
        )
    elif args.command == "export":
        export_sft(args.data)
    else:
        split = read_json(args.split)
        validate_split(split["train_ids"], split["eval_ids"])
        raw = read_json(args.config)
        if not raw.get("llm_agent") or not raw.get("llm_user"):
            raise ValueError("Explicit llm_agent and llm_user are required")
        config = TextRunConfig.model_validate(raw)
        if config.domain != split["domain"] or config.domain not in SUPPORTED_DOMAINS:
            raise ValueError("Domain mismatch or unsupported domain")
        if config.user != "user_simulator":
            raise ValueError("Use native text user_simulator")
        selected = (
            split["train_ids"] if args.command == "collect" else split["eval_ids"]
        )
        if args.command == "evaluate":
            manifest = read_json(args.manifest)
            if manifest["tau_commit"] != get_commit_hash():
                raise ValueError("Benchmark revision differs from training manifest")
            if any(
                manifest[key] != split[key]
                for key in ("domain", "train_ids", "eval_ids")
            ):
                raise ValueError("Evaluation split differs from training manifest")
            config.agent = "ee_agent"
        elif config.agent != "llm_agent":
            raise ValueError(
                "Collect with llm_agent; privileged ground-truth prompts are not supported"
            )
        if Path(args.output).exists():
            raise ValueError("Output exists; choose a new results path")
        registry.register_agent_factory(create_agent, "ee_agent")
        config.task_ids = selected
        config.task_split_name = None
        config.task_set_name = config.domain
        config.num_tasks = None
        results = run_domain(config)
        results.save(Path(args.output))
        print(f"Saved {len(results.simulations)} native simulations to {args.output}")


if __name__ == "__main__":
    main()
