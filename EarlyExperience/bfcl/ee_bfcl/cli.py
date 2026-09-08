import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Early Experience / Qwen3 non-thinking / BFCL v4")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Harvest public expert logs and split successful Base cases")
    prep.add_argument("--results", required=True)
    prep.add_argument("--scores", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=42)
    collection = commands.add_parser("collect", help="Generate actual IWM and SR data via an API")
    evaluation = commands.add_parser("evaluate", help="Run a checkpoint on heldout Base or OOD")
    for sub in (collection, evaluation):
        sub.add_argument("--prepared", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--config", required=True)
        sub.add_argument("--limit", type=int)
        sub.add_argument("--execute", action="store_true", help="Explicitly make model API calls (default is dry run)")
    collection.add_argument("--filter-leaks", action="store_true", help="Apply the upstream SR vocabulary filter; default audit only")
    evaluation.add_argument("--category", default="multi_turn_base",
                            choices=["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"])
    evaluation.add_argument("--max-steps", type=int, default=20)
    planning = commands.add_parser("plan-training")
    planning.add_argument("--prepared", required=True)
    planning.add_argument("--collected", required=True)
    planning.add_argument("--output", required=True)
    planning.add_argument("--model", default="Qwen/Qwen3-32B")
    planning.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        result = prepare(args.results, args.scores, args.output, args.seed)
    elif args.command == "plan-training":
        from .train import plan
        result = plan(args.prepared, args.collected, args.output, args.model, args.allow_partial)
    else:
        from .io import read_jsonl
        config = json.loads(Path(args.config).read_text())
        if config.get("enable_thinking") is not False:
            parser.error("enable_thinking must be false")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if not args.execute:
            manifest = json.loads((Path(args.prepared) / "manifest.json").read_text())
            if args.command == "collect":
                records = [r for r in read_jsonl(Path(args.prepared) / "expert_records.jsonl") if r["calls"]]
                n = min(len(records), args.limit) if args.limit else len(records)
                result = {"dry_run": True, "states": n, "k_iwm": config.get("k_iwm", 10),
                          "k_sr": config.get("k_sr", 3), "api_requests_without_cache_or_retries": n * (config.get("k_iwm", 10) + 3),
                          "model": config["model"], "base_url": config["base_url"],
                          "enable_thinking": False, "note": "1 proposal + K alternative summaries + 1 expert summary + 1 reflection per state. Token cost depends on real prompts and provider; no calls made."}
            else:
                from .environment import load_cases
                cases, _ = load_cases(args.category)
                if args.category == "multi_turn_base":
                    cases = [c for c in cases if c["id"] in set(manifest["heldout_ids"])]
                if args.limit:
                    cases = cases[:args.limit]
                result = {"dry_run": True, "cases": len(cases), "category": args.category,
                          "max_requests": sum(len(c["question"]) for c in cases) * args.max_steps,
                          "model": config["model"], "enable_thinking": False}
        else:
            from .client import Client
            client = Client(config, Path(args.output) / "api_cache")
            if args.command == "collect":
                from .collect import collect
                result = collect(args.prepared, args.output, config, client, args.limit, args.filter_leaks)
            else:
                from .evaluate import evaluate
                result = evaluate(args.prepared, args.output, args.category, client, config, args.limit, args.max_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
