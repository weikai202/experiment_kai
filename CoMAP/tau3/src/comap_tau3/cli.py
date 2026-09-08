"""Reproducible preparation, rollout, training and three-round evolution CLI."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from .splits import digest, make_manifest, official_splits, save_new_json, validate_manifest


def read(path):
    return yaml.safe_load(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-splits")
    prepare.add_argument("--tau-root", required=True)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument(
        "--protocol", choices=["official_full_train", "heldout_dev"], default="official_full_train"
    )
    prepare.add_argument("--output", required=True)
    validate = commands.add_parser("validate-splits")
    validate.add_argument("--tau-root", required=True)
    validate.add_argument("--manifest", required=True)
    for command in ("run", "train", "evolve"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--domain", choices=["airline", "retail", "telecom"], required=True)
        sub.add_argument("--output", required=True)
        if command == "run":
            sub.add_argument("--split", choices=["train", "dev", "test"], default="train")
            sub.add_argument("--round", type=int, choices=[0, 1, 2])
            sub.add_argument("--limit", type=int)
        if command == "train":
            sub.add_argument("--round", type=int, choices=[0, 1, 2], required=True)
            sub.add_argument("--transitions", required=True)
            sub.add_argument("--kind", choices=["world_model", "policy"], required=True)
    args = parser.parse_args()
    if args.command == "prepare-splits":
        manifest = make_manifest(args.tau_root, args.seed, args.protocol)
        save_new_json(args.output, manifest)
        print(f"Split manifest written: {args.output}\nsha256: {digest(manifest)}")
        return
    manifest = read(args.manifest)
    if args.command == "validate-splits":
        validate_manifest(manifest, official_splits(args.tau_root))
        print(
            json.dumps(
                {
                    "valid": True,
                    "sha256": digest(manifest),
                    "counts": {
                        name: {
                            "rounds": [len(s) for s in value["rounds"]],
                            "dev": len(value["dev"]),
                            "test": len(value["test"]),
                        }
                        for name, value in manifest["domains"].items()
                    },
                },
                indent=2,
            )
        )
        return
    config = read(args.config)
    validate_manifest(manifest, official_splits(config["tau_root"]))
    if args.command == "run":
        if args.split == "test" and config.get("manifest_sha256") != digest(manifest):
            raise ValueError(
                "Final test requires manifest_sha256 in config to match the fixed experiment split"
            )
        from .runner import run

        result = run(
            config,
            manifest,
            args.domain,
            args.split,
            args.output,
            round_index=args.round,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
        return
    from .training import load_transitions, train

    if args.command == "train":
        rows = load_transitions(args.transitions, manifest, args.domain, args.round)
        print(
            train(
                rows, config[args.kind], config["training"][args.kind], args.output, kind=args.kind
            )
        )
        return
    from .runner import run

    if any(config[key]["backend"] != "hf" for key in ("policy", "world_model")):
        raise ValueError(
            "Automatic evolve currently manages only HF inference. For vLLM use run/train stages and restart endpoints with each updated checkpoint (see VLLM.md)."
        )
    if config.get("manifest_sha256") != digest(manifest):
        raise ValueError(
            "Set manifest_sha256 to the shared fixed pipeline split before full evolution"
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_new_json(output / "manifest.json", manifest)
    current = copy.deepcopy(config)
    teacher_paths = {"policy": None, "world_model": None}
    for index in range(3):
        stage = output / f"round_{index}"
        stage.mkdir()
        save_new_json(stage / "input_config.json", current)
        run(current, manifest, args.domain, "train", stage / "rollout", round_index=index)
        rows = load_transitions(stage / "rollout/transitions.jsonl", manifest, args.domain, index)
        for kind in ("world_model", "policy"):
            student, teacher = train(
                rows,
                current[kind],
                current["training"][kind],
                stage / kind,
                kind=kind,
                teacher_path=teacher_paths[kind],
            )
            current[kind]["model"] = student
            teacher_paths[kind] = teacher
        save_new_json(stage / "output_config.json", current)
    save_new_json(output / "final_config.json", current)
    print(f"Three rounds complete. Evaluation configuration: {output / 'final_config.json'}")


if __name__ == "__main__":
    main()
