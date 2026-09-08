import copy
import json
from pathlib import Path

from .environment import (Environment, add_observations, initial_messages, load_cases,
                          parse_calls, score_case, turn_messages)
from .io import digest, read_jsonl, write_json, write_jsonl


def evaluate(prepared, output, category, client, config, limit=None, max_steps=20):
    prepared, output = Path(prepared), Path(output)
    manifest = json.loads((prepared / "manifest.json").read_text())
    from .provenance import verify_manifest
    verify_manifest(manifest)
    if category not in ["multi_turn_base"] + manifest["ood_categories"]:
        raise ValueError("Supported evaluation: heldout Base and three multi-turn OOD categories")
    cases, ground_truth = load_cases(category)
    if category == "multi_turn_base":
        cases = [c for c in cases if c["id"] in set(manifest["heldout_ids"])]
    if limit is not None:
        cases = cases[:limit]
    if not cases or max_steps < 1:
        raise ValueError("Empty evaluation set or invalid max_steps")
    if set(c["id"] for c in cases) & set(manifest["train_ids"]):
        raise ValueError("Evaluation overlaps with training cases")
    settings = {"manifest_sha256": digest(prepared / "manifest.json"), "category": category,
                "case_ids": [c["id"] for c in cases], "config": config, "max_steps": max_steps}
    metadata = output / "evaluation.json"
    if metadata.exists() and json.loads(metadata.read_text()) != settings:
        raise ValueError("Evaluation settings changed; use a new output directory")
    write_json(metadata, settings)
    path = output / "results.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    completed = {r["id"] for r in rows}
    if len(completed) != len(rows) or not completed <= set(settings["case_ids"]):
        raise ValueError("Invalid resumed result IDs")
    for case in cases:
        if case["id"] in completed:
            continue
        env = Environment(case)
        messages = initial_messages(case)
        predictions, logs = [], []
        failure = None
        for turn in range(len(case["question"])):
            if turn:
                messages.extend(turn_messages(case, turn))
            steps = []
            for step in range(max_steps):
                # API outages/truncation raise, leaving resumable results; never silently score them.
                text = client.complete(messages)
                messages.append({"role": "assistant", "content": text})
                try:
                    calls = parse_calls(text)
                except (ValueError, TypeError) as error:
                    failure = {"valid": False, "error_type": "ee:decode_error", "message": str(error)}
                    logs.append({"turn": turn, "step": step, "response": text, "error": str(error)})
                    break
                logs.append({"turn": turn, "step": step, "response": text, "calls": calls})
                if not calls:
                    break
                outputs = env.execute(calls)
                logs[-1]["outputs"] = outputs
                steps.append(calls)
                add_observations(messages, calls, outputs)
            else:
                failure = {"valid": False, "error_type": "ee:step_limit"}
            predictions.append(steps)
            if failure:
                break
        score = failure or score_case(case, predictions, ground_truth[case["id"]])
        rows.append({"id": case["id"], "result": predictions, "score": score, "inference_log": logs})
        write_jsonl(path, rows)
        print(f"{len(rows)}/{len(cases)} {case['id']}: {score['valid']}", flush=True)
    report = {"category": category, "correct": sum(r["score"]["valid"] for r in rows),
              "total": len(rows), "accuracy": sum(r["score"]["valid"] for r in rows) / len(rows),
              "scope": "heldout Base" if category == "multi_turn_base" else "OOD",
              "partial": limit is not None, "model": config["model"], "enable_thinking": False,
              "scorer": "official BFCL multi_turn_checker + multi_turn_irrelevance_checker; literal-only executor"}
    write_json(output / "metrics.json", report)
    return report
