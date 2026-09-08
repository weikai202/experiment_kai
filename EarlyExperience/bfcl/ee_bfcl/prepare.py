"""Harvest published Opus inference logs; split by case before producing SFT."""
import copy
import json
import random
import subprocess
from pathlib import Path

from .environment import (Environment, action_text, add_observations, initial_messages,
                          load_cases, parse_calls, turn_messages)
from .io import digest, read_jsonl, write_json, write_jsonl


def successful_ids(results, scores):
    ids = [r["id"] for r in results]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate expert case IDs")
    headers = [s for s in scores if "total_count" in s and "correct_count" in s]
    if len(headers) != 1 or headers[0]["total_count"] != len(ids):
        raise ValueError("Score summary must cover the entire result corpus")
    details = {s["id"]: s for s in scores if "id" in s}
    if not set(details) <= set(ids):
        raise ValueError("Scores contain unknown case IDs")
    if len(details) != sum("id" in s for s in scores):
        raise ValueError("Duplicate score IDs")
    # Official score JSONL stores an aggregate header plus failed cases only.
    if len(details) == len(ids):
        valid = [i for i in ids if details[i].get("valid") is True]
    else:
        if any(s.get("valid") is not False for s in details.values()):
            raise ValueError("Incomplete scores must use official failures-only format")
        valid = [i for i in ids if i not in details]
    if len(valid) != headers[0]["correct_count"]:
        raise ValueError("Successful case count disagrees with official summary")
    return valid


def same_output(left, right):
    try:
        return json.loads(left) == json.loads(right)
    except (ValueError, TypeError):
        return left == right


def harvest(case, result):
    logs = [x for x in result["inference_log"] if isinstance(x, dict) and "begin_of_turn_query" in x]
    if len(logs) != len(case["question"]):
        raise ValueError(f"{case['id']}: expert turn count mismatch")
    messages = initial_messages(case)
    env = Environment(case)
    records, mismatches = [], []
    for turn, log in enumerate(logs):
        if turn:
            messages.extend(turn_messages(case, turn))
        steps = sorted((k for k in log if k.startswith("step_")), key=lambda k: int(k[5:]))
        for step_index, key in enumerate(steps):
            step = log[key]
            decoded = [x["model_response_decoded"] for x in step if "model_response_decoded" in x]
            if not decoded:
                # Textual final responses are normalized to [] for all baselines.
                break
            if len(decoded) != 1 or not isinstance(decoded[0], list):
                raise ValueError(f"Ambiguous decoded expert calls: {case['id']}/{key}")
            calls = parse_calls(action_text(decoded[0]))
            if not calls:
                break
            outputs = [x["content"] for x in step if x.get("role") == "tool"]
            if len(outputs) != len(calls):
                raise ValueError(f"Missing real tool observations: {case['id']}/{key}")
            replayed = env.execute(calls)
            for call, logged, current in zip(calls, outputs, replayed):
                if not same_output(logged, current):
                    mismatches.append({"case_id": case["id"], "turn": turn, "step": step_index,
                                       "call": call, "logged": logged, "replayed": current})
            target = {"role": "assistant", "content": action_text(calls)}
            records.append({"id": f"{case['id']}/{turn}/{step_index}", "case_id": case["id"],
                            "turn": turn, "calls": calls, "outputs": outputs,
                            "messages": copy.deepcopy(messages) + [target]})
            messages.append(target)
            add_observations(messages, calls, outputs)
        stop = {"role": "assistant", "content": "[]"}
        records.append({"id": f"{case['id']}/{turn}/stop", "case_id": case["id"],
                        "turn": turn, "calls": [], "outputs": [],
                        "messages": copy.deepcopy(messages) + [stop]})
        messages.append(stop)
    return records, mismatches


def prepare(result_path, score_path, output, seed=42):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Output already has a manifest; use a new output directory")
    results, scores = read_jsonl(result_path), read_jsonl(score_path)
    valid = successful_ids(results, scores)
    if any(not i.startswith("multi_turn_base_") for i in valid):
        raise ValueError("Only Base expert cases may enter the training split")
    cases, _ = load_cases("multi_turn_base")
    by_id = {c["id"]: c for c in cases}
    # Preserve numeric input order, then use a dedicated RNG; save exact IDs.
    shuffled = sorted(valid, key=lambda x: int(x.rsplit("_", 1)[1]))
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * 0.75)
    if cut == 0 or cut == len(shuffled):
        raise ValueError("Need enough successful cases for train and heldout splits")
    train_ids, heldout_ids = shuffled[:cut], shuffled[cut:]
    train_set = set(train_ids)
    train, heldout, mismatches = [], [], []
    for result in results:
        if result["id"] not in valid:
            continue
        records, differences = harvest(by_id[result["id"]], result)
        (train if result["id"] in train_set else heldout).extend(records)
        mismatches.extend(differences)
    from bfcl_eval import __file__ as bfcl_file
    package = Path(bfcl_file).resolve().parent
    commit = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"],
                            text=True, capture_output=True)
    from .provenance import source_hashes
    data_dir = package / "data"
    manifest = {"schema_version": 1, "seed": seed, "expert_source": "published_opus_2025_12_16",
                "result_sha256": digest(result_path), "score_sha256": digest(score_path),
                "bfcl_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
                "bfcl_source_sha256": source_hashes(),
                "bfcl_data_sha256": {str(p.relative_to(data_dir)): digest(p) for p in sorted(data_dir.rglob("*.json"))
                                      if "multi_turn" in str(p.relative_to(data_dir))},
                "train_ids": train_ids, "heldout_ids": heldout_ids,
                "excluded_base_ids": sorted(set(by_id) - set(valid)),
                "ood_categories": ["multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"],
                "counts": {"successful_cases": len(valid), "train_cases": len(train_ids),
                           "heldout_cases": len(heldout_ids), "train_samples": len(train),
                           "heldout_samples": len(heldout), "replay_mismatches": len(mismatches)},
                "format": "BFCL Python prompting, tool messages, canonical [] turn completion",
                "enable_thinking": False}
    write_jsonl(output / "expert_records.jsonl", train)
    write_jsonl(output / "heldout_records.jsonl", heldout)
    write_jsonl(output / "expert_sft_text.jsonl", ({"messages": r["messages"]} for r in train))
    write_jsonl(output / "heldout_sft_text.jsonl", ({"messages": r["messages"]} for r in heldout))
    write_jsonl(output / "replay_mismatches.jsonl", mismatches)
    write_json(output / "manifest.json", manifest)
    return manifest["counts"]
