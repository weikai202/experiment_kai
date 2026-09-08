import ast
import copy
import json
import random
import re
from pathlib import Path

from .environment import Environment, action_text, load_cases, parse_calls
from .io import digest, read_jsonl, stable_seed, write_json, write_jsonl
from .prepare import same_output

IWM_SYSTEM = (
    "You are a world model of a function-calling environment. Predict the observed "
    "consequence of the specified action given the conversation. Do not choose an "
    "action or answer the user. Return Observation: followed by a descriptive outcome."
)
LEAK = re.compile(r"\b(expert|selected|chosen|correct|best|optimal|preferred)\b|"
                  r"\b(?:Action|Alternative)\s*#?\s*\d+", re.I)


def parse_proposals(text, names):
    stripped = text.strip()
    if stripped.startswith("```json\n") and stripped.endswith("```"):
        stripped = stripped[8:-3]
    rows = json.loads(stripped)
    if not isinstance(rows, list) or len(rows) != len(names):
        raise ValueError("Proposer must return exactly one argument set per sampled name")
    by_name = {}
    for row in rows:
        name, arguments = row["name"], row["arguments"]
        if name not in names or name in by_name or not isinstance(arguments, dict):
            raise ValueError("Unknown/duplicate name or non-object arguments")
        if any(not key.isidentifier() or key.startswith("_") for key in arguments):
            raise ValueError("Invalid argument name")
        by_name[name] = parse_calls(name + "(" + ", ".join(f"{k}={v!r}" for k, v in arguments.items()) + ")")
    return [by_name[name] for name in names]


def build_iwm(messages, calls, summary):
    return {"messages": [{"role": "system", "content": IWM_SYSTEM},
                         {"role": "user", "content": "Conversation:\n" + json.dumps(messages, ensure_ascii=False)
                          + "\nAction:\n" + action_text(calls)},
                         {"role": "assistant", "content": "Observation:\n" + summary.strip()}]}


def summarize(client, observation):
    return client.ask(
        "Describe the observed result of this tool action in one concise factual sentence. "
        "Preserve errors, names, values, and relevant state changes. Do not rank actions, "
        "invent outcomes, or discuss whether it helps the task.\n" + json.dumps(observation, ensure_ascii=False)
    ).strip()


def collect(prepared, output, config, client, limit=None, filter_leaks=False):
    prepared, output = Path(prepared), Path(output)
    manifest = json.loads((prepared / "manifest.json").read_text())
    from .provenance import verify_manifest
    verify_manifest(manifest)
    if manifest["counts"]["replay_mismatches"]:
        raise ValueError("Expert replay mismatches exist; use matching BFCL version and prepare again")
    records = read_jsonl(prepared / "expert_records.jsonl")
    allowed = set(manifest["train_ids"])
    if any(r["case_id"] not in allowed for r in records):
        raise ValueError("Training records contain heldout/OOD cases")
    cases, _ = load_cases("multi_turn_base")
    cases = {c["id"]: c for c in cases}
    k, sr_k = config.get("k_iwm", 10), config.get("k_sr", 3)
    if not 1 <= sr_k <= k:
        raise ValueError("Require 1 <= k_sr <= k_iwm")
    settings = {"manifest_sha256": digest(prepared / "manifest.json"),
                "records_sha256": digest(prepared / "expert_records.jsonl"),
                "config": config, "limit_states": limit, "filter_leaks": filter_leaks}
    state_dir = output / "states"
    metadata_path = output / "collection.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != settings:
        raise ValueError("Collection settings changed; use a new output directory")
    write_json(metadata_path, settings)
    iwm, reflection, audit, ids = [], [], [], []
    env, previous_case, count = None, None, 0
    for record in records:
        case_id = record["case_id"]
        if case_id != previous_case:
            env, previous_case = Environment(cases[case_id]), case_id
        calls = record["calls"]
        if not calls:
            continue
        if limit is not None and count >= limit:
            break
        messages = record["messages"][:-1]
        rng = random.Random(stable_seed(config.get("seed", 42), record["id"]))
        state_path = state_dir / (record["id"].replace("/", "_") + ".json")
        if state_path.exists():
            state = json.loads(state_path.read_text())
        else:
            expert_names = {ast.parse(c, mode="eval").body.func.id for c in calls}
            docs = {d["name"]: d for d in cases[case_id]["function"]}
            pool = sorted(set(env.method_owner) & set(docs) - expert_names)
            if len(pool) < k:
                raise ValueError(f"{record['id']}: only {len(pool)} distinct alternative names for K={k}")
            names = rng.sample(pool, k)
            proposal_prompt = (
                "Given the conversation and tool documentation, fill plausible arguments for EVERY listed "
                "function. Return only a JSON array of objects {\"name\": string, \"arguments\": object}, "
                "one for each function in the listed order. Do not add, omit, or rename functions.\n"
                "Conversation:\n" + json.dumps(messages, ensure_ascii=False) + "\nFunctions:\n"
                + json.dumps([docs[n] for n in names], ensure_ascii=False))
            for attempt in range(3):
                proposal = client.ask(proposal_prompt)
                try:
                    proposed_calls = parse_proposals(proposal, names)
                    break
                except (ValueError, KeyError, TypeError, SyntaxError) as error:
                    if attempt == 2:
                        raise ValueError(f"Invalid proposals after 3 attempts at {record['id']}: {error}") from error
                    proposal_prompt += ("\nPrevious response:\n" + proposal + "\nValidation error: " + str(error)
                                        + "\nReturn a corrected JSON array for exactly the listed function names.")
            alternatives = [env.probe(alt) for alt in proposed_calls]
            expert = env.probe(calls)
            expert["summary"] = summarize(client, expert)
            for alternative in alternatives:
                alternative["summary"] = summarize(client, alternative)
            sr_alternatives = rng.sample(alternatives, sr_k)
            monologue = client.ask(
                "Write a self-reflection grounded only in the observations below, leading to the provided "
                "action. Analyze the user's goal, compare consequences, justify the action, and discuss "
                "relevant constraints. Use a natural first-person explanation. No meta-commentary about "
                "being an AI. Do not mention a teacher, expert, ranked options, or labels such as "
                "selected/chosen/correct/best/optimal/preferred or Action 1. Do not include tool-call code, "
                "headings, or XML tags. Aim for " + str(config.get("reflection_words", 200)) +
                " words, at most about 500 words.\nConversation:\n" + json.dumps(messages, ensure_ascii=False)
                + "\nAction to take:\n" + action_text(calls) + "\nObserved outcome:\n" + expert["summary"]
                + "\nOther actions and their observed outcomes:\n"
                + json.dumps([{"calls": x["calls"], "outcome": x["summary"]} for x in sr_alternatives], ensure_ascii=False))
            if "<reflection>" in monologue or "</reflection>" in monologue:
                raise ValueError("Reflection generator emitted reserved delimiters")
            state = {"id": record["id"], "case_id": case_id, "expert": expert,
                     "alternatives": alternatives, "sr_alternatives": sr_alternatives,
                     "reflection": monologue, "leak_terms": sorted(set(LEAK.findall(monologue)))}
            write_json(state_path, state)
        replayed = env.execute(calls)
        if len(replayed) != len(record["outputs"]) or any(
                not same_output(a, b) for a, b in zip(replayed, record["outputs"])):
            raise ValueError(f"Simulator drift at {record['id']}; regenerate on matching environment")
        # IWM uses all non-expert branches, including simulator errors; no reward filtering.
        for alternative in state["alternatives"]:
            iwm.append(build_iwm(messages, alternative["calls"], alternative["summary"]))
        if not filter_leaks or not state["leak_terms"]:
            target = "<reflection>\n" + state["reflection"].strip() + "\n</reflection>\n" + action_text(calls)
            reflection.append({"messages": copy.deepcopy(messages) + [{"role": "assistant", "content": target}]})
        audit.append({"id": record["id"], "leak_terms": state["leak_terms"],
                      "reflection_kept": not filter_leaks or not state["leak_terms"]})
        ids.append(record["id"])
        count += 1
        print(f"Collected {count} states: {record['id']}", flush=True)
    write_jsonl(output / "iwm_sft_text.jsonl", iwm)
    write_jsonl(output / "reflection_sft_text.jsonl", reflection)
    write_jsonl(output / "reflection_audit.jsonl", audit)
    report = {"states": count, "state_ids": ids, "iwm_samples": len(iwm),
              "reflection_samples": len(reflection), "flagged_reflections": sum(bool(r["leak_terms"]) for r in audit),
              "partial": count < sum(bool(r["calls"]) for r in records)}
    write_json(output / "report.json", report)
    return report
