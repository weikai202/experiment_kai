import hashlib
import json
from pathlib import Path

from .protocol import digest, sft_records


class Recorder:
    def __init__(self, scenario_id, proposer, user, k, emit):
        if k < 1:
            raise ValueError("Paper-style EE requires K >= 1")
        self.scenario_id, self.proposer, self.user = scenario_id, proposer, user
        self.k, self.emit, self.step = k, emit, 0

    def __call__(self, state, action, context):
        from .adapter import probe

        alternatives = self.proposer.alternatives(state, action, self.k)
        record = {
            "schema_version": 1,
            "scenario_id": self.scenario_id,
            "step": self.step,
            "state": state,
            "state_hash": digest(state),
            "expert_action": action,
            "expert_observation": probe(context, action, self.user),
            "alternatives": [],
        }
        for alternative in alternatives:
            record["alternatives"].append(
                {
                    "action": alternative,
                    "observation": probe(context, alternative, self.user),
                }
            )
        record["id"] = digest(
            {"scenario": self.scenario_id, "step": self.step, "state": state}
        )
        self.emit(record)
        self.step += 1


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def prepare(raw, output, provenance, reflector=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {"expert": 0, "iwm": 0, "reflection": 0}
    handles = {name: (output / (name + "_sft.jsonl")).open("w") for name in counts}
    enriched = (output / "grounded_records.jsonl").open("w")
    ids, seen = [], set()
    try:
        for record in read_jsonl(raw):
            if record["scenario_id"] not in provenance["scenario_ids"]:
                raise ValueError("Record is outside train provenance")
            if record["scenario_id"] not in provenance.get(
                "accepted_scenario_ids", provenance["scenario_ids"]
            ):
                continue
            if record["id"] in seen or record["state_hash"] != digest(record["state"]):
                raise ValueError("Duplicate record or modified state")
            seen.add(record["id"])
            if reflector:
                for alternative in record["alternatives"]:
                    alternative["reflection"] = reflector.reflect(record, alternative)
            enriched.write(json.dumps(record, ensure_ascii=False) + "\n")
            expert, iwm, reflection = sft_records(record)
            for name, values in [
                ("expert", [expert]),
                ("iwm", iwm),
                ("reflection", reflection),
            ]:
                for value in values:
                    handles[name].write(json.dumps(value, ensure_ascii=False) + "\n")
                    counts[name] += 1
                    ids.append(
                        {
                            "category": name,
                            "record_id": record["id"],
                            "scenario_id": record["scenario_id"],
                        }
                    )
        if not counts["expert"]:
            raise ValueError("No expert records")
        for handle in handles.values():
            handle.flush()
        write_json(
            output / "provenance.json",
            {
                "format": "ee-json-action-v1",
                "files": {
                    name: hashlib.sha256(
                        (output / (name + "_sft.jsonl")).read_bytes()
                    ).hexdigest()
                    for name in counts
                },
                "source": provenance,
                "counts": counts,
                "examples": ids,
                "raw_sha256": __import__("hashlib")
                .sha256(Path(raw).read_bytes())
                .hexdigest(),
                "complete": True,
            },
        )
    finally:
        enriched.close()
        for handle in handles.values():
            handle.close()
    return counts
