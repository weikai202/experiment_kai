"""Collect native demonstrations; replay training states and sample real branches."""

import hashlib
import json
from pathlib import Path

import litellm

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import Results
from tau2.runner import build_environment, build_user
from tau2.utils.utils import get_commit_hash

from .core import (
    SUPPORTED_DOMAINS,
    action_key,
    api_messages,
    dumps,
    make_expert,
    make_iwm,
    make_reflection,
    parse_action,
    probe,
    validate_split,
    visible_context,
)


def read_json(path):
    """Read UTF-8 configuration or manifest."""
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    """Load an intermediate or SFT file, failing on incomplete records."""
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    """Atomically publish a completed SFT artifact."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(dumps(row) + "\n")
    temporary.replace(path)


class Generator:
    """Explicit model configuration; one multi-candidate request per state."""

    def __init__(self, model, llm_args=None):
        self.model = model
        self.args = dict(llm_args or {})

    def text(self, messages, json_mode=False):
        """Generate complete text and preserve provider usage for audit."""
        kwargs = dict(self.args)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = litellm.completion(model=self.model, messages=messages, **kwargs)
        if response.choices[0].finish_reason != "stop":
            raise RuntimeError(
                f"Incomplete generation: {response.choices[0].finish_reason}"
            )
        text = response.choices[0].message.content
        if not text:
            raise ValueError("Empty generation")
        return text, response.usage.model_dump() if response.usage else {}

    def alternatives(self, context, tools, expert, k):
        """Generate distinct non-expert actions without future observations."""
        text, usage = self.text(
            [
                {
                    "role": "system",
                    "content": 'Propose plausible alternative customer-service actions at the given state. Return JSON {"actions": [...]}. Each action has either {"content": "message to customer"} or {"tool_calls": [{"name": "tool name", "arguments": {}}]}. Use only the supplied tools and information known at this state. Do not predict outcomes.',
                },
                {
                    "role": "user",
                    "content": dumps(
                        {
                            "state": context,
                            "tools": tools,
                            "excluded_action": api_messages([expert])[0],
                            "number_of_distinct_candidates": k + 2,
                        }
                    ),
                },
            ],
            json_mode=True,
        )
        seen = {action_key(expert)}
        actions = []
        for i, value in enumerate(json.loads(text)["actions"]):
            action = parse_action(value, i)
            key = action_key(action)
            if key not in seen:
                seen.add(key)
                actions.append(action)
            if len(actions) == k:
                return actions, usage
        raise ValueError(
            f"Only {len(actions)} distinct alternatives; need {k}. Adjust generation settings and resume."
        )

    def reflection(self, context, expert, outcome, alternative, alt_outcome):
        """Ground one reflection in the two actual transitions."""
        return self.text(
            [
                {
                    "role": "system",
                    "content": "Write a private decision-time self-reflection of 200-400 words (soft limit 500). Analyze the current situation and goal, compare the possible actions, justify the demonstrated action using constraints and consequences, and highlight relevant clues. Stay strictly within the provided information; avoid AI meta-commentary; use natural step-by-step reasoning focused on the decision. The outcomes are evidence for you as the data author, not facts already available to the acting agent. Reason anticipatorily about expected effects. Never refer to an expert, answer label, numbered alternatives, or future observations as already seen. Output plain reflection text only, without tags or a final action; the demonstrated action will be appended unchanged.",
                },
                {
                    "role": "user",
                    "content": dumps(
                        {
                            "state": context,
                            "demonstrated_action": api_messages([expert])[0],
                            "observed_outcome": outcome,
                            "other_action": api_messages([alternative])[0],
                            "other_observed_outcome": alt_outcome,
                        }
                    ),
                },
            ]
        )


def generate_records(results_path, split_path, output, generator, k=3, max_states=None):
    """Replay native Results, resume completed states, and save all real branches.

    Demonstrations are accepted as supplied, with no reward filtering. The user
    must curate demonstrations first if only successful trajectories are desired.
    """
    if k < 1 or (max_states is not None and max_states < 1):
        raise ValueError("k and max_states must be positive")
    source = Results.load(Path(results_path))
    if source.info.git_commit != get_commit_hash():
        raise ValueError(
            "Demonstration commit differs from checkout; replay with the original benchmark revision"
        )
    split = read_json(split_path)
    validate_split(split["train_ids"], split["eval_ids"])
    domain = source.info.environment_info.domain_name
    if domain not in SUPPORTED_DOMAINS or split["domain"] != domain:
        raise ValueError("Domain mismatch or unsupported non-memory/voice domain")
    if source.info.agent_info.implementation not in {"llm_agent", "ee_agent"}:
        raise ValueError(
            "Only ordinary half-duplex LLMAgent demonstrations are supported"
        )
    if source.info.user_info.implementation != "user_simulator":
        raise ValueError("Only the native text user_simulator is supported")
    if any(s.task_id not in split["train_ids"] for s in source.simulations):
        raise ValueError("Demonstrations contain tasks outside train_ids")
    if not source.simulations:
        raise ValueError("No demonstration simulations")
    if any(s.messages is None or s.ticks for s in source.simulations):
        raise ValueError("Voice/full-duplex trajectories are unsupported")
    tasks = {task.id: task for task in source.tasks}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "domain": domain,
        "train_ids": split["train_ids"],
        "eval_ids": split["eval_ids"],
        "tau_commit": source.info.git_commit,
        "generator_model": generator.model,
        "generator_args": generator.args,
        "k": k,
        "source_sha256": hashlib.sha256(Path(results_path).read_bytes()).hexdigest(),
        "source_agent": source.info.agent_info.model_dump(mode="json"),
        "source_user": source.info.user_info.model_dump(mode="json"),
        "format_version": 1,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError(
            "Resume configuration differs from manifest; use a new output directory"
        )
    manifest_path.write_text(dumps(manifest) + "\n")
    records_path = output / "transitions.jsonl"
    done = (
        {row["id"] for row in read_jsonl(records_path)}
        if records_path.exists()
        else set()
    )
    processed = 0
    with records_path.open("a") as stream:
        for simulation in source.simulations:
            task = tasks[simulation.task_id]
            history = simulation.messages
            initial = task.initial_state
            initial_count = len(initial.message_history or []) if initial else 0
            for index, action in enumerate(history):
                # Initial scripted history/greeting is context, not a policy target.
                if index < max(1, initial_count) or not isinstance(
                    action, AssistantMessage
                ):
                    continue
                key = f"{simulation.id}:{index}"
                if key in done:
                    continue
                env = build_environment(domain)
                env.set_state(
                    initialization_data=initial.initialization_data
                    if initial
                    else None,
                    initialization_actions=initial.initialization_actions
                    if initial
                    else None,
                    message_history=history[:index],
                )
                if (
                    simulation.policy is not None
                    and simulation.policy != env.get_policy()
                ):
                    raise ValueError(
                        "Demonstration policy differs from the replay environment"
                    )
                context = visible_context(env.get_policy(), history[:index])
                tools = [t.openai_schema for t in env.get_tools()]
                expert = make_expert(context, tools, action)
                if action.is_tool_call():
                    following = history[index + 1 : index + 1 + len(action.tool_calls)]
                    if len(following) != len(action.tool_calls) or any(
                        not isinstance(m, ToolMessage)
                        or m.id != tc.id
                        or m.requestor != "assistant"
                        for m, tc in zip(following, action.tool_calls)
                    ):
                        raise ValueError(
                            f"Missing or misaligned expert tool outcomes: {key}"
                        )
                    outcome = {"messages": api_messages(following), "terminal": False}
                    replayed = probe(env, action)
                    # IDs are identical for expert replay; errors remain useful signal.
                    if replayed != outcome:
                        raise ValueError(f"Non-reproducible expert transition: {key}")
                else:
                    # User private tool activity can intervene before a spoken response.
                    response = next(
                        (
                            m
                            for m in history[index + 1 :]
                            if isinstance(m, (AssistantMessage, UserMessage))
                            and not m.is_tool_call()
                        ),
                        None,
                    )
                    if not isinstance(response, UserMessage):
                        raise ValueError(
                            f"No observed user response for expert action: {key}"
                        )
                    from tau2.user.user_simulator import UserSimulator

                    outcome = {
                        "messages": api_messages([response]),
                        "terminal": UserSimulator.is_stop(response),
                    }
                user_info = source.info.user_info
                user = build_user(
                    "user_simulator",
                    env,
                    task,
                    llm=user_info.llm,
                    llm_args=user_info.llm_args,
                    persona_config=user_info.persona_config,
                )
                user.set_seed(simulation.seed if simulation.seed is not None else 0)
                alternatives, proposal_usage = generator.alternatives(
                    context, tools, action, k
                )
                variants = []
                for alternative in alternatives:
                    alt_outcome = probe(env, alternative, user, history[:index])
                    reflection, usage = generator.reflection(
                        context, action, outcome, alternative, alt_outcome
                    )
                    variants.append(
                        {
                            "action": alternative.model_dump(mode="json"),
                            "outcome": alt_outcome,
                            "iwm": make_iwm(context, tools, alternative, alt_outcome),
                            "reflection": make_reflection(expert, reflection),
                            "reflection_usage": usage,
                        }
                    )
                row = {
                    "id": key,
                    "task_id": task.id,
                    "expert": expert,
                    "expert_outcome": outcome,
                    "alternatives": variants,
                    "proposal_usage": proposal_usage,
                }
                stream.write(dumps(row) + "\n")
                stream.flush()
                done.add(key)
                processed += 1
                print(f"saved {key}: {k} alternatives", flush=True)
                if max_states is not None and processed >= max_states:
                    return


def export_sft(directory):
    """Export separate IL, IWM and SR categories; no implicit filtering."""
    directory = Path(directory)
    rows = read_jsonl(directory / "transitions.jsonl")
    if not rows:
        raise ValueError("No completed transitions")
    write_jsonl(directory / "expert_sft.jsonl", [r["expert"] for r in rows])
    write_jsonl(
        directory / "iwm_sft.jsonl", [v["iwm"] for r in rows for v in r["alternatives"]]
    )
    write_jsonl(
        directory / "reflection_sft.jsonl",
        [v["reflection"] for r in rows for v in r["alternatives"]],
    )
    print(
        dumps(
            {
                "states": len(rows),
                "alternatives": sum(len(r["alternatives"]) for r in rows),
            }
        )
    )
