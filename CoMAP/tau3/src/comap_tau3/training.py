"""Single-device reference trainer for WMSD and feedback-conditioned policy SDPO.

Full-vocabulary losses follow CoMAP's public loss definitions. This backend
replaces the external verl launcher; it does not claim identical verl numerics.
"""

import copy
import json
import math
import random
from pathlib import Path

from .backends import load_hf, prompt_ids
from .prompts import dumps, world_prompt
from .splits import digest, save_new_json


def load_transitions(path, manifest, domain, round_index):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    allowed = set(manifest["domains"][domain]["rounds"][round_index])
    if not rows:
        raise ValueError("No observed training transitions")
    seen = set()
    for row in rows:
        key = (row["episode_id"], row["step_id"])
        if key in seen:
            raise ValueError("Duplicate transition")
        seen.add(key)
        if (
            row["split"] != "train"
            or row["domain"] != domain
            or row["round_index"] != round_index
            or row["task_id"] not in allowed
            or row["manifest_sha256"] != digest(manifest)
        ):
            raise ValueError("Transition split/provenance mismatch; dev/test training is forbidden")
        if not row.get("next_observation"):
            raise ValueError("World model targets must be real observed transitions")
    return rows


def policy_examples(rows):
    # Public successful_or_all episode filter.
    successful = [row for row in rows if row["success"]]
    selected = successful or rows
    examples = []
    for row in selected:
        feedback = [
            {
                "role": "user",
                "content": "Feedback from the recorded attempt (offline teacher context only):\n"
                "Executed action:\n"
                + dumps(row["action"])
                + "\nActual next observation:\n"
                + dumps(row["next_observation"])
                + "\nEpisode success: "
                + str(row["success"])
                + "\nUse this feedback to improve the original response.",
            }
        ]
        try:
            reflection = json.loads(row["raw_reflection"])
            explanation = reflection.get("reflection", "Review the predicted consequence.")
        except (ValueError, AttributeError):
            explanation = "Keep the executable draft action."
        target = dumps(
            {
                "reflection": explanation,
                "decision": "REVISE" if row["used_revised_action"] else "KEEP",
                "revise_probability": 1.0 if row["used_revised_action"] else 0.0,
                "action": row["action"],
            }
        )
        examples.extend(
            [
                {
                    "prompt": row["policy_prompt"],
                    "teacher_prompt": row["policy_prompt"] + feedback,
                    "target": dumps(row["action"]),
                    "weight": 0.5,
                    "origin": "expert_anchor",
                },
                {
                    "prompt": row["reflection_prompt"],
                    "teacher_prompt": row["reflection_prompt"] + feedback,
                    "target": target,
                    "weight": 1.0,
                    "origin": "reflect",
                },
            ]
        )
        if row["success"] and row["used_revised_action"] and not row["reflection_parse_error"]:
            examples.append(
                {
                    "prompt": row["policy_prompt"],
                    "teacher_prompt": row["policy_prompt"] + feedback,
                    "target": dumps(row["action"]),
                    "weight": 0.25,
                    "origin": "draft_distill",
                }
            )
    return examples


def divergence(student_logits, teacher_logits, alpha):
    """CoMAP full-logit KL endpoints and generalized Jensen-Shannon divergence."""
    import torch
    import torch.nn.functional as F

    student = F.log_softmax(student_logits.float(), dim=-1)
    teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    if alpha == 0:
        loss = F.kl_div(student, teacher, log_target=True, reduction="none")
    elif alpha == 1:
        loss = F.kl_div(teacher, student, log_target=True, reduction="none")
    elif 0 < alpha < 1:
        mixture = torch.logaddexp(student + math.log(1 - alpha), teacher + math.log(alpha))
        loss = (1 - alpha) * F.kl_div(
            mixture, student, log_target=True, reduction="none"
        ) + alpha * F.kl_div(mixture, teacher, log_target=True, reduction="none")
    else:
        raise ValueError("alpha must be in [0,1]")
    return loss.sum(-1).mean()


def ema_update(teacher, student, rate):
    """Actual parameter EMA, unlike the public orchestration snapshot placeholder."""
    import torch

    if not 0 <= rate <= 1:
        raise ValueError("EMA update rate must be in [0,1]")
    with torch.no_grad():
        for target, source in zip(teacher.parameters(), student.parameters(), strict=True):
            target.lerp_(source.detach(), rate)
        for target, source in zip(teacher.buffers(), student.buffers(), strict=True):
            target.copy_(source)


def continuation_logits(model, prefix, continuation):
    """Align logits by continuation position even when teacher prefix is longer."""
    import torch

    if not prefix or not continuation:
        raise ValueError("Empty prefix or continuation")
    inputs = torch.tensor([prefix + continuation], device=model.device)
    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    return output.logits[0, len(prefix) - 1 : len(prefix) - 1 + len(continuation)]


def train(rows, model_config, train_config, output, *, kind, teacher_path=None):
    import torch
    import torch.nn.functional as F
    from transformers import set_seed

    if kind not in {"world_model", "policy"}:
        raise ValueError("Unknown training stage")
    if model_config["backend"] == "vllm":
        if not model_config.get("checkpoint"):
            raise ValueError("vLLM training requires the checkpoint field, not just an endpoint")
        model_config = {**model_config, "backend": "hf", "model": model_config["checkpoint"]}
    if model_config["backend"] != "hf":
        raise ValueError(
            "Co-evolution training requires local HF weights; API-only inference is not training"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(train_config.get("seed", 42))
    model, tokenizer = load_hf(
        model_config["model"],
        model_config.get("device", "cuda"),
        model_config.get("dtype", "bfloat16"),
        revision=model_config.get("revision"),
    )
    if teacher_path:
        teacher, teacher_tokenizer = load_hf(
            teacher_path, str(model.device), model_config.get("dtype", "bfloat16")
        )
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError("Teacher and student tokenizers differ")
    else:
        teacher = copy.deepcopy(model)
    teacher.eval().requires_grad_(False)
    # Disable dropout for student sampling and teacher-aligned logits; gradients remain enabled.
    model.eval()
    if train_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    if kind == "world_model":
        examples = [
            {
                "prompt": world_prompt(row["policy"], row["tools"], row["history"], row["action"]),
                "target": dumps(row["next_observation"]),
                "weight": 1.0,
                "origin": "transition",
            }
            for row in rows
        ]
    else:
        examples = policy_examples(rows)
    if not examples:
        raise ValueError("No training examples")
    max_context = train_config.get("max_context", model.config.max_position_embeddings)
    max_new = train_config.get("max_response_tokens", 512)
    encoded = []
    for example in examples:
        prefix = prompt_ids(tokenizer, example["prompt"])
        target = tokenizer.encode(example["target"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        teacher_prefix = prompt_ids(tokenizer, example.get("teacher_prompt", example["prompt"]))
        if max(len(prefix), len(teacher_prefix)) + max(len(target), max_new) > max_context:
            raise ValueError(
                "Training context overflow; increase max_context (no silent truncation)"
            )
        encoded.append((prefix, teacher_prefix, target, example["weight"], example["origin"]))
    epochs = train_config.get("epochs", 3)
    accumulation = train_config.get("gradient_accumulation_steps", 16)
    if epochs < 1 or accumulation < 1:
        raise ValueError("epochs and accumulation must be positive integers")
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.get("learning_rate", 2e-5))
    steps = epochs * math.ceil(len(encoded) / accumulation)
    warmup = int(steps * train_config.get("warmup_ratio", 0.03))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    logs = []
    rng = random.Random(train_config.get("seed", 42))
    for epoch in range(epochs):
        rng.shuffle(encoded)
        for start in range(0, len(encoded), accumulation):
            group = encoded[start : start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for prefix, teacher_prefix, target, weight, origin in group:
                if kind == "policy":
                    inputs = torch.tensor([prefix], device=model.device)
                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        sampled = model.generate(
                            inputs,
                            attention_mask=torch.ones_like(inputs),
                            max_new_tokens=max_new,
                            do_sample=True,
                            temperature=train_config.get("temperature", 0.7),
                            top_p=1.0,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    model.train(was_training)
                    continuation = sampled[0, len(prefix) :].tolist()
                else:
                    continuation = target
                student_logits = continuation_logits(model, prefix, continuation)
                with torch.no_grad():
                    teacher_logits = continuation_logits(teacher, teacher_prefix, continuation)
                distill = divergence(
                    student_logits,
                    teacher_logits,
                    train_config.get("alpha", 0.5) if kind == "policy" else 0.0,
                )
                if kind == "world_model":
                    ce = F.cross_entropy(
                        student_logits.float(), torch.tensor(target, device=model.device)
                    )
                    loss = ce + train_config.get("distill_weight", 0.5) * distill
                else:
                    del student_logits, teacher_logits
                    supervised = continuation_logits(model, prefix, target)
                    ce = F.cross_entropy(
                        supervised.float(), torch.tensor(target, device=model.device)
                    )
                    loss = distill + weight * ce
                (loss / len(group)).backward()
                logs.append(
                    {
                        "epoch": epoch,
                        "origin": origin,
                        "loss": loss.item(),
                        "ce": ce.item(),
                        "distillation": distill.item(),
                        "sampled_tokens": len(continuation),
                    }
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.get("max_grad_norm", 1.0)
            )
            optimizer.step()
            scheduler.step()
            ema_update(teacher, model, train_config.get("teacher_update_rate", 0.01))
    model.save_pretrained(output / "student")
    tokenizer.save_pretrained(output / "student")
    teacher.save_pretrained(output / "teacher")
    tokenizer.save_pretrained(output / "teacher")
    save_new_json(
        output / "training.json",
        {
            "kind": kind,
            "examples": len(examples),
            "config": train_config,
            "metrics": logs,
            "implementation": "tau3_single_device_full_logit_adaptation",
        },
    )
    return str(output / "student"), str(output / "teacher")
