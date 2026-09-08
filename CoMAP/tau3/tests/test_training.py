import json

import pytest
from test_splits import fixture_manifest

from comap_tau3.splits import digest
from comap_tau3.training import divergence, ema_update, load_transitions, train

torch = pytest.importorskip("torch")


def test_divergence_endpoints_gradients_and_teacher_detached():
    student = torch.tensor([[0.3, -0.2, 0.8]], requires_grad=True)
    teacher = torch.tensor([[0.9, 0.4, -0.5]], requires_grad=True)
    p, q = student.softmax(-1), teacher.softmax(-1)
    for alpha, expected in [
        (0, (q * (q.log() - p.log())).sum()),
        (1, (p * (p.log() - q.log())).sum()),
    ]:
        assert torch.allclose(divergence(student, teacher, alpha), expected, atol=1e-6)
    loss = divergence(student, teacher, 0.5)
    assert loss.item() > 0
    loss.backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None
    assert abs(divergence(student, student, 0.5).item()) < 1e-6


def test_ema_is_parameter_average():
    student = torch.nn.Linear(2, 2, bias=False)
    teacher = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        student.weight.fill_(4)
        teacher.weight.zero_()
    ema_update(teacher, student, 0.25)
    assert torch.equal(teacher.weight, torch.ones_like(teacher.weight))


def test_rejects_dev_and_cross_round_training(tmp_path):
    manifest, _ = fixture_manifest()
    row = {
        "episode_id": "a",
        "step_id": 0,
        "split": "dev",
        "domain": "airline",
        "round_index": 0,
        "task_id": "0",
        "manifest_sha256": digest(manifest),
        "next_observation": [{"content": "observed"}],
    }
    path = tmp_path / "rows.jsonl"
    for split, round_index in [("dev", 0), ("test", 0), ("train", 1)]:
        path.write_text(json.dumps({**row, "split": split, "round_index": round_index}) + "\n")
        with pytest.raises(ValueError, match="provenance"):
            load_transitions(path, manifest, "airline", 0)


def tiny_checkpoint(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[EOS]": 2,
        "user": 3,
        "assistant": 4,
        "system": 5,
        "hello": 6,
        "observed": 7,
    }
    inner = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    inner.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=inner, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=2048,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    model.save_pretrained(path)
    return model


@pytest.mark.parametrize("kind", ["world_model", "policy"])
def test_actual_tiny_model_optimizer_and_checkpoint(tmp_path, kind):
    torch.set_num_threads(1)
    source = tmp_path / "source"
    original = tiny_checkpoint(source)
    action = {"type": "message", "content": "hello"}
    row = {
        "policy": "hello",
        "tools": [],
        "history": [{"role": "user", "content": "hello"}],
        "action": action,
        "next_observation": [{"role": "user", "content": "observed"}],
        "success": True,
        "raw_reflection": "{}",
        "used_revised_action": False,
        "reflection_parse_error": None,
        "policy_prompt": [{"role": "user", "content": "hello"}],
        "reflection_prompt": [{"role": "user", "content": "hello observed"}],
    }
    output, teacher = train(
        [row],
        {"backend": "hf", "model": str(source), "device": "cpu", "dtype": "float32"},
        {
            "epochs": 1,
            "gradient_accumulation_steps": 2,
            "max_response_tokens": 2,
            "max_context": 2048,
            "learning_rate": 0.01,
        },
        tmp_path / "trained",
        kind=kind,
    )
    from transformers import AutoModelForCausalLM

    updated = AutoModelForCausalLM.from_pretrained(output)
    assert any(not torch.equal(a, b) for a, b in zip(original.parameters(), updated.parameters()))
    assert (tmp_path / "trained/training.json").is_file()
    assert AutoModelForCausalLM.from_pretrained(teacher) is not None
