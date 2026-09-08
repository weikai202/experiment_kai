"""Real CPU training smoke tests with a locally created random tiny model."""

import sys

import pytest

from experiments.early_experience.pipeline import write_jsonl


@pytest.mark.parametrize("method", ["il", "iwm", "sr"])
def test_tiny_cpu_training(tmp_path, monkeypatch, method):
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from experiments.early_experience.train import main

    torch.set_num_threads(1)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "<eos>": 2,
                "user": 3,
                "assistant": 4,
                "request": 5,
                "action": 6,
                "observation": 7,
                "reflection": 8,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    hf = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<eos>",
    )
    hf.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' <eos> ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    config = transformers.GPT2Config(
        vocab_size=9,
        n_positions=128,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=1,
    )
    model = transformers.GPT2LMHeadModel(config)
    base = tmp_path / "base"
    model.save_pretrained(base)
    hf.save_pretrained(base)
    data = tmp_path / "data"
    data.mkdir()
    for name, target in [
        ("expert", "action"),
        ("iwm", "observation"),
        ("reflection", "reflection action"),
    ]:
        write_jsonl(
            data / f"{name}_sft.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": "request"},
                        {"role": "assistant", "content": target},
                    ]
                }
            ],
        )
    output = tmp_path / "trained"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--method",
            method,
            "--model",
            str(base),
            "--data",
            str(data),
            "--output",
            str(output),
            "--batch-size",
            "1",
            "--gradient-accumulation",
            "1",
            "--max-length",
            "128",
        ],
    )
    monkeypatch.setenv("WANDB_DISABLED", "true")
    main()
    final = transformers.GPT2LMHeadModel.from_pretrained(output / "final")
    assert not torch.equal(model.transformer.wte.weight, final.transformer.wte.weight)
    if method == "iwm":
        warm = transformers.GPT2LMHeadModel.from_pretrained(output / "iwm_warmup")
        assert not torch.equal(
            warm.transformer.wte.weight, final.transformer.wte.weight
        )
    assert (output / "training_config.json").exists()
