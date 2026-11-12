"""A tiny randomly initialised BERT saved locally, so dense tests need no network."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from groundlab.io_utils import sha256_file
from groundlab.models import ModelSpec
from groundlab.retrieval.dense import DenseParams

TINY_KEY = "tiny-bert-test"
TINY_REVISION = "0" * 40
WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "omega",
    "psi",
    "title",
    "body",
    "text",
    "claim",
    "protein",
    "cell",
    "gene",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
]

TINY_PARAMS = DenseParams(
    model=TINY_KEY,
    revision=TINY_REVISION,
    pooling="cls",
    normalize=True,
    max_length=32,
    truncation_side="right",
    query_instruction="Represent this sentence for searching relevant passages: ",
    document_instruction="",
    document_separator="\n",
    attn_implementation="eager",
    dtype="float32",
    device="cuda",
    batch_size=4,
)


def make_tiny_model(models_dir: Path) -> ModelSpec:
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    root = models_dir / TINY_KEY
    root.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ":", "represent", "this"]
    vocab += ["sentence", "for", "searching", "relevant", "passages", *WORDS]
    (root / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab={t: i for i, t in enumerate(vocab)}, do_lower_case=True)
    if tokenizer("alpha omega")["input_ids"][1:3] == [1, 1]:
        raise RuntimeError("tiny tokenizer maps known words to [UNK]")
    tokenizer.save_pretrained(root)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    BertModel(config).save_pretrained(root)
    files = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    return ModelSpec(TINY_KEY, "local/tiny-bert-test", TINY_REVISION, "test-only", files)


TINY_CE_KEY = "tiny-ce-test"


def make_tiny_cross_encoder(models_dir: Path) -> ModelSpec:
    """A tiny random BertForSequenceClassification (one logit) saved as safetensors."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    root = models_dir / TINY_CE_KEY
    root.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ":", "represent", "this"]
    vocab += ["sentence", "for", "searching", "relevant", "passages", *WORDS]
    (root / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab={t: i for i, t in enumerate(vocab)}, do_lower_case=True)
    tokenizer.save_pretrained(root)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=1,
    )
    torch.manual_seed(0)
    BertForSequenceClassification(config).save_pretrained(root)
    files = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    assert "model.safetensors" in files
    return ModelSpec(TINY_CE_KEY, "local/tiny-ce-test", TINY_REVISION, "test-only", files)


def params(**overrides: object) -> DenseParams:
    return replace(TINY_PARAMS, **overrides)  # type: ignore[arg-type]
