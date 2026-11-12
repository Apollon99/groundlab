"""Checks against the real pinned model files (D-019a, D-020).

These need `groundlab models fetch` to have been run (scripts/setup.ps1 does it). If the
files are absent the tests are skipped with an explicit reason; they never download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groundlab.errors import VerificationError
from groundlab.models import BGE_SMALL, MINILM_CE, ModelSpec, model_dir, verify_model
from groundlab.text import QueryTooLongError, encode_cross_encoder_pair

MODELS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "models"

pytestmark = pytest.mark.pinned_models


def _require(spec: ModelSpec) -> Path:
    try:
        verify_model(spec, MODELS_DIR)
    except VerificationError as exc:
        pytest.skip(f"pinned model files unavailable ({exc}); run `groundlab models fetch`")
    return model_dir(spec, MODELS_DIR)


@pytest.fixture(scope="module")
def ce_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_require(MINILM_CE))


def test_bge_model_card_settings_match_our_explicit_choices() -> None:
    root = _require(BGE_SMALL)
    pooling = json.loads((root / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    assert pooling["pooling_mode_cls_token"] is True
    assert not any(
        v for k, v in pooling.items() if k.startswith("pooling_mode_") and "cls" not in k
    )
    modules = json.loads((root / "modules.json").read_text(encoding="utf-8"))
    assert [m["type"].rsplit(".", 1)[-1] for m in modules] == [
        "Transformer",
        "Pooling",
        "Normalize",
    ]
    st = json.loads((root / "sentence_bert_config.json").read_text(encoding="utf-8"))
    assert st["max_seq_length"] == 512


def test_bge_tokenizer_treats_separator_like_whitespace() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_require(BGE_SMALL))
    assert tok.truncation_side == "right"
    assert tok("Title\nBody text")["input_ids"] == tok("Title Body text")["input_ids"]


def test_cross_encoder_budget_is_512_with_three_special_tokens(ce_tokenizer: Any) -> None:
    assert ce_tokenizer.model_max_length == 512
    assert ce_tokenizer.num_special_tokens_to_add(pair=True) == 3


def test_cross_encoder_truncates_only_the_document(ce_tokenizer: Any) -> None:
    query = "Does vitamin D supplementation reduce fracture risk in older adults?"
    document = "Title\n" + " ".join(["randomized controlled trial of vitamin D"] * 200)
    enc = encode_cross_encoder_pair(ce_tokenizer, query, document)
    assert len(enc.input_ids) == 512
    assert enc.document_truncated
    assert enc.document_tokens_kept == 512 - 3 - enc.query_tokens
    q_ids = ce_tokenizer(query, add_special_tokens=False)["input_ids"]
    assert enc.input_ids[1 : 1 + len(q_ids)] == q_ids  # query kept intact after [CLS]
    d_ids = ce_tokenizer(document, add_special_tokens=False)["input_ids"]
    start = 2 + len(q_ids)
    assert enc.input_ids[start:-1] == d_ids[: enc.document_tokens_kept]  # a prefix of the doc
    assert enc.input_ids[-1] == ce_tokenizer.sep_token_id


def test_cross_encoder_short_pair_is_not_truncated(ce_tokenizer: Any) -> None:
    enc = encode_cross_encoder_pair(ce_tokenizer, "vitamin D", "Title\nshort body")
    assert not enc.document_truncated
    assert len(enc.input_ids) == enc.query_tokens + enc.document_tokens_total + 3


def test_cross_encoder_query_exhausting_budget_is_an_error(ce_tokenizer: Any) -> None:
    long_query = " ".join(["vitamin"] * 509)  # 509 + 3 special = 512: no room for the doc
    assert len(ce_tokenizer(long_query, add_special_tokens=False)["input_ids"]) == 509
    with pytest.raises(QueryTooLongError, match="exhausts"):
        encode_cross_encoder_pair(ce_tokenizer, long_query, "Title\nbody")
    ok = encode_cross_encoder_pair(ce_tokenizer, " ".join(["vitamin"] * 508), "Title\nbody")
    assert ok.document_tokens_kept == 1
