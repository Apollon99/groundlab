"""Canonical text representations (D-019, D-019a).

- Document text for every arm: `title + "\\n" + text`; an empty title gives `text`.
  No further normalization (no lowercasing, no whitespace changes).
- Cross-encoder input: the pair (query, document_text) with a 512-token budget under the
  pinned tokenizer. Only the document is truncated. If the query plus the special tokens
  leaves no room for at least one document token, an explicit error is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from groundlab.errors import VerificationError
from groundlab.types import Document

DOCUMENT_SEPARATOR = "\n"
CROSS_ENCODER_MAX_LENGTH = 512


def canonical_document_text(doc: Document) -> str:
    if not doc.title:
        return doc.text
    return f"{doc.title}{DOCUMENT_SEPARATOR}{doc.text}"


class QueryTooLongError(VerificationError):
    """The query alone (with special tokens) exhausts the cross-encoder token budget."""


@dataclass(frozen=True)
class PairEncoding:
    input_ids: list[int]
    token_type_ids: list[int]
    attention_mask: list[int]
    query_tokens: int
    document_tokens_kept: int
    document_tokens_total: int

    @property
    def document_truncated(self) -> bool:
        return self.document_tokens_kept < self.document_tokens_total


def encode_cross_encoder_pair(
    tokenizer: Any, query: str, document_text: str, max_length: int = CROSS_ENCODER_MAX_LENGTH
) -> PairEncoding:
    """Encode one (query, document) pair, truncating only the document."""
    special = int(tokenizer.num_special_tokens_to_add(pair=True))
    q_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
    d_ids = tokenizer(document_text, add_special_tokens=False)["input_ids"]
    if len(q_ids) + special >= max_length:
        raise QueryTooLongError(
            f"query uses {len(q_ids)} tokens + {special} special tokens, which exhausts the "
            f"{max_length}-token cross-encoder budget; refusing to truncate the query"
        )
    enc = tokenizer(
        query,
        document_text,
        truncation="only_second",
        max_length=max_length,
        return_token_type_ids=True,
    )
    ids = list(enc["input_ids"])
    types = list(enc["token_type_ids"])
    kept = sum(1 for t in types if t == 1) - 1  # minus the final [SEP]
    q_kept = sum(1 for t in types if t == 0) - 2  # minus [CLS] and the first [SEP]
    if len(ids) > max_length or q_kept != len(q_ids):
        raise VerificationError("cross-encoder encoding violated the truncation contract")
    return PairEncoding(
        input_ids=ids,
        token_type_ids=types,
        attention_mask=list(enc["attention_mask"]),
        query_tokens=len(q_ids),
        document_tokens_kept=kept,
        document_tokens_total=len(d_ids),
    )
