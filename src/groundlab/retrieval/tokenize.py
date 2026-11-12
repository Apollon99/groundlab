"""BM25 tokenizer: regex `[^\\W_]+`, Unicode casefold, English Snowball stemming, no stopwords."""

from __future__ import annotations

import re
from functools import lru_cache

import Stemmer

TOKENIZER_NAME = "regex_casefold_snowball_en"
_TOKEN_RE = re.compile(r"[^\W_]+")


@lru_cache(maxsize=1)
def _stemmer() -> Stemmer.Stemmer:
    # PyStemmer's "english" algorithm is Snowball English (Porter2).
    return Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.casefold())
    return list(_stemmer().stemWords(tokens))
