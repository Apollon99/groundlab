"""BM25 retrieval via bm25s, with project-controlled tokenization and tie-breaking.

Semantics:
- Document text is the canonical `title + "\n" + text` (D-019). The tokenizer splits on the
  separator exactly as it did on the former single space (D-009), so indexes are unchanged.
- Query terms absent from the vocabulary contribute 0 (standard BM25).
- Only documents with score > 0 (i.e. matching at least one term) are returned,
  as in Lucene. A query with no matching document gets an empty ranking.
- Ranking: score descending, then doc ID ascending.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import bm25s
import numpy as np

from groundlab.errors import VerificationError
from groundlab.io_utils import loads_strict, sha256_file, write_json
from groundlab.retrieval.tokenize import TOKENIZER_NAME, tokenize
from groundlab.text import canonical_document_text
from groundlab.types import Document, Query, Run, ScoredDoc


@dataclass(frozen=True)
class BM25Params:
    method: str = "lucene"
    k1: float = 1.2
    b: float = 0.75
    tokenizer: str = TOKENIZER_NAME

    def __post_init__(self) -> None:
        if self.tokenizer != TOKENIZER_NAME:
            raise ValueError(f"unsupported tokenizer {self.tokenizer!r}")


def document_text(doc: Document) -> str:
    return canonical_document_text(doc)


class BM25Index:
    def __init__(self, model: Any, doc_ids: list[str], params: BM25Params, corpus_sha256: str):
        self._model = model
        self.doc_ids = doc_ids
        self.params = params
        self.corpus_sha256 = corpus_sha256
        # Filled by save()/load(); empty for an index that only exists in memory.
        self.file_sha256: dict[str, str] = {}
        # Position of each doc in doc-ID string order, used as the tie-breaker.
        order = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i])
        self._id_rank = np.empty(len(doc_ids), dtype=np.int64)
        self._id_rank[order] = np.arange(len(doc_ids))

    @classmethod
    def build(cls, docs: Sequence[Document], params: BM25Params, corpus_sha256: str) -> BM25Index:
        if not docs:
            raise ValueError("cannot index an empty corpus")
        model = bm25s.BM25(method=params.method, k1=params.k1, b=params.b)
        model.index([tokenize(document_text(d)) for d in docs], show_progress=False)
        return cls(model, [d.doc_id for d in docs], params, corpus_sha256)

    def scores(self, query: Query) -> np.ndarray:
        vocab = self._model.vocab_dict
        tokens = [t for t in tokenize(query.text) if t in vocab]
        if not tokens:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        return np.asarray(self._model.get_scores(tokens))

    def search(self, queries: Sequence[Query], depth: int) -> Run:
        if depth <= 0:
            raise ValueError("depth must be positive")
        run: Run = {}
        for query in queries:
            scores = self.scores(query)
            if not np.all(np.isfinite(scores)):
                raise VerificationError(f"non-finite BM25 score for query {query.query_id}")
            hits = np.flatnonzero(scores > 0)
            order = np.lexsort((self._id_rank[hits], -scores[hits].astype(np.float64)))
            top = hits[order[:depth]]
            run[query.query_id] = [ScoredDoc(self.doc_ids[i], float(scores[i])) for i in top]
        return run

    # ------------------------------------------------------------------ persistence

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": "bm25",
            "params": asdict(self.params),
            "corpus_sha256": self.corpus_sha256,
            "n_docs": len(self.doc_ids),
            "file_sha256": self.file_sha256,
            "versions": {"bm25s": version("bm25s"), "PyStemmer": version("PyStemmer")},
        }

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "manifest.json").unlink(missing_ok=True)
        model_dir = index_dir / "bm25s"
        if model_dir.exists():
            shutil.rmtree(model_dir)  # no stale files from an earlier build
        self._model.save(str(model_dir), show_progress=False)
        write_json(index_dir / "doc_ids.json", self.doc_ids)
        self.file_sha256 = index_file_hashes(index_dir)
        write_json(index_dir / "manifest.json", self.manifest())

    @classmethod
    def load(cls, index_dir: Path, params: BM25Params, corpus_sha256: str) -> BM25Index:
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise VerificationError(f"no index at {index_dir}; run `groundlab build` first")
        manifest = loads_strict(manifest_path.read_text(encoding="utf-8"))
        if manifest["params"] != asdict(params):
            raise VerificationError("index was built with different BM25 params; rebuild it")
        if manifest["corpus_sha256"] != corpus_sha256:
            raise VerificationError("index was built from a different corpus; rebuild it")
        # Every index file, including doc_ids.json, must match its recorded SHA-256, and no
        # file may be missing or added. Checked before anything is deserialized.
        recorded = manifest.get("file_sha256")
        if not isinstance(recorded, dict) or not recorded:
            raise VerificationError("index manifest has no file hashes; rebuild the index")
        actual = index_file_hashes(index_dir)
        if actual != recorded:
            changed = sorted(
                name
                for name in recorded.keys() | actual.keys()
                if recorded.get(name) != actual.get(name)
            )
            raise VerificationError(f"index files differ from manifest: {changed}; rebuild")
        model = bm25s.BM25.load(str(index_dir / "bm25s"), show_progress=False)
        doc_ids = loads_strict((index_dir / "doc_ids.json").read_text(encoding="utf-8"))
        if len(doc_ids) != manifest["n_docs"]:
            raise VerificationError("doc_ids.json does not match the index manifest")
        index = cls(model, list(doc_ids), params, corpus_sha256)
        index.file_sha256 = dict(recorded)
        return index


def index_file_hashes(index_dir: Path) -> dict[str, str]:
    """SHA-256 of every file in an index directory except manifest.json (POSIX paths)."""
    return {
        path.relative_to(index_dir).as_posix(): sha256_file(path)
        for path in sorted(index_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
