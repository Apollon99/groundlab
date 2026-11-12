"""Read processed datasets. All split access goes through here.

Queries and qrels are loaded by separate functions so callers that rank
(retrievers) never need to touch qrels. Test-split access requires a verified
`FreezeManifest` (see groundlab.freeze).
"""

from __future__ import annotations

from pathlib import Path

from groundlab.data.specs import SPLITS, VIEWS
from groundlab.errors import SplitAccessError, VerificationError
from groundlab.freeze import FreezeManifest
from groundlab.io_utils import loads_strict, read_jsonl, sha256_file
from groundlab.types import Document, Qrels, Query


class ProcessedDataset:
    def __init__(self, data_dir: Path, dataset: str, freeze: FreezeManifest | None = None) -> None:
        if dataset not in VIEWS:
            raise VerificationError(f"unknown dataset {dataset!r}")
        self.dataset = dataset
        self.root = data_dir / "processed" / dataset
        audit_path = self.root / "audit.json"
        if not audit_path.exists():
            raise VerificationError(f"{audit_path} missing; run `groundlab data prepare` first")
        audit = loads_strict(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            raise VerificationError(f"{audit_path}: audit status is {audit.get('status')!r}")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise VerificationError(f"{manifest_path} missing; run data prepare first")
        if freeze is not None:
            if freeze.data_manifest_sha256 != sha256_file(manifest_path):
                raise SplitAccessError(
                    "test split refused: prepared-data manifest differs from freeze"
                )
            if freeze.protocol_sha256 is None:
                raise SplitAccessError("test split refused: analysis protocol is not frozen")
        self.manifest: dict[str, object] = loads_strict(manifest_path.read_text(encoding="utf-8"))

    def _checked(self, rel: str) -> Path:
        files = self.manifest["files"]
        assert isinstance(files, dict)
        path = self.root / rel
        if rel not in files:
            raise VerificationError(f"{rel} not listed in manifest")
        if sha256_file(path) != files[rel]:
            raise VerificationError(f"{path}: content differs from manifest; re-run data prepare")
        return path

    @property
    def corpus_sha256(self) -> str:
        files = self.manifest["files"]
        assert isinstance(files, dict)
        return str(files["corpus.jsonl"])

    def corpus(self) -> list[Document]:
        rows = read_jsonl(self._checked("corpus.jsonl"))
        return [Document(r["doc_id"], r["title"], r["text"]) for r in rows]

    def _split_dir(self, view: str, split: str, freeze: FreezeManifest | None) -> str:
        if view not in VIEWS[self.dataset]:
            raise VerificationError(f"dataset {self.dataset!r} has no view {view!r}")
        if split not in SPLITS:
            raise SplitAccessError(f"unknown split {split!r}")
        if split == "test" and freeze is None:
            raise SplitAccessError("test split requires a verified freeze manifest")
        return f"views/{view}/{split}"

    def input_sha256(
        self, view: str, split: str, freeze: FreezeManifest | None = None
    ) -> dict[str, str]:
        """Verified SHA-256 of the corpus, query and qrels files for one view/split.

        Only hashes are computed; no labels are parsed. `run` records these so that
        `evaluate` can refuse to score a run against inputs that changed afterwards,
        even if the processed-data manifest was regenerated consistently.
        """
        base = self._split_dir(view, split, freeze)
        rels = ("corpus.jsonl", f"{base}/queries.jsonl", f"{base}/qrels.tsv")
        return {rel: sha256_file(self._checked(rel)) for rel in rels}

    def queries(self, view: str, split: str, freeze: FreezeManifest | None = None) -> list[Query]:
        base = self._split_dir(view, split, freeze)
        rows = read_jsonl(self._checked(f"{base}/queries.jsonl"))
        for row in rows:
            if set(row) != {"query_id", "text"}:
                raise VerificationError(f"query row has unexpected fields {sorted(row)}")
        return [Query(r["query_id"], r["text"]) for r in rows]

    def qrels(self, view: str, split: str, freeze: FreezeManifest | None = None) -> Qrels:
        base = self._split_dir(view, split, freeze)
        lines = self._checked(f"{base}/qrels.tsv").read_text(encoding="utf-8").splitlines()
        if lines[0] != "query_id\tdoc_id\tgrade":
            raise VerificationError("unexpected processed qrels header")
        qrels: Qrels = {}
        for line in lines[1:]:
            qid, did, grade = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(grade)
        return qrels
