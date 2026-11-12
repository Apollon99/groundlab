"""TREC run file IO: `qid Q0 docid rank score tag`, one line per hit."""

from __future__ import annotations

from pathlib import Path

from groundlab.errors import VerificationError
from groundlab.io_utils import atomic_write_bytes
from groundlab.types import Run, ScoredDoc


def dumps_run(run: Run, tag: str) -> bytes:
    if any(ch.isspace() for ch in tag):
        raise ValueError("run tag must not contain whitespace")
    lines = []
    for qid in sorted(run):
        for rank, hit in enumerate(run[qid], start=1):
            if any(ch.isspace() for ch in qid + hit.doc_id):
                raise ValueError(f"whitespace in id: {qid!r}/{hit.doc_id!r}")
            lines.append(f"{qid} Q0 {hit.doc_id} {rank} {hit.score!r} {tag}\n")
    return "".join(lines).encode("utf-8")


def write_run(path: Path, run: Run, tag: str) -> None:
    atomic_write_bytes(path, dumps_run(run, tag))


def read_run(path: Path) -> Run:
    """Read a run, requiring ranks 1..n in file order for each query."""
    run: Run = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 6 or parts[1] != "Q0":
            raise VerificationError(f"{path}:{lineno}: malformed run line")
        qid, _, did, rank, score, _tag = parts
        hits = run.setdefault(qid, [])
        if int(rank) != len(hits) + 1:
            raise VerificationError(f"{path}:{lineno}: rank {rank} out of order for {qid}")
        hits.append(ScoredDoc(did, float(score)))
    return run


def rankings(run: Run) -> dict[str, list[str]]:
    return {qid: [h.doc_id for h in hits] for qid, hits in run.items()}
