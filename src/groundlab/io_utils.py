"""Small file helpers: hashing, strict JSON/JSONL, atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def loads_strict(text: str) -> Any:
    """json.loads that rejects duplicate keys instead of silently keeping the last."""
    return json.loads(text, object_pairs_hook=_no_duplicate_keys)


def iter_jsonl_lines(lines: Iterable[str], source: str) -> Iterator[dict[str, Any]]:
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = loads_strict(line)
        except ValueError as exc:
            raise ValueError(f"{source}:{lineno}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{source}:{lineno}: expected a JSON object")
        yield obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(iter_jsonl_lines(handle, str(path)))


def dumps_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def dumps_json(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, dumps_json(obj))
