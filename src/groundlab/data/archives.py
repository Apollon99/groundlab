"""Download, hash-verify and read pinned archives.

Members are read straight from the archive into memory; nothing is extracted to
disk, so archive path-traversal is not possible.
"""

from __future__ import annotations

import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from groundlab.data.specs import ArchiveSpec
from groundlab.errors import VerificationError
from groundlab.io_utils import sha256_bytes, sha256_file

Downloader = Callable[[str, Path], None]

_USER_AGENT = "groundlab-data-prepare/0.1"


def http_download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def verify_archive(spec: ArchiveSpec, path: Path) -> None:
    size = path.stat().st_size
    if size != spec.size:
        raise VerificationError(f"{path}: size {size} != pinned {spec.size} ({spec.key})")
    digest = sha256_file(path)
    if digest != spec.sha256:
        raise VerificationError(f"{path}: SHA-256 {digest} != pinned {spec.sha256} ({spec.key})")


def ensure_archive(
    spec: ArchiveSpec,
    raw_dir: Path,
    *,
    allow_download: bool,
    downloader: Downloader = http_download,
) -> Path:
    """Return the path of a verified archive, downloading it first if allowed."""
    path = raw_dir / spec.filename
    if not path.exists():
        if not allow_download:
            raise VerificationError(
                f"archive missing: {path} ({spec.key}); run without --verify-only"
            )
        downloader(spec.url, path)
    # A bad file is left in place for inspection, never silently re-downloaded.
    verify_archive(spec, path)
    return path


def read_members(spec: ArchiveSpec, path: Path) -> dict[str, bytes]:
    """Read every pinned member and verify its SHA-256."""
    data: dict[str, bytes] = {}
    if spec.kind == "zip":
        with zipfile.ZipFile(path) as zf:
            for name in spec.members:
                try:
                    data[name] = zf.read(name)
                except KeyError as exc:
                    raise VerificationError(f"{spec.key}: member {name!r} not in archive") from exc
    elif spec.kind == "tar.gz":
        with tarfile.open(path, "r:gz") as tf:
            for name in spec.members:
                try:
                    member = tf.getmember(name)
                except KeyError as exc:
                    raise VerificationError(f"{spec.key}: member {name!r} not in archive") from exc
                if not member.isfile():
                    raise VerificationError(f"{spec.key}: member {name!r} is not a regular file")
                handle = tf.extractfile(member)
                if handle is None:
                    raise VerificationError(f"{spec.key}: cannot read member {name!r}")
                data[name] = handle.read()
    else:
        raise ValueError(f"unknown archive kind {spec.kind!r}")
    for name, expected in spec.members.items():
        actual = sha256_bytes(data[name])
        if actual != expected:
            raise VerificationError(f"{spec.key}:{name}: SHA-256 {actual} != pinned {expected}")
    return data
