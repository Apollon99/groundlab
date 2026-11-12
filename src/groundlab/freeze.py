"""Test-split guard.

The confirmatory (test) split may only be touched by an experiment whose config,
evaluation-relevant code and dependency lock were frozen first. A freeze manifest
lives at `freeze/<experiment-name>.json` in the repository and records:

    {"experiment": str, "config_sha256": str, "code_tree": str, "uv_lock_sha256": str}

For experiments with parent configs, `config_sha256` is the combined digest from
`ExperimentConfig.freeze_sha256`. For other experiments it is the config file hash.
`code_tree` is the git tree hash of `src/groundlab` at freeze time and
`uv_lock_sha256` is the SHA-256 of `uv.lock` (D-015). The guard refuses unless the
manifest exists, the config hash matches, the lock hash matches, the committed tree
matches, and `src/`, `configs/`, `freeze/` and `uv.lock` have no uncommitted changes.

The study freeze also pins the prepared-data manifest and analysis protocol.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from groundlab.errors import SplitAccessError
from groundlab.io_utils import loads_strict, sha256_file

SPLITS = ("dev", "test")
PROTOCOL_PATH = Path("docs/EXPERIMENT_PLAN.md")
FROZEN_PATHS = ("src", "configs", "freeze", "uv.lock", PROTOCOL_PATH.as_posix())


@dataclass(frozen=True)
class FreezeManifest:
    experiment: str
    config_sha256: str
    code_tree: str
    uv_lock_sha256: str
    data_manifest_sha256: str | None = None
    protocol_sha256: str | None = None


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SplitAccessError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def require_frozen(experiment: str, config_sha256: str, repo_root: Path) -> FreezeManifest:
    path = repo_root / "freeze" / f"{experiment}.json"
    if not path.exists():
        raise SplitAccessError(
            f"test split refused: no freeze manifest at {path}. "
            "Run groundlab freeze and commit its manifests first."
        )
    raw = loads_strict(path.read_text(encoding="utf-8"))
    try:
        manifest = FreezeManifest(
            experiment=str(raw["experiment"]),
            config_sha256=str(raw["config_sha256"]),
            code_tree=str(raw["code_tree"]),
            uv_lock_sha256=str(raw["uv_lock_sha256"]),
            data_manifest_sha256=raw.get("data_manifest_sha256"),
            protocol_sha256=raw.get("protocol_sha256"),
        )
    except KeyError as exc:
        raise SplitAccessError(f"freeze manifest {path} lacks field {exc}") from exc
    if manifest.experiment != experiment:
        raise SplitAccessError(
            f"freeze manifest is for {manifest.experiment!r}, not {experiment!r}"
        )
    if manifest.config_sha256 != config_sha256:
        raise SplitAccessError("test split refused: config changed since freeze")
    lock = repo_root / "uv.lock"
    if not lock.exists() or sha256_file(lock) != manifest.uv_lock_sha256:
        raise SplitAccessError("test split refused: uv.lock missing or changed since freeze")
    if manifest.protocol_sha256 is not None:
        protocol = repo_root / PROTOCOL_PATH
        if not protocol.is_file() or sha256_file(protocol) != manifest.protocol_sha256:
            raise SplitAccessError("test split refused: analysis protocol changed since freeze")
    dirty = _git(repo_root, "status", "--porcelain", "--", *FROZEN_PATHS)
    if dirty:
        raise SplitAccessError(
            "test split refused: uncommitted changes to frozen code, configs or protocol"
        )
    tree = _git(repo_root, "rev-parse", "HEAD:src/groundlab")
    if tree != manifest.code_tree:
        raise SplitAccessError("test split refused: evaluation code changed since freeze")
    return manifest


def check_split_access(
    split: str, experiment: str, config_sha256: str, repo_root: Path
) -> FreezeManifest | None:
    """Return None for dev; for test return the verified freeze or raise."""
    if split not in SPLITS:
        raise SplitAccessError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "dev":
        return None
    return require_frozen(experiment, config_sha256, repo_root)
