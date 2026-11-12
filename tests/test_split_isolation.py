"""Split isolation: test data is unreachable without a valid freeze; splits never overlap."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from groundlab.data.loader import ProcessedDataset
from groundlab.errors import SplitAccessError
from groundlab.freeze import FreezeManifest, check_split_access, require_frozen
from groundlab.io_utils import sha256_file
from groundlab.types import Query


def test_query_type_carries_only_id_and_text() -> None:
    assert [f.name for f in dataclasses.fields(Query)] == ["query_id", "text"]


def test_test_split_refused_without_freeze(prepared_data_dir: Path) -> None:
    ds = ProcessedDataset(prepared_data_dir, "scifact")
    for view in ("evidence", "beir"):
        with pytest.raises(SplitAccessError):
            ds.queries(view, "test")
        with pytest.raises(SplitAccessError):
            ds.qrels(view, "test")


def test_dev_and_test_queries_and_ids_are_disjoint(prepared_data_dir: Path) -> None:
    ds = ProcessedDataset(prepared_data_dir, "scifact")
    token = FreezeManifest("unit-test", "-", "-", "-")  # loader-level test only, no git checks
    for view in ("evidence", "beir"):
        dev = {q.query_id for q in ds.queries(view, "dev")}
        test = {q.query_id for q in ds.queries(view, "test", token)}
        assert dev and test and not dev & test
        assert set(ds.qrels(view, "dev")) == dev
        assert set(ds.qrels(view, "test", token)) == test


def test_unknown_split_rejected(tmp_path: Path) -> None:
    with pytest.raises(SplitAccessError):
        check_split_access("train", "x", "y", tmp_path)
    assert check_split_access("dev", "x", "y", tmp_path) is None


# --------------------------------------------------------------------------- freeze guard


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.fixture
def frozen_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "groundlab").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "freeze").mkdir()
    (repo / "src" / "groundlab" / "m.py").write_text("x = 1\n")
    (repo / "uv.lock").write_text("lock v1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "code")
    tree = _git(repo, "rev-parse", "HEAD:src/groundlab")
    manifest = {
        "experiment": "exp",
        "config_sha256": "cfg",
        "code_tree": tree,
        "uv_lock_sha256": sha256_file(repo / "uv.lock"),
    }
    (repo / "freeze" / "exp.json").write_text(json.dumps(manifest))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "freeze")
    return repo


def test_freeze_accepts_matching_state(frozen_repo: Path) -> None:
    assert require_frozen("exp", "cfg", frozen_repo).experiment == "exp"


def test_freeze_refuses_changed_config(frozen_repo: Path) -> None:
    with pytest.raises(SplitAccessError, match="config changed"):
        require_frozen("exp", "other", frozen_repo)


def test_freeze_refuses_dirty_code(frozen_repo: Path) -> None:
    (frozen_repo / "src" / "groundlab" / "m.py").write_text("x = 2\n")
    with pytest.raises(SplitAccessError, match="uncommitted"):
        require_frozen("exp", "cfg", frozen_repo)


def test_freeze_refuses_committed_code_change(frozen_repo: Path) -> None:
    (frozen_repo / "src" / "groundlab" / "m.py").write_text("x = 2\n")
    _git(frozen_repo, "commit", "-qam", "change")
    with pytest.raises(SplitAccessError, match="code changed"):
        require_frozen("exp", "cfg", frozen_repo)


def test_freeze_refuses_changed_lockfile(frozen_repo: Path) -> None:
    (frozen_repo / "uv.lock").write_text("lock v2\n")
    _git(frozen_repo, "commit", "-qam", "bump deps")
    with pytest.raises(SplitAccessError, match=r"uv\.lock"):
        require_frozen("exp", "cfg", frozen_repo)


def test_freeze_refuses_manifest_without_lock_hash(frozen_repo: Path) -> None:
    path = frozen_repo / "freeze" / "exp.json"
    manifest = json.loads(path.read_text())
    del manifest["uv_lock_sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(SplitAccessError, match="lacks field"):
        require_frozen("exp", "cfg", frozen_repo)


def test_freeze_refuses_missing_manifest(frozen_repo: Path) -> None:
    with pytest.raises(SplitAccessError, match="no freeze manifest"):
        require_frozen("other-exp", "cfg", frozen_repo)


def test_loader_rejects_regenerated_manifest_after_freeze(prepared_data_dir: Path) -> None:
    path = prepared_data_dir / "processed" / "scifact" / "manifest.json"
    token = FreezeManifest(
        "unit-test", "-", "-", "-", data_manifest_sha256=sha256_file(path), protocol_sha256="p"
    )
    assert ProcessedDataset(prepared_data_dir, "scifact", token).dataset == "scifact"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(SplitAccessError, match="prepared-data manifest"):
        ProcessedDataset(prepared_data_dir, "scifact", token)


def test_guard_rejects_protocol_change_after_freeze(frozen_repo: Path) -> None:
    protocol = frozen_repo / "docs" / "EXPERIMENT_PLAN.md"
    protocol.parent.mkdir()
    protocol.write_text("protocol v1\n")
    path = frozen_repo / "freeze" / "exp.json"
    manifest = json.loads(path.read_text())
    manifest["protocol_sha256"] = sha256_file(protocol)
    path.write_text(json.dumps(manifest))
    _git(frozen_repo, "add", ".")
    _git(frozen_repo, "commit", "-qm", "freeze protocol")
    assert require_frozen("exp", "cfg", frozen_repo).protocol_sha256 == sha256_file(protocol)
    protocol.write_text("protocol v2\n")
    _git(frozen_repo, "commit", "-qam", "change protocol")
    with pytest.raises(SplitAccessError, match="analysis protocol changed"):
        require_frozen("exp", "cfg", frozen_repo)
