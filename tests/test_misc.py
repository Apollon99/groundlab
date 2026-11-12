from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from groundlab import cli, envinfo
from groundlab.config import load_config
from groundlab.errors import EXIT_CUDA_UNAVAILABLE, ConfigError, VerificationError
from groundlab.io_utils import loads_strict
from groundlab.runio import read_run, write_run
from groundlab.types import ScoredDoc

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["bm25_scifact_evidence", "bm25_scifact_beir", "bm25_nfcorpus"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(REPO / "configs" / f"{name}.toml")
    assert cfg.name == name
    assert (cfg.bm25.k1, cfg.bm25.b, cfg.bm25.method) == (1.2, 0.75, "lucene")


def test_config_rejects_unknown_and_missing_keys(tmp_path: Path) -> None:
    text = (REPO / "configs" / "bm25_scifact_evidence.toml").read_text(encoding="utf-8")
    (tmp_path / "a.toml").write_text(text.replace("depth = 100", "depth = 100\nseed = 1"))
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(tmp_path / "a.toml")
    (tmp_path / "b.toml").write_text(text.replace("b = 0.75\n", ""))
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(tmp_path / "b.toml")


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        loads_strict('{"a": 1, "a": 2}')


def test_run_roundtrip_and_rank_validation(tmp_path: Path) -> None:
    run = {"q1": [ScoredDoc("d1", 2.5), ScoredDoc("d2", 1.0)], "q2": [ScoredDoc("d3", 0.1)]}
    write_run(tmp_path / "r.trec", run, tag="t")
    assert read_run(tmp_path / "r.trec") == run
    (tmp_path / "bad.trec").write_text("q1 Q0 d1 2 1.0 t\n")
    with pytest.raises(VerificationError, match="out of order"):
        read_run(tmp_path / "bad.trec")


def test_env_collect_has_required_fields() -> None:
    info = envinfo.collect(REPO)
    for key in ("os", "python", "cpu", "memory_total_bytes", "disk", "gpu", "packages", "git"):
        assert key in info
    assert str(Path.home()) not in info["python"]["executable"]


def test_require_cuda_fails_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_gpu() -> dict[str, Any]:
        return {"cuda_available": False}

    monkeypatch.setattr(envinfo, "gpu_info", no_gpu)
    assert cli.main(["env", "check", "--require-cuda"]) == EXIT_CUDA_UNAVAILABLE
    assert cli.main(["env", "check"]) == 0


@pytest.mark.cuda
def test_cuda_visible_when_present() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    assert envinfo.gpu_info()["cuda_available"] is True
