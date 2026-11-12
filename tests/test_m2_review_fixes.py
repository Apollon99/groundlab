"""Regressions for the three M2 review fixes. Deterministic, no CUDA or network needed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from groundlab import cli
from groundlab.config import load_config
from groundlab.errors import ConfigError
from groundlab.retrieval import dense
from groundlab.types import Query, ScoredDoc

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "configs" / "dense_scifact_beir.toml"
INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ------------------------------------------------------------------ fakes


class _FakeCuda:
    """Mimics torch's global peak counters: a probe entry resets them."""

    peak = (0, 0)


class _FakeProbe:
    def __init__(self, device: str) -> None:
        pass

    def __enter__(self) -> _FakeProbe:
        _FakeCuda.peak = (0, 0)
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def result(self) -> dict[str, int]:
        allocated, reserved = _FakeCuda.peak
        return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}


class _FakeEncoder:
    device = "cuda"
    dim = 2

    def __init__(self) -> None:
        self.params = SimpleNamespace(batch_size=4)
        self.model_file_sha256: dict[str, str] = {"model.safetensors": "0" * 64}

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        _FakeCuda.peak = (111, 222)  # query-batch phase
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(queries), 1))

    def query_texts(self, queries: list[Query]) -> list[str]:
        return [q.text for q in queries]

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        _FakeCuda.peak = (333, 444)  # latency phase (probe reset happened on entry)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def describe(self) -> dict[str, Any]:
        return {}


class _FakeIndex:
    """Batched search and timed single-query rank deliberately disagree."""

    def __init__(self, warmup: int = 0) -> None:
        self.manifest: dict[str, Any] = {"gpu_memory": None}
        self.warmup = warmup
        self.rank_calls = 0

    def search_embeddings(self, queries: list[Query], emb: np.ndarray, depth: int) -> Any:
        return {q.query_id: [ScoredDoc("batch-a", 1.0), ScoredDoc("batch-b", 0.5)] for q in queries}

    def scores(self, emb: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def rank(self, scores: np.ndarray, depth: int) -> list[ScoredDoc]:
        """Warm-up and timed calls return distinct, numbered doc IDs."""
        self.rank_calls += 1
        if self.rank_calls <= self.warmup:
            return [ScoredDoc(f"warmup-{self.rank_calls}", 1.0)]
        return [ScoredDoc(f"timed-{self.rank_calls - self.warmup}", 1.0)]


def _run_fake(monkeypatch: pytest.MonkeyPatch, index: _FakeIndex, n: int, warmup: int) -> Any:
    monkeypatch.setattr(cli, "GpuMemoryProbe", _FakeProbe)
    monkeypatch.setattr(dense, "_sync", lambda device: None)
    monkeypatch.setattr(cli, "_dense_encoder", lambda cfg, models_dir: _FakeEncoder())
    monkeypatch.setattr(cli.DenseIndex, "load", lambda *a, **k: index)
    cfg = SimpleNamespace(depth=2, latency_warmup=warmup)
    queries = [Query(f"q{i}", "alpha") for i in range(n)]
    return cli._run_dense(cfg, Path("."), Path("."), queries, "corpus")  # type: ignore[arg-type]


# ------------------------------------------------------------------ fix 1


def test_per_phase_gpu_counters_survive_report_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    _, details = _run_fake(monkeypatch, _FakeIndex(), n=3, warmup=1)
    assert details["query_encoding"]["gpu_memory"] == {
        "peak_allocated_bytes": 111,
        "peak_reserved_bytes": 222,
    }
    assert details["latency"]["gpu_memory"] == {
        "peak_allocated_bytes": 333,
        "peak_reserved_bytes": 444,
    }


# ------------------------------------------------------------------ fix 2


def test_agreement_uses_timed_single_query_rankings(monkeypatch: pytest.MonkeyPatch) -> None:
    index = _FakeIndex(warmup=2)
    _, details = _run_fake(monkeypatch, index, n=3, warmup=2)
    # Batched reconstruction would give ["batch-a", "batch-b"] == batch ranking (3 identical);
    # the timed calls return different IDs, so none agree.
    assert details["batch_vs_single_query"]["queries_with_identical_ranking"] == 0
    assert details["batch_vs_single_query"]["n_queries"] == 3
    # 2 warm-up rank calls + 3 timed; only the timed ones are compared.
    assert index.rank_calls == 5
    assert "single_query_rankings" not in details["latency"]
    assert "single_query_embeddings" not in details["latency"]


def test_measure_latency_returns_timed_rankings_and_excludes_warmup() -> None:
    index = _FakeIndex(warmup=3)
    ticks = iter(range(1000))
    result = dense.measure_latency(
        _FakeEncoder(),  # type: ignore[arg-type]
        index,  # type: ignore[arg-type]
        [Query(f"q{i}", "a") for i in range(4)],
        2,
        warmup=3,
        clock=lambda: next(ticks),
    )
    # Exactly the four timed calls, in order; any warm-up ID or shift would break equality.
    assert result["single_query_rankings"] == [[f"timed-{i}"] for i in range(1, 5)]
    assert index.rank_calls == 3 + 4
    assert result["n"] == 4


# ------------------------------------------------------------------ fix 3


def _config_with(tmp_path: Path, old: str, new: str) -> Path:
    text = SHIPPED.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "cfg.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_shipped_bge_config_loads() -> None:
    cfg = load_config(SHIPPED)
    assert cfg.dense is not None
    assert cfg.dense.max_length == 512 and cfg.dense.query_instruction == INSTRUCTION


def test_bge_config_rejects_128_token_limit(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="max_length"):
        load_config(_config_with(tmp_path, "max_length = 512", "max_length = 128"))


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " ",
        "Represent this sentence for searching relevant passages:",  # no trailing space
        "Represent this query for searching relevant passages: ",
        INSTRUCTION + " ",
    ],
)
def test_bge_config_rejects_blank_or_wrong_prefix(tmp_path: Path, bad: str) -> None:
    path = _config_with(
        tmp_path, f'query_instruction = "{INSTRUCTION}"', f'query_instruction = "{bad}"'
    )
    with pytest.raises(ConfigError, match="query_instruction"):
        load_config(path)


def test_cli_rejects_bad_bge_config_with_usage_exit(tmp_path: Path) -> None:
    path = _config_with(tmp_path, "max_length = 512", "max_length = 128")
    assert cli.main(["build", "--config", str(path), "--data-dir", str(tmp_path)]) == 2


def test_synthetic_encoder_keeps_flexibility(tmp_path: Path) -> None:
    from tests.test_dense_cli import CONFIG

    path = tmp_path / "synthetic.toml"
    blanked = CONFIG.replace(f'"{INSTRUCTION}"', '""')
    path.write_text(blanked, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.dense is not None
    assert cfg.dense.max_length == 32
    assert cfg.dense.query_instruction == ""
