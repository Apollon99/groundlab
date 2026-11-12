"""Strict TOML experiment configs. Unknown or missing keys are errors."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groundlab.data.specs import VIEWS
from groundlab.errors import ConfigError
from groundlab.evaluation.metrics import Metric, parse_metric
from groundlab.io_utils import dumps_json, sha256_bytes, sha256_file
from groundlab.models import BGE_SMALL, MINILM_CE
from groundlab.retrieval.bm25 import BM25Params
from groundlab.retrieval.dense import DenseParams
from groundlab.retrieval.hybrid import HybridParams
from groundlab.retrieval.rerank import RERANK_DEPTH, RerankParams
from groundlab.text import CROSS_ENCODER_MAX_LENGTH

_COMMON: dict[str, set[str]] = {
    "experiment": {"name", "description"},
    "dataset": {"name", "view"},
    "evaluation": {"metrics"},
}
_HYBRID_KEYS = {"kind", "fusion", "rrf_k", "parent_depth", "bm25_config", "dense_config"}
_RETRIEVER_KEYS: dict[str, set[str]] = {
    "hybrid": _HYBRID_KEYS,
    "rerank": _HYBRID_KEYS
    | {
        "model",
        "revision",
        "max_length",
        "rerank_depth",
        "batch_size",
        "attn_implementation",
        "dtype",
        "device",
    },
    "bm25": {"kind", "method", "k1", "b", "tokenizer"},
    "dense": {
        "kind",
        "model",
        "revision",
        "pooling",
        "normalize",
        "max_length",
        "truncation_side",
        "query_instruction",
        "document_instruction",
        "document_separator",
        "attn_implementation",
        "dtype",
        "device",
        "batch_size",
    },
}
_RUN_KEYS: dict[str, set[str]] = {
    "bm25": {"depth"},
    "dense": {"depth", "latency_warmup"},
    "hybrid": {"depth", "latency_warmup"},
    "rerank": {"depth", "latency_warmup"},
}
FUSED_KINDS = ("hybrid", "rerank")


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    sha256: str
    name: str
    description: str
    dataset: str
    view: str
    kind: str
    bm25: BM25Params | None
    dense: DenseParams | None
    depth: int
    latency_warmup: int
    metrics: tuple[Metric, ...]
    raw: dict[str, Any]
    hybrid: HybridParams | None = None
    rerank: RerankParams | None = None
    # hybrid and rerank configs: the loaded parent configs, keyed "bm25" and "dense".
    parents: dict[str, ExperimentConfig] = field(default_factory=dict)

    @property
    def parent_sha256(self) -> dict[str, str]:
        return {key: parent.sha256 for key, parent in sorted(self.parents.items())}

    @property
    def freeze_sha256(self) -> str:
        """Bind the entire experiment, including referenced parent configs, at freeze time."""
        if not self.parents:
            return self.sha256
        return sha256_bytes(dumps_json({"config": self.sha256, "parents": self.parent_sha256}))


def _require(table: dict[str, Any], section: str, keys: set[str]) -> None:
    unknown = table.keys() - keys
    missing = keys - table.keys()
    if unknown:
        raise ConfigError(f"[{section}] unknown keys: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"[{section}] missing keys: {sorted(missing)}")


def _typed(table: dict[str, Any], key: str, typ: type) -> Any:
    value = table[key]
    # bool is a subclass of int; keep them apart.
    if typ is int and isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    if typ is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, typ):
        raise ConfigError(f"{key} must be of type {typ.__name__}")
    return value


# Approved D-019a (512 tokens) and D-020 (fixed query instruction, trailing space included).
BGE_MAX_LENGTH = 512
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _check_shipped_bge(dense: DenseParams) -> None:
    if dense.max_length != BGE_MAX_LENGTH:
        raise ValueError(
            f"dense.max_length must be {BGE_MAX_LENGTH} for BGE, got {dense.max_length}"
        )
    if dense.query_instruction != BGE_QUERY_INSTRUCTION:
        raise ValueError(
            f"dense.query_instruction must be exactly {BGE_QUERY_INSTRUCTION!r} for BGE, "
            f"got {dense.query_instruction!r}"
        )


def _check_shipped_cross_encoder(rerank: RerankParams) -> None:
    if rerank.max_length != CROSS_ENCODER_MAX_LENGTH:
        raise ValueError(f"rerank.max_length must be {CROSS_ENCODER_MAX_LENGTH} for MiniLM")
    if rerank.rerank_depth != RERANK_DEPTH:
        raise ValueError(f"rerank.rerank_depth must be {RERANK_DEPTH} for MiniLM")


def _load_parent(
    child: Path, ref: Any, kind: str, dataset: str, view: str, parent_depth: int
) -> ExperimentConfig:
    """Load a parent config named next to the child config; it must match dataset and view."""
    if not isinstance(ref, str) or not ref or Path(ref).name != ref:
        raise ConfigError(f"{kind}_config must be a file name next to the config, got {ref!r}")
    parent = load_config(child.parent / ref, expected_kind=kind)
    if (parent.dataset, parent.view) != (dataset, view):
        raise ConfigError(f"{ref}: dataset/view differ from the {child.name} config")
    if parent.depth != parent_depth:
        raise ConfigError(f"{ref}: run.depth {parent.depth} != hybrid parent_depth {parent_depth}")
    return parent


def load_config(path: Path, *, expected_kind: str | None = None) -> ExperimentConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    _require(raw, "<root>", {*_COMMON, "retriever", "run"})
    for section in (*_COMMON, "retriever", "run"):
        if not isinstance(raw[section], dict):
            raise ConfigError(f"[{section}] must be a table")
    for section, keys in _COMMON.items():
        _require(raw[section], section, keys)
    kind = raw["retriever"].get("kind")
    if not isinstance(kind, str) or kind not in _RETRIEVER_KEYS:
        raise ConfigError(f"retriever.kind must be one of {sorted(_RETRIEVER_KEYS)}")
    if expected_kind is not None and kind != expected_kind:
        raise ConfigError(f"{path.name}: expected a {expected_kind} config, got {kind}")
    _require(raw["retriever"], "retriever", _RETRIEVER_KEYS[kind])
    _require(raw["run"], "run", _RUN_KEYS[kind])

    name = raw["experiment"]["name"]
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        raise ConfigError("experiment.name must be alphanumeric with - or _")
    dataset, view = raw["dataset"]["name"], raw["dataset"]["view"]
    if dataset not in VIEWS or view not in VIEWS[dataset]:
        raise ConfigError(f"unknown dataset/view {dataset!r}/{view!r}")
    depth = _typed(raw["run"], "depth", int)
    if depth <= 0:
        raise ConfigError("run.depth must be a positive integer")
    warmup = _typed(raw["run"], "latency_warmup", int) if kind != "bm25" else 0
    if warmup < 0:
        raise ConfigError("run.latency_warmup must be >= 0")
    r = raw["retriever"]
    bm25: BM25Params | None = None
    dense: DenseParams | None = None
    hybrid: HybridParams | None = None
    rerank: RerankParams | None = None
    parents: dict[str, ExperimentConfig] = {}
    try:
        metrics = tuple(parse_metric(m) for m in raw["evaluation"]["metrics"])
        if kind == "bm25":
            bm25 = BM25Params(
                method=_typed(r, "method", str),
                k1=_typed(r, "k1", float),
                b=_typed(r, "b", float),
                tokenizer=_typed(r, "tokenizer", str),
            )
        elif kind in FUSED_KINDS:
            hybrid = HybridParams(
                fusion=_typed(r, "fusion", str),
                rrf_k=_typed(r, "rrf_k", int),
                parent_depth=_typed(r, "parent_depth", int),
            )
            if depth != hybrid.parent_depth:
                raise ConfigError(f"fused run.depth must be {hybrid.parent_depth}")
            for key in ("bm25", "dense"):
                parents[key] = _load_parent(
                    path, r[f"{key}_config"], key, dataset, view, hybrid.parent_depth
                )
            if kind == "rerank":
                rerank = RerankParams(
                    model=_typed(r, "model", str),
                    revision=_typed(r, "revision", str),
                    max_length=_typed(r, "max_length", int),
                    rerank_depth=_typed(r, "rerank_depth", int),
                    batch_size=_typed(r, "batch_size", int),
                    attn_implementation=_typed(r, "attn_implementation", str),
                    dtype=_typed(r, "dtype", str),
                    device=_typed(r, "device", str),
                )
                if rerank.model == MINILM_CE.key:
                    _check_shipped_cross_encoder(rerank)
        else:
            dense = DenseParams(
                model=_typed(r, "model", str),
                revision=_typed(r, "revision", str),
                pooling=_typed(r, "pooling", str),
                normalize=_typed(r, "normalize", bool),
                max_length=_typed(r, "max_length", int),
                truncation_side=_typed(r, "truncation_side", str),
                query_instruction=_typed(r, "query_instruction", str),
                document_instruction=_typed(r, "document_instruction", str),
                document_separator=_typed(r, "document_separator", str),
                attn_implementation=_typed(r, "attn_implementation", str),
                dtype=_typed(r, "dtype", str),
                device=_typed(r, "device", str),
                batch_size=_typed(r, "batch_size", int),
            )
        if dense is not None and dense.model == BGE_SMALL.key:
            _check_shipped_bge(dense)
    except (ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc
    return ExperimentConfig(
        path=path,
        sha256=sha256_file(path),
        name=name,
        description=str(raw["experiment"]["description"]),
        dataset=dataset,
        view=view,
        kind=kind,
        bm25=bm25,
        dense=dense,
        depth=depth,
        latency_warmup=warmup,
        metrics=metrics,
        raw=raw,
        hybrid=hybrid,
        rerank=rerank,
        parents=parents,
    )
