"""Pinned pretrained models: download at a fixed revision and verify every file by SHA-256.

Provenance of the pinned hashes: computed from files downloaded on 2026-11-12 at the
pinned revisions and independently checked against the Hugging Face API metadata
(LFS SHA-256 for weights, git blob SHA-1 for all other files). All matched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from groundlab.errors import VerificationError
from groundlab.io_utils import sha256_file


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    license: str
    files: dict[str, str]  # path inside the repo -> SHA-256


BGE_SMALL = ModelSpec(
    key="bge-small-en-v1.5",
    repo_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    license="MIT",
    files={
        "config.json": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
        "model.safetensors": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "tokenizer_config.json": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
        "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
        "1_Pooling/config.json": "d1caf60c96f5fba2157c0c26b76d80818fad6cf0b8eb5e73ec372ff9818eba5c",
        "sentence_bert_config.json": "84e39fda68ccbff05bfa723ae9c0e70e23e2ec373b76e0f8c6e71af72a693cbf",
        "config_sentence_transformers.json": "940d5f50db195fa6e5e6a4f122c095f77880de259d74b14a65779ed48bdd7c56",
    },
)

# Tokenizer files since M2; the weights were added in M3 (D-027). `model.safetensors` is
# 90,870,598 bytes; its SHA-256 equals the Hugging Face LFS SHA-256 at this revision.
MINILM_CE = ModelSpec(
    key="ms-marco-MiniLM-L6-v2",
    repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    license="Apache-2.0",
    files={
        "config.json": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
        "model.safetensors": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "tokenizer_config.json": "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
        "special_tokens_map.json": "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    },
)

MODELS: dict[str, ModelSpec] = {m.key: m for m in (BGE_SMALL, MINILM_CE)}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise VerificationError(f"unknown model {key!r}; known: {sorted(MODELS)}") from exc


def model_dir(spec: ModelSpec, models_dir: Path) -> Path:
    return models_dir / spec.key


def verify_model(spec: ModelSpec, models_dir: Path) -> dict[str, str]:
    """Check every pinned file exists and matches its SHA-256. Returns the hashes."""
    root = model_dir(spec, models_dir)
    for name, expected in spec.files.items():
        path = root / name
        if not path.is_file():
            raise VerificationError(
                f"{spec.key}: missing {name}; run `groundlab models fetch --model {spec.key}`"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise VerificationError(f"{spec.key}:{name}: SHA-256 {actual} != pinned {expected}")
    return dict(spec.files)


def fetch_model(spec: ModelSpec, models_dir: Path) -> dict[str, str]:
    """Download the pinned files at the pinned revision, then verify them."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import hf_hub_download

    root = model_dir(spec, models_dir)
    for name in spec.files:
        hf_hub_download(spec.repo_id, name, revision=spec.revision, local_dir=root)
    return verify_model(spec, models_dir)
