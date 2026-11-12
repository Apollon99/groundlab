# Architecture

## Data flow

```
pinned archives ──► data prepare ──► processed views + manifest + audit
 (raw/, SHA-256)     (verify, build,      (processed/<dataset>/)
                      audit counts)              │
                                                 ▼
                    configs/*.toml ──► build ──► index (artifacts/indexes/<exp>/)
                                                 │
                         split guard ──► run ──► run.trec + run_manifest.json
                     (test needs freeze)         │            (outputs/runs/<exp>/<split>/)
                                                 ▼
                         split guard ──► evaluate ──► metrics.json (native, cross-checked)
                                                 │
                                    (M4+) stats ──► (M6) report
```

## Modules (`src/groundlab/`)

| Module | Responsibility | Knows about qrels? |
|---|---|---|
| `data/specs.py` | Pinned URLs, sizes, SHA-256 (archives and members), expected counts | — |
| `data/archives.py` | Download, verify, read members in memory (no extraction to disk) | — |
| `data/prepare.py` | Build processed corpus and views; audit measured vs expected | builds them |
| `data/loader.py` | Read processed files; verify manifest hashes; enforce split access | yes, separate function |
| `freeze.py` | Test-split guard (freeze manifest + git state) | — |
| `retrieval/tokenize.py`, `retrieval/bm25.py` | BM25 indexing and search | **no** |
| `text.py` | Canonical document text (`title + "\n" + text`); cross-encoder pair encoding with document-only truncation | **no** |
| `models.py` | Pinned model specs (repo, revision, per-file SHA-256); fetch and verify | — |
| `retrieval/dense.py` | CUDA bi-encoder, verified embedding cache, exact float64 dot-product search, latency and GPU-memory measurement | **no** |
| `retrieval/hybrid.py` | Exact RRF over the BM25 and dense lists; `HybridCandidates` (parent retrieval + fusion, shared by both fused arms); `PhaseTimer`, single-query latency | **no** |
| `retrieval/rerank.py` | Pinned CUDA cross-encoder; head-only reordering (`rerank_candidates`) | **no** |
| `evaluation/metrics.py` | Native nDCG/Recall/MRR | yes |
| `evaluation/trec_check.py` | pytrec_eval cross-check (raises on mismatch) | yes |
| `runio.py` | TREC run read/write with rank validation | — |
| `envinfo.py` | OS, CPU, RAM, disk, GPU, driver, packages, git state | — |
| `config.py` | Strict TOML loader (unknown or missing keys are errors) | — |
| `cli.py` | Command wiring and exit codes | — |

Retrievers receive `Query(query_id, text)` objects and documents only. `Query` has no other
fields, and `BM25Index.search(queries, depth)` has no other parameters (both are tested).

## Split isolation

- Processed files are per view and split: `views/<view>/<split>/{queries.jsonl,qrels.tsv}`.
- `ProcessedDataset.queries/qrels(view, "test", freeze)` raise `SplitAccessError` unless a
  `FreezeManifest` is passed. `run` and `evaluate` call `check_split_access` before reading
  any data, and only obtain a manifest from `freeze.require_frozen`, which checks:
  the manifest file exists in `freeze/`, the config SHA-256 matches, the SHA-256 of `uv.lock`
  matches, `src/`, `configs/`, `freeze/` and `uv.lock` are clean, and the git tree hash of
  `src/groundlab` equals the frozen one.
- For fused experiments, the frozen config digest binds the child config and both parent
  config hashes. Changing a parent invalidates the freeze even if the child file is unchanged.
- The study freeze also pins the prepared-data manifest and the analysis protocol.
- `data prepare` asserts that dev and test query IDs never overlap within a view.

## Integrity chain

archive SHA-256 → member SHA-256 → expected counts (audit.json, any difference fails) →
processed-file SHA-256 (manifest.json, checked on every load) → corpus SHA-256 recorded in the
index manifest (a stale index is refused) → config SHA-256 + run SHA-256 recorded in the run
manifest (evaluate refuses a run produced by a different config or a modified run file).

## Determinism

- BM25 is deterministic on CPU. Ties are broken by doc ID ascending, implemented once in
  `types.rank_scores` and as a lexsort in `BM25Index.search`.
- GPU stages make no cross-machine bit-identity claim; reproducibility is defined by the
  committed config, manifests and code revision, and checked with tolerances. On this machine,
  two independent dense builds gave byte-identical `embeddings.npy` and a repeated dev run gave
  an identical `run.trec` (D-022). Batched and single-query encodings differ by at most ~2e-7,
  which changed the top-100 order for 2 of 505 SciFact dev queries; the evaluated run always
  comes from the batched pass.
- BM25 index files are not byte-identical across builds (vocabulary ID order, D-026); scores
  and rankings are.

## Dense pipeline (M2)

`build` (dense config): verify model files → load tokenizer/model from the pinned local copy
→ encode canonical document texts on CUDA (FP32, no TF32, eager attention, CLS pooling, L2
norm, right truncation at 512) → save `embeddings.npy` + `doc_ids.json` + manifest (params,
model file hashes, corpus hash, file hashes, token-length stats, encode time, peak GPU memory).

`run` (dense config): refuse test split without freeze → require CUDA (exit 5) → verify model
files and the embedding cache → batched query encoding with the fixed instruction → exact
float64 dot product on CPU over all docs → write run. Then a separate single-query latency pass
(D-024) and a batch-vs-single agreement check are recorded in the run manifest.

## Hybrid and rerank pipeline (M3, D-027 to D-030)

Config: `kind = "hybrid"` or `"rerank"` names its parents (`bm25_config`, `dense_config`, file
names next to the config). The parents must have the same dataset and view and `run.depth` 100.
Their SHA-256 values are recorded in the run manifest and re-checked by `evaluate`. The six
shipped fused configs of one view share an identical fusion block and identical parents (tested),
so both arms always start from the same fused ordering.

`build` (fused config): require CUDA (exit 5), then only verify: both parent indexes (every file
hash, params, corpus hash, model files) and, for rerank, the reranker files. Nothing is built or
rewritten; run `build` on the two parent configs first. Parent indexes are read from
`<--index-root>/<parent name>` (default `artifacts/indexes`).

`run` (fused config): refuse test split without freeze (exit 4) → require CUDA (exit 5) → load
the verified parent indexes → throughput pass → latency pass → write run and manifest.

```
queries ─► BM25 top-100 (CPU, as the bm25 arm) ─┐
        └► BGE encode (CUDA) ► exact top-100 ───┴► RRF k=60 ─► fused top-100 ──► hybrid run
                                                                  │
                                          first 30 ► MiniLM pairs (CUDA) ► reorder head ──► rerank run
                                          positions 31..100: unchanged
```

- **Fusion**: first 100 real hits of each parent, 1-based ranks, `1/(60 + rank)` summed, absent
  contribution zero, no padding. The sum is an exact rational (`fractions.Fraction`), so
  mathematically equal scores tie exactly. Order: score descending, then doc ID ascending
  (string order); the top 100 are emitted. The stored score is the rounded float of the exact
  sum, so equal-looking scores in the file are possible; the rank column decides.
- **Rerank**: score the first `min(30, len(fused))` candidates with raw logits; reorder only
  that head (logit descending, doc ID ascending on exact ties); positions after 30 keep
  order and membership. File scores of the rerank run are rank-derived and strictly descending
  (`len - position`) because logits and RRF scores are on different scales; raw logits are not
  stored in the run file. Evaluation reads the rank column.
- **Numerics and batching (rerank)**: FP32, TF32 off, eager attention, `inference_mode`; weights
  loaded with `use_safetensors=True, local_files_only=True` after SHA-256 verification; input
  is `encode_cross_encoder_pair(raw query, canonical document text)`, 512 tokens, document-only
  truncation, no BGE instruction. The pairs of one query form one batch (chunks of 32, right
  padded to the longest pair). For identical candidate lists, the two paths use the same batches.
  Batched and single-query dense retrieval can yield different candidate lists.
- **Throughput path** (the evaluated run): BM25 loop, batched dense query encoding (batch 64),
  batched exact search, fusion, then one rerank call per query. `PhaseTimer` synchronizes CUDA at
  both ends of every phase and reads each phase's CUDA peak counters before the next phase can
  reset them.
- **Timed single-query path**: the whole pipeline for one query (BM25, one-query encoding,
  exact search, fusion, rerank), 10 untimed warm-up queries, every query timed once with CUDA
  synchronized; mean/p50/p90/p99/min/max overall and per phase. One outer probe reads the peak
  allocated/reserved counters right after the pass. Rankings from this path are compared with
  the evaluated run (`batch_vs_single_query`); dense encoding in a batch can differ from the
  one-query encoding by ~2e-7 (D-022), so a few rankings may differ.
- **Manifest** records: parent configs and hashes, parent index manifests (BM25 file hashes, dense
  cache hashes, model hashes), fusion settings, `fused_candidates_sha256` (identical for the two
  arms of a view), parent-list statistics (empty/short BM25 lists), encoder and reranker
  descriptions, throughput and latency blocks, input hashes.
- GPU memory figures include the resident weights of both models and exclude the CUDA context.

## Metric conventions

- Rank order comes from the run file, not from scores.
- Relevant = grade ≥ 1. nDCG: gain = grade, discount 1/log2(rank+1), ideal from all judged
  grades ≥ 1 (trec_eval `ndcg_cut` semantics).
- The evaluated query set is the qrels query set. A qrels query missing from the run scores 0
  and counts in the mean. A run query not in the qrels is an error.
- Every evaluation is cross-checked per query against pytrec_eval (tolerance 1e-9). trec_eval
  breaks ties by its own rule, so the run is passed with synthetic strictly decreasing scores
  derived from our rank order. MRR@k uses trec_eval `recip_rank` on the run truncated to k.

## Windows note

Python 3.11 decodes `.pth` files with the locale codec, which breaks a plain editable install
when the repository path contains non-ASCII characters (for example a Windows profile or folder name with accented letters). The project uses hatchling's
`dev-mode-exact` editable mode (import hook, needs the `editables` package) instead.
