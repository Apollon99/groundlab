# GroundLab Milestone 0: Read-only dataset audit

Date: 2026-11-12 (UTC). Run in Claude's Linux cloud sandbox, Python 3.13.18, standard library only.
Scope: the project owner's authorization of 2026-11-12 (download, hash, inspect, map, count, check integrity, measure overlap). No repo, no project stack, no retrieval code, no experiments.
Every number below was computed in this run unless it is marked **not verified**. The original raw output (`audit_output.json`) and script (`audit.py`) were local-only and are not included in this public snapshot.

## 1. Sources, hashes, licenses

| Archive | URL | HTTP Last-Modified | Size (bytes) | SHA-256 (archive) | MD5 (archive) |
|---|---|---|---|---|---|
| BEIR SciFact | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip | 2021-04-20 14:21:44 GMT | 2,816,079 | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` | `5f7d1de60b170fc8027bb7898e2efca1` |
| BEIR NFCorpus | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip | 2021-01-15 13:32:50 GMT | 2,448,432 | `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b` | `a89dba18a62ef92f7d323ec890a0d38d` |
| Ai2 SciFact | https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz | 2021-01-26 02:28:59 GMT | 3,115,079 | `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be` | `cb7da4d8609e30f2c7483b61aa447f7e` (= S3 ETag) |

- All three SHA-256 values match the planning document.
- Both BEIR MD5s match the md5 column of the BEIR README (beir-cellar/beir, main branch).
- Versioning: none of the archives carries a version string. Ai2's URL is `release/latest`, so its content can change without notice. The SHA-256 plus Last-Modified is the only pin. BEIR zip entry dates are 2021-03-19 (SciFact) and 2021-01-13 (NFCorpus).
- Ai2 tarball also contains macOS AppleDouble files (`._data`, `cross_validation/._fold_1`, `._fold_4`) and a 5-fold `cross_validation/` directory. Both are harmless and should be ignored by the loader.

Extracted-file SHA-256 (key files):

| File | SHA-256 |
|---|---|
| beir scifact/corpus.jsonl | `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6` |
| beir scifact/queries.jsonl | `8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313` |
| beir scifact/qrels/test.tsv | `0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c` |
| beir scifact/qrels/train.tsv | `a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d` |
| ai2 data/corpus.jsonl | `b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62` |
| ai2 data/claims_train.jsonl | `f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0` |
| ai2 data/claims_dev.jsonl | `86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217` |
| ai2 data/claims_test.jsonl | `558930d75215c73f84a28fe538307d6d397c9de1ec7239514cf45f80d75d2ca3` |
| beir nfcorpus/corpus.jsonl | `10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f` |
| beir nfcorpus/queries.jsonl | `d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a` |
| beir nfcorpus/qrels/test.tsv | `f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a` |
| beir nfcorpus/qrels/dev.tsv | `b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8` |
| beir nfcorpus/qrels/train.tsv | `6336b80f9bffc4f063f3aa450047ad35c0b7c534efe4a6ba35e16dbace047f6a` |

Licenses:
- None of the three archives contains a license file.
- Ai2 SciFact repo `LICENSE.md`: claims and evidence annotations (`claims_*.jsonl`) are CC BY 4.0. Abstracts (`corpus.jsonl`) come from S2ORC and are ODC-By 1.0. Code is Apache 2.0.
- BEIR repo `LICENSE` is Apache 2.0, which covers the code. The BEIR README disclaims dataset licensing ("we do not ... claim that you have license to use the dataset"). So the BEIR SciFact copy inherits Ai2's terms. No conflict found, but BEIR redistributes without any license notice.
- NFCorpus: **not verified**. The NFCorpus homepage (www.cl.uni-heidelberg.de) is blocked by this sandbox's network policy, and I did not request access to it. The license must be checked from the homepage before the README cites it.
  - *Prepublication clarification (2026-11-12):* the open item above was pending at the time of this audit. The official NFCorpus terms have since been checked and are recorded in `NOTICES.md` (Heidelberg terms page: academic use permitted; other uses of the included NutritionFacts.org data require consulting its terms and its author). This is no longer an open release item.

## 2. Ai2 to BEIR mapping

- BEIR `test` (300 qids) = Ai2 `claims_dev` (300). IDs match 300/300 and claim text matches exactly 300/300. Overlap with Ai2 train is 0.
- BEIR `train` (809 qids) = Ai2 `claims_train` (809). IDs match 809/809 and text matches exactly 809/809.
- BEIR `queries.jsonl` has 1,109 rows, which are exactly Ai2 train + dev. None of the 300 Ai2 `claims_test` IDs appear in BEIR.
- BEIR qrels equal the Ai2 `cited_doc_ids` set for every claim: 300/300 in test and 809/809 in train. They equal the evidence set for only 175/300 and 480/809. All BEIR scores are 1.
- **Leakage hazard:** BEIR `queries.jsonl` `metadata` is a copy of the Ai2 evidence map, including labels and sentence indices. It is identical to Ai2 for 1,109/1,109 queries and non-empty for 693. The loader must drop `metadata` before anything reaches a retriever.
- Corpus: 5,183 docs in both, with all IDs matching. Titles match exactly 5,183/5,183. For the text, 4,128 BEIR documents equal `" ".join(abstract)` exactly. The other 1,055 are equal once whitespace is ignored but differ in whitespace. That count is close to Ai2's 977 `structured=True` abstracts, but I did not test that link.
- Ai2 test labels are not public. `claims_test.jsonl` has only `id` and `claim`. The Ai2 README says test gold labels are not released and test evaluation goes through the leaderboard.

## 3. Primary view (Ai2 dev = BEIR test)

Rule: a doc is relevant to a claim iff the claim's Ai2 evidence map has that doc with at least one non-empty evidence set. Stance is not used, duplicates are collapsed, and unjudged docs count as non-relevant.

- 188 evidence-bearing claims and 209 positive (claim, doc) pairs.
- 0 duplicate pairs were collapsed. I also parsed the files with a duplicate-JSON-key check and found none. The Ai2 files have no empty evidence sets.
- Pairs per claim: 1 pair for 178 claims, 2 for 4, 3 for 2, 4 for 3, and 5 for 1. So 10 claims have more than one evidence doc.
- 83 claims have more than one evidence set in total.
- Stance: 124 claims are SUPPORT only and 64 are CONTRADICT only. No claim has both, and no (claim, doc) pair mixes labels. By pair: 138 SUPPORT and 71 CONTRADICT.
- 112 of the 300 claims have no evidence.
- All 209 primary pairs are in the BEIR test qrels. 130 of the 339 BEIR positives have no evidence: 113 on 112 no-evidence claims, and 17 on 13 evidence-bearing claims.
- One Ai2 dev claim lists a cited doc twice (individual identifiers omitted from this public copy). It has no evidence, so the primary view is unaffected, and BEIR stores it once.

## 4. Clusters (188-claim cohort)

Connected components of claims that share a doc ID:

| Linking IDs | Clusters | Size distribution | Max | Singletons | Effective count (Σn)²/Σn² |
|---|---|---|---|---|---|
| Cited + evidence (Ai2) | 164 | 1:144, 2:17, 3:2, 4:1 | 4 | 144 | 143.67 |
| Evidence only | 164 | identical | 4 | 144 | 143.67 |
| Cited only (BEIR qrels) | 164 | identical | 4 | 144 | 143.67 |
| (all 300 claims, cited + evidence) | 247 | 1:205, 2:32, 3:9, 4:1 | 4 | 205 | 209.3 |

- On this cohort the cluster definition does not matter: all three linkings give the same partition.
- 18 of the 20 multi-claim clusters mix SUPPORT and CONTRADICT claims. These are claim/negation pairs written from the same paper, which confirms that clustering is needed.
- Within dev, 11 near-duplicate claim pairs (defined in §5) fall in the cohort, and all 11 are already in the same cluster.

## 5. Train/test overlap (Ai2 train vs Ai2 dev)

- Docs cited or used as evidence: 565 unique in Ai2 train and 283 in dev. 181 are shared.
- The primary view has 182 unique positive docs. Of these, 119 appear in Ai2 train in some role, and 110 are Ai2 train evidence docs.
- 135 of the 209 primary pairs have a doc that appears in train.
- 123 of the 188 primary claims have at least one positive doc in train (112/188 counting only train evidence docs).
- Across all 300 dev claims, 196 have a cited doc that appears in train.
- Near-duplicate claim texts. Method: Jaccard ≥ 0.6 prefilter, then difflib ratio ≥ 0.8 on lowercased alphanumeric text.

  | Comparison | Exact duplicates (normalized) | Near-duplicate pairs | Claims affected |
  |---|---|---|---|
  | dev vs train | 2 | 123 | 108 of 300 dev claims |
  | Ai2 test vs train | 1 | 123 | 109 of 300 test claims |
  | Ai2 test vs dev | 0 | 44 | 40 of 300 test claims |
  | primary cohort vs train | not computed | not computed | 72 of 188 cohort claims |

  Most hits are deliberate negations or number swaps (example claim texts omitted from this public copy; only the aggregate counts and method are retained).
- Implication: anything fitted or tuned on Ai2 train ("dev" in our protocol) sees many of the same documents and near-paraphrased claims as the confirmatory set. This is harmless only while no component is trained or tuned on Ai2 train labels in a way that can memorize documents.

## 6. Integrity (SciFact and NFCorpus)

- SciFact checks, all clean. No duplicate qrels rows (339 test and 919 train rows, all unique). No qrels IDs are missing from the corpus or the queries. No empty titles or texts. No cited or evidence IDs are missing from the corpus. No evidence sentence index is out of range. No ID overlaps between Ai2 splits, and no qid overlap between BEIR test and train.
- NFCorpus basics:
  - 3,633 docs, all unique, with no empty text or title.
  - 3,237 queries, which cover all splits.
  - Test: 323 qids and 12,334 qrels rows, all unique. Grades: 1 × 11,758 and 2 × 576, with no grade 0. 119 test queries have at least one grade-2 doc and 204 have only grade 1. Relevant docs per query: min 1, median 16, max 475.
  - Dev: 324 qids and 11,385 rows, with grades 1 and 2.
  - Train: 2,590 qids and 110,575 rows, all grade 1.
  - No query or doc IDs are missing, and test shares no qids with train or dev.

## 7. Discrepancies against the planning document

Every count claim was reproduced exactly, and all three hashes match. Nothing here needs a change to the hypotheses. The following must be written into the protocol:
1. Drop BEIR `queries.jsonl` `metadata` in the loader, and test that it was dropped. Otherwise the gold labels leak.
2. Fix which corpus text is canonical. My recommendation is BEIR `corpus.jsonl` title + text, the same for every arm. Record that 1,055 docs differ from Ai2 in whitespace only.
3. Pin the Ai2 archive by SHA-256, because the URL is `release/latest`.
4. Clustering: state that cited+evidence linkage is used and that it is identical to evidence-only on this cohort.
5. State that Ai2 train shares 110 of the 182 positive docs and near-paraphrased claims with the confirmatory set. Any use of Ai2 train is limited to fixed, non-learned settings: GPU batch profiling and smoke tests, with no tuning of k, depth or models against train qrels.
6. NFCorpus license was not yet verified at the time of this audit (since resolved; see the clarification in section 1 and `NOTICES.md`).

## 8. Suitability for the planned confirmatory analysis

The data is usable, with caveats:
- Sample size: 188 claims in 164 clusters. The effective count is about 144, and 144 clusters are singletons, so the cluster-level unit is nearly the claim. That gives plenty of sign-flip permutations. However, power to detect a 0.02 difference in nDCG@10 is likely limited, because 178 of 188 claims have exactly one relevant doc. Per-claim scores are then a step function of one rank, and many paired differences will be exactly 0. I did not compute power because that needs run scores. A simulation on synthetic fixtures before the freeze is the right check.
- Margin test: shifting differences by 0.02 and sign-flipping assumes the cluster differences are symmetric about 0.02 under the null. With bounded, discrete, zero-inflated differences that is not plausible by default. I recommend keeping the margin test exploratory unless the synthetic simulation shows correct size. One alternative is a cluster-bootstrap lower bound that exceeds 0.02, also checked by simulation.
- Label incompleteness:
  - 112 dev claims have no evidence and are excluded.
  - 17 cited-but-no-rationale docs on cohort claims count as non-relevant, so a system that retrieves them is penalized.
  - Anything else Ai2 did not cite is unjudged. Absolute scores therefore understate retrieval quality. Paired contrasts are less affected, but the BEIR-qrels secondary view remains a needed sensitivity check.
- Pretrained-model contamination (bge-small, MiniLM-L6 CE) with SciFact/BEIR is **not verified**. The README should flag it as an unexamined risk.

## Commands run

Historical record of the original audit. Files referenced here (`audit.py`, `audit_output.json`, the scratch directory and downloaded archives) were local-only and are not part of this snapshot.

```
S=<scratchpad>; mkdir -p $S/dl_scifact $S/dl_nfcorpus $S/dl_ai2 $S/dl_docs $S/scripts
curl -sS -L -o $S/dl_ai2/data.tar.gz -D $S/dl_ai2/headers.txt https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
curl -sS -L -o $S/dl_scifact/scifact.zip -D $S/dl_scifact/headers.txt https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip     # first attempt: 403 from sandbox egress policy; succeeded after host was allowlisted
curl -sS -L -o $S/dl_nfcorpus/nfcorpus.zip -D $S/dl_nfcorpus/headers.txt https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip # same
sha256sum $S/dl_*/*.zip $S/dl_*/*.tar.gz; md5sum $S/dl_*/*.zip $S/dl_*/*.tar.gz
tar -tzvf $S/dl_ai2/data.tar.gz; mkdir $S/x_ai2; tar -xzf $S/dl_ai2/data.tar.gz -C $S/x_ai2 --no-same-owner
unzip -l <each zip>; mkdir $S/x_scifact $S/x_nfcorpus; unzip -q <zip> -d <its own dir>
find $S/x_* -type f | sort | xargs sha256sum
curl -sS -o $S/dl_docs/{beir_README.md,beir_LICENSE,scifact_README.md,scifact_LICENSE} https://raw.githubusercontent.com/{beir-cellar/beir/main,allenai/scifact/master}/...
python3 -I $S/scripts/audit.py $S/x_ai2/data $S/x_scifact/scifact $S/x_nfcorpus/nfcorpus > audit_output.json
python3 -I -c '<duplicate-JSON-key and duplicate cited_doc_ids check>' $S/x_ai2/data
curl https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/   # 403, sandbox policy; not pursued
```
HuggingFace mirrors (BeIR/*) were also blocked and were not used. No substitute sources were used.
