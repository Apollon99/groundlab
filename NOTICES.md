# Third-party data and model notices

GroundLab's own code and documentation are MIT-licensed (see `LICENSE`). That license
applies only to files in this repository written for GroundLab. It does not cover, and grants
no rights to, any dataset, annotation, abstract or model weight listed below.

GroundLab does not redistribute any dataset or model weights. `groundlab data prepare`
downloads archives from the sources below and verifies them against pinned SHA-256 values
(see `src/groundlab/data/specs.py`).

| Source | Used for | Dataset terms |
|---|---|---|
| Ai2 SciFact `data.tar.gz` (scifact.s3-us-west-2.amazonaws.com, `release/latest`) | Claims and evidence annotations | Claims and annotations: CC BY 4.0. Abstracts (`corpus.jsonl`): ODC-By 1.0 (from S2ORC). Code: Apache 2.0. |
| BEIR SciFact `scifact.zip` (public.ukp.informatik.tu-darmstadt.de) | Canonical corpus text, queries, BEIR qrels | BEIR code is Apache 2.0. BEIR disclaims dataset licensing; the SciFact copy inherits Ai2's terms above. |
| BEIR NFCorpus `nfcorpus.zip` (public.ukp.informatik.tu-darmstadt.de) | Secondary dataset | Checked after the M0 audit; see "NFCorpus terms" below. No dataset license is granted by BEIR or by this repository. |

## NFCorpus terms

The SciFact rows in the table above reflect the M0 audit. The M0 audit could not reach the
NFCorpus homepage, so NFCorpus was pending then. It was checked before publication
(2026-11-12) against the [official NFCorpus terms](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/#terms).
That page permits academic
use. For any other use of the included NutritionFacts.org data, consult NutritionFacts.org's
own terms and contact its author, Michael Greger. This repository does not state or imply any
blanket commercial-use or redistribution permission for NFCorpus.

Citation: Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler, "A Full-Text
Learning to Rank Dataset for Medical Information Retrieval", ECIR 2016
([PDF](https://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf)).

BEIR only provides a reformatted copy of NFCorpus (and SciFact). It does not itself grant any
rights in the underlying datasets. GroundLab keeps the source data unbundled: nothing is
committed or redistributed here, and users obtain it from the original sources and are
responsible for complying with the terms above.

Models are downloaded by `groundlab models fetch` from Hugging Face at a pinned revision into
`artifacts/models/` (git-ignored) and verified per file by SHA-256 (`src/groundlab/models.py`).
Licenses below are the `license` field of each model's Hugging Face card at the pinned revision.

| Model | Revision | Used for | License |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | Dense retrieval (M2) | MIT |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | Reranking (M3); tokenizer files and `model.safetensors` (SHA-256 pinned) | Apache-2.0 |

Known unexamined risk: these pretrained models may have seen SciFact/BEIR data during training.
This has not been checked.

Python dependencies and their versions are pinned in `uv.lock`.

## M7 local generator

The [Qwen2.5-1.5B-Instruct-GGUF checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9)
used in M7 is Apache-2.0. The [llama.cpp b12308 runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b12308)
is distributed under the llama.cpp MIT license with notices in its archive. Both
stay under ignored local artifacts. The M7 freeze records their versions and hashes.
