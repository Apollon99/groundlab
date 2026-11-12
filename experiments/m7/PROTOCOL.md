# GroundLab M7 protocol

M7 is a separate descriptive local generation study. It starts after the retrieval
study and its aggregate results are known. The M6 protocol, code tree, and runs
stay frozen. The goal is to see whether an evidence-bearing claim is classified
correctly and whether the model cites the annotated document for that stance.

- Select 60 claims from the 188 SciFact evidence-bearing held-out claims. Within each
  gold stance, select the first 30 by SHA-256 of UTF-8 `groundlab-v1|<query_id>` with
  string query ID as the tie-break. Freeze the ordered sample hash before reading
  per-query rankings. Preserve the IDs only in local artifacts.
- Four arms use the same claims. Closed book supplies no abstracts. BM25 and RRF
  plus MiniLM supply their saved top three abstracts. Oracle supplies up to three
  annotated evidence abstracts ordered by string document ID. Missing hits stay
  missing. Documents use their canonical prepared titles and abstracts.
- Use the pinned Qwen2.5-1.5B-Instruct Q4_K_M GGUF and llama.cpp b12308 Windows
  CPU binary. The backend runs offline on loopback with eight threads, a 4096-token
  context and one request at a time. Use the model chat template, the same system
  prompt, greedy decoding, seed 20261111 and at most 128 generated tokens.
- Include each document ID and title in the 768-token budget. Preserve document
  order and the title. Trim the abstract at a token boundary when needed. Check
  the complete templated prompt plus output budget. Record presented IDs, token
  counts, stops, truncation, response validation and timing.
- Request schema-constrained JSON with a stance and up to three citations. Validate
  unique presented document IDs and the stance/citation rules. Grounded correctness
  requires a correct stance and at least one citation, all of which must be annotated
  for that stance. Record invalid and timed-out outputs as failures. Citation precision
  uses attempted citation occurrences from parsed output, including invalid attempts.
- Report accuracy, abstention, invalid/timeout counts and paired outcomes. Context
  arms also report evidence coverage, citation precision, grounded accuracy, and
  complete annotated rationale visibility. Visibility uses whitespace-normalized
  sentence matching against the displayed abstract. These are document-level proxies.
- Use Ai2 training claims and synthetic examples for smoke tests before freeze.
  The CPU gate extrapolates the slowest measured train-only trial over 240 trials.
  The held-out run has a two-hour execution budget and resumes only from verified
  trial records. Keep item-level data and model assets in ignored local paths.

The exact artifact versions and SHA-256 values are in `pins.json`. Freeze binds
the code, prompt, data, executable and DLLs, sample and saved ranking hashes.
Model and runtime sources are the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
and the [llama.cpp b12308 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12308).
