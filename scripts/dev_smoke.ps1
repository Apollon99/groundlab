# Dev-split smoke run of every config: prepare -> build -> run -> evaluate.
# Dev split only. These numbers are smoke checks, not results.
# Dense configs need CUDA and the pinned models (verified here, never silently skipped).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$g = ".\.venv\Scripts\groundlab.exe"

function Invoke-Step([string[]]$StepArgs) {
    # Library warnings on stderr must not abort the script; the exit code decides.
    $ErrorActionPreference = "Continue"
    & $g @StepArgs
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "groundlab $($StepArgs -join ' ') failed with exit code $code" }
}

foreach ($ds in "scifact", "nfcorpus") {
    Invoke-Step @("data", "prepare", "--dataset", $ds, "--data-dir", "data")
}
Invoke-Step @("models", "fetch", "--model", "bge-small-en-v1.5", "--verify-only")
Invoke-Step @("models", "fetch", "--model", "ms-marco-MiniLM-L6-v2", "--verify-only")
# Order matters: hybrid/rerank `build` verifies the BM25 and dense indexes built just before.
$configs = @(
    "bm25_scifact_evidence", "bm25_scifact_beir", "bm25_nfcorpus",
    "dense_scifact_evidence", "dense_scifact_beir", "dense_nfcorpus",
    "hybrid_scifact_evidence", "hybrid_scifact_beir", "hybrid_nfcorpus",
    "rerank_scifact_evidence", "rerank_scifact_beir", "rerank_nfcorpus"
)
foreach ($cfg in $configs) {
    $c = "configs\$cfg.toml"
    Invoke-Step @("build", "--config", $c)
    Invoke-Step @("run", "--config", $c, "--split", "dev")
    Invoke-Step @("evaluate", "--config", $c, "--split", "dev")
}
