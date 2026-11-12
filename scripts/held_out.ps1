# Run the frozen held-out study, then generate its offline report.
# Completed rankings are reused on retry and validated by evaluate.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$g = ".\.venv\Scripts\groundlab.exe"

function Invoke-Step([string[]]$StepArgs) {
    & $g @StepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "groundlab $($StepArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

foreach ($view in "scifact_evidence", "scifact_beir", "nfcorpus") {
    foreach ($arm in "bm25", "dense", "hybrid", "rerank") {
        $name = "$($arm)_$view"
        $config = "configs\$name.toml"
        $dir = "outputs\runs\$name\test"
        if (!(Test-Path -LiteralPath (Join-Path $dir "run_manifest.json"))) {
            if ($arm -in "bm25", "dense" -and !(Test-Path -LiteralPath "artifacts\indexes\$name\manifest.json")) {
                Invoke-Step @("build", "--config", $config)
            }
            Invoke-Step @("run", "--config", $config, "--split", "test")
        }
        Invoke-Step @("evaluate", "--config", $config, "--split", "test")
    }
}
Invoke-Step @("report")
