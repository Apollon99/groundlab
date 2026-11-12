# Create the locked environment and record environment metadata.
# Usage: .\scripts\setup.ps1 [-RequireCuda]
param([switch]$RequireCuda)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install it from https://docs.astral.sh/uv/ and re-run."
}
uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw "uv python install failed" }
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw "uv sync --locked failed (lockfile out of date or install error)" }

$envArgs = @("env", "check", "--out", "outputs\env_check.json")
if ($RequireCuda) { $envArgs += "--require-cuda" }
& .\.venv\Scripts\groundlab.exe @envArgs | Out-Null
if ($LASTEXITCODE -ne 0) { throw "groundlab env check failed with exit code $LASTEXITCODE" }

# Pinned model files (downloaded at the pinned revision, verified by SHA-256).
foreach ($m in "bge-small-en-v1.5", "ms-marco-MiniLM-L6-v2") {
    & .\.venv\Scripts\groundlab.exe models fetch --model $m | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "groundlab models fetch --model $m failed with exit code $LASTEXITCODE" }
}
Write-Host "Environment ready. Metadata written to outputs\env_check.json; models in artifacts\models"
