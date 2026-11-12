# Run the test suite, lint and type checks. Stops at the first failure.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

& .\.venv\Scripts\python.exe -m pytest
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
& .\.venv\Scripts\ruff.exe check src tests
if ($LASTEXITCODE -ne 0) { throw "ruff check failed" }
& .\.venv\Scripts\ruff.exe format --check src tests
if ($LASTEXITCODE -ne 0) { throw "ruff format check failed" }
& .\.venv\Scripts\mypy.exe
if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
Write-Host "All checks passed."
