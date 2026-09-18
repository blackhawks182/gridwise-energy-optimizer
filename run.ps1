$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path "gridwise\.env")) {
    Write-Host "Missing gridwise\.env. Copy gridwise\.env.example to gridwise\.env and add at least one LLM provider key." -ForegroundColor Yellow
    exit 1
}

python -m gridwise
