$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/4] Python" -ForegroundColor Cyan
python --version

Write-Host "[2/4] Compile" -ForegroundColor Cyan
python -m compileall -q gridwise

Write-Host "[3/4] Tests" -ForegroundColor Cyan
python -m pytest -q

Write-Host "[4/4] Import" -ForegroundColor Cyan
python -c "import gridwise.main; print('GRIDWISE IMPORT OK')"

Write-Host "" 
Write-Host "Core verification complete." -ForegroundColor Green
Write-Host "For the 10 official public cases, set GRIDWISE_PUBLIC_CASES and rerun: python -m pytest -q" -ForegroundColor Yellow
Write-Host "For a live LLM paraphrase check, set GRIDWISE_LIVE_LLM=1 and run: python -m pytest -q gridwise/tests/test_live_llm.py" -ForegroundColor Yellow
