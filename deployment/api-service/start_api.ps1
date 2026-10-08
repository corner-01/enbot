$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Test-Path -LiteralPath ".env")) {
    throw "Missing $projectRoot\.env"
}

$python = Join-Path $projectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "No Windows virtual-environment Python was found"
}

try {
    Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null
} catch {
    throw "Ollama is not reachable on 127.0.0.1:11434"
}

& $python -X utf8 main.py
