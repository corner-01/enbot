$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
$bridge = Join-Path $projectRoot "integrations\feishu_bot.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}
if (-not (Test-Path -LiteralPath $bridge)) {
    throw "Feishu bridge not found: $bridge"
}

# Bypass EnBot /chat and send Feishu messages directly to Ollama.
$env:FEISHU_BACKEND = "ollama"
$env:FEISHU_OLLAMA_MODEL = "enbot-optuna:7b"

Set-Location -LiteralPath $projectRoot
& $python $bridge
exit $LASTEXITCODE
