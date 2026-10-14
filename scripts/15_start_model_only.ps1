param(
    [string]$OllamaModel = "enbot-dpo-v1-lab",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8002
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

Write-Host "Checking local Ollama..."
$ollama = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
$modelNames = @($ollama.models | ForEach-Object { $_.name })
if ($modelNames -notcontains $OllamaModel -and $modelNames -notcontains "$OllamaModel`:latest") {
    throw "Ollama model was not found: $OllamaModel"
}

# 纯模型模式：不初始化知识库、不连接 Milvus、不调用 Agent 工具。
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = $OllamaModel
$env:OLLAMA_NUM_GPU = ""
$env:ENBOT_MODEL_ONLY = "true"
$env:ENBOT_DISABLE_RAG = "true"
$env:ENBOT_REQUIRE_MILVUS = "false"
$env:ENBOT_EAGER_LOAD = "true"

Write-Host ("Starting EnBot model-only service on http://{0}:{1} with {2}" -f $HostAddress, $Port, $OllamaModel)
& $Python -m uvicorn app.service_api:app --host $HostAddress --port $Port --workers 1