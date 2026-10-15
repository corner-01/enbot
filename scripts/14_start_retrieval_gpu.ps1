param(
    [Parameter(Mandatory = $true)]
    [string]$ApiKey,

    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8100,
    [int]$CudaDevice = 1
)
# CudaDevice 指定 BEG 执行位置, 默认 1: GPU, 0: CPU

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
#$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

#if (-not (Test-Path -LiteralPath $Python)) {
#    throw "Python environment not found: $Python"
#}

$requiredPaths = @(
    (Join-Path $ProjectRoot "models\Qwen3-8B-native"),
    (Join-Path $ProjectRoot "models\bge-reranker-v2-m3"),
    (Join-Path $ProjectRoot "knowledge_data")
)
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required retrieval asset not found: $path"
    }
}

$env:RETRIEVAL_MODE = "local"
$env:RETRIEVAL_API_KEY = $ApiKey
$env:RETRIEVAL_EAGER_LOAD = "true"
$env:RETRIEVAL_EAGER_WARMUP = "true"
$env:RETRIEVAL_MAX_CONCURRENCY = "1"

$env:CUDA_VISIBLE_DEVICES = "$CudaDevice"

$env:EMBEDDING_DEVICE = "cuda"
$env:EMBEDDING_MODEL_PATH = ".\models\Qwen3-8B-native"
$env:EMBEDDING_BATCH_SIZE = "4"

$env:RERANKER_DEVICE = "cuda"
$env:RERANKER_MODEL_PATH = ".\models\bge-reranker-v2-m3"
$env:RERANKER_ENABLED = "true"

$env:KNOWLEDGE_FILE = ".\knowledge_data"
$env:MILVUS_URI = "http://127.0.0.1:19530"
$env:MILVUS_COLLECTION = "enbot_knowledge_v1"

Write-Host "Checking CUDA..."
& python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"

Write-Host "Starting Milvus..."
& (Join-Path $PSScriptRoot "09_start_milvus.ps1")

Write-Host ("Starting retrieval GPU node on http://{0}:{1}" -f $HostAddress, $Port)
& python -m uvicorn services.retrieval.app:app --host $HostAddress --port $Port --workers 1

