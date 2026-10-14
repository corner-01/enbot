param(
    [Parameter(Mandatory = $true)]
    [string]$RetrievalApiUrl,

    [Parameter(Mandatory = $true)]
    [string]$RetrievalApiKey,

    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [ValidateSet("ollama", "openai")]
    [string]$LlmProvider = "ollama",
    [string]$OllamaModel = "enbot-v9:7b"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

$RetrievalApiUrl = $RetrievalApiUrl.TrimEnd("/")
if (-not [Uri]::IsWellFormedUriString($RetrievalApiUrl, [UriKind]::Absolute)) {
    throw "RetrievalApiUrl must be an absolute URL, for example http://192.168.1.120:8100"
}
if ([string]::IsNullOrWhiteSpace($RetrievalApiKey)) {
    throw "RetrievalApiKey must not be empty"
}
if (($RetrievalApiKey.ToCharArray() | Where-Object { [int]$_ -gt 127 }).Count -gt 0) {
    throw "RetrievalApiKey must contain ASCII characters only; replace placeholder text with the real key"
}

$headers = @{ "X-API-Key" = $RetrievalApiKey }
Write-Host "Checking retrieval GPU node: $RetrievalApiUrl"
Invoke-RestMethod -Uri "$RetrievalApiUrl/health/ready" -Headers $headers -TimeoutSec 15 | Out-Host

if ($LlmProvider -eq "ollama") {
    Write-Host "Checking local Ollama..."
    $ollama = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
    $modelNames = @($ollama.models | ForEach-Object { $_.name })
    if ($modelNames -notcontains $OllamaModel) {
        Write-Warning "Ollama is reachable, but $OllamaModel was not found."
    }
} else {
    Write-Host "Using OpenAI provider; API key and model are read from .env."
}

$env:LLM_PROVIDER = $LlmProvider
$env:RETRIEVAL_MODE = "remote"
$env:RETRIEVAL_API_URL = $RetrievalApiUrl
$env:RETRIEVAL_API_KEY = $RetrievalApiKey
$env:RETRIEVAL_TIMEOUT_SECONDS = "8"
$env:RETRIEVAL_LOCAL_FALLBACK = "false"
$env:OLLAMA_NUM_GPU = ""
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = $OllamaModel
$env:ENBOT_EAGER_LOAD = "true"

Write-Host ("Starting EnBot primary node on http://{0}:{1}" -f $HostAddress, $Port)
& $Python -m uvicorn app.service_api:app --host $HostAddress --port $Port --workers 1
