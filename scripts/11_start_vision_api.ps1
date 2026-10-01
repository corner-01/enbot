param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8011
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "services\vision\.venv\Scripts\python.exe"
$uploadDir = Join-Path $projectRoot "tmp\vision-uploads"
$preprocessedDir = Join-Path $projectRoot "tmp\vision-preprocessed"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Vision virtual environment not found: $python"
}

New-Item -ItemType Directory -Path $uploadDir -Force | Out-Null
New-Item -ItemType Directory -Path $preprocessedDir -Force | Out-Null

# Keep multipart spool files and all service temporary files on D:.
$env:TEMP = $uploadDir
$env:TMP = $uploadDir
$env:VISION_UPLOAD_DIR = $uploadDir
if ([string]::IsNullOrWhiteSpace($env:PADDLE_PDX_CACHE_HOME)) {
    $env:PADDLE_PDX_CACHE_HOME = "D:\AI-Cache\paddle"
}
if ([string]::IsNullOrWhiteSpace($env:PADDLE_PDX_MODEL_SOURCE)) {
    $env:PADDLE_PDX_MODEL_SOURCE = "BOS"
}
if ([string]::IsNullOrWhiteSpace($env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK)) {
    $env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
}

Set-Location -LiteralPath $projectRoot
Write-Host "Starting EnBot Vision API..."
Write-Host "API docs: http://${HostAddress}:${Port}/docs"
Write-Host "Health:   http://${HostAddress}:${Port}/health/ready"
Write-Host "Uploads:  $uploadDir (request files are deleted after processing)"

& $python -m uvicorn services.vision.app:app `
    --host $HostAddress `
    --port $Port `
    --workers 1
exit $LASTEXITCODE
