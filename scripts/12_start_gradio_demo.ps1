param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
$tempDir = Join-Path $projectRoot "tmp\gradio"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
$env:TEMP = $tempDir
$env:TMP = $tempDir
$env:GRADIO_TEMP_DIR = $tempDir
$env:GRADIO_SERVER_NAME = $HostAddress
$env:GRADIO_SERVER_PORT = "$Port"

& $python -c "import gradio" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Gradio is not installed. Run: .\venv\Scripts\python.exe -m pip install -r .\demo\requirements.txt"
}

Set-Location -LiteralPath $projectRoot
Write-Host "Starting EnBot Gradio demo..."
Write-Host "URL: http://${HostAddress}:${Port}"
Write-Host "Temporary files: $tempDir"
& $python -m demo.gradio_app
exit $LASTEXITCODE
