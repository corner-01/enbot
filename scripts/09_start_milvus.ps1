param(
    [string]$MilvusRoot = $env:ENBOT_MILVUS_HOME
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($MilvusRoot)) {
    $MilvusRoot = "D:\enbot-milvus"
}

$launcherFile = Join-Path $MilvusRoot "standalone_embed.bat"
$launcherUrl = "https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat"

New-Item -ItemType Directory -Path $MilvusRoot -Force | Out-Null

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not running. Start it first, then run this script again."
}

if (-not (Test-Path -LiteralPath $launcherFile)) {
    Write-Host "Downloading the official Milvus Standalone launcher..."
    Invoke-WebRequest -Uri $launcherUrl -OutFile $launcherFile
}

Push-Location $MilvusRoot
try {
    & cmd.exe /c "standalone_embed.bat start"
    if ($LASTEXITCODE -ne 0) {
        throw "Milvus container failed to start. Run 'docker logs milvus-standalone' for details."
    }
}
finally {
    Pop-Location
}

docker ps --filter "name=milvus-standalone"

Write-Host "Milvus endpoint: http://127.0.0.1:19530"
Write-Host "Milvus WebUI:    http://127.0.0.1:9091/webui/"
Write-Host "Data directory:  $MilvusRoot\volumes"
