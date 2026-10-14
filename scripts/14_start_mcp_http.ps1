param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8020
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv-mcp\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到虚拟环境 Python：$Python"
}

Set-Location -LiteralPath $ProjectRoot
& $Python -m integrations.mcp_server `
    --transport streamable-http `
    --host $HostAddress `
    --port $Port
