param(
    [string]$DestinationRoot = "D:\WSL"
)

$ErrorActionPreference = "Stop"

function Move-WslDistribution {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $destination = [System.IO.Path]::GetFullPath((Join-Path $DestinationRoot $Name))
    $expectedRoot = [System.IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\') + '\'
    if (-not $destination.StartsWith($expectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected WSL destination: $destination"
    }
    if (Test-Path -LiteralPath $destination) {
        throw "Destination already exists: $destination"
    }

    Write-Host "Moving $Name to $destination ..."
    & wsl.exe --manage $Name --move $destination
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to move $Name (exit code $LASTEXITCODE)."
    }
}

Write-Host "Close every VS Code window and quit Docker Desktop before running this script."
New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

& wsl.exe --shutdown
if ($LASTEXITCODE -ne 0) {
    throw "WSL shutdown failed. Restart Windows, keep VS Code and Docker closed, then run again."
}

Move-WslDistribution -Name "Ubuntu"
Move-WslDistribution -Name "Ubuntu-22.04"

Write-Host "Verifying distributions ..."
& wsl.exe --list --verbose
& wsl.exe -d Ubuntu -- bash -lc "test -d /home/yft/projects/enbot-finetune && echo ENBOT_FINETUNE_OK"

Write-Host "WSL migration completed. Both distributions now store their virtual disks under $DestinationRoot."
