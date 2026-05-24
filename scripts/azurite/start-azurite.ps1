param(
    [int]$WaitSeconds = 30,
    [string]$BlobEndpoint = "http://127.0.0.1:10010/devstoreaccount1?comp=list"
)

$ErrorActionPreference = "Stop"

$ConnectionString = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10010/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10011/devstoreaccount1;"

function Test-DockerAvailable {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    docker version --format '{{.Server.Version}}' 2>$null | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    return ($exitCode -eq 0)
}

Write-Host ">>> Checking Docker availability..."
if (-not (Test-DockerAvailable)) {
    throw "Docker is not available. Install Docker Desktop and ensure the engine is running, then retry."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ">>> Starting Azurite via docker compose..."
docker compose -f docker-compose.azurite.yml up -d
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up failed (exit $LASTEXITCODE)."
}

Write-Host ">>> Waiting up to $WaitSeconds`s for Azurite blob endpoint..."
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri $BlobEndpoint -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 750
    }
}

if (-not $ready) {
    Write-Warning "Azurite did not respond within $WaitSeconds seconds. Check 'docker logs gtog-azurite'."
} else {
    Write-Host ">>> Azurite is ready on ports 10000 (blob), 10001 (queue), 10002 (table)."
}

Write-Host ""
Write-Host "Connection string (well-known Azurite credentials):"
Write-Host $ConnectionString
