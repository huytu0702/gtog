param(
    [string]$ResourceGroup = "rg-gtog-prod",
    [string]$Subscription = "1095803e-80bf-47e0-961f-3d74cb4c605c",
    [string]$ApiAppName = "ca-gtog-api-prod",
    [string]$WorkerAppName = "ca-gtog-worker-prod",
    [string]$TunnelAppName = "ca-gtog-tunnel-prod",
    [string]$TunnelSecretRefName = "tunnel-token",
    [string]$AppPublicHostname = "",
    [string]$ApiPublicHostname = "",
    [string]$ApiHealthUrl = "",
    [string]$ProbeOriginUrls = "",
    [string]$OriginBypassWorkspace = "",
    [string]$OriginBypassLogQuery = ""
)

$ErrorActionPreference = "Stop"

if (-not $env:AZURE_CONFIG_DIR) {
    $env:AZURE_CONFIG_DIR = Join-Path (Get-Location) ".azure"
}
New-Item -ItemType Directory -Path $env:AZURE_CONFIG_DIR -Force | Out-Null

function Require-Value {
    param(
        [string]$Name,
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Name is required for this validation step."
    }
}

function Write-Phase3Check {
    param([string]$Message)
    Write-Host "[phase3] $Message"
}

Require-Value -Name "ResourceGroup" -Value $ResourceGroup
Require-Value -Name "ApiAppName" -Value $ApiAppName
Require-Value -Name "WorkerAppName" -Value $WorkerAppName
Require-Value -Name "TunnelAppName" -Value $TunnelAppName
Require-Value -Name "AppPublicHostname" -Value $AppPublicHostname
Require-Value -Name "ApiPublicHostname" -Value $ApiPublicHostname

if (-not $ApiHealthUrl) {
    $ApiHealthUrl = "https://$ApiPublicHostname/health"
}

Write-Phase3Check "Using subscription $Subscription"
az account set --subscription $Subscription --output none | Out-Null

Write-Phase3Check "Reading current API app, worker app, and tunnel app state"
$apiApp = az containerapp show --resource-group $ResourceGroup --name $ApiAppName --output json | ConvertFrom-Json
$workerApp = az containerapp show --resource-group $ResourceGroup --name $WorkerAppName --output json | ConvertFrom-Json
$tunnelApp = az containerapp show --resource-group $ResourceGroup --name $TunnelAppName --output json | ConvertFrom-Json

$expectedAppOrigin = "https://$AppPublicHostname"
$errors = [System.Collections.Generic.List[string]]::new()

$apiIngress = $apiApp.properties.configuration.ingress
if ($apiIngress.external -ne $false) {
    $errors.Add("API app ingress is not internal-only.")
}
if ($apiIngress.targetPort -ne 8000) {
    $errors.Add("API app target port is not 8000.")
}
$corsPolicy = $apiIngress.corsPolicy
$corsAllowedOrigins = @($corsPolicy.allowedOrigins) | Sort-Object
if (($corsAllowedOrigins -join "|") -ne $expectedAppOrigin) {
    $errors.Add("Ingress CORS allowed origins do not match the expected app origin.")
}
if ($corsPolicy.allowCredentials -ne $true) {
    $errors.Add("Ingress CORS allowCredentials is not enabled.")
}
$corsAllowedMethods = @($corsPolicy.allowedMethods) | Sort-Object
if (($corsAllowedMethods -join "|") -ne ((@("DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT") | Sort-Object) -join "|")) {
    $errors.Add("Ingress CORS allowed methods do not match the expected browser contract.")
}
$corsAllowedHeaders = @($corsPolicy.allowedHeaders) | Sort-Object
if (($corsAllowedHeaders -join "|") -ne "*") {
    $errors.Add("Ingress CORS allowed headers do not match the expected browser contract.")
}
if ($corsPolicy.maxAge -ne 600) {
    $errors.Add("Ingress CORS maxAge is not 600.")
}

$apiContainers = @($apiApp.properties.template.containers)
if ($apiContainers.Count -ne 1) {
    $errors.Add("API app must define exactly one container.")
} else {
    $apiEnv = @{}
    foreach ($item in @($apiContainers[0].env)) {
        if ($item.name) {
            $apiEnv[$item.name] = if ($item.secretRef) { $item.secretRef } else { $item.value }
        }
    }
    if ($apiEnv["APP_ROLE"] -ne "api") {
        $errors.Add("API app APP_ROLE must be api.")
    }
    if ($apiEnv["CORS_ORIGINS"] -ne $expectedAppOrigin) {
        $errors.Add("API app CORS_ORIGINS does not match the expected app origin.")
    }
    if ($apiEnv["REQUIRE_EDGE_AUTH"] -ne "true") {
        $errors.Add("API app REQUIRE_EDGE_AUTH must be true.")
    }
}

$workerIngress = $workerApp.properties.configuration.ingress
if ($null -ne $workerIngress -and $workerIngress.PSObject.Properties.Count -gt 0) {
    $errors.Add("Worker app still exposes ingress.")
}
$workerContainers = @($workerApp.properties.template.containers)
if ($workerContainers.Count -ne 1) {
    $errors.Add("Worker app must define exactly one container.")
} else {
    $workerEnv = @{}
    foreach ($item in @($workerContainers[0].env)) {
        if ($item.name) {
            $workerEnv[$item.name] = if ($item.secretRef) { $item.secretRef } else { $item.value }
        }
    }
    if ($workerEnv["APP_ROLE"] -ne "worker") {
        $errors.Add("Worker app APP_ROLE must be worker.")
    }
}

$tunnelContainers = @($tunnelApp.properties.template.containers)
if ($tunnelContainers.Count -ne 1) {
    $errors.Add("Tunnel app must define exactly one container.")
} else {
    $tunnelContainer = $tunnelContainers[0]
    if (-not $tunnelContainer.image) {
        $errors.Add("Tunnel app image is not configured.")
    }
    if ($null -ne $tunnelContainer.command -and @($tunnelContainer.command).Count -gt 0) {
        $errors.Add("Unexpected tunnel command: $($tunnelContainer.command -join ' ').")
    }
    if ((@($tunnelContainer.args) -join "|") -ne "tunnel|--no-autoupdate|run") {
        $errors.Add("Unexpected tunnel args: $($tunnelContainer.args -join ' ').")
    }
    $tunnelEnv = @{}
    foreach ($item in @($tunnelContainer.env)) {
        if ($item.name) {
            $tunnelEnv[$item.name] = if ($item.secretRef) { $item.secretRef } else { $item.value }
        }
    }
    if ($tunnelEnv["TUNNEL_TOKEN"] -ne $TunnelSecretRefName) {
        $errors.Add("Tunnel app TUNNEL_TOKEN must reference the $TunnelSecretRefName secret.")
    }
}

$tunnelScale = $tunnelApp.properties.template.scale
if ($tunnelScale.minReplicas -ne 2) {
    $errors.Add("Tunnel app minReplicas is not 2.")
}
if ($tunnelScale.maxReplicas -ne 2) {
    $errors.Add("Tunnel app maxReplicas is not 2.")
}
$tunnelIngress = $tunnelApp.properties.configuration.ingress
if ($null -ne $tunnelIngress -and $tunnelIngress.PSObject.Properties.Count -gt 0) {
    $errors.Add("Tunnel app should not expose ingress.")
}

if ($errors.Count -gt 0) {
    throw ($errors -join [Environment]::NewLine)
}

Write-Phase3Check "Checking public health endpoint through the intended route"
& curl.exe -sS -f $ApiHealthUrl | Out-Null

if ($ProbeOriginUrls) {
    Write-Phase3Check "Checking direct-origin public probes"
    foreach ($probeUrl in ($ProbeOriginUrls -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $probeStatus = & curl.exe -sS -o NUL -w "%{http_code}" --max-time 10 $probeUrl 2>$null
        if ($probeStatus -match "^[2-5][0-9][0-9]$") {
            throw "Direct-origin probe reached an HTTP handler at $probeUrl with status $probeStatus"
        }
    }
}

if ($OriginBypassWorkspace -or $OriginBypassLogQuery) {
    Require-Value -Name "OriginBypassWorkspace" -Value $OriginBypassWorkspace
    Require-Value -Name "OriginBypassLogQuery" -Value $OriginBypassLogQuery
    Write-Phase3Check "Running backend log query for origin-bypass evidence"
    az monitor log-analytics query `
        --workspace $OriginBypassWorkspace `
        --analytics-query $OriginBypassLogQuery `
        --output table
}

Write-Phase3Check "Phase 3 private-origin validation checks completed successfully"
