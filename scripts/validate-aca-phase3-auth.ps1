param(
    [string]$ResourceGroup = "rg-gtog-prod",
    [string]$Subscription = "1095803e-80bf-47e0-961f-3d74cb4c605c",
    [string]$ApiAppName = "ca-gtog-api-prod",
    [string]$WorkerAppName = "ca-gtog-worker-prod",
    [string]$TunnelAppName = "ca-gtog-tunnel-prod",
    [string]$TunnelSecretRefName = "tunnel-token",
    [string]$ApiPublicHostname = "",
    [string]$ApiHealthUrl = "",
    [string]$AuthMeUrl = "",
    [string]$ExpectedClientId = "",
    [string]$ExpectedIssuerUrl = "",
    [string]$ExpectedAllowedAudiences = "",
    [string]$ExpectedLoginParametersJson = "",
    [string]$ProbeOriginUrls = "",
    [string]$OriginBypassWorkspace = "",
    [string]$OriginBypassLogQuery = "",
    [string]$TestAccessToken = "",
    [string]$WrongAudienceToken = "",
    [string]$ProductionRejectionToken = ""
)

$ErrorActionPreference = "Stop"

if (-not $env:AZURE_CONFIG_DIR) {
    $env:AZURE_CONFIG_DIR = (Join-Path (Get-Location) ".azure")
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

function Convert-CsvToArray {
    param([string]$Value)

    if (-not $Value) {
        return @()
    }

    return $Value.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

function Get-DefaultLoginParameters {
    param([string[]]$AllowedAudiences)

    if ($AllowedAudiences.Count -eq 0) {
        throw "At least one allowed audience is required."
    }

    return @("scope=openid profile email offline_access $($AllowedAudiences[0])/access_as_user")
}

function Invoke-StatusCheck {
    param(
        [string]$Url,
        [hashtable]$Headers = @{}
    )

    try {
        $response = Invoke-WebRequest -Uri $Url -Method Get -Headers $Headers -MaximumRedirection 0 -ErrorAction Stop
        return [int]$response.StatusCode
    } catch {
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            return [int]$_.Exception.Response.StatusCode
        }
        throw
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
Require-Value -Name "ApiPublicHostname" -Value $ApiPublicHostname
Require-Value -Name "ExpectedClientId" -Value $ExpectedClientId
Require-Value -Name "ExpectedIssuerUrl" -Value $ExpectedIssuerUrl
Require-Value -Name "ExpectedAllowedAudiences" -Value $ExpectedAllowedAudiences

if (-not $ApiHealthUrl) {
    $ApiHealthUrl = "https://$ApiPublicHostname/health"
}
if (-not $AuthMeUrl) {
    $AuthMeUrl = "https://$ApiPublicHostname/.auth/me"
}
$expectedAllowedAudienceArray = Convert-CsvToArray -Value $ExpectedAllowedAudiences
if (-not $ExpectedLoginParametersJson) {
    $expectedLoginParameters = Get-DefaultLoginParameters -AllowedAudiences $expectedAllowedAudienceArray
} else {
    $expectedLoginParameters = $ExpectedLoginParametersJson | ConvertFrom-Json
}

Write-Phase3Check "Using subscription $Subscription"
az account set --subscription $Subscription --output none | Out-Null

Write-Phase3Check "Reading current API app, worker app, tunnel app, and auth settings"
$apiApp = az containerapp show --resource-group $ResourceGroup --name $ApiAppName --output json | ConvertFrom-Json -Depth 20
$workerApp = az containerapp show --resource-group $ResourceGroup --name $WorkerAppName --output json | ConvertFrom-Json -Depth 20
$tunnelApp = az containerapp show --resource-group $ResourceGroup --name $TunnelAppName --output json | ConvertFrom-Json -Depth 20
$auth = az containerapp auth show --resource-group $ResourceGroup --name $ApiAppName --output json | ConvertFrom-Json -Depth 20
$microsoftAuth = az containerapp auth microsoft show --resource-group $ResourceGroup --name $ApiAppName --output json | ConvertFrom-Json -Depth 20

$errors = [System.Collections.Generic.List[string]]::new()
if (-not $auth.platform.enabled) {
    $errors.Add("Easy Auth is not enabled.")
}
if ($auth.globalValidation.unauthenticatedClientAction -ne "Return401") {
    $errors.Add("Unauthenticated action is not Return401.")
}
if (-not $auth.httpSettings.requireHttps) {
    $errors.Add("Easy Auth does not require HTTPS.")
}
if ($auth.httpSettings.forwardProxy.convention -ne "Standard") {
    $errors.Add("Forward proxy convention is not Standard.")
}
if ((@($auth.globalValidation.excludedPaths) | Sort-Object) -join "|" -ne (@("/health", "/health/readiness") | Sort-Object) -join "|") {
    $errors.Add("Excluded paths are not exactly /health and /health/readiness.")
}
if ((@($auth.identityProviders.azureActiveDirectory.login.loginParameters)) -join "|" -ne (@($expectedLoginParameters)) -join "|") {
    $errors.Add("Login parameters do not match the expected environment contract.")
}
if ($microsoftAuth.registration.clientId -ne $ExpectedClientId) {
    $errors.Add("Configured Entra clientId does not match the expected app registration.")
}
if ($microsoftAuth.registration.openIdIssuer -ne $ExpectedIssuerUrl) {
    $errors.Add("Configured issuer URI does not match the expected tenant issuer.")
}
if ((@($microsoftAuth.validation.allowedAudiences) | Sort-Object) -join "|" -ne ($expectedAllowedAudienceArray | Sort-Object) -join "|") {
    $errors.Add("Allowed audiences do not match the expected environment-specific values.")
}
if ($apiApp.properties.configuration.ingress.external -ne $false) {
    $errors.Add("API app ingress is not internal-only.")
}
if ($apiApp.properties.configuration.ingress.targetPort -ne 8000) {
    $errors.Add("API app target port is not 8000.")
}
if ($workerApp.properties.configuration.ingress) {
    $errors.Add("Worker app still exposes ingress.")
}

$tunnelContainers = @($tunnelApp.properties.template.containers)
if ($tunnelContainers.Count -ne 1) {
    $errors.Add("Tunnel app must define exactly one container.")
} else {
    $tunnelContainer = $tunnelContainers[0]
    if (-not $tunnelContainer.image) {
        $errors.Add("Tunnel app image is not configured.")
    }

    $command = @($tunnelContainer.command)
    if (($command -join "|") -ne @("/bin/sh") -join "|") {
        $errors.Add("Unexpected tunnel command: $($command -join ', ')")
    }

    $args = @($tunnelContainer.args)
    $expectedArgs = @("-c", 'cloudflared tunnel --no-autoupdate run --token "$TUNNEL_TOKEN"')
    if (($args -join "|") -ne ($expectedArgs -join "|")) {
        $errors.Add("Unexpected tunnel args: $($args -join ', ')")
    }

    $tunnelTokenEnv = $tunnelContainer.env | Where-Object { $_.name -eq "TUNNEL_TOKEN" } | Select-Object -First 1
    if (-not $tunnelTokenEnv -or $tunnelTokenEnv.secretRef -ne $TunnelSecretRefName) {
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
if ($tunnelApp.properties.configuration.ingress) {
    $errors.Add("Tunnel app should not expose ingress.")
}

if ($errors.Count -gt 0) {
    throw ($errors -join [Environment]::NewLine)
}

Write-Phase3Check "Checking public health endpoint through the intended route"
$healthStatus = Invoke-StatusCheck -Url $ApiHealthUrl
if ($healthStatus -lt 200 -or $healthStatus -ge 400) {
    throw "Expected public health endpoint to succeed, got status $healthStatus."
}

Write-Phase3Check "Checking /.auth/me without forcing a login redirect"
$authMeStatus = Invoke-StatusCheck -Url $AuthMeUrl
if ($authMeStatus -notin @(200, 401, 403)) {
    throw "Unexpected /.auth/me status: $authMeStatus"
}

Write-Phase3Check "Checking unauthenticated API access returns 401 through Easy Auth"
$unauthStatus = Invoke-StatusCheck -Url "https://$ApiPublicHostname/api/collections"
if ($unauthStatus -ne 401) {
    throw "Expected unauthenticated /api/* request to return 401, got $unauthStatus"
}

if ($TestAccessToken) {
    Write-Phase3Check "Checking authenticated API access with environment token"
    $authOkStatus = Invoke-StatusCheck -Url "https://$ApiPublicHostname/api/collections" -Headers @{ Authorization = "Bearer $TestAccessToken" }
    if ($authOkStatus -ne 200) {
        throw "Expected environment token to return 200, got $authOkStatus"
    }
}

if ($WrongAudienceToken) {
    Write-Phase3Check "Checking wrong-audience token rejection"
    $wrongAudienceStatus = Invoke-StatusCheck -Url "https://$ApiPublicHostname/api/collections" -Headers @{ Authorization = "Bearer $WrongAudienceToken" }
    if ($wrongAudienceStatus -notin @(401, 403)) {
        throw "Expected wrong-audience token to be rejected, got $wrongAudienceStatus"
    }
}

if ($ProductionRejectionToken) {
    Write-Phase3Check "Checking cross-environment token rejection"
    $crossEnvironmentStatus = Invoke-StatusCheck -Url "https://$ApiPublicHostname/api/collections" -Headers @{ Authorization = "Bearer $ProductionRejectionToken" }
    if ($crossEnvironmentStatus -notin @(401, 403)) {
        throw "Expected cross-environment token to be rejected, got $crossEnvironmentStatus"
    }
}

if ($ProbeOriginUrls) {
    Write-Phase3Check "Checking direct-origin public probes"
    foreach ($probeUrl in (Convert-CsvToArray -Value $ProbeOriginUrls)) {
        try {
            $probeStatus = Invoke-StatusCheck -Url $probeUrl
            throw "Direct-origin probe reached an HTTP handler at $probeUrl with status $probeStatus"
        } catch {
            if ($_.Exception.Message -like "Direct-origin probe reached*") {
                throw
            }
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

Write-Phase3Check "Phase 3 validation checks completed successfully"
