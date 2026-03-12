param(
    [string]$ApiBaseUrl = "",
    [string]$ApiPublicHostname = "",
    [string]$AuthBearerToken = "",
    [string]$ResourceGroup = "",
    [string]$Subscription = "",
    [string]$ApiAppName = "",
    [string]$WorkerAppName = "",
    [string]$TunnelAppName = "",
    [string]$ExpectedClientId = "",
    [string]$ExpectedIssuerUrl = "",
    [string]$ExpectedAllowedAudiences = "",
    [string]$WrongAudienceToken = "",
    [string]$ProductionRejectionToken = "",
    [string]$ProbeOriginUrls = "",
    [string]$SmokeArtifactName = "smoke-staging-report",
    [string]$Phase3ValidationArtifactName = "phase3-auth-origin-validation",
    [string]$SmokePhaseLabel = "staging",
    [string]$RolloutStateFile = "",
    [string]$EvidenceDir = "",
    [string]$CollectionId = "",
    [string]$SampleQuery = "What does this collection contain?",
    [string]$TunnelSecretRefName = "tunnel-token"
)

$ErrorActionPreference = "Stop"

if (-not $EvidenceDir) {
    $EvidenceDir = Join-Path (Get-Location) "artifacts/$SmokeArtifactName"
}
if (-not $CollectionId) {
    $CollectionId = "smoke-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
}

$phase3ArtifactName = $Phase3ValidationArtifactName
$smokeArtifactName = $SmokeArtifactName
$phase3OutputFile = Join-Path $EvidenceDir "$phase3ArtifactName.txt"
$smokeReportFile = Join-Path $EvidenceDir "$smokeArtifactName.json"
$uploadFile = Join-Path $EvidenceDir "smoke-upload.txt"
$sseOutputFile = Join-Path $EvidenceDir "sse-output.txt"
$results = [System.Collections.Generic.List[object]]::new()

function Require-Value {
    param([string]$Name, [string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Name is required"
    }
}

function Add-Result {
    param([string]$Name, [string]$Status)
    $results.Add(@{ name = $Name; status = $Status }) | Out-Null
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

function Invoke-JsonRequest {
    param(
        [string]$Method,
        [string]$Url,
        [string]$Body = "",
        [string]$OutFile = ""
    )

    $headers = @{ Authorization = "Bearer $AuthBearerToken" }
    if ($Body) {
        $headers["Content-Type"] = "application/json"
        $response = Invoke-WebRequest -Method $Method -Uri $Url -Headers $headers -Body $Body
    } else {
        $response = Invoke-WebRequest -Method $Method -Uri $Url -Headers $headers
    }

    if ($OutFile) {
        $response.Content | Set-Content -Path $OutFile
    }

    return [pscustomobject]@{
        StatusCode = [int]$response.StatusCode
        Json = if ($response.Content) { $response.Content | ConvertFrom-Json -Depth 20 } else { $null }
    }
}

New-Item -ItemType Directory -Path $EvidenceDir -Force | Out-Null
Set-Content -Path $uploadFile -Value "smoke document for $CollectionId"

Require-Value -Name "ApiBaseUrl" -Value $ApiBaseUrl
Require-Value -Name "ApiPublicHostname" -Value $ApiPublicHostname
Require-Value -Name "AuthBearerToken" -Value $AuthBearerToken
Require-Value -Name "ResourceGroup" -Value $ResourceGroup
Require-Value -Name "Subscription" -Value $Subscription
Require-Value -Name "ApiAppName" -Value $ApiAppName
Require-Value -Name "WorkerAppName" -Value $WorkerAppName
Require-Value -Name "TunnelAppName" -Value $TunnelAppName
Require-Value -Name "ExpectedClientId" -Value $ExpectedClientId
Require-Value -Name "ExpectedIssuerUrl" -Value $ExpectedIssuerUrl
Require-Value -Name "ExpectedAllowedAudiences" -Value $ExpectedAllowedAudiences
Require-Value -Name "WrongAudienceToken" -Value $WrongAudienceToken
Require-Value -Name "ProductionRejectionToken" -Value $ProductionRejectionToken
Require-Value -Name "ProbeOriginUrls" -Value $ProbeOriginUrls

$healthStatus = Invoke-StatusCheck -Url "$ApiBaseUrl/health"
if ($healthStatus -ne 200) { throw "health expected 200, got $healthStatus" }
Add-Result -Name "health" -Status "$healthStatus"

$readinessStatus = Invoke-StatusCheck -Url "$ApiBaseUrl/health/readiness"
if ($readinessStatus -ne 200) { throw "readiness expected 200, got $readinessStatus" }
Add-Result -Name "readiness" -Status "$readinessStatus"

$authMeStatus = Invoke-StatusCheck -Url "$ApiBaseUrl/.auth/me"
if ($authMeStatus -notin @(200, 401, 403)) { throw "/.auth/me returned unexpected status $authMeStatus" }
Add-Result -Name "auth_me" -Status "$authMeStatus"

$unauthStatus = Invoke-StatusCheck -Url "$ApiBaseUrl/api/collections"
if ($unauthStatus -ne 401) { throw "Expected unauthenticated /api/collections to return 401, got $unauthStatus" }
Add-Result -Name "unauthenticated_collections" -Status "$unauthStatus"

$createBody = @{ name = $CollectionId; description = "Phase 5 smoke collection" } | ConvertTo-Json -Compress
$createResponse = Invoke-JsonRequest -Method Post -Url "$ApiBaseUrl/api/collections" -Body $createBody -OutFile (Join-Path $EvidenceDir "create-collection.json")
if ($createResponse.StatusCode -ne 201) { throw "Create collection expected 201, got $($createResponse.StatusCode)" }
Add-Result -Name "create_collection" -Status "$($createResponse.StatusCode)"

$uploadHeaders = @{ Authorization = "Bearer $AuthBearerToken" }
$uploadResponse = Invoke-WebRequest -Uri "$ApiBaseUrl/api/collections/$CollectionId/documents" -Method Post -Headers $uploadHeaders -Form @{ file = Get-Item $uploadFile } -OutFile (Join-Path $EvidenceDir "upload-document.json") -PassThru
if ([int]$uploadResponse.StatusCode -ne 201) { throw "Upload document expected 201, got $($uploadResponse.StatusCode)" }
Add-Result -Name "upload_document" -Status "$([int]$uploadResponse.StatusCode)"

$listResponse = Invoke-JsonRequest -Method Get -Url "$ApiBaseUrl/api/collections/$CollectionId/documents" -OutFile (Join-Path $EvidenceDir "list-documents.json")
if ($listResponse.StatusCode -ne 200) { throw "List documents expected 200, got $($listResponse.StatusCode)" }
Add-Result -Name "list_documents" -Status "$($listResponse.StatusCode)"

$indexResponse = Invoke-JsonRequest -Method Post -Url "$ApiBaseUrl/api/collections/$CollectionId/index" -OutFile (Join-Path $EvidenceDir "start-indexing.json")
if ($indexResponse.StatusCode -ne 202) { throw "Start indexing expected 202, got $($indexResponse.StatusCode)" }
$jobId = $indexResponse.Json.job_id
Add-Result -Name "start_indexing" -Status "$($indexResponse.StatusCode)"

$jobStatus = "queued"
$jobPollStatusCode = 0
for ($i = 0; $i -lt 20; $i++) {
    $jobResponse = Invoke-JsonRequest -Method Get -Url "$ApiBaseUrl/api/index-jobs/$jobId" -OutFile (Join-Path $EvidenceDir "job-status.json")
    $jobPollStatusCode = $jobResponse.StatusCode
    if ($jobPollStatusCode -ne 200) { throw "Job status expected 200, got $jobPollStatusCode" }
    $jobStatus = [string]$jobResponse.Json.status
    if ($jobStatus -eq "completed") { break }
    if ($jobStatus -in @("failed", "cancelled")) {
        throw "Index job entered terminal failure state: $jobStatus"
    }
    Start-Sleep -Seconds 5
}
if ($jobStatus -ne "completed") {
    throw "Index job did not complete within the polling window"
}
Add-Result -Name "job_status_polling" -Status "$jobPollStatusCode"

$searchBody = @{ query = $SampleQuery } | ConvertTo-Json -Compress
$localResponse = Invoke-JsonRequest -Method Post -Url "$ApiBaseUrl/api/collections/$CollectionId/search/local" -Body $searchBody -OutFile (Join-Path $EvidenceDir "local-search.json")
if ($localResponse.StatusCode -ne 200) { throw "Local search expected 200, got $($localResponse.StatusCode)" }
Add-Result -Name "query_local" -Status "$($localResponse.StatusCode)"
$globalResponse = Invoke-JsonRequest -Method Post -Url "$ApiBaseUrl/api/collections/$CollectionId/search/global" -Body $searchBody -OutFile (Join-Path $EvidenceDir "global-search.json")
if ($globalResponse.StatusCode -ne 200) { throw "Global search expected 200, got $($globalResponse.StatusCode)" }
Add-Result -Name "query_global" -Status "$($globalResponse.StatusCode)"
$togResponse = Invoke-JsonRequest -Method Post -Url "$ApiBaseUrl/api/collections/$CollectionId/search/tog" -Body $searchBody -OutFile (Join-Path $EvidenceDir "tog-search.json")
if ($togResponse.StatusCode -ne 200) { throw "ToG search expected 200, got $($togResponse.StatusCode)" }
Add-Result -Name "query_tog" -Status "$($togResponse.StatusCode)"

$encodedQuery = [System.Uri]::EscapeDataString($SampleQuery)
$sseResponse = Invoke-WebRequest -Uri "$ApiBaseUrl/api/collections/$CollectionId/search/agent/stream?query=$encodedQuery" -Headers @{ Authorization = "Bearer $AuthBearerToken" } -OutFile $sseOutputFile -PassThru
if ([int]$sseResponse.StatusCode -ne 200) { throw "SSE endpoint expected 200, got $($sseResponse.StatusCode)" }
$sseContent = Get-Content -Path $sseOutputFile -Raw
if ($sseContent -notmatch "heartbeat|data:") {
    throw "SSE output did not contain expected event data"
}
Add-Result -Name "sse" -Status "$([int]$sseResponse.StatusCode)"

& (Join-Path $PSScriptRoot "validate-aca-phase3-auth.ps1") `
    -ResourceGroup $ResourceGroup `
    -Subscription $Subscription `
    -ApiAppName $ApiAppName `
    -WorkerAppName $WorkerAppName `
    -TunnelAppName $TunnelAppName `
    -TunnelSecretRefName $TunnelSecretRefName `
    -ApiPublicHostname $ApiPublicHostname `
    -ExpectedClientId $ExpectedClientId `
    -ExpectedIssuerUrl $ExpectedIssuerUrl `
    -ExpectedAllowedAudiences $ExpectedAllowedAudiences `
    -TestAccessToken $AuthBearerToken `
    -WrongAudienceToken $WrongAudienceToken `
    -ProductionRejectionToken $ProductionRejectionToken `
    -ProbeOriginUrls $ProbeOriginUrls *> $phase3OutputFile

$payload = [ordered]@{
    artifact = $smokeArtifactName
    phase = $SmokePhaseLabel
    collection_id = $CollectionId
    checks = $results
}
if ($RolloutStateFile) {
    $payload.rollout_state_file = $RolloutStateFile
    if (Test-Path -Path $RolloutStateFile) {
        $payload.rollout_state = Get-Content -Path $RolloutStateFile -Raw | ConvertFrom-Json -Depth 20
    }
}
$payload | ConvertTo-Json -Depth 10 | Set-Content -Path $smokeReportFile

Write-Host "Smoke report written to $smokeReportFile"
Write-Host "Phase 3 validation evidence written to $phase3OutputFile"
