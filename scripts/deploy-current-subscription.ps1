param(
    [Parameter(Mandatory = $true)]
    [string]$TunnelToken,
    [string]$ResourceGroup = "rg-gtog-prod",
    [string]$Location = "southeastasia",
    [string]$AppPublicHostname = "app.gtog.id.vn",
    [string]$ApiPublicHostname = "api.gtog.id.vn",
    [string]$ContainerAppEnvironment = "cae-gtog-prod",
    [string]$InfrastructureResourceGroup = "rg-gtog-prod-aca-infra",
    [string]$LogAnalyticsWorkspace = "law-gtog-prod",
    [string]$VnetName = "vnet-gtog-prod-aca",
    [string]$InfrastructureSubnetName = "snet-aca-infra",
    [string]$InfrastructureSubnetPrefix = "10.30.0.0/23",
    [string]$PrivateEndpointSubnetName = "snet-aca-private-endpoints",
    [string]$PrivateEndpointSubnetPrefix = "10.30.2.0/27",
    [string]$PrivateEndpointName = "pe-cae-gtog-prod",
    [string]$FrontendAppName = "ca-gtog-frontend-prod",
    [string]$ApiAppName = "ca-gtog-api-prod",
    [string]$WorkerAppName = "ca-gtog-worker-prod",
    [string]$TunnelAppName = "ca-gtog-tunnel-prod",
    [string]$ManagedIdentityName = "mi-gtog-backend",
    [string]$KeyVaultName = "",
    [string]$AcrName = "",
    [string]$StorageAccountName = "",
    [string]$CosmosAccountName = "",
    [string]$CosmosDatabaseName = "",
    [string]$QueueName = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envPath = Join-Path $repoRoot "backend\.env"
if (-not (Test-Path $envPath)) {
    throw "backend/.env not found at $envPath"
}

function Parse-DotEnvValue {
    param([string]$RawValue)

    $value = $RawValue.Trim()
    if ($value.Length -ge 2) {
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            return $value.Substring(1, $value.Length - 2)
        }
    }

    $inSingle = $false
    $inDouble = $false
    for ($i = 0; $i -lt $value.Length; $i++) {
        $ch = $value[$i]
        if ($ch -eq "'" -and -not $inDouble) { $inSingle = -not $inSingle; continue }
        if ($ch -eq '"' -and -not $inSingle) { $inDouble = -not $inDouble; continue }
        if ($ch -eq '#' -and -not $inSingle -and -not $inDouble) {
            if ($i -eq 0 -or [char]::IsWhiteSpace($value[$i - 1])) {
                return $value.Substring(0, $i).TrimEnd()
            }
        }
    }

    return $value
}

function Get-EnvMap {
    param([string]$Path)
    $map = @{}
    foreach ($line in Get-Content -Path $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) { continue }
        if ($trimmed.StartsWith("#")) { continue }
        $idx = $trimmed.IndexOf("=")
        if ($idx -lt 1) { continue }
        $k = $trimmed.Substring(0, $idx).Trim()
        $v = Parse-DotEnvValue -RawValue $trimmed.Substring($idx + 1)
        $map[$k] = $v
    }
    return $map
}

function Require-Tool {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "$Name is required but not found in PATH" }
}

function Ensure-AzLogin {
    az account show --output none
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI is not logged in. Run: az login --use-device-code"
    }
}

function Ensure-Provider {
    param([string]$Namespace)
    az provider register --namespace $Namespace --wait --output none
}

function Test-AzCommand {
    param([scriptblock]$Command)
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    return ($exitCode -eq 0)
}

function Get-AzTsv {
    param([scriptblock]$Command)
    $result = & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI command failed while reading TSV output."
    }
    if ($null -eq $result) {
        return ""
    }
    return ([string]$result).Trim()
}

function First-OrDefault {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return "" }
    $parts = @($Value -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($parts.Count -eq 0) { return "" }
    return ([string]$parts[0]).Trim()
}

function Ensure-Acr {
    param(
        [string]$Name,
        [string]$Rg,
        [string]$Loc
    )

    $candidate = $Name
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        $existing = First-OrDefault (az acr list --resource-group $Rg --query "[].name" -o tsv)
        if ($existing) {
            return $existing
        }

        $subId = (Get-AzTsv { az account show --query id -o tsv })
        $suffix = $subId.Replace("-", "").Substring(0, 8).ToLower()
        $candidate = "acrgtog$suffix"
    }

    if (-not (Test-AzCommand { az acr show --name $candidate --resource-group $Rg --output none 2>$null })) {
        az acr create --resource-group $Rg --name $candidate --sku Basic --admin-enabled true --location $Loc --output none
    }
    else {
        az acr update --resource-group $Rg --name $candidate --admin-enabled true --output none
    }

    return $candidate
}

$envMap = Get-EnvMap -Path $envPath

$graphragApiKey = $envMap["GRAPHRAG_API_KEY"]
$openaiApiKey = $envMap["OPENAI_API_KEY"]
$openaiApiBase = $envMap["OPENAI_API_BASE"]
$googleApiKey = $envMap["GOOGLE_API_KEY"]
$tavilyApiKey = $envMap["TAVILY_API_KEY"]
$storageConnectionString = $envMap["AZURE_STORAGE_CONNECTION_STRING"]
$storageAccountKey = $envMap["AZURE_STORAGE_ACCOUNT_KEY"]
$cosmosConnectionString = $envMap["AZURE_COSMOS_CONNECTION_STRING"]
$cosmosKey = $envMap["AZURE_COSMOS_KEY"]

if ([string]::IsNullOrWhiteSpace($StorageAccountName)) { $StorageAccountName = $envMap["AZURE_STORAGE_ACCOUNT_NAME"] }
if ([string]::IsNullOrWhiteSpace($CosmosAccountName)) { $CosmosAccountName = "cdb-gtog-prod" }
if ([string]::IsNullOrWhiteSpace($CosmosDatabaseName)) { $CosmosDatabaseName = $envMap["AZURE_COSMOS_DATABASE_NAME"] }
if ([string]::IsNullOrWhiteSpace($QueueName)) { $QueueName = $envMap["AZURE_STORAGE_QUEUE_NAME"] }
if ([string]::IsNullOrWhiteSpace($CosmosDatabaseName)) { $CosmosDatabaseName = "gtog-control" }
if ([string]::IsNullOrWhiteSpace($QueueName)) { $QueueName = "indexing-jobs" }

if ([string]::IsNullOrWhiteSpace($StorageAccountName)) { throw "AZURE_STORAGE_ACCOUNT_NAME is missing in backend/.env" }
if ([string]::IsNullOrWhiteSpace($graphragApiKey)) { throw "GRAPHRAG_API_KEY is missing in backend/.env" }
if ([string]::IsNullOrWhiteSpace($openaiApiKey)) { throw "OPENAI_API_KEY is missing in backend/.env" }
if ([string]::IsNullOrWhiteSpace($openaiApiBase)) { throw "OPENAI_API_BASE is missing in backend/.env" }
if ([string]::IsNullOrWhiteSpace($googleApiKey)) { throw "GOOGLE_API_KEY is missing in backend/.env" }
if ([string]::IsNullOrWhiteSpace($tavilyApiKey)) { throw "TAVILY_API_KEY is missing in backend/.env" }

Require-Tool -Name "az"
Require-Tool -Name "docker"

Ensure-AzLogin
$subscriptionId = Get-AzTsv { az account show --query id --output tsv }
Write-Host ">>> Using subscription: $subscriptionId"

az group create --name $ResourceGroup --location $Location --output none

Ensure-Provider -Namespace "Microsoft.App"
Ensure-Provider -Namespace "Microsoft.OperationalInsights"
Ensure-Provider -Namespace "Microsoft.Network"
Ensure-Provider -Namespace "Microsoft.KeyVault"
Ensure-Provider -Namespace "Microsoft.ManagedIdentity"
Ensure-Provider -Namespace "Microsoft.ContainerRegistry"
Ensure-Provider -Namespace "Microsoft.Storage"
Ensure-Provider -Namespace "Microsoft.DocumentDB"

$AcrName = Ensure-Acr -Name $AcrName -Rg $ResourceGroup -Loc $Location

if ([string]::IsNullOrWhiteSpace($KeyVaultName)) {
    $suffix = $subscriptionId.Replace("-", "").Substring(0, 8).ToLower()
    $KeyVaultName = "kvgtog$suffix"
}

if (-not (Test-AzCommand { az identity show --resource-group $ResourceGroup --name $ManagedIdentityName --output none 2>$null })) {
    az identity create --resource-group $ResourceGroup --name $ManagedIdentityName --location $Location --output none
}

if (-not (Test-AzCommand { az keyvault show --resource-group $ResourceGroup --name $KeyVaultName --output none 2>$null })) {
    az keyvault create --resource-group $ResourceGroup --name $KeyVaultName --location $Location --enable-rbac-authorization false --output none
}

$userObjectId = ""
if (Test-AzCommand { az ad signed-in-user show --query id -o tsv 2>$null }) {
    $userObjectId = Get-AzTsv { az ad signed-in-user show --query id -o tsv }
}
if (-not [string]::IsNullOrWhiteSpace($userObjectId)) {
    az keyvault set-policy --name $KeyVaultName --resource-group $ResourceGroup --object-id $userObjectId --secret-permissions get list set delete recover backup restore purge --output none
}

$miPrincipalId = Get-AzTsv { az identity show --resource-group $ResourceGroup --name $ManagedIdentityName --query principalId -o tsv }
if (-not [string]::IsNullOrWhiteSpace($miPrincipalId)) {
    az keyvault set-policy --name $KeyVaultName --resource-group $ResourceGroup --object-id $miPrincipalId --secret-permissions get list --output none
}

if (-not (Test-AzCommand { az cosmosdb show --name $CosmosAccountName --resource-group $ResourceGroup --output none 2>$null })) {
    throw "Cosmos account '$CosmosAccountName' does not exist in resource group '$ResourceGroup'. Run scripts/provision-azure-db.ps1 first."
}

$identityClientId = Get-AzTsv { az identity show --resource-group $ResourceGroup --name $ManagedIdentityName --query clientId -o tsv }
$keyVaultUrl = "https://$KeyVaultName.vault.azure.net/"
$cosmosEndpoint = Get-AzTsv { az cosmosdb show --name $CosmosAccountName --resource-group $ResourceGroup --query documentEndpoint -o tsv }

$edgeOriginSecret = [guid]::NewGuid().ToString("N")

az keyvault secret set --vault-name $KeyVaultName --name "graphrag-api-key" --value $graphragApiKey --output none
az keyvault secret set --vault-name $KeyVaultName --name "openai-api-key" --value $openaiApiKey --output none
az keyvault secret set --vault-name $KeyVaultName --name "google-api-key" --value $googleApiKey --output none
az keyvault secret set --vault-name $KeyVaultName --name "tavily-api-key" --value $tavilyApiKey --output none
if (-not [string]::IsNullOrWhiteSpace($storageConnectionString)) { az keyvault secret set --vault-name $KeyVaultName --name "storage-connection-string" --value $storageConnectionString --output none }
if (-not [string]::IsNullOrWhiteSpace($storageAccountKey)) { az keyvault secret set --vault-name $KeyVaultName --name "storage-account-key" --value $storageAccountKey --output none }
if (-not [string]::IsNullOrWhiteSpace($cosmosConnectionString)) { az keyvault secret set --vault-name $KeyVaultName --name "cosmos-connection-string" --value $cosmosConnectionString --output none }
if (-not [string]::IsNullOrWhiteSpace($cosmosKey)) { az keyvault secret set --vault-name $KeyVaultName --name "cosmos-key" --value $cosmosKey --output none }
az keyvault secret set --vault-name $KeyVaultName --name "edge-origin-secret" --value $edgeOriginSecret --output none

az acr login --name $AcrName

$acrLoginServer = Get-AzTsv { az acr show --name $AcrName --resource-group $ResourceGroup --query loginServer -o tsv }
$acrUsername = Get-AzTsv { az acr credential show --name $AcrName --query username -o tsv }
$acrPassword = Get-AzTsv { az acr credential show --name $AcrName --query "passwords[0].value" -o tsv }
if ([string]::IsNullOrWhiteSpace($acrLoginServer) -or [string]::IsNullOrWhiteSpace($acrUsername) -or [string]::IsNullOrWhiteSpace($acrPassword)) {
    throw "Unable to resolve ACR credentials for $AcrName"
}

$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$backendImage = "$AcrName.azurecr.io/gtog-backend:manual-$tag"
$frontendImage = "$AcrName.azurecr.io/gtog-frontend:manual-$tag"

Push-Location $repoRoot
try {
    docker build -f backend/Dockerfile -t $backendImage .
    if ($LASTEXITCODE -ne 0) { throw "Backend image build failed" }

    docker build --build-arg "NEXT_PUBLIC_API_BASE_URL=https://$ApiPublicHostname" -f frontend/Dockerfile -t $frontendImage frontend
    if ($LASTEXITCODE -ne 0) { throw "Frontend image build failed" }

    docker push $backendImage
    if ($LASTEXITCODE -ne 0) { throw "Backend image push failed" }

    docker push $frontendImage
    if ($LASTEXITCODE -ne 0) { throw "Frontend image push failed" }
}
finally {
    Pop-Location
}

$provisionAcaScript = Join-Path $repoRoot "scripts\provision-aca-private-origin.ps1"
if (-not (Test-Path $provisionAcaScript)) {
    throw "Required script not found: $provisionAcaScript"
}
& $provisionAcaScript `
    -ResourceGroup $ResourceGroup `
    -Location $Location `
    -Subscription $subscriptionId `
    -ContainerAppEnvironment $ContainerAppEnvironment `
    -InfrastructureResourceGroup $InfrastructureResourceGroup `
    -LogAnalyticsWorkspace $LogAnalyticsWorkspace `
    -VnetName $VnetName `
    -InfrastructureSubnetName $InfrastructureSubnetName `
    -InfrastructureSubnetPrefix $InfrastructureSubnetPrefix `
    -PrivateEndpointSubnetName $PrivateEndpointSubnetName `
    -PrivateEndpointSubnetPrefix $PrivateEndpointSubnetPrefix `
    -PrivateEndpointName $PrivateEndpointName `
    -FrontendAppName $FrontendAppName `
    -ApiAppName $ApiAppName `
    -WorkerAppName $WorkerAppName `
    -TunnelAppName $TunnelAppName `
    -FrontendImage $frontendImage `
    -ApiImage $backendImage `
    -WorkerImage $backendImage `
    -UserAssignedIdentityName $ManagedIdentityName `
    -KeyVaultName $KeyVaultName `
    -TunnelToken $TunnelToken `
    -EdgeOriginSecret $edgeOriginSecret `
    -RegistryServer $acrLoginServer `
    -RegistryUsername $acrUsername `
    -RegistryPassword $acrPassword `
    -CreateApps `
    -AppPublicHostname $AppPublicHostname `
    -ApiPublicHostname $ApiPublicHostname

az containerapp secret set --resource-group $ResourceGroup --name $ApiAppName --secrets "graphrag-api-key=$graphragApiKey" "openai-api-key=$openaiApiKey" "google-api-key=$googleApiKey" "tavily-api-key=$tavilyApiKey" --output none
az containerapp secret set --resource-group $ResourceGroup --name $WorkerAppName --secrets "graphrag-api-key=$graphragApiKey" "openai-api-key=$openaiApiKey" "google-api-key=$googleApiKey" "tavily-api-key=$tavilyApiKey" --output none

$apiEnvVars = @(
    "APP_ROLE=api",
    "CORS_ORIGINS=https://$AppPublicHostname",
    "REQUIRE_EDGE_AUTH=true",
    "EDGE_ORIGIN_SECRET=secretref:edge-origin-secret",
    "AZURE_USE_MANAGED_IDENTITY=true",
    "AZURE_MANAGED_IDENTITY_CLIENT_ID=$identityClientId",
    "AZURE_KEY_VAULT_URL=$keyVaultUrl",
    "AZURE_KEY_VAULT_GRAPHRAG_API_KEY_SECRET_NAME=graphrag-api-key",
    "AZURE_KEY_VAULT_OPENAI_API_KEY_SECRET_NAME=openai-api-key",
    "OPENAI_API_BASE=$openaiApiBase",
    "AZURE_KEY_VAULT_GOOGLE_API_KEY_SECRET_NAME=google-api-key",
    "AZURE_KEY_VAULT_TAVILY_API_KEY_SECRET_NAME=tavily-api-key",
    "AZURE_KEY_VAULT_STORAGE_CONNECTION_STRING_SECRET_NAME=storage-connection-string",
    "AZURE_KEY_VAULT_STORAGE_ACCOUNT_KEY_SECRET_NAME=storage-account-key",
    "AZURE_KEY_VAULT_COSMOS_CONNECTION_STRING_SECRET_NAME=cosmos-connection-string",
    "AZURE_KEY_VAULT_COSMOS_KEY_SECRET_NAME=cosmos-key",
    "AZURE_STORAGE_ACCOUNT_NAME=$StorageAccountName",
    "AZURE_STORAGE_QUEUE_NAME=$QueueName",
    "AZURE_COSMOS_ENDPOINT=$cosmosEndpoint",
    "AZURE_COSMOS_DATABASE_NAME=$CosmosDatabaseName",
    "AZURE_COSMOS_COLLECTIONS_CONTAINER=collections",
    "AZURE_COSMOS_DOCUMENTS_CONTAINER=documents",
    "AZURE_COSMOS_INDEXING_JOBS_CONTAINER=indexingJobs",
    "AZURE_COSMOS_JOB_EVENTS_CONTAINER=jobEvents",
    "AZURE_COSMOS_ARTIFACT_MANIFEST_CONTAINER=artifactManifest",
    "AZURE_COSMOS_CONVERSATION_SESSIONS_CONTAINER=conversationSessions",
    "AZURE_COSMOS_CONVERSATION_TURNS_CONTAINER=conversationTurns"
)

$workerEnvVars = @(
    "APP_ROLE=worker",
    "AZURE_USE_MANAGED_IDENTITY=true",
    "AZURE_MANAGED_IDENTITY_CLIENT_ID=$identityClientId",
    "AZURE_KEY_VAULT_URL=$keyVaultUrl",
    "AZURE_KEY_VAULT_GRAPHRAG_API_KEY_SECRET_NAME=graphrag-api-key",
    "AZURE_KEY_VAULT_OPENAI_API_KEY_SECRET_NAME=openai-api-key",
    "OPENAI_API_BASE=$openaiApiBase",
    "AZURE_KEY_VAULT_GOOGLE_API_KEY_SECRET_NAME=google-api-key",
    "AZURE_KEY_VAULT_TAVILY_API_KEY_SECRET_NAME=tavily-api-key",
    "AZURE_KEY_VAULT_STORAGE_CONNECTION_STRING_SECRET_NAME=storage-connection-string",
    "AZURE_KEY_VAULT_STORAGE_ACCOUNT_KEY_SECRET_NAME=storage-account-key",
    "AZURE_KEY_VAULT_COSMOS_CONNECTION_STRING_SECRET_NAME=cosmos-connection-string",
    "AZURE_KEY_VAULT_COSMOS_KEY_SECRET_NAME=cosmos-key",
    "AZURE_STORAGE_ACCOUNT_NAME=$StorageAccountName",
    "AZURE_STORAGE_QUEUE_NAME=$QueueName",
    "AZURE_COSMOS_ENDPOINT=$cosmosEndpoint",
    "AZURE_COSMOS_DATABASE_NAME=$CosmosDatabaseName",
    "AZURE_COSMOS_COLLECTIONS_CONTAINER=collections",
    "AZURE_COSMOS_DOCUMENTS_CONTAINER=documents",
    "AZURE_COSMOS_INDEXING_JOBS_CONTAINER=indexingJobs",
    "AZURE_COSMOS_JOB_EVENTS_CONTAINER=jobEvents",
    "AZURE_COSMOS_ARTIFACT_MANIFEST_CONTAINER=artifactManifest",
    "AZURE_COSMOS_CONVERSATION_SESSIONS_CONTAINER=conversationSessions",
    "AZURE_COSMOS_CONVERSATION_TURNS_CONTAINER=conversationTurns"
)

az containerapp update --resource-group $ResourceGroup --name $ApiAppName --set-env-vars $apiEnvVars --output none
az containerapp update --resource-group $ResourceGroup --name $WorkerAppName --set-env-vars $workerEnvVars --output none

$frontendOrigin = Get-AzTsv { az containerapp show --resource-group $ResourceGroup --name $FrontendAppName --query properties.configuration.ingress.fqdn -o tsv }
$apiOrigin = Get-AzTsv { az containerapp show --resource-group $ResourceGroup --name $ApiAppName --query properties.configuration.ingress.fqdn -o tsv }

Write-Host ""
Write-Host "=========================================="
Write-Host "Deployment complete"
Write-Host "=========================================="
Write-Host "ResourceGroup: $ResourceGroup"
Write-Host "Subscription: $subscriptionId"
Write-Host "ACR: $AcrName"
Write-Host "KeyVault: $KeyVaultName"
Write-Host "ManagedIdentity: $ManagedIdentityName"
Write-Host "BackendImage: $backendImage"
Write-Host "FrontendImage: $frontendImage"
Write-Host "FrontendOrigin: https://$frontendOrigin"
Write-Host "ApiOrigin: https://$apiOrigin"
Write-Host ""
Write-Host "Validate API public endpoint:"
Write-Host "curl -i \"https://$ApiPublicHostname/api/collections\" -H \"Origin: https://$AppPublicHostname\""
