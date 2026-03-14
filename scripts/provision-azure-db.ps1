param(
    [string]$ResourceGroup = "rg-gtog-prod",
    [string]$Location = "southeastasia",
    [string]$Subscription = "1095803e-80bf-47e0-961f-3d74cb4c605c",
    [string]$StorageAccount = "stgtogprod",
    [string]$SearchService = "srch-gtog-prod",
    [string]$SearchSku = "free",
    [string]$CosmosAccount = "cdb-gtog-prod",
    [string]$CosmosDatabase = "gtog-control",
    [string]$CollectionsContainer = "collections",
    [string]$DocumentsContainer = "documents",
    [string]$IndexingJobsContainer = "indexingJobs",
    [string]$JobEventsContainer = "jobEvents",
    [string]$ArtifactManifestContainer = "artifactManifest",
    [string]$EntitiesContainer = "entities",
    [string]$RelationshipsContainer = "relationships",
    [string]$TextUnitsContainer = "textUnits",
    [string]$CommunitiesContainer = "communities",
    [string]$CommunityReportsContainer = "communityReports",
    [string]$CovariatesContainer = "covariates"
)

$ErrorActionPreference = "Stop"

if (-not $env:AZURE_CONFIG_DIR) {
    $env:AZURE_CONFIG_DIR = (Join-Path (Get-Location) ".azure")
}
New-Item -ItemType Directory -Path $env:AZURE_CONFIG_DIR -Force | Out-Null

function Test-AzCommand {
    param([scriptblock]$Command)
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    return ($exitCode -eq 0)
}

Write-Host ">>> Checking Azure login context..."
if (-not (Test-AzCommand { az account show --output none 2>$null })) {
    throw "Azure CLI is not logged in. Run: az login --use-device-code"
}

Write-Host ">>> Setting subscription: $Subscription"
az account set --subscription $Subscription --output none

Write-Host ">>> Ensuring resource group: $ResourceGroup"
az group create --name $ResourceGroup --location $Location --output none

Write-Host ">>> Ensuring storage account: $StorageAccount"
az storage account create `
    --name $StorageAccount `
    --resource-group $ResourceGroup `
    --location $Location `
    --sku Standard_LRS `
    --kind StorageV2 `
    --allow-blob-public-access false `
    --output none
if (-not (Test-AzCommand { az storage account show --name $StorageAccount --resource-group $ResourceGroup --output none 2>$null })) {
    throw "Storage account '$StorageAccount' was not created."
}

Write-Host ">>> Fetching storage account key"
$storageAccountKey = az storage account keys list `
    --account-name $StorageAccount `
    --resource-group $ResourceGroup `
    --query "[0].value" `
    --output tsv
if (-not $storageAccountKey) {
    throw "Failed to retrieve storage account key."
}
$storageConnectionString = "DefaultEndpointsProtocol=https;AccountName=$StorageAccount;AccountKey=$storageAccountKey;EndpointSuffix=core.windows.net"

Write-Host ">>> Ensuring blob containers"
@("gtog-input", "gtog-output", "gtog-cache", "gtog-logs") | ForEach-Object {
    az storage container create `
        --name $_ `
        --connection-string $storageConnectionString `
        --output none
}

Write-Host ">>> Ensuring storage queues"
@("indexing-jobs") | ForEach-Object {
    az storage queue create `
        --name $_ `
        --connection-string $storageConnectionString `
        --output none
}

Write-Host ">>> Ensuring Azure AI Search: $SearchService (sku=$SearchSku)"
if (-not (Test-AzCommand { az search service show --name $SearchService --resource-group $ResourceGroup --output none 2>$null })) {
    az search service create `
        --name $SearchService `
        --resource-group $ResourceGroup `
        --location $Location `
        --sku $SearchSku `
        --output none
}

$searchEndpoint = "https://$SearchService.search.windows.net"

function Test-CosmosAccountIsServerless {
    $capabilities = az cosmosdb show `
        --name $CosmosAccount `
        --resource-group $ResourceGroup `
        --query "capabilities[].name" `
        --output tsv 2>$null
    return ($capabilities -split "`r?`n" | Where-Object { $_ -eq "EnableServerless" }).Count -gt 0
}

Write-Host ">>> Ensuring Cosmos DB account: $CosmosAccount"
if (Test-AzCommand { az cosmosdb show --name $CosmosAccount --resource-group $ResourceGroup --output none 2>$null }) {
    if (-not (Test-CosmosAccountIsServerless)) {
        throw "Cosmos DB account '$CosmosAccount' already exists but is not configured for serverless. Capacity mode cannot be changed in place."
    }
    Write-Host "    Cosmos DB account already exists and is configured for serverless."
} else {
    az cosmosdb create `
        --name $CosmosAccount `
        --resource-group $ResourceGroup `
        --locations regionName=$Location failoverPriority=0 isZoneRedundant=False `
        --kind GlobalDocumentDB `
        --capabilities EnableServerless `
        --default-consistency-level Session `
        --output none
}
if (-not (Test-AzCommand { az cosmosdb show --name $CosmosAccount --resource-group $ResourceGroup --output none 2>$null })) {
    throw "Cosmos DB account '$CosmosAccount' was not created."
}

Write-Host ">>> Ensuring Cosmos DB SQL database: $CosmosDatabase"
az cosmosdb sql database create `
    --account-name $CosmosAccount `
    --resource-group $ResourceGroup `
    --name $CosmosDatabase `
    --output none

function Ensure-CosmosContainer {
    param([string]$ContainerName)

    if (-not (Test-AzCommand {
        az cosmosdb sql container show `
            --account-name $CosmosAccount `
            --resource-group $ResourceGroup `
            --database-name $CosmosDatabase `
            --name $ContainerName `
            --output none 2>$null
    })) {
        az cosmosdb sql container create `
            --account-name $CosmosAccount `
            --resource-group $ResourceGroup `
            --database-name $CosmosDatabase `
            --name $ContainerName `
            --partition-key-path "/collectionId" `
            --output none
    }
}

Write-Host ">>> Ensuring Cosmos DB containers (serverless)"
Ensure-CosmosContainer -ContainerName $CollectionsContainer
Ensure-CosmosContainer -ContainerName $DocumentsContainer
Ensure-CosmosContainer -ContainerName $IndexingJobsContainer
Ensure-CosmosContainer -ContainerName $JobEventsContainer
Ensure-CosmosContainer -ContainerName $ArtifactManifestContainer
Ensure-CosmosContainer -ContainerName $EntitiesContainer
Ensure-CosmosContainer -ContainerName $RelationshipsContainer
Ensure-CosmosContainer -ContainerName $TextUnitsContainer
Ensure-CosmosContainer -ContainerName $CommunitiesContainer
Ensure-CosmosContainer -ContainerName $CommunityReportsContainer
Ensure-CosmosContainer -ContainerName $CovariatesContainer

$cosmosEndpoint = az cosmosdb show `
    --name $CosmosAccount `
    --resource-group $ResourceGroup `
    --query documentEndpoint `
    --output tsv
if (-not $cosmosEndpoint) {
    throw "Failed to retrieve Cosmos endpoint."
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Provisioning complete."
Write-Host "=========================================="
Write-Host ""
Write-Host "Add these non-secret values to backend/.env:"
Write-Host "AZURE_STORAGE_ACCOUNT_NAME=`"$StorageAccount`""
Write-Host "AZURE_STORAGE_QUEUE_NAME=`"indexing-jobs`""
Write-Host "AZURE_SEARCH_ENDPOINT=`"$searchEndpoint`""
Write-Host "AZURE_COSMOS_ENDPOINT=`"$cosmosEndpoint`""
Write-Host "AZURE_COSMOS_DATABASE_NAME=`"$CosmosDatabase`""
Write-Host "AZURE_COSMOS_COLLECTIONS_CONTAINER=`"$CollectionsContainer`""
Write-Host "AZURE_COSMOS_DOCUMENTS_CONTAINER=`"$DocumentsContainer`""
Write-Host "AZURE_COSMOS_INDEXING_JOBS_CONTAINER=`"$IndexingJobsContainer`""
Write-Host "AZURE_COSMOS_JOB_EVENTS_CONTAINER=`"$JobEventsContainer`""
Write-Host "AZURE_COSMOS_ARTIFACT_MANIFEST_CONTAINER=`"$ArtifactManifestContainer`""
Write-Host "AZURE_COSMOS_ENTITIES_CONTAINER=`"$EntitiesContainer`""
Write-Host "AZURE_COSMOS_RELATIONSHIPS_CONTAINER=`"$RelationshipsContainer`""
Write-Host "AZURE_COSMOS_TEXT_UNITS_CONTAINER=`"$TextUnitsContainer`""
Write-Host "AZURE_COSMOS_COMMUNITIES_CONTAINER=`"$CommunitiesContainer`""
Write-Host "AZURE_COSMOS_COMMUNITY_REPORTS_CONTAINER=`"$CommunityReportsContainer`""
Write-Host "AZURE_COSMOS_COVARIATES_CONTAINER=`"$CovariatesContainer`""
Write-Host ""
Write-Host "Retrieve secret values separately via Azure CLI before writing backend/.env."
Write-Host "Required secret keys: AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_ACCOUNT_KEY, AZURE_SEARCH_API_KEY, AZURE_COSMOS_CONNECTION_STRING, AZURE_COSMOS_KEY"
Write-Host "Search SKU in use: $SearchSku"
Write-Host "Do not commit secrets to git."
