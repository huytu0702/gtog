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
$searchApiKey = az search admin-key show `
    --service-name $SearchService `
    --resource-group $ResourceGroup `
    --query primaryKey `
    --output tsv
if (-not $searchApiKey) {
    throw "Failed to retrieve Azure AI Search admin key."
}

Write-Host ">>> Ensuring Cosmos DB account: $CosmosAccount"
if (-not (Test-AzCommand { az cosmosdb show --name $CosmosAccount --resource-group $ResourceGroup --output none 2>$null })) {
    az cosmosdb create `
        --name $CosmosAccount `
        --resource-group $ResourceGroup `
        --locations regionName=$Location failoverPriority=0 isZoneRedundant=False `
        --kind GlobalDocumentDB `
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
    param(
        [string]$ContainerName,
        [string]$MaxThroughput
    )

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
            --max-throughput $MaxThroughput `
            --output none
    }
}

Write-Host ">>> Ensuring Cosmos DB containers (autoscale)"
Ensure-CosmosContainer -ContainerName $CollectionsContainer -MaxThroughput "1000"
Ensure-CosmosContainer -ContainerName $DocumentsContainer -MaxThroughput "1000"
Ensure-CosmosContainer -ContainerName $IndexingJobsContainer -MaxThroughput "4000"
Ensure-CosmosContainer -ContainerName $JobEventsContainer -MaxThroughput "4000"
Ensure-CosmosContainer -ContainerName $ArtifactManifestContainer -MaxThroughput "1000"
Ensure-CosmosContainer -ContainerName $EntitiesContainer -MaxThroughput "4000"
Ensure-CosmosContainer -ContainerName $RelationshipsContainer -MaxThroughput "4000"
Ensure-CosmosContainer -ContainerName $TextUnitsContainer -MaxThroughput "4000"
Ensure-CosmosContainer -ContainerName $CommunitiesContainer -MaxThroughput "1000"
Ensure-CosmosContainer -ContainerName $CommunityReportsContainer -MaxThroughput "1000"
Ensure-CosmosContainer -ContainerName $CovariatesContainer -MaxThroughput "1000"

$cosmosEndpoint = az cosmosdb show `
    --name $CosmosAccount `
    --resource-group $ResourceGroup `
    --query documentEndpoint `
    --output tsv
if (-not $cosmosEndpoint) {
    throw "Failed to retrieve Cosmos endpoint."
}
$cosmosKey = az cosmosdb keys list `
    --name $CosmosAccount `
    --resource-group $ResourceGroup `
    --query primaryMasterKey `
    --output tsv
if (-not $cosmosKey) {
    throw "Failed to retrieve Cosmos primary key."
}
$cosmosConnectionString = az cosmosdb keys list `
    --name $CosmosAccount `
    --resource-group $ResourceGroup `
    --type connection-strings `
    --query "connectionStrings[0].connectionString" `
    --output tsv
if (-not $cosmosConnectionString) {
    throw "Failed to retrieve Cosmos connection string."
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Provisioning complete."
Write-Host "=========================================="
Write-Host ""
Write-Host "Add these to backend/.env:"
Write-Host "AZURE_STORAGE_CONNECTION_STRING=`"$storageConnectionString`""
Write-Host "AZURE_STORAGE_ACCOUNT_NAME=`"$StorageAccount`""
Write-Host "AZURE_STORAGE_ACCOUNT_KEY=`"$storageAccountKey`""
Write-Host "AZURE_STORAGE_QUEUE_NAME=`"indexing-jobs`""
Write-Host "AZURE_SEARCH_ENDPOINT=`"$searchEndpoint`""
Write-Host "AZURE_SEARCH_API_KEY=`"$searchApiKey`""
Write-Host "AZURE_COSMOS_CONNECTION_STRING=`"$cosmosConnectionString`""
Write-Host "AZURE_COSMOS_ENDPOINT=`"$cosmosEndpoint`""
Write-Host "AZURE_COSMOS_KEY=`"$cosmosKey`""
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
Write-Host "Search SKU in use: $SearchSku"
Write-Host "Do not commit secrets to git."
