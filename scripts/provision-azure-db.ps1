#Requires -Version 5.1
# Provision Azure CosmosDB NoSQL (serverless + vector search) for GraphRAG backend.
# Usage: .\scripts\provision-azure-db.ps1

param(
    [string]$ResourceGroup    = "rg-gtog-prod",
    [string]$Location         = "southeastasia",
    [string]$Subscription     = "",
    [string]$StorageAccount   = "stgtogprod",
    [string]$CosmosAccount    = "cdb-gtog-prod",
    [string]$CosmosDatabase   = "gtog-control",

    # Control-plane containers (partition key: /collectionId)
    [string]$CollectionsContainer     = "collections",
    [string]$DocumentsContainer       = "documents",
    [string]$IndexingJobsContainer    = "indexingJobs",
    [string]$JobEventsContainer       = "jobEvents",
    [string]$ArtifactManifestContainer = "artifactManifest",

)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Resolve-AzArgs {
    param([object[]]$InputArgs)

    $resolved = @()
    foreach ($item in $InputArgs) {
        if ($null -eq $item) { continue }
        if ($item -is [System.Array]) {
            foreach ($nested in $item) {
                if ($null -ne $nested) { $resolved += [string]$nested }
            }
        } else {
            $resolved += [string]$item
        }
    }

    return ,$resolved
}

function Invoke-Az {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [object[]]$Args
    )
    $azArgs = Resolve-AzArgs $Args
    $output = & az @azArgs
    if ($LASTEXITCODE -ne 0) { throw "az $($azArgs -join ' ') failed: $output" }
    return $output
}

function Test-Az {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [object[]]$Args
    )

    $azArgs = Resolve-AzArgs $Args
    try {
        & az @azArgs --output none --only-show-errors 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# ---------------------------------------------------------------------------
# Login check + subscription
# ---------------------------------------------------------------------------

Write-Host ">>> Checking Azure login context..."
if (-not (Test-Az "account", "show")) {
    throw "Azure CLI is not logged in. Run: az login --use-device-code"
}

if (-not $Subscription) {
    $Subscription = (Invoke-Az "account", "show", "--query", "id", "--output", "tsv").Trim()
}

Write-Host ">>> Setting subscription: $Subscription"
Invoke-Az "account", "set", "--subscription", $Subscription | Out-Null

# ---------------------------------------------------------------------------
# Resource group
# ---------------------------------------------------------------------------

Write-Host ">>> Ensuring resource group: $ResourceGroup"
Invoke-Az "group", "create", "--name", $ResourceGroup, "--location", $Location, "--output", "none" | Out-Null

# ---------------------------------------------------------------------------
# Storage account + blob containers + queues
# ---------------------------------------------------------------------------

Write-Host ">>> Ensuring storage account: $StorageAccount"
Invoke-Az "storage", "account", "create",
    "--name", $StorageAccount,
    "--resource-group", $ResourceGroup,
    "--location", $Location,
    "--sku", "Standard_LRS",
    "--kind", "StorageV2",
    "--allow-blob-public-access", "false",
    "--output", "none" | Out-Null

Write-Host ">>> Fetching storage account key"
$storageKey = (Invoke-Az "storage", "account", "keys", "list",
    "--account-name", $StorageAccount,
    "--resource-group", $ResourceGroup,
    "--query", "[0].value",
    "--output", "tsv").Trim()
if (-not $storageKey) { throw "Failed to retrieve storage account key." }

$storageConnStr = "DefaultEndpointsProtocol=https;AccountName=$StorageAccount;AccountKey=$storageKey;EndpointSuffix=core.windows.net"

Write-Host ">>> Ensuring blob containers"
foreach ($c in @("pipeline-input", "pipeline-logs")) {
    Invoke-Az "storage", "container", "create",
        "--name", $c,
        "--connection-string", $storageConnStr,
        "--output", "none" | Out-Null
}

Write-Host ">>> Ensuring storage queues"
Invoke-Az "storage", "queue", "create",
    "--name", "indexing-jobs",
    "--connection-string", $storageConnStr,
    "--output", "none" | Out-Null

# ---------------------------------------------------------------------------
# CosmosDB account — serverless + NoSQL vector search
# ---------------------------------------------------------------------------

function Test-CosmosHasRequiredCapabilities {
    $caps = & az cosmosdb show `
        --name $CosmosAccount `
        --resource-group $ResourceGroup `
        --query "capabilities[].name" `
        --output tsv 2>$null
    $capList = $caps -split "`r?`n"
    $isServerless = ($capList | Where-Object { $_ -eq "EnableServerless" }).Count -gt 0
    $hasVectorSearch = ($capList | Where-Object { $_ -eq "EnableNoSQLVectorSearch" }).Count -gt 0
    return $isServerless -and $hasVectorSearch
}

Write-Host ">>> Ensuring Cosmos DB account: $CosmosAccount (serverless + vector search)"
if (Test-Az "cosmosdb", "show", "--name", $CosmosAccount, "--resource-group", $ResourceGroup) {
    if (-not (Test-CosmosHasRequiredCapabilities)) {
        throw "Cosmos DB account '$CosmosAccount' exists but is missing required capabilities (EnableServerless, EnableNoSQLVectorSearch)."
    }
    Write-Host "    Account already exists and has required serverless + vector capabilities."
} else {
    Invoke-Az "cosmosdb", "create",
        "--name", $CosmosAccount,
        "--resource-group", $ResourceGroup,
        "--locations", "regionName=$Location", "failoverPriority=0", "isZoneRedundant=False",
        "--kind", "GlobalDocumentDB",
        "--capabilities", "EnableServerless", "EnableNoSQLVectorSearch",
        "--default-consistency-level", "Session",
        "--output", "none" | Out-Null
    Write-Host "    Created Cosmos DB account."
}

# ---------------------------------------------------------------------------
# CosmosDB SQL database
# ---------------------------------------------------------------------------

Write-Host ">>> Ensuring Cosmos DB SQL database: $CosmosDatabase"
$dbExists = Test-Az "cosmosdb", "sql", "database", "show",
    "--account-name", $CosmosAccount,
    "--resource-group", $ResourceGroup,
    "--name", $CosmosDatabase
if (-not $dbExists) {
    Invoke-Az "cosmosdb", "sql", "database", "create",
        "--account-name", $CosmosAccount,
        "--resource-group", $ResourceGroup,
        "--name", $CosmosDatabase,
        "--output", "none" | Out-Null
    Write-Host "    Created database '$CosmosDatabase'."
} else {
    Write-Host "    Database '$CosmosDatabase' already exists."
}

# ---------------------------------------------------------------------------
# Helper: standard container (partition key: /collectionId)
# ---------------------------------------------------------------------------

function Ensure-ControlContainer {
    param([string]$Name)

    if (Test-Az "cosmosdb", "sql", "container", "show",
        "--account-name", $CosmosAccount,
        "--resource-group", $ResourceGroup,
        "--database-name", $CosmosDatabase,
        "--name", $Name) {
        Write-Host "    Container '$Name' already exists."
        return
    }

    Invoke-Az "cosmosdb", "sql", "container", "create",
        "--account-name", $CosmosAccount,
        "--resource-group", $ResourceGroup,
        "--database-name", $CosmosDatabase,
        "--name", $Name,
        "--partition-key-path", "/collectionId",
        "--output", "none" | Out-Null
    Write-Host "    Created container '$Name'."
}

# ---------------------------------------------------------------------------
# Control-plane containers
# ---------------------------------------------------------------------------

Write-Host ">>> Ensuring control-plane containers (partition key: /collectionId)"
Ensure-ControlContainer $CollectionsContainer
Ensure-ControlContainer $DocumentsContainer
Ensure-ControlContainer $IndexingJobsContainer
Ensure-ControlContainer $JobEventsContainer
Ensure-ControlContainer $ArtifactManifestContainer

# ---------------------------------------------------------------------------
# Vector store containers
# ---------------------------------------------------------------------------

Write-Host ">>> Skipping vector container provisioning (created on-demand during indexing per collection)"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

$cosmosEndpoint = (Invoke-Az "cosmosdb", "show",
    "--name", $CosmosAccount,
    "--resource-group", $ResourceGroup,
    "--query", "documentEndpoint",
    "--output", "tsv").Trim()
if (-not $cosmosEndpoint) { throw "Failed to retrieve Cosmos DB endpoint." }

Write-Host ""
Write-Host "=========================================="
Write-Host "Provisioning complete."
Write-Host "=========================================="
Write-Host ""
Write-Host "Add these non-secret values to backend/.env:"
Write-Host "AZURE_STORAGE_ACCOUNT_NAME=`"$StorageAccount`""
Write-Host "AZURE_STORAGE_QUEUE_NAME=`"indexing-jobs`""
Write-Host "AZURE_COSMOS_ENDPOINT=`"$cosmosEndpoint`""
Write-Host "AZURE_COSMOS_DATABASE_NAME=`"$CosmosDatabase`""
Write-Host "AZURE_COSMOS_COLLECTIONS_CONTAINER=`"$CollectionsContainer`""
Write-Host "AZURE_COSMOS_DOCUMENTS_CONTAINER=`"$DocumentsContainer`""
Write-Host "AZURE_COSMOS_INDEXING_JOBS_CONTAINER=`"$IndexingJobsContainer`""
Write-Host "AZURE_COSMOS_JOB_EVENTS_CONTAINER=`"$JobEventsContainer`""
Write-Host "AZURE_COSMOS_ARTIFACT_MANIFEST_CONTAINER=`"$ArtifactManifestContainer`""
Write-Host ""
Write-Host "Retrieve secret values separately (do not commit to git):"
Write-Host "  az storage account keys list --account-name $StorageAccount --resource-group $ResourceGroup --query '[0].value' -o tsv"
Write-Host "  az cosmosdb keys list --name $CosmosAccount --resource-group $ResourceGroup --query primaryMasterKey -o tsv"
