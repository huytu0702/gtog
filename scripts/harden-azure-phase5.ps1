param(
    [string]$ResourceGroup = "rg-gtog-prod",
    [string]$Location = "southeastasia",
    [string]$Subscription = "1095803e-80bf-47e0-961f-3d74cb4c605c",
    [string]$StorageAccount = "stgtogprod",
    [string]$SearchService = "srch-gtog-prod",
    [string]$CosmosAccount = "cdb-gtog-prod",
    [string]$CosmosDatabase = "gtog-control",
    [string]$KeyVaultName = "kv-gtog-prod",
    [string]$ManagedIdentityName = "mi-gtog-backend",
    [string]$LogAnalyticsWorkspace = "law-gtog-prod",
    [string]$ActionGroupName = "ag-gtog-prod",
    [string]$AlertEmail = "",
    [switch]$ApplyNetworkRestrictions,
    [switch]$EnablePrivateEndpoints,
    [string]$VnetName = "vnet-gtog-prod",
    [string]$PrivateEndpointSubnetName = "snet-private-endpoints"
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

function Ensure-RoleAssignment {
    param(
        [string]$Scope,
        [string]$RoleName,
        [string]$AssigneeObjectId
    )
    $exists = az role assignment list `
        --scope $Scope `
        --assignee-object-id $AssigneeObjectId `
        --query "[?roleDefinitionName=='$RoleName'] | length(@)" `
        --output tsv
    if ($exists -eq "0") {
        az role assignment create `
            --scope $Scope `
            --role $RoleName `
            --assignee-object-id $AssigneeObjectId `
            --assignee-principal-type ServicePrincipal `
            --output none
    }
}

function Upsert-KeyVaultSecret {
    param(
        [string]$VaultName,
        [string]$SecretName,
        [string]$SecretValue
    )
    if (-not $SecretValue) {
        return
    }
    az keyvault secret set `
        --vault-name $VaultName `
        --name $SecretName `
        --value $SecretValue `
        --output none
}

function Ensure-MetricAlert {
    param(
        [string]$Name,
        [string]$ResourceId,
        [string]$Condition,
        [string]$Description,
        [string]$Severity = "2",
        [string]$WindowSize = "PT5M",
        [string]$EvaluationFrequency = "PT5M",
        [string]$ActionGroupId = ""
    )
    if (-not (Test-AzCommand { az monitor metrics alert show --resource-group $ResourceGroup --name $Name --output none 2>$null })) {
        if ($ActionGroupId) {
            az monitor metrics alert create `
                --resource-group $ResourceGroup `
                --name $Name `
                --scopes $ResourceId `
                --condition $Condition `
                --description $Description `
                --severity $Severity `
                --window-size $WindowSize `
                --evaluation-frequency $EvaluationFrequency `
                --action $ActionGroupId `
                --output none
        } else {
            az monitor metrics alert create `
                --resource-group $ResourceGroup `
                --name $Name `
                --scopes $ResourceId `
                --condition $Condition `
                --description $Description `
                --severity $Severity `
                --window-size $WindowSize `
                --evaluation-frequency $EvaluationFrequency `
                --output none
        }
    }
}

Write-Host ">>> Checking Azure login context..."
if (-not (Test-AzCommand { az account show --output none 2>$null })) {
    throw "Azure CLI is not logged in. Run: az login --use-device-code"
}

Write-Host ">>> Setting subscription: $Subscription"
az account set --subscription $Subscription --output none

$subId = az account show --query id --output tsv
$tenantId = az account show --query tenantId --output tsv
$searchSku = az search service show --name $SearchService --resource-group $ResourceGroup --query "sku.name" --output tsv

Write-Host ""
Write-Host "=========================================="
Write-Host "Phase 5 Recommended Profile"
Write-Host "=========================================="
Write-Host "Managed Identity: enabled (user-assigned)"
Write-Host "Key Vault: enabled with RBAC + purge protection"
Write-Host "Cosmos retry: total=9, backoff_max=30s, status_codes=429,503"
Write-Host "Storage TLS minimum: TLS1_2"
Write-Host "Alerts: Cosmos/Search/Storage metric alerts"
if ($searchSku -eq "free") {
    Write-Host "Azure AI Search sku=free: private endpoint/public network lock is skipped."
    Write-Host "Upgrade to Basic/Standard to fully enforce private networking."
}
Write-Host "=========================================="
Write-Host ""

Write-Host ">>> Ensuring user-assigned managed identity: $ManagedIdentityName"
if (-not (Test-AzCommand { az identity show --name $ManagedIdentityName --resource-group $ResourceGroup --output none 2>$null })) {
    az identity create `
        --name $ManagedIdentityName `
        --resource-group $ResourceGroup `
        --location $Location `
        --output none
}
$miPrincipalId = az identity show `
    --name $ManagedIdentityName `
    --resource-group $ResourceGroup `
    --query principalId `
    --output tsv
$miClientId = az identity show `
    --name $ManagedIdentityName `
    --resource-group $ResourceGroup `
    --query clientId `
    --output tsv

Write-Host ">>> Ensuring Key Vault: $KeyVaultName"
if (-not (Test-AzCommand { az keyvault show --name $KeyVaultName --resource-group $ResourceGroup --output none 2>$null })) {
    az keyvault create `
        --name $KeyVaultName `
        --resource-group $ResourceGroup `
        --location $Location `
        --enable-rbac-authorization true `
        --retention-days 90 `
        --enable-purge-protection true `
        --public-network-access Enabled `
        --output none
}
$keyVaultId = az keyvault show --name $KeyVaultName --resource-group $ResourceGroup --query id --output tsv

Write-Host ">>> Assigning managed identity access to Key Vault secrets"
Ensure-RoleAssignment -Scope $keyVaultId -RoleName "Key Vault Secrets User" -AssigneeObjectId $miPrincipalId

Write-Host ">>> Collecting current service secrets for Key Vault bootstrap"
$storageAccountKey = az storage account keys list `
    --account-name $StorageAccount `
    --resource-group $ResourceGroup `
    --query "[0].value" `
    --output tsv
$storageConnectionString = "DefaultEndpointsProtocol=https;AccountName=$StorageAccount;AccountKey=$storageAccountKey;EndpointSuffix=core.windows.net"

$searchApiKey = az search admin-key show `
    --service-name $SearchService `
    --resource-group $ResourceGroup `
    --query primaryKey `
    --output tsv

$cosmosEndpoint = az cosmosdb show `
    --name $CosmosAccount `
    --resource-group $ResourceGroup `
    --query documentEndpoint `
    --output tsv
$cosmosKey = az cosmosdb keys list `
    --name $CosmosAccount `
    --resource-group $ResourceGroup `
    --query primaryMasterKey `
    --output tsv
$cosmosConnectionString = az cosmosdb keys list `
    --name $CosmosAccount `
    --resource-group $ResourceGroup `
    --type connection-strings `
    --query "connectionStrings[0].connectionString" `
    --output tsv

Write-Host ">>> Upserting Key Vault secrets"
Upsert-KeyVaultSecret -VaultName $KeyVaultName -SecretName "storage-connection-string" -SecretValue $storageConnectionString
Upsert-KeyVaultSecret -VaultName $KeyVaultName -SecretName "storage-account-key" -SecretValue $storageAccountKey
Upsert-KeyVaultSecret -VaultName $KeyVaultName -SecretName "search-api-key" -SecretValue $searchApiKey
Upsert-KeyVaultSecret -VaultName $KeyVaultName -SecretName "cosmos-connection-string" -SecretValue $cosmosConnectionString
Upsert-KeyVaultSecret -VaultName $KeyVaultName -SecretName "cosmos-key" -SecretValue $cosmosKey

Write-Host ">>> Hardening baseline service settings"
az storage account update `
    --name $StorageAccount `
    --resource-group $ResourceGroup `
    --https-only true `
    --min-tls-version TLS1_2 `
    --allow-blob-public-access false `
    --output none
az cosmosdb update `
    --name $CosmosAccount `
    --resource-group $ResourceGroup `
    --enable-automatic-failover true `
    --output none

Write-Host ">>> Ensuring Log Analytics workspace: $LogAnalyticsWorkspace"
if (-not (Test-AzCommand { az monitor log-analytics workspace show --resource-group $ResourceGroup --workspace-name $LogAnalyticsWorkspace --output none 2>$null })) {
    az monitor log-analytics workspace create `
        --resource-group $ResourceGroup `
        --workspace-name $LogAnalyticsWorkspace `
        --location $Location `
        --output none
}

$actionGroupId = ""
if ($AlertEmail) {
    Write-Host ">>> Ensuring action group: $ActionGroupName"
    if (-not (Test-AzCommand { az monitor action-group show --name $ActionGroupName --resource-group $ResourceGroup --output none 2>$null })) {
        az monitor action-group create `
            --name $ActionGroupName `
            --resource-group $ResourceGroup `
            --short-name "gtogops" `
            --action email default $AlertEmail `
            --output none
    }
    $actionGroupId = az monitor action-group show `
        --name $ActionGroupName `
        --resource-group $ResourceGroup `
        --query id `
        --output tsv
}

$cosmosResourceId = "/subscriptions/$subId/resourceGroups/$ResourceGroup/providers/Microsoft.DocumentDB/databaseAccounts/$CosmosAccount"
$searchResourceId = "/subscriptions/$subId/resourceGroups/$ResourceGroup/providers/Microsoft.Search/searchServices/$SearchService"
$storageResourceId = "/subscriptions/$subId/resourceGroups/$ResourceGroup/providers/Microsoft.Storage/storageAccounts/$StorageAccount"

Write-Host ">>> Ensuring metric alerts"
Ensure-MetricAlert `
    -Name "alert-cosmos-ru-high" `
    -ResourceId $cosmosResourceId `
    -Condition "avg `"NormalizedRUConsumption`" > 80" `
    -Description "Cosmos normalized RU consumption high"
Ensure-MetricAlert `
    -Name "alert-cosmos-latency-high" `
    -ResourceId $cosmosResourceId `
    -Condition "avg `"ServerSideLatencyGateway`" > 100" `
    -Description "Cosmos gateway latency high"
Ensure-MetricAlert `
    -Name "alert-search-throttle" `
    -ResourceId $searchResourceId `
    -Condition "avg `"ThrottledSearchQueriesPercentage`" > 1" `
    -Description "Search throttled query percentage high"
Ensure-MetricAlert `
    -Name "alert-search-latency" `
    -ResourceId $searchResourceId `
    -Condition "avg `"SearchLatency`" > 1000" `
    -Description "Search latency high"
Ensure-MetricAlert `
    -Name "alert-storage-availability" `
    -ResourceId $storageResourceId `
    -Condition "avg `"Availability`" < 99.9" `
    -Description "Storage availability dropped below target"

if ($EnablePrivateEndpoints) {
    Write-Host ">>> Ensuring VNet/subnet for private endpoints"
    if (-not (Test-AzCommand { az network vnet show --resource-group $ResourceGroup --name $VnetName --output none 2>$null })) {
        az network vnet create `
            --resource-group $ResourceGroup `
            --name $VnetName `
            --location $Location `
            --address-prefixes "10.20.0.0/16" `
            --subnet-name $PrivateEndpointSubnetName `
            --subnet-prefixes "10.20.1.0/24" `
            --output none
    }
    az network vnet subnet update `
        --resource-group $ResourceGroup `
        --vnet-name $VnetName `
        --name $PrivateEndpointSubnetName `
        --disable-private-endpoint-network-policies true `
        --output none

    $storageId = az storage account show --name $StorageAccount --resource-group $ResourceGroup --query id --output tsv
    $cosmosId = az cosmosdb show --name $CosmosAccount --resource-group $ResourceGroup --query id --output tsv
    $keyVaultResourceId = az keyvault show --name $KeyVaultName --resource-group $ResourceGroup --query id --output tsv

    if (-not (Test-AzCommand { az network private-endpoint show --resource-group $ResourceGroup --name "pe-$StorageAccount-blob" --output none 2>$null })) {
        az network private-endpoint create `
            --resource-group $ResourceGroup `
            --name "pe-$StorageAccount-blob" `
            --location $Location `
            --vnet-name $VnetName `
            --subnet $PrivateEndpointSubnetName `
            --private-connection-resource-id $storageId `
            --group-id blob `
            --connection-name "pec-$StorageAccount-blob" `
            --output none
    }
    if (-not (Test-AzCommand { az network private-endpoint show --resource-group $ResourceGroup --name "pe-$CosmosAccount-sql" --output none 2>$null })) {
        az network private-endpoint create `
            --resource-group $ResourceGroup `
            --name "pe-$CosmosAccount-sql" `
            --location $Location `
            --vnet-name $VnetName `
            --subnet $PrivateEndpointSubnetName `
            --private-connection-resource-id $cosmosId `
            --group-id Sql `
            --connection-name "pec-$CosmosAccount-sql" `
            --output none
    }
    if (-not (Test-AzCommand { az network private-endpoint show --resource-group $ResourceGroup --name "pe-$KeyVaultName-vault" --output none 2>$null })) {
        az network private-endpoint create `
            --resource-group $ResourceGroup `
            --name "pe-$KeyVaultName-vault" `
            --location $Location `
            --vnet-name $VnetName `
            --subnet $PrivateEndpointSubnetName `
            --private-connection-resource-id $keyVaultResourceId `
            --group-id vault `
            --connection-name "pec-$KeyVaultName-vault" `
            --output none
    }

    if ($searchSku -ne "free") {
        $searchId = az search service show --name $SearchService --resource-group $ResourceGroup --query id --output tsv
        if (-not (Test-AzCommand { az network private-endpoint show --resource-group $ResourceGroup --name "pe-$SearchService-search" --output none 2>$null })) {
            az network private-endpoint create `
                --resource-group $ResourceGroup `
                --name "pe-$SearchService-search" `
                --location $Location `
                --vnet-name $VnetName `
                --subnet $PrivateEndpointSubnetName `
                --private-connection-resource-id $searchId `
                --group-id searchService `
                --connection-name "pec-$SearchService-search" `
                --output none
        }
    } else {
        Write-Host "Skipping Search private endpoint because sku=free."
    }
}

if ($ApplyNetworkRestrictions) {
    Write-Host ">>> Applying public network lockdown"
    az storage account update `
        --name $StorageAccount `
        --resource-group $ResourceGroup `
        --public-network-access Disabled `
        --output none
    az cosmosdb update `
        --name $CosmosAccount `
        --resource-group $ResourceGroup `
        --public-network-access DISABLED `
        --is-virtual-network-filter-enabled true `
        --output none
    az keyvault update `
        --name $KeyVaultName `
        --resource-group $ResourceGroup `
        --public-network-access Disabled `
        --output none
    if ($searchSku -ne "free") {
        az search service update `
            --name $SearchService `
            --resource-group $ResourceGroup `
            --public-network-access Disabled `
            --output none
    } else {
        Write-Host "Skipping Search public network lockdown because sku=free."
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Phase 5 baseline hardening complete."
Write-Host "=========================================="
Write-Host "Tenant: $tenantId"
Write-Host "Subscription: $subId"
Write-Host "Managed Identity ClientId: $miClientId"
Write-Host "Key Vault URL: https://$KeyVaultName.vault.azure.net/"
Write-Host ""
Write-Host "Add these to backend/.env for MI + Key Vault:"
Write-Host "AZURE_USE_MANAGED_IDENTITY=true"
Write-Host "AZURE_MANAGED_IDENTITY_CLIENT_ID=$miClientId"
Write-Host "AZURE_KEY_VAULT_URL=https://$KeyVaultName.vault.azure.net/"
Write-Host "AZURE_KEY_VAULT_STORAGE_CONNECTION_STRING_SECRET_NAME=storage-connection-string"
Write-Host "AZURE_KEY_VAULT_SEARCH_API_KEY_SECRET_NAME=search-api-key"
Write-Host "AZURE_KEY_VAULT_COSMOS_CONNECTION_STRING_SECRET_NAME=cosmos-connection-string"
Write-Host "AZURE_KEY_VAULT_COSMOS_KEY_SECRET_NAME=cosmos-key"
Write-Host "AZURE_COSMOS_CONNECTION_TIMEOUT_SECONDS=15"
Write-Host "AZURE_COSMOS_RETRY_TOTAL=9"
Write-Host "AZURE_COSMOS_RETRY_BACKOFF_MAX_SECONDS=30"
Write-Host "AZURE_COSMOS_RETRY_ON_STATUS_CODES=429,503"
