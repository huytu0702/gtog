#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-gtog-prod}"
LOCATION="${LOCATION:-southeastasia}"
SUBSCRIPTION="${SUBSCRIPTION:-1095803e-80bf-47e0-961f-3d74cb4c605c}"
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-stgtogprod}"
SEARCH_SERVICE="${SEARCH_SERVICE:-srch-gtog-prod}"
COSMOS_ACCOUNT="${COSMOS_ACCOUNT:-cdb-gtog-prod}"
KEY_VAULT_NAME="${KEY_VAULT_NAME:-kvgtogp57594}"
MANAGED_IDENTITY_NAME="${MANAGED_IDENTITY_NAME:-mi-gtog-backend}"
LOG_ANALYTICS_WORKSPACE="${LOG_ANALYTICS_WORKSPACE:-law-gtog-prod}"
ACTION_GROUP_NAME="${ACTION_GROUP_NAME:-ag-gtog-prod}"
ALERT_EMAIL="${ALERT_EMAIL:-}"
ENABLE_PRIVATE_ENDPOINTS="${ENABLE_PRIVATE_ENDPOINTS:-false}"
APPLY_NETWORK_RESTRICTIONS="${APPLY_NETWORK_RESTRICTIONS:-false}"
VNET_NAME="${VNET_NAME:-vnet-gtog-prod}"
PRIVATE_ENDPOINT_SUBNET="${PRIVATE_ENDPOINT_SUBNET:-snet-private-endpoints}"

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  export AZURE_CONFIG_DIR="$(pwd)/.azure"
fi
mkdir -p "$AZURE_CONFIG_DIR"

echo ">>> Checking Azure login context"
az account show --output none
az account set --subscription "$SUBSCRIPTION"

SUB_ID="$(az account show --query id -o tsv)"
SEARCH_SKU="$(az search service show --name "$SEARCH_SERVICE" --resource-group "$RESOURCE_GROUP" --query sku.name -o tsv)"

echo ">>> Phase 5 recommended profile"
echo "Managed Identity + Key Vault + Cosmos retry tuning + metric alerts"
if [[ "$SEARCH_SKU" == "free" ]]; then
  echo "NOTE: Search SKU is free. Full private endpoint/network lockdown for Search is skipped."
fi

echo ">>> Ensuring managed identity: $MANAGED_IDENTITY_NAME"
if ! az identity show --name "$MANAGED_IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null; then
  az identity create \
    --name "$MANAGED_IDENTITY_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
fi
MI_PRINCIPAL_ID="$(az identity show --name "$MANAGED_IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" --query principalId -o tsv)"
MI_CLIENT_ID="$(az identity show --name "$MANAGED_IDENTITY_NAME" --resource-group "$RESOURCE_GROUP" --query clientId -o tsv)"

echo ">>> Registering provider Microsoft.KeyVault (idempotent)"
az provider register --namespace Microsoft.KeyVault --wait --output none

echo ">>> Ensuring key vault: $KEY_VAULT_NAME"
if ! az keyvault show --name "$KEY_VAULT_NAME" --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null; then
  az keyvault create \
    --name "$KEY_VAULT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --enable-rbac-authorization true \
    --retention-days 90 \
    --enable-purge-protection true \
    --public-network-access Enabled \
    --output none
fi
KEY_VAULT_ID="$(az keyvault show --name "$KEY_VAULT_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv)"
KEY_VAULT_URL="$(az keyvault show --name "$KEY_VAULT_NAME" --resource-group "$RESOURCE_GROUP" --query properties.vaultUri -o tsv)"

if [[ "$(az role assignment list --scope "$KEY_VAULT_ID" --assignee-object-id "$MI_PRINCIPAL_ID" --query "[?roleDefinitionName=='Key Vault Secrets User'] | length(@)" -o tsv)" == "0" ]]; then
  az role assignment create \
    --scope "$KEY_VAULT_ID" \
    --role "Key Vault Secrets User" \
    --assignee-object-id "$MI_PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal \
    --output none
fi

echo ">>> Upserting Key Vault secrets (requires caller to have Key Vault secrets write role)"
STORAGE_ACCOUNT_KEY="$(az storage account keys list --account-name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" --query "[0].value" -o tsv)"
STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=${STORAGE_ACCOUNT};AccountKey=${STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
SEARCH_API_KEY="$(az search admin-key show --service-name "$SEARCH_SERVICE" --resource-group "$RESOURCE_GROUP" --query primaryKey -o tsv)"
COSMOS_CONNECTION_STRING="$(az cosmosdb keys list --name "$COSMOS_ACCOUNT" --resource-group "$RESOURCE_GROUP" --type connection-strings --query "connectionStrings[0].connectionString" -o tsv)"
COSMOS_KEY="$(az cosmosdb keys list --name "$COSMOS_ACCOUNT" --resource-group "$RESOURCE_GROUP" --query primaryMasterKey -o tsv)"

az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name storage-connection-string --value "$STORAGE_CONNECTION_STRING" --output none
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name storage-account-key --value "$STORAGE_ACCOUNT_KEY" --output none
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name search-api-key --value "$SEARCH_API_KEY" --output none
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name cosmos-connection-string --value "$COSMOS_CONNECTION_STRING" --output none
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name cosmos-key --value "$COSMOS_KEY" --output none

echo ">>> Applying baseline service hardening"
az storage account update \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --https-only true \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false \
  --output none
az cosmosdb update \
  --name "$COSMOS_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --enable-automatic-failover true \
  --output none

if ! az monitor log-analytics workspace show --resource-group "$RESOURCE_GROUP" --workspace-name "$LOG_ANALYTICS_WORKSPACE" --output none 2>/dev/null; then
  az monitor log-analytics workspace create \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LOG_ANALYTICS_WORKSPACE" \
    --location "$LOCATION" \
    --output none
fi

ACTION_GROUP_ID=""
if [[ -n "$ALERT_EMAIL" ]]; then
  if ! az monitor action-group show --name "$ACTION_GROUP_NAME" --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null; then
    az monitor action-group create \
      --name "$ACTION_GROUP_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --short-name "gtogops" \
      --action email default "$ALERT_EMAIL" \
      --output none
  fi
  ACTION_GROUP_ID="$(az monitor action-group show --name "$ACTION_GROUP_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv)"
fi

ensure_alert() {
  local name="$1"
  local scope="$2"
  local condition="$3"
  local description="$4"
  if ! az monitor metrics alert show --resource-group "$RESOURCE_GROUP" --name "$name" --output none 2>/dev/null; then
    if [[ -n "$ACTION_GROUP_ID" ]]; then
      az monitor metrics alert create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$name" \
        --scopes "$scope" \
        --condition "$condition" \
        --description "$description" \
        --severity 2 \
        --window-size PT5M \
        --evaluation-frequency PT5M \
        --action "$ACTION_GROUP_ID" \
        --output none
    else
      az monitor metrics alert create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$name" \
        --scopes "$scope" \
        --condition "$condition" \
        --description "$description" \
        --severity 2 \
        --window-size PT5M \
        --evaluation-frequency PT5M \
        --output none
    fi
  fi
}

COSMOS_RESOURCE_ID="/subscriptions/${SUB_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.DocumentDB/databaseAccounts/${COSMOS_ACCOUNT}"
SEARCH_RESOURCE_ID="/subscriptions/${SUB_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.Search/searchServices/${SEARCH_SERVICE}"
STORAGE_RESOURCE_ID="/subscriptions/${SUB_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.Storage/storageAccounts/${STORAGE_ACCOUNT}"

ensure_alert "alert-cosmos-ru-high" "$COSMOS_RESOURCE_ID" "avg \"NormalizedRUConsumption\" > 80" "Cosmos normalized RU consumption high"
ensure_alert "alert-cosmos-latency-high" "$COSMOS_RESOURCE_ID" "avg \"ServerSideLatencyGateway\" > 100" "Cosmos gateway latency high"
ensure_alert "alert-search-throttle" "$SEARCH_RESOURCE_ID" "avg \"ThrottledSearchQueriesPercentage\" > 1" "Search throttled query percentage high"
ensure_alert "alert-search-latency" "$SEARCH_RESOURCE_ID" "avg \"SearchLatency\" > 1000" "Search latency high"
ensure_alert "alert-storage-availability" "$STORAGE_RESOURCE_ID" "avg \"Availability\" < 99.9" "Storage availability dropped below target"

if [[ "$ENABLE_PRIVATE_ENDPOINTS" == "true" ]]; then
  echo ">>> Ensuring VNet/subnet for private endpoints"
  if ! az network vnet show --resource-group "$RESOURCE_GROUP" --name "$VNET_NAME" --output none 2>/dev/null; then
    az network vnet create \
      --resource-group "$RESOURCE_GROUP" \
      --name "$VNET_NAME" \
      --location "$LOCATION" \
      --address-prefixes "10.20.0.0/16" \
      --subnet-name "$PRIVATE_ENDPOINT_SUBNET" \
      --subnet-prefixes "10.20.1.0/24" \
      --output none
  fi
  az network vnet subnet update \
    --resource-group "$RESOURCE_GROUP" \
    --vnet-name "$VNET_NAME" \
    --name "$PRIVATE_ENDPOINT_SUBNET" \
    --disable-private-endpoint-network-policies true \
    --output none
  echo "Private endpoint creation should be completed with the PowerShell script for full zone-group mapping."
fi

if [[ "$APPLY_NETWORK_RESTRICTIONS" == "true" ]]; then
  echo ">>> Applying public network lockdown"
  az storage account update --name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" --public-network-access Disabled --output none
  az cosmosdb update --name "$COSMOS_ACCOUNT" --resource-group "$RESOURCE_GROUP" --public-network-access DISABLED --is-virtual-network-filter-enabled true --output none
  az keyvault update --name "$KEY_VAULT_NAME" --resource-group "$RESOURCE_GROUP" --public-network-access Disabled --output none
  if [[ "$SEARCH_SKU" != "free" ]]; then
    az search service update --name "$SEARCH_SERVICE" --resource-group "$RESOURCE_GROUP" --public-network-access Disabled --output none
  else
    echo "Skipping Search lockdown because SKU is free."
  fi
fi

echo "=========================================="
echo "Phase 5 baseline hardening complete"
echo "ManagedIdentityClientId=$MI_CLIENT_ID"
echo "KeyVaultUrl=$KEY_VAULT_URL"
echo "=========================================="
