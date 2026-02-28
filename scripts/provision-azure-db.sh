#!/bin/bash
# provision-azure-db.sh
# Provisions Azure storage and search resources for GraphRAG (gtog)
# Region: southeastasia | Subscription: Azure for Students
#
# Usage: bash scripts/provision-azure-db.sh

set -e

# ── Config ────────────────────────────────────────────────────────────────────
RESOURCE_GROUP="rg-gtog-prod"
LOCATION="southeastasia"
STORAGE_ACCOUNT="stgtogprod"
SEARCH_SERVICE="srch-gtog-prod"
SUBSCRIPTION="1095803e-80bf-47e0-961f-3d74cb4c605c"

CONTAINERS=("gtog-input" "gtog-output" "gtog-cache" "gtog-logs")

# ── Set subscription ──────────────────────────────────────────────────────────
echo ">>> Setting subscription..."
az account set --subscription "$SUBSCRIPTION"

# ── Resource Group ────────────────────────────────────────────────────────────
echo ">>> Creating resource group: $RESOURCE_GROUP..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output table

# ── Storage Account ───────────────────────────────────────────────────────────
echo ">>> Creating storage account: $STORAGE_ACCOUNT..."
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --allow-blob-public-access false \
  --output table

# ── Blob Containers ───────────────────────────────────────────────────────────
echo ">>> Fetching storage connection string..."
CONN_STR=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString \
  --output tsv)

echo ">>> Creating blob containers..."
for CONTAINER in "${CONTAINERS[@]}"; do
  echo "    - $CONTAINER"
  az storage container create \
    --name "$CONTAINER" \
    --connection-string "$CONN_STR" \
    --output none
done

# ── Azure AI Search ───────────────────────────────────────────────────────────
echo ">>> Creating Azure AI Search (Free tier): $SEARCH_SERVICE..."
az search service create \
  --name "$SEARCH_SERVICE" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku free \
  --output table

echo ">>> Fetching AI Search admin key..."
SEARCH_KEY=$(az search admin-key show \
  --service-name "$SEARCH_SERVICE" \
  --resource-group "$RESOURCE_GROUP" \
  --query primaryKey \
  --output tsv)

SEARCH_ENDPOINT="https://${SEARCH_SERVICE}.search.windows.net"

# ── Output ────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Provisioning complete!"
echo "=========================================="
echo ""
echo "Add these to your .env file:"
echo ""
echo "AZURE_STORAGE_CONNECTION_STRING=\"$CONN_STR\""
echo "AZURE_SEARCH_ENDPOINT=\"$SEARCH_ENDPOINT\""
echo "AZURE_SEARCH_API_KEY=\"$SEARCH_KEY\""
echo ""
echo "Keep these secret — do not commit to git."
