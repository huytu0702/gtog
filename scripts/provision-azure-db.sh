#!/usr/bin/env bash
# Provision Azure resources for GraphRAG backend database layer (Phase 1).
# Usage: bash scripts/provision-azure-db.sh

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-gtog-prod}"
LOCATION="${LOCATION:-southeastasia}"
SUBSCRIPTION="${SUBSCRIPTION:-1095803e-80bf-47e0-961f-3d74cb4c605c}"

STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-stgtogprod}"
SEARCH_SERVICE="${SEARCH_SERVICE:-srch-gtog-prod}"
SEARCH_SKU="${SEARCH_SKU:-free}"

COSMOS_ACCOUNT="${COSMOS_ACCOUNT:-cdb-gtog-prod}"
COSMOS_DATABASE="${COSMOS_DATABASE:-gtog-control}"

COLLECTIONS_CONTAINER="${COLLECTIONS_CONTAINER:-collections}"
DOCUMENTS_CONTAINER="${DOCUMENTS_CONTAINER:-documents}"
INDEXING_JOBS_CONTAINER="${INDEXING_JOBS_CONTAINER:-indexingJobs}"
JOB_EVENTS_CONTAINER="${JOB_EVENTS_CONTAINER:-jobEvents}"
ARTIFACT_MANIFEST_CONTAINER="${ARTIFACT_MANIFEST_CONTAINER:-artifactManifest}"

BLOB_CONTAINERS=("gtog-input" "gtog-output" "gtog-cache" "gtog-logs")

echo ">>> Setting subscription: ${SUBSCRIPTION}"
az account set --subscription "${SUBSCRIPTION}"

echo ">>> Ensuring resource group: ${RESOURCE_GROUP}"
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output none

echo ">>> Ensuring storage account: ${STORAGE_ACCOUNT}"
az storage account create \
  --name "${STORAGE_ACCOUNT}" \
  --resource-group "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --allow-blob-public-access false \
  --output none

echo ">>> Fetching storage connection string"
STORAGE_CONNECTION_STRING="$(
  az storage account show-connection-string \
    --name "${STORAGE_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query connectionString \
    --output tsv
)"

echo ">>> Ensuring blob containers"
for container in "${BLOB_CONTAINERS[@]}"; do
  az storage container create \
    --name "${container}" \
    --connection-string "${STORAGE_CONNECTION_STRING}" \
    --output none
done

echo ">>> Ensuring Azure AI Search service: ${SEARCH_SERVICE} (sku=${SEARCH_SKU})"
if az search service show --name "${SEARCH_SERVICE}" --resource-group "${RESOURCE_GROUP}" --output none 2>/dev/null; then
  echo "    Search service already exists, skipping create."
else
  az search service create \
    --name "${SEARCH_SERVICE}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --sku "${SEARCH_SKU}" \
    --output none
fi

SEARCH_ENDPOINT="https://${SEARCH_SERVICE}.search.windows.net"
SEARCH_API_KEY="$(
  az search admin-key show \
    --service-name "${SEARCH_SERVICE}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query primaryKey \
    --output tsv
)"

echo ">>> Ensuring Cosmos DB account: ${COSMOS_ACCOUNT}"
if az cosmosdb show --name "${COSMOS_ACCOUNT}" --resource-group "${RESOURCE_GROUP}" --output none 2>/dev/null; then
  echo "    Cosmos account already exists, skipping create."
else
  az cosmosdb create \
    --name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --locations regionName="${LOCATION}" failoverPriority=0 isZoneRedundant=False \
    --kind GlobalDocumentDB \
    --default-consistency-level Session \
    --output none
fi

echo ">>> Ensuring Cosmos database: ${COSMOS_DATABASE}"
az cosmosdb sql database create \
  --account-name "${COSMOS_ACCOUNT}" \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${COSMOS_DATABASE}" \
  --output none

create_container() {
  local container_name="$1"
  local max_throughput="$2"

  if az cosmosdb sql container show \
    --account-name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --database-name "${COSMOS_DATABASE}" \
    --name "${container_name}" \
    --output none 2>/dev/null; then
    echo "    Container ${container_name} already exists, skipping create."
    return 0
  fi

  az cosmosdb sql container create \
    --account-name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --database-name "${COSMOS_DATABASE}" \
    --name "${container_name}" \
    --partition-key-path "/collectionId" \
    --max-throughput "${max_throughput}" \
    --output none
}

echo ">>> Ensuring Cosmos containers (autoscale throughput)"
create_container "${COLLECTIONS_CONTAINER}" "1000"
create_container "${DOCUMENTS_CONTAINER}" "1000"
create_container "${INDEXING_JOBS_CONTAINER}" "4000"
create_container "${JOB_EVENTS_CONTAINER}" "4000"
create_container "${ARTIFACT_MANIFEST_CONTAINER}" "1000"

COSMOS_ENDPOINT="$(
  az cosmosdb show \
    --name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query documentEndpoint \
    --output tsv
)"
COSMOS_KEY="$(
  az cosmosdb keys list \
    --name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query primaryMasterKey \
    --output tsv
)"
COSMOS_CONNECTION_STRING="$(
  az cosmosdb keys list \
    --name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --type connection-strings \
    --query "connectionStrings[0].connectionString" \
    --output tsv
)"

echo
echo "=========================================="
echo "Provisioning complete."
echo "=========================================="
echo
echo "Add these to backend/.env:"
echo "AZURE_STORAGE_CONNECTION_STRING=\"${STORAGE_CONNECTION_STRING}\""
echo "AZURE_SEARCH_ENDPOINT=\"${SEARCH_ENDPOINT}\""
echo "AZURE_SEARCH_API_KEY=\"${SEARCH_API_KEY}\""
echo "AZURE_COSMOS_CONNECTION_STRING=\"${COSMOS_CONNECTION_STRING}\""
echo "AZURE_COSMOS_ENDPOINT=\"${COSMOS_ENDPOINT}\""
echo "AZURE_COSMOS_KEY=\"${COSMOS_KEY}\""
echo "AZURE_COSMOS_DATABASE_NAME=\"${COSMOS_DATABASE}\""
echo "AZURE_COSMOS_COLLECTIONS_CONTAINER=\"${COLLECTIONS_CONTAINER}\""
echo "AZURE_COSMOS_DOCUMENTS_CONTAINER=\"${DOCUMENTS_CONTAINER}\""
echo "AZURE_COSMOS_INDEXING_JOBS_CONTAINER=\"${INDEXING_JOBS_CONTAINER}\""
echo "AZURE_COSMOS_JOB_EVENTS_CONTAINER=\"${JOB_EVENTS_CONTAINER}\""
echo "AZURE_COSMOS_ARTIFACT_MANIFEST_CONTAINER=\"${ARTIFACT_MANIFEST_CONTAINER}\""
echo
echo "Search SKU in use: ${SEARCH_SKU}"
echo "Do not commit secrets to git."
