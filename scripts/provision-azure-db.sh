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
ENTITIES_CONTAINER="${ENTITIES_CONTAINER:-entities}"
RELATIONSHIPS_CONTAINER="${RELATIONSHIPS_CONTAINER:-relationships}"
TEXT_UNITS_CONTAINER="${TEXT_UNITS_CONTAINER:-textUnits}"
COMMUNITIES_CONTAINER="${COMMUNITIES_CONTAINER:-communities}"
COMMUNITY_REPORTS_CONTAINER="${COMMUNITY_REPORTS_CONTAINER:-communityReports}"
COVARIATES_CONTAINER="${COVARIATES_CONTAINER:-covariates}"

BLOB_CONTAINERS=("gtog-input" "gtog-output" "gtog-cache" "gtog-logs")
QUEUE_NAMES=("indexing-jobs")

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

echo ">>> Fetching storage account key"
STORAGE_ACCOUNT_KEY="$(
  az storage account keys list \
    --account-name "${STORAGE_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query "[0].value" \
    --output tsv
)"
STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=${STORAGE_ACCOUNT};AccountKey=${STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"

echo ">>> Ensuring blob containers"
for container in "${BLOB_CONTAINERS[@]}"; do
  az storage container create \
    --name "${container}" \
    --connection-string "${STORAGE_CONNECTION_STRING}" \
    --output none
done

echo ">>> Ensuring storage queues"
for queue_name in "${QUEUE_NAMES[@]}"; do
  az storage queue create \
    --name "${queue_name}" \
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

cosmos_account_is_serverless() {
  local capabilities
  capabilities="$(
    az cosmosdb show \
      --name "${COSMOS_ACCOUNT}" \
      --resource-group "${RESOURCE_GROUP}" \
      --query "capabilities[].name" \
      --output tsv 2>/dev/null || true
  )"
  [[ "$capabilities" == *"EnableServerless"* ]]
}

echo ">>> Ensuring Cosmos DB account: ${COSMOS_ACCOUNT}"
if az cosmosdb show --name "${COSMOS_ACCOUNT}" --resource-group "${RESOURCE_GROUP}" --output none 2>/dev/null; then
  if cosmos_account_is_serverless; then
    echo "    Cosmos account already exists and is configured for serverless."
  else
    echo "Cosmos account ${COSMOS_ACCOUNT} already exists but is not configured for serverless; capacity mode cannot be changed in place." >&2
    exit 1
  fi
else
  az cosmosdb create \
    --name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --locations regionName="${LOCATION}" failoverPriority=0 isZoneRedundant=False \
    --kind GlobalDocumentDB \
    --capabilities EnableServerless \
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
    --output none
}

echo ">>> Ensuring Cosmos containers (serverless)"
create_container "${COLLECTIONS_CONTAINER}"
create_container "${DOCUMENTS_CONTAINER}"
create_container "${INDEXING_JOBS_CONTAINER}"
create_container "${JOB_EVENTS_CONTAINER}"
create_container "${ARTIFACT_MANIFEST_CONTAINER}"
create_container "${ENTITIES_CONTAINER}"
create_container "${RELATIONSHIPS_CONTAINER}"
create_container "${TEXT_UNITS_CONTAINER}"
create_container "${COMMUNITIES_CONTAINER}"
create_container "${COMMUNITY_REPORTS_CONTAINER}"
create_container "${COVARIATES_CONTAINER}"

COSMOS_ENDPOINT="$(
  az cosmosdb show \
    --name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query documentEndpoint \
    --output tsv
)"

echo
echo "=========================================="
echo "Provisioning complete."
echo "=========================================="
echo
echo "Add these non-secret values to backend/.env:"
echo "AZURE_STORAGE_ACCOUNT_NAME=\"${STORAGE_ACCOUNT}\""
echo "AZURE_STORAGE_QUEUE_NAME=\"indexing-jobs\""
echo "AZURE_SEARCH_ENDPOINT=\"${SEARCH_ENDPOINT}\""
echo "AZURE_COSMOS_ENDPOINT=\"${COSMOS_ENDPOINT}\""
echo "AZURE_COSMOS_DATABASE_NAME=\"${COSMOS_DATABASE}\""
echo "AZURE_COSMOS_COLLECTIONS_CONTAINER=\"${COLLECTIONS_CONTAINER}\""
echo "AZURE_COSMOS_DOCUMENTS_CONTAINER=\"${DOCUMENTS_CONTAINER}\""
echo "AZURE_COSMOS_INDEXING_JOBS_CONTAINER=\"${INDEXING_JOBS_CONTAINER}\""
echo "AZURE_COSMOS_JOB_EVENTS_CONTAINER=\"${JOB_EVENTS_CONTAINER}\""
echo "AZURE_COSMOS_ARTIFACT_MANIFEST_CONTAINER=\"${ARTIFACT_MANIFEST_CONTAINER}\""
echo "AZURE_COSMOS_ENTITIES_CONTAINER=\"${ENTITIES_CONTAINER}\""
echo "AZURE_COSMOS_RELATIONSHIPS_CONTAINER=\"${RELATIONSHIPS_CONTAINER}\""
echo "AZURE_COSMOS_TEXT_UNITS_CONTAINER=\"${TEXT_UNITS_CONTAINER}\""
echo "AZURE_COSMOS_COMMUNITIES_CONTAINER=\"${COMMUNITIES_CONTAINER}\""
echo "AZURE_COSMOS_COMMUNITY_REPORTS_CONTAINER=\"${COMMUNITY_REPORTS_CONTAINER}\""
echo "AZURE_COSMOS_COVARIATES_CONTAINER=\"${COVARIATES_CONTAINER}\""
echo
echo "Retrieve secret values separately via Azure CLI before writing backend/.env."
echo "Required secret keys: AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_ACCOUNT_KEY, AZURE_SEARCH_API_KEY, AZURE_COSMOS_CONNECTION_STRING, AZURE_COSMOS_KEY"
echo "Search SKU in use: ${SEARCH_SKU}"
echo "Do not commit secrets to git."
