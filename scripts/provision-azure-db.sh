#!/usr/bin/env bash
# Provision Azure CosmosDB NoSQL (serverless + vector search) for GraphRAG backend.
# Usage: bash scripts/provision-azure-db.sh

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-gtog-prod}"
LOCATION="${LOCATION:-southeastasia}"
SUBSCRIPTION="${SUBSCRIPTION:-$(az account show --query id --output tsv)}"

STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-stgtogprod}"

COSMOS_ACCOUNT="${COSMOS_ACCOUNT:-cdb-gtog-prod}"
COSMOS_DATABASE="${COSMOS_DATABASE:-gtog-control}"

# Control-plane containers (partition key: /collectionId)
COLLECTIONS_CONTAINER="${COLLECTIONS_CONTAINER:-collections}"
DOCUMENTS_CONTAINER="${DOCUMENTS_CONTAINER:-documents}"
INDEXING_JOBS_CONTAINER="${INDEXING_JOBS_CONTAINER:-indexingJobs}"
JOB_EVENTS_CONTAINER="${JOB_EVENTS_CONTAINER:-jobEvents}"
ARTIFACT_MANIFEST_CONTAINER="${ARTIFACT_MANIFEST_CONTAINER:-artifactManifest}"

# Serving containers (partition key: /collectionId)
ENTITIES_CONTAINER="${ENTITIES_CONTAINER:-entities}"
RELATIONSHIPS_CONTAINER="${RELATIONSHIPS_CONTAINER:-relationships}"
TEXT_UNITS_CONTAINER="${TEXT_UNITS_CONTAINER:-textUnits}"
COMMUNITIES_CONTAINER="${COMMUNITIES_CONTAINER:-communities}"
COMMUNITY_REPORTS_CONTAINER="${COMMUNITY_REPORTS_CONTAINER:-communityReports}"
COVARIATES_CONTAINER="${COVARIATES_CONTAINER:-covariates}"

# Vector store containers (partition key: /id, vector dimension: 3072)
# Names match embeddings_schema keys in settings.yaml
VECTOR_ENTITY_CONTAINER="entity.description"
VECTOR_COMMUNITY_CONTAINER="community.full_content"
VECTOR_TEXT_UNIT_CONTAINER="text_unit.text"
VECTOR_DIMENSION=3072

BLOB_CONTAINERS=("pipeline-input" "pipeline-logs")
QUEUE_NAMES=("indexing-jobs")

# ---------------------------------------------------------------------------

echo ">>> Setting subscription: ${SUBSCRIPTION}"
az account set --subscription "${SUBSCRIPTION}"

echo ">>> Ensuring resource group: ${RESOURCE_GROUP}"
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output none

# ---------------------------------------------------------------------------
# Storage account + blob containers + queues
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# CosmosDB account — serverless + NoSQL vector search
# ---------------------------------------------------------------------------

cosmos_account_is_serverless() {
  local capabilities
  capabilities="$(
    az cosmosdb show \
      --name "${COSMOS_ACCOUNT}" \
      --resource-group "${RESOURCE_GROUP}" \
      --query "capabilities[].name" \
      --output tsv 2>/dev/null || true
  )"
  [[ "${capabilities}" == *"EnableServerless"* ]]
}

echo ">>> Ensuring Cosmos DB account: ${COSMOS_ACCOUNT} (serverless + vector search)"
if az cosmosdb show --name "${COSMOS_ACCOUNT}" --resource-group "${RESOURCE_GROUP}" --output none 2>/dev/null; then
  if cosmos_account_is_serverless; then
    echo "    Cosmos DB account already exists and is configured for serverless."
  else
    echo "ERROR: Cosmos DB account '${COSMOS_ACCOUNT}' exists but is NOT serverless. Capacity mode cannot be changed in place." >&2
    exit 1
  fi
else
  az cosmosdb create \
    --name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --locations regionName="${LOCATION}" failoverPriority=0 isZoneRedundant=False \
    --kind GlobalDocumentDB \
    --capabilities EnableServerless EnableNoSQLVectorSearch \
    --default-consistency-level Session \
    --output none
fi

# ---------------------------------------------------------------------------
# CosmosDB SQL database
# ---------------------------------------------------------------------------

echo ">>> Ensuring Cosmos DB SQL database: ${COSMOS_DATABASE}"
if az cosmosdb sql database show \
  --account-name "${COSMOS_ACCOUNT}" \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${COSMOS_DATABASE}" \
  --output none 2>/dev/null; then
  echo "    Database '${COSMOS_DATABASE}' already exists."
else
  az cosmosdb sql database create \
    --account-name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${COSMOS_DATABASE}" \
    --output none
  echo "    Created database '${COSMOS_DATABASE}'."
fi

# ---------------------------------------------------------------------------
# Helper: create a standard container (partition key: /collectionId)
# ---------------------------------------------------------------------------

ensure_control_container() {
  local container_name="$1"

  if az cosmosdb sql container show \
    --account-name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --database-name "${COSMOS_DATABASE}" \
    --name "${container_name}" \
    --output none 2>/dev/null; then
    echo "    Container '${container_name}' already exists."
    return 0
  fi

  az cosmosdb sql container create \
    --account-name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --database-name "${COSMOS_DATABASE}" \
    --name "${container_name}" \
    --partition-key-path "/collectionId" \
    --output none
  echo "    Created container '${container_name}'."
}

# ---------------------------------------------------------------------------
# Helper: create a vector-search container (partition key: /id, diskANN index)
# ---------------------------------------------------------------------------
# Vector embedding policy and indexing policy are passed as inline JSON.
# diskANN index requires the NoSQL Vector Search capability enabled above.

ensure_vector_container() {
  local container_name="$1"
  local dimension="$2"

  if az cosmosdb sql container show \
    --account-name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --database-name "${COSMOS_DATABASE}" \
    --name "${container_name}" \
    --output none 2>/dev/null; then
    echo "    Vector container '${container_name}' already exists."
    return 0
  fi

  local vector_embedding_policy
  vector_embedding_policy=$(cat <<EOF
{
  "vectorEmbeddings": [
    {
      "path": "/vector",
      "dataType": "float32",
      "distanceFunction": "cosine",
      "dimensions": ${dimension}
    }
  ]
}
EOF
)

  local indexing_policy
  indexing_policy=$(cat <<EOF
{
  "indexingMode": "consistent",
  "automatic": true,
  "includedPaths": [{"path": "/*"}],
  "excludedPaths": [
    {"path": "/_etag/?"},
    {"path": "/vector/*"}
  ],
  "vectorIndexes": [
    {"path": "/vector", "type": "diskANN"}
  ]
}
EOF
)

  az cosmosdb sql container create \
    --account-name "${COSMOS_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --database-name "${COSMOS_DATABASE}" \
    --name "${container_name}" \
    --partition-key-path "/id" \
    --vector-embeddings "${vector_embedding_policy}" \
    --idx "${indexing_policy}" \
    --output none
  echo "    Created vector container '${container_name}' (dim=${dimension}, diskANN)."
}

# ---------------------------------------------------------------------------
# Control-plane + serving containers
# ---------------------------------------------------------------------------

echo ">>> Ensuring control-plane containers (partition key: /collectionId)"
ensure_control_container "${COLLECTIONS_CONTAINER}"
ensure_control_container "${DOCUMENTS_CONTAINER}"
ensure_control_container "${INDEXING_JOBS_CONTAINER}"
ensure_control_container "${JOB_EVENTS_CONTAINER}"
ensure_control_container "${ARTIFACT_MANIFEST_CONTAINER}"

echo ">>> Ensuring serving containers (partition key: /collectionId)"
ensure_control_container "${ENTITIES_CONTAINER}"
ensure_control_container "${RELATIONSHIPS_CONTAINER}"
ensure_control_container "${TEXT_UNITS_CONTAINER}"
ensure_control_container "${COMMUNITIES_CONTAINER}"
ensure_control_container "${COMMUNITY_REPORTS_CONTAINER}"
ensure_control_container "${COVARIATES_CONTAINER}"

# ---------------------------------------------------------------------------
# Vector store containers (match embeddings_schema keys in settings.yaml)
# ---------------------------------------------------------------------------

echo ">>> Ensuring vector store containers (partition key: /id, dim=${VECTOR_DIMENSION}, diskANN)"
ensure_vector_container "${VECTOR_ENTITY_CONTAINER}"    "${VECTOR_DIMENSION}"
ensure_vector_container "${VECTOR_COMMUNITY_CONTAINER}" "${VECTOR_DIMENSION}"
ensure_vector_container "${VECTOR_TEXT_UNIT_CONTAINER}" "${VECTOR_DIMENSION}"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

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
echo "Retrieve secret values separately (do not commit to git):"
echo "  az storage account keys list --account-name ${STORAGE_ACCOUNT} --resource-group ${RESOURCE_GROUP} --query \"[0].value\" -o tsv"
echo "  az cosmosdb keys list --name ${COSMOS_ACCOUNT} --resource-group ${RESOURCE_GROUP} --query primaryMasterKey -o tsv"
