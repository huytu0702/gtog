#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-gtog-prod}"
LOCATION="${LOCATION:-southeastasia}"
SUBSCRIPTION="${SUBSCRIPTION:-1095803e-80bf-47e0-961f-3d74cb4c605c}"
CONTAINER_APP_ENVIRONMENT="${CONTAINER_APP_ENVIRONMENT:-cae-gtog-prod}"
INFRASTRUCTURE_RESOURCE_GROUP="${INFRASTRUCTURE_RESOURCE_GROUP:-rg-gtog-prod-aca-infra}"
LOG_ANALYTICS_WORKSPACE="${LOG_ANALYTICS_WORKSPACE:-law-gtog-prod}"
VNET_NAME="${VNET_NAME:-vnet-gtog-prod-aca}"
INFRASTRUCTURE_SUBNET_NAME="${INFRASTRUCTURE_SUBNET_NAME:-snet-aca-infra}"
INFRASTRUCTURE_SUBNET_PREFIX="${INFRASTRUCTURE_SUBNET_PREFIX:-10.30.0.0/23}"
PRIVATE_ENDPOINT_SUBNET_NAME="${PRIVATE_ENDPOINT_SUBNET_NAME:-snet-aca-private-endpoints}"
PRIVATE_ENDPOINT_SUBNET_PREFIX="${PRIVATE_ENDPOINT_SUBNET_PREFIX:-10.30.2.0/27}"
PRIVATE_ENDPOINT_NAME="${PRIVATE_ENDPOINT_NAME:-pe-cae-gtog-prod}"
PRIVATE_DNS_ZONE="${PRIVATE_DNS_ZONE:-privatelink.${LOCATION}.azurecontainerapps.io}"
PRIVATE_DNS_LINK_NAME="${PRIVATE_DNS_LINK_NAME:-link-cae-gtog-prod}"
API_APP_NAME="${API_APP_NAME:-ca-gtog-api-prod}"
WORKER_APP_NAME="${WORKER_APP_NAME:-ca-gtog-worker-prod}"
TUNNEL_APP_NAME="${TUNNEL_APP_NAME:-ca-gtog-tunnel-prod}"
API_IMAGE="${API_IMAGE:-}"
WORKER_IMAGE="${WORKER_IMAGE:-}"
TUNNEL_IMAGE="${TUNNEL_IMAGE:-cloudflare/cloudflared:latest}"
USER_ASSIGNED_IDENTITY_NAME="${USER_ASSIGNED_IDENTITY_NAME:-}"
KEY_VAULT_NAME="${KEY_VAULT_NAME:-}"
TUNNEL_TOKEN="${TUNNEL_TOKEN:-}"
TUNNEL_TOKEN_SECRET_NAME="${TUNNEL_TOKEN_SECRET_NAME:-cloudflare-tunnel-token}"
EDGE_ORIGIN_SECRET="${EDGE_ORIGIN_SECRET:-}"
EDGE_ORIGIN_SECRET_NAME="${EDGE_ORIGIN_SECRET_NAME:-edge-origin-secret}"
CREATE_APPS="${CREATE_APPS:-false}"

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  export AZURE_CONFIG_DIR="$(pwd)/.azure"
fi
mkdir -p "$AZURE_CONFIG_DIR"

ensure_subnet() {
  local name="$1"
  local prefix="$2"
  local delegation="$3"

  if az network vnet subnet show \
    --resource-group "$RESOURCE_GROUP" \
    --vnet-name "$VNET_NAME" \
    --name "$name" \
    --output none 2>/dev/null; then
    return 0
  fi

  if [[ -n "$delegation" ]]; then
    az network vnet subnet create \
      --resource-group "$RESOURCE_GROUP" \
      --vnet-name "$VNET_NAME" \
      --name "$name" \
      --address-prefixes "$prefix" \
      --delegations "$delegation" \
      --output none
  else
    az network vnet subnet create \
      --resource-group "$RESOURCE_GROUP" \
      --vnet-name "$VNET_NAME" \
      --name "$name" \
      --address-prefixes "$prefix" \
      --output none
  fi
}

upsert_key_vault_secret() {
  local vault_name="$1"
  local secret_name="$2"
  local secret_value="$3"

  if [[ -z "$vault_name" || -z "$secret_value" ]]; then
    return 0
  fi

  az keyvault secret set \
    --vault-name "$vault_name" \
    --name "$secret_name" \
    --value "$secret_value" \
    --output none
}

echo ">>> Checking Azure login context"
az account show --output none

echo ">>> Ensuring containerapp extension"
az extension add --name containerapp --upgrade --allow-preview true --output none

echo ">>> Setting subscription: ${SUBSCRIPTION}"
az account set --subscription "$SUBSCRIPTION"

echo ">>> Registering providers"
az provider register --namespace Microsoft.App --wait --output none
az provider register --namespace Microsoft.OperationalInsights --wait --output none
az provider register --namespace Microsoft.Network --wait --output none

echo ">>> Ensuring resource group: ${RESOURCE_GROUP}"
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

echo ">>> Ensuring Log Analytics workspace: ${LOG_ANALYTICS_WORKSPACE}"
if ! az monitor log-analytics workspace show \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$LOG_ANALYTICS_WORKSPACE" \
  --output none 2>/dev/null; then
  az monitor log-analytics workspace create \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LOG_ANALYTICS_WORKSPACE" \
    --location "$LOCATION" \
    --output none
fi

WORKSPACE_ID="$(
  az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LOG_ANALYTICS_WORKSPACE" \
    --query customerId \
    --output tsv
)"
WORKSPACE_KEY="$(
  az monitor log-analytics workspace get-shared-keys \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LOG_ANALYTICS_WORKSPACE" \
    --query primarySharedKey \
    --output tsv
)"

echo ">>> Ensuring VNet: ${VNET_NAME}"
if ! az network vnet show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VNET_NAME" \
  --output none 2>/dev/null; then
  az network vnet create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VNET_NAME" \
    --location "$LOCATION" \
    --address-prefixes "10.30.0.0/16" \
    --output none
fi

echo ">>> Ensuring infrastructure subnet"
ensure_subnet "$INFRASTRUCTURE_SUBNET_NAME" "$INFRASTRUCTURE_SUBNET_PREFIX" "Microsoft.App/environments"

echo ">>> Ensuring private endpoint subnet"
ensure_subnet "$PRIVATE_ENDPOINT_SUBNET_NAME" "$PRIVATE_ENDPOINT_SUBNET_PREFIX" ""
az network vnet subnet update \
  --resource-group "$RESOURCE_GROUP" \
  --vnet-name "$VNET_NAME" \
  --name "$PRIVATE_ENDPOINT_SUBNET_NAME" \
  --disable-private-endpoint-network-policies true \
  --output none

INFRA_SUBNET_ID="$(
  az network vnet subnet show \
    --resource-group "$RESOURCE_GROUP" \
    --vnet-name "$VNET_NAME" \
    --name "$INFRASTRUCTURE_SUBNET_NAME" \
    --query id \
    --output tsv
)"
PRIVATE_ENDPOINT_SUBNET_ID="$(
  az network vnet subnet show \
    --resource-group "$RESOURCE_GROUP" \
    --vnet-name "$VNET_NAME" \
    --name "$PRIVATE_ENDPOINT_SUBNET_NAME" \
    --query id \
    --output tsv
)"

echo ">>> Ensuring ACA environment: ${CONTAINER_APP_ENVIRONMENT}"
if ! az containerapp env show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$CONTAINER_APP_ENVIRONMENT" \
  --output none 2>/dev/null; then
  az containerapp env create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_APP_ENVIRONMENT" \
    --location "$LOCATION" \
    --enable-workload-profiles true \
    --infrastructure-resource-group "$INFRASTRUCTURE_RESOURCE_GROUP" \
    --infrastructure-subnet-resource-id "$INFRA_SUBNET_ID" \
    --internal-only true \
    --logs-workspace-id "$WORKSPACE_ID" \
    --logs-workspace-key "$WORKSPACE_KEY" \
    --output none
fi

ENVIRONMENT_ID="$(
  az containerapp env show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_APP_ENVIRONMENT" \
    --query id \
    --output tsv
)"
DEFAULT_DOMAIN="$(
  az containerapp env show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_APP_ENVIRONMENT" \
    --query properties.defaultDomain \
    --output tsv
)"

echo ">>> Disabling ACA public network access"
az rest \
  --method patch \
  --uri "https://management.azure.com${ENVIRONMENT_ID}?api-version=2024-03-01" \
  --body '{"properties":{"publicNetworkAccess":"Disabled"}}' \
  --headers "Content-Type=application/json" \
  --output none

echo ">>> Ensuring private endpoint: ${PRIVATE_ENDPOINT_NAME}"
if ! az network private-endpoint show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$PRIVATE_ENDPOINT_NAME" \
  --output none 2>/dev/null; then
  az network private-endpoint create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$PRIVATE_ENDPOINT_NAME" \
    --location "$LOCATION" \
    --subnet "$PRIVATE_ENDPOINT_SUBNET_ID" \
    --private-connection-resource-id "$ENVIRONMENT_ID" \
    --group-id managedEnvironments \
    --connection-name "${PRIVATE_ENDPOINT_NAME}-connection" \
    --output none
fi

echo ">>> Ensuring private DNS zone: ${PRIVATE_DNS_ZONE}"
if ! az network private-dns zone show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$PRIVATE_DNS_ZONE" \
  --output none 2>/dev/null; then
  az network private-dns zone create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$PRIVATE_DNS_ZONE" \
    --output none
fi

if ! az network private-dns link vnet show \
  --resource-group "$RESOURCE_GROUP" \
  --zone-name "$PRIVATE_DNS_ZONE" \
  --name "$PRIVATE_DNS_LINK_NAME" \
  --output none 2>/dev/null; then
  az network private-dns link vnet create \
    --resource-group "$RESOURCE_GROUP" \
    --zone-name "$PRIVATE_DNS_ZONE" \
    --name "$PRIVATE_DNS_LINK_NAME" \
    --virtual-network "$VNET_NAME" \
    --registration-enabled false \
    --output none
fi

if ! az network private-endpoint dns-zone-group show \
  --resource-group "$RESOURCE_GROUP" \
  --endpoint-name "$PRIVATE_ENDPOINT_NAME" \
  --name default \
  --output none 2>/dev/null; then
  az network private-endpoint dns-zone-group create \
    --resource-group "$RESOURCE_GROUP" \
    --endpoint-name "$PRIVATE_ENDPOINT_NAME" \
    --name default \
    --private-dns-zone "$PRIVATE_DNS_ZONE" \
    --zone-name default \
    --output none
fi

upsert_key_vault_secret "$KEY_VAULT_NAME" "$TUNNEL_TOKEN_SECRET_NAME" "$TUNNEL_TOKEN"
upsert_key_vault_secret "$KEY_VAULT_NAME" "$EDGE_ORIGIN_SECRET_NAME" "$EDGE_ORIGIN_SECRET"

IDENTITY_RESOURCE_ID=""
if [[ -n "$USER_ASSIGNED_IDENTITY_NAME" ]]; then
  IDENTITY_RESOURCE_ID="$(
    az identity show \
      --resource-group "$RESOURCE_GROUP" \
      --name "$USER_ASSIGNED_IDENTITY_NAME" \
      --query id \
      --output tsv
  )"
fi

if [[ "$CREATE_APPS" == "true" ]]; then
  if [[ -z "$API_IMAGE" ]]; then
    echo "API_IMAGE is required when CREATE_APPS=true" >&2
    exit 1
  fi
  if [[ -z "$WORKER_IMAGE" ]]; then
    echo "WORKER_IMAGE is required when CREATE_APPS=true" >&2
    exit 1
  fi
  if [[ -z "$TUNNEL_TOKEN" ]]; then
    echo "TUNNEL_TOKEN is required when CREATE_APPS=true" >&2
    exit 1
  fi

  echo ">>> Ensuring API app: ${API_APP_NAME}"
  if ! az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --output none 2>/dev/null; then
    API_ARGS=(
      containerapp create
      --resource-group "$RESOURCE_GROUP"
      --name "$API_APP_NAME"
      --environment "$CONTAINER_APP_ENVIRONMENT"
      --image "$API_IMAGE"
      --ingress internal
      --target-port 8000
      --transport auto
      --cpu 1.0
      --memory 2.0Gi
      --min-replicas 1
      --max-replicas 2
      --output none
    )
    if [[ -n "$IDENTITY_RESOURCE_ID" ]]; then
      API_ARGS+=(--user-assigned "$IDENTITY_RESOURCE_ID")
    fi
    az "${API_ARGS[@]}"
  fi

  echo ">>> Ensuring worker app: ${WORKER_APP_NAME}"
  if ! az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$WORKER_APP_NAME" \
    --output none 2>/dev/null; then
    WORKER_ARGS=(
      containerapp create
      --resource-group "$RESOURCE_GROUP"
      --name "$WORKER_APP_NAME"
      --environment "$CONTAINER_APP_ENVIRONMENT"
      --image "$WORKER_IMAGE"
      --cpu 1.0
      --memory 2.0Gi
      --min-replicas 1
      --max-replicas 1
      --output none
    )
    if [[ -n "$IDENTITY_RESOURCE_ID" ]]; then
      WORKER_ARGS+=(--user-assigned "$IDENTITY_RESOURCE_ID")
    fi
    az "${WORKER_ARGS[@]}"
  fi

  echo ">>> Ensuring tunnel connector app: ${TUNNEL_APP_NAME}"
  if ! az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$TUNNEL_APP_NAME" \
    --output none 2>/dev/null; then
    az containerapp create \
      --resource-group "$RESOURCE_GROUP" \
      --name "$TUNNEL_APP_NAME" \
      --environment "$CONTAINER_APP_ENVIRONMENT" \
      --image "$TUNNEL_IMAGE" \
      --cpu 0.5 \
      --memory 1.0Gi \
      --min-replicas 2 \
      --max-replicas 2 \
      --secrets "tunnel-token=${TUNNEL_TOKEN}" \
      --env-vars "TUNNEL_TOKEN=secretref:tunnel-token" \
      --command /bin/sh \
      --args -c 'cloudflared tunnel --no-autoupdate run --token "$TUNNEL_TOKEN"' \
      --output none
  fi
fi

echo
echo "=========================================="
echo "Private-origin ACA provisioning complete."
echo "=========================================="
echo "ACA environment: ${CONTAINER_APP_ENVIRONMENT}"
echo "Default private domain: ${DEFAULT_DOMAIN}"
echo "Private DNS zone: ${PRIVATE_DNS_ZONE}"
echo
echo "Next Cloudflare steps:"
echo "1. Create a remotely managed tunnel for this environment."
echo "2. Add public hostname api.<domain> to the tunnel."
echo "3. Point the tunnel service to the API private origin in ACA."
echo "4. If Easy Auth depends on the public host, set the origin request host header to api.<domain>."
echo "5. Keep WAF, rate limiting, cache bypass, and optional X-Edge-Secret injection on api.<domain>."
