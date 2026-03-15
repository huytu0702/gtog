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
FRONTEND_APP_NAME="${FRONTEND_APP_NAME:-ca-gtog-frontend-prod}"
API_APP_NAME="${API_APP_NAME:-ca-gtog-api-prod}"
WORKER_APP_NAME="${WORKER_APP_NAME:-ca-gtog-worker-prod}"
TUNNEL_APP_NAME="${TUNNEL_APP_NAME:-ca-gtog-tunnel-prod}"
TUNNEL_SECRET_REF_NAME="${TUNNEL_SECRET_REF_NAME:-tunnel-token}"
FRONTEND_IMAGE="${FRONTEND_IMAGE:-}"
API_IMAGE="${API_IMAGE:-}"
WORKER_IMAGE="${WORKER_IMAGE:-}"
TUNNEL_IMAGE="${TUNNEL_IMAGE:-cloudflare/cloudflared:latest}"
FRONTEND_CPU="${FRONTEND_CPU:-1.0}"
FRONTEND_MEMORY="${FRONTEND_MEMORY:-2.0Gi}"
FRONTEND_MIN_REPLICAS="${FRONTEND_MIN_REPLICAS:-1}"
FRONTEND_MAX_REPLICAS="${FRONTEND_MAX_REPLICAS:-2}"
API_CPU="${API_CPU:-1.0}"
API_MEMORY="${API_MEMORY:-2.0Gi}"
API_MIN_REPLICAS="${API_MIN_REPLICAS:-1}"
API_MAX_REPLICAS="${API_MAX_REPLICAS:-2}"
WORKER_CPU="${WORKER_CPU:-1.0}"
WORKER_MEMORY="${WORKER_MEMORY:-2.0Gi}"
WORKER_MIN_REPLICAS="${WORKER_MIN_REPLICAS:-1}"
WORKER_MAX_REPLICAS="${WORKER_MAX_REPLICAS:-1}"
TUNNEL_CPU="${TUNNEL_CPU:-0.5}"
TUNNEL_MEMORY="${TUNNEL_MEMORY:-1.0Gi}"
TUNNEL_MIN_REPLICAS="${TUNNEL_MIN_REPLICAS:-2}"
TUNNEL_MAX_REPLICAS="${TUNNEL_MAX_REPLICAS:-2}"
ROLLOUT_MODE="${ROLLOUT_MODE:-reconcile}"
CANARY_TRAFFIC_PERCENT="${CANARY_TRAFFIC_PERCENT:-10}"
STABLE_TRAFFIC_PERCENT="${STABLE_TRAFFIC_PERCENT:-90}"
ROLLOUT_STATE_FILE="${ROLLOUT_STATE_FILE:-}"
USER_ASSIGNED_IDENTITY_NAME="${USER_ASSIGNED_IDENTITY_NAME:-}"
KEY_VAULT_NAME="${KEY_VAULT_NAME:-}"
TUNNEL_TOKEN="${TUNNEL_TOKEN:-}"
TUNNEL_TOKEN_SECRET_NAME="${TUNNEL_TOKEN_SECRET_NAME:-cloudflare-tunnel-token}"
EDGE_ORIGIN_SECRET="${EDGE_ORIGIN_SECRET:-}"
EDGE_ORIGIN_SECRET_NAME="${EDGE_ORIGIN_SECRET_NAME:-edge-origin-secret}"
CREATE_APPS="${CREATE_APPS:-false}"
CONFIGURE_EASY_AUTH="${CONFIGURE_EASY_AUTH:-false}"
CREATE_ENTRA_APP="${CREATE_ENTRA_APP:-false}"
RESET_ENTRA_CLIENT_SECRET="${RESET_ENTRA_CLIENT_SECRET:-false}"
APP_PUBLIC_HOSTNAME="${APP_PUBLIC_HOSTNAME:-}"
API_PUBLIC_HOSTNAME="${API_PUBLIC_HOSTNAME:-}"
ENTRA_APP_DISPLAY_NAME="${ENTRA_APP_DISPLAY_NAME:-}"
ENTRA_APP_ID="${ENTRA_APP_ID:-}"
ENTRA_TENANT_ID="${ENTRA_TENANT_ID:-}"
ENTRA_ISSUER_URL="${ENTRA_ISSUER_URL:-}"
ENTRA_CLIENT_SECRET="${ENTRA_CLIENT_SECRET:-}"
ENTRA_CLIENT_SECRET_NAME="${ENTRA_CLIENT_SECRET_NAME:-entra-client-secret}"
ENTRA_CLIENT_SECRET_DISPLAY_NAME="${ENTRA_CLIENT_SECRET_DISPLAY_NAME:-${API_APP_NAME}-easy-auth}"
API_APP_ID_URI="${API_APP_ID_URI:-}"
ALLOWED_AUDIENCES="${ALLOWED_AUDIENCES:-}"
ENTRA_SCOPE_NAME="${ENTRA_SCOPE_NAME:-access_as_user}"
ENTRA_SCOPE_ADMIN_CONSENT_DISPLAY_NAME="${ENTRA_SCOPE_ADMIN_CONSENT_DISPLAY_NAME:-Access ${API_APP_NAME}}"
ENTRA_SCOPE_ADMIN_CONSENT_DESCRIPTION="${ENTRA_SCOPE_ADMIN_CONSENT_DESCRIPTION:-Allow the signed-in user to access ${API_APP_NAME}.}"
ENTRA_SCOPE_USER_CONSENT_DISPLAY_NAME="${ENTRA_SCOPE_USER_CONSENT_DISPLAY_NAME:-Access ${API_APP_NAME}}"
ENTRA_SCOPE_USER_CONSENT_DESCRIPTION="${ENTRA_SCOPE_USER_CONSENT_DESCRIPTION:-Allow the app to access ${API_APP_NAME} on your behalf.}"
AAD_LOGIN_PARAMETERS_JSON="${AAD_LOGIN_PARAMETERS_JSON:-}"

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  export AZURE_CONFIG_DIR="$(pwd)/.azure"
fi
mkdir -p "$AZURE_CONFIG_DIR"

ACCOUNT_TENANT_ID=""
IDENTITY_RESOURCE_ID=""
IDENTITY_PRINCIPAL_ID=""
EXPECTED_ALLOWED_AUDIENCES_JSON="[]"
EXPECTED_LOGIN_PARAMETERS_JSON="[]"
EXPECTED_SCOPE_VALUE=""
EXPECTED_CALLBACK_URL=""

bool_true() {
  local value="${1:-}"
  [[ "${value,,}" == "true" ]]
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required for this operation" >&2
    exit 1
  fi
}

require_frontend_runtime_contract_hostnames() {
  require_var APP_PUBLIC_HOSTNAME
  require_var API_PUBLIC_HOSTNAME
}

require_api_runtime_contract_hostname() {
  require_var APP_PUBLIC_HOSTNAME
}

require_easy_auth_hostname() {
  require_var API_PUBLIC_HOSTNAME
}

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

container_app_exists() {
  local app_name="$1"
  az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$app_name" \
    --output none 2>/dev/null
}

ensure_container_app_identity() {
  local app_name="$1"
  if [[ -z "$IDENTITY_RESOURCE_ID" ]]; then
    return 0
  fi

  az containerapp identity assign \
    --resource-group "$RESOURCE_GROUP" \
    --name "$app_name" \
    --user-assigned "$IDENTITY_RESOURCE_ID" \
    --output none
}

csv_to_json_array() {
  local raw="$1"
  RAW="$raw" python - <<'PY'
import json
import os

raw = os.environ.get("RAW", "")
values = [item.strip() for item in raw.split(",") if item.strip()]
print(json.dumps(values))
PY
}

json_array_to_lines() {
  local raw_json="$1"
  RAW_JSON="$raw_json" python - <<'PY'
import json
import os

for item in json.loads(os.environ["RAW_JSON"]):
    print(item)
PY
}

build_default_login_parameters_json() {
  local scope_value="$1"
  SCOPE_VALUE="$scope_value" python - <<'PY'
import json
import os

scope_value = os.environ["SCOPE_VALUE"]
print(json.dumps([f"scope=openid profile email offline_access {scope_value}"]))
PY
}

ensure_frontend_ingress_contract() {
  local frontend_args=(
    containerapp update
    --resource-group "$RESOURCE_GROUP"
    --name "$FRONTEND_APP_NAME"
    --cpu "$FRONTEND_CPU"
    --memory "$FRONTEND_MEMORY"
    --min-replicas "$FRONTEND_MIN_REPLICAS"
    --max-replicas "$FRONTEND_MAX_REPLICAS"
    --set-env-vars "NEXT_PUBLIC_API_BASE_URL=https://${API_PUBLIC_HOSTNAME}" "CORS_ORIGINS=https://${APP_PUBLIC_HOSTNAME}"
    --output none
  )
  if [[ -n "$FRONTEND_IMAGE" ]]; then
    frontend_args+=(--image "$FRONTEND_IMAGE")
  fi
  az "${frontend_args[@]}"

  az containerapp ingress enable \
    --resource-group "$RESOURCE_GROUP" \
    --name "$FRONTEND_APP_NAME" \
    --type internal \
    --target-port 3000 \
    --transport auto \
    --output none
}

ensure_api_ingress_contract() {
  local api_args=(
    containerapp update
    --resource-group "$RESOURCE_GROUP"
    --name "$API_APP_NAME"
    --cpu "$API_CPU"
    --memory "$API_MEMORY"
    --min-replicas "$API_MIN_REPLICAS"
    --max-replicas "$API_MAX_REPLICAS"
    --set-env-vars "APP_ROLE=api" "CORS_ORIGINS=https://${APP_PUBLIC_HOSTNAME}" "REQUIRE_PLATFORM_AUTH=true"
    --output none
  )
  if [[ -n "$API_IMAGE" ]]; then
    api_args+=(--image "$API_IMAGE")
  fi
  az "${api_args[@]}"

  az containerapp ingress enable \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --type internal \
    --target-port 8000 \
    --transport auto \
    --output none
}

ensure_worker_ingress_contract() {
  local worker_args=(
    containerapp update
    --resource-group "$RESOURCE_GROUP"
    --name "$WORKER_APP_NAME"
    --cpu "$WORKER_CPU"
    --memory "$WORKER_MEMORY"
    --min-replicas "$WORKER_MIN_REPLICAS"
    --max-replicas "$WORKER_MAX_REPLICAS"
    --set-env-vars "APP_ROLE=worker"
    --output none
  )
  if [[ -n "$WORKER_IMAGE" ]]; then
    worker_args+=(--image "$WORKER_IMAGE")
  fi
  az "${worker_args[@]}"

  az containerapp ingress disable \
    --resource-group "$RESOURCE_GROUP" \
    --name "$WORKER_APP_NAME" \
    --output none
}

ensure_tunnel_connector_contract() {
  if ! container_app_exists "$TUNNEL_APP_NAME"; then
    return 0
  fi

  if [[ -n "$TUNNEL_TOKEN" ]]; then
    az containerapp secret set \
      --resource-group "$RESOURCE_GROUP" \
      --name "$TUNNEL_APP_NAME" \
      --secrets "${TUNNEL_SECRET_REF_NAME}=${TUNNEL_TOKEN}" \
      --output none
  fi

  az containerapp update \
    --resource-group "$RESOURCE_GROUP" \
    --name "$TUNNEL_APP_NAME" \
    --image "$TUNNEL_IMAGE" \
    --cpu "$TUNNEL_CPU" \
    --memory "$TUNNEL_MEMORY" \
    --min-replicas "$TUNNEL_MIN_REPLICAS" \
    --max-replicas "$TUNNEL_MAX_REPLICAS" \
    --set-env-vars "TUNNEL_TOKEN=secretref:${TUNNEL_SECRET_REF_NAME}" \
    --command "" \
    --args tunnel --no-autoupdate run \
    --output none

  az containerapp ingress disable \
    --resource-group "$RESOURCE_GROUP" \
    --name "$TUNNEL_APP_NAME" \
    --output none
}

write_rollout_state() {
  local rollout_mode="$1"
  local stable_revision="$2"
  local candidate_revision="$3"

  if [[ -z "$ROLLOUT_STATE_FILE" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$ROLLOUT_STATE_FILE")"
  python - "$ROLLOUT_STATE_FILE" "$rollout_mode" "$stable_revision" "$candidate_revision" "$CANARY_TRAFFIC_PERCENT" "$STABLE_TRAFFIC_PERCENT" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
state_path.write_text(
    json.dumps(
        {
            "rollout_mode": sys.argv[2],
            "stable_revision": sys.argv[3],
            "candidate_revision": sys.argv[4],
            "canary_traffic_percent": int(sys.argv[5]),
            "stable_traffic_percent": int(sys.argv[6]),
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
}

latest_revision_name() {
  az containerapp revision list \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --query "sort_by([].{name:name, created:properties.createdTime}, &created)[-1].name" \
    --output tsv
}

stable_revision_name() {
  local candidate_revision="$1"
  local stable_revision
  stable_revision="$({
    az containerapp revision list \
      --resource-group "$RESOURCE_GROUP" \
      --name "$API_APP_NAME" \
      --query "sort_by([?name!='${candidate_revision}'].{name:name, created:properties.createdTime}, &created)[-1].name" \
      --output tsv
  } || true)"
  if [[ -n "$stable_revision" ]]; then
    printf '%s' "$stable_revision"
  else
    printf '%s' "$candidate_revision"
  fi
}

read_rollout_state_field() {
  local field_name="$1"
  if [[ -z "$ROLLOUT_STATE_FILE" || ! -f "$ROLLOUT_STATE_FILE" ]]; then
    echo "ROLLOUT_STATE_FILE is required for ${ROLLOUT_MODE} rollout mode" >&2
    exit 1
  fi

  python - "$ROLLOUT_STATE_FILE" "$field_name" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get(sys.argv[2], ""))
PY
}

validate_rollout_percentages() {
  if ! [[ "$CANARY_TRAFFIC_PERCENT" =~ ^[0-9]+$ ]]; then
    echo "CANARY_TRAFFIC_PERCENT must be an integer between 0 and 100" >&2
    exit 1
  fi

  if ! [[ "$STABLE_TRAFFIC_PERCENT" =~ ^[0-9]+$ ]]; then
    echo "STABLE_TRAFFIC_PERCENT must be an integer between 0 and 100" >&2
    exit 1
  fi

  if (( CANARY_TRAFFIC_PERCENT < 0 || CANARY_TRAFFIC_PERCENT > 100 )); then
    echo "CANARY_TRAFFIC_PERCENT must be between 0 and 100" >&2
    exit 1
  fi

  if (( STABLE_TRAFFIC_PERCENT < 0 || STABLE_TRAFFIC_PERCENT > 100 )); then
    echo "STABLE_TRAFFIC_PERCENT must be between 0 and 100" >&2
    exit 1
  fi

  if (( CANARY_TRAFFIC_PERCENT + STABLE_TRAFFIC_PERCENT != 100 )); then
    echo "CANARY_TRAFFIC_PERCENT and STABLE_TRAFFIC_PERCENT must sum to 100" >&2
    exit 1
  fi
}

apply_canary_traffic_split() {
  local candidate_revision="$1"
  local stable_revision="$2"

  validate_rollout_percentages

  if [[ "$stable_revision" == "$candidate_revision" ]]; then
    echo "No previous stable revision found for canary traffic split" >&2
    exit 1
  fi

  az containerapp revision set-mode \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --mode multiple \
    --output none
  az containerapp revision activate \
    --resource-group "$RESOURCE_GROUP" \
    --revision "$stable_revision" \
    --output none
  az containerapp revision activate \
    --resource-group "$RESOURCE_GROUP" \
    --revision "$candidate_revision" \
    --output none
  az containerapp ingress traffic set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --revision-weight "${stable_revision}=${STABLE_TRAFFIC_PERCENT}" "${candidate_revision}=${CANARY_TRAFFIC_PERCENT}" \
    --output none

  write_rollout_state "canary" "$stable_revision" "$candidate_revision"
}

promote_full_traffic() {
  local stable_revision="$1"
  local candidate_revision="$2"

  az containerapp revision set-mode \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --mode multiple \
    --output none
  az containerapp revision activate \
    --resource-group "$RESOURCE_GROUP" \
    --revision "$candidate_revision" \
    --output none
  az containerapp ingress traffic set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --revision-weight "${candidate_revision}=100" \
    --output none

  write_rollout_state "promote" "$stable_revision" "$candidate_revision"
}

rollback_to_stable() {
  local stable_revision="$1"
  local candidate_revision="$2"

  az containerapp revision set-mode \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --mode multiple \
    --output none
  az containerapp revision activate \
    --resource-group "$RESOURCE_GROUP" \
    --revision "$stable_revision" \
    --output none
  az containerapp ingress traffic set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --revision-weight "${stable_revision}=100" "${candidate_revision}=0" \
    --output none

  write_rollout_state "rollback" "$stable_revision" "$candidate_revision"
}

ensure_entra_app_contract() {
  require_var API_PUBLIC_HOSTNAME

  if [[ -z "$ENTRA_APP_ID" && -n "$ENTRA_APP_DISPLAY_NAME" ]]; then
    ENTRA_APP_ID="$(
      az ad app list \
        --display-name "$ENTRA_APP_DISPLAY_NAME" \
        --query "[?displayName=='${ENTRA_APP_DISPLAY_NAME}'] | [0].appId" \
        --output tsv
    )"
  fi

  if [[ -z "$ENTRA_APP_ID" ]]; then
    if ! bool_true "$CREATE_ENTRA_APP"; then
      echo "ENTRA_APP_ID or ENTRA_APP_DISPLAY_NAME is required unless CREATE_ENTRA_APP=true" >&2
      exit 1
    fi
    require_var ENTRA_APP_DISPLAY_NAME
    echo ">>> Creating Entra app registration: ${ENTRA_APP_DISPLAY_NAME}"
    ENTRA_APP_ID="$(
      az ad app create \
        --display-name "$ENTRA_APP_DISPLAY_NAME" \
        --sign-in-audience AzureADMyOrg \
        --query appId \
        --output tsv
    )"
  fi

  local entra_object_id
  entra_object_id="$(
    az ad app show \
      --id "$ENTRA_APP_ID" \
      --query id \
      --output tsv
  )"

  if [[ -z "$API_APP_ID_URI" ]]; then
    API_APP_ID_URI="api://${ENTRA_APP_ID}"
  fi
  if [[ -z "$ALLOWED_AUDIENCES" ]]; then
    ALLOWED_AUDIENCES="$API_APP_ID_URI"
  fi

  EXPECTED_SCOPE_VALUE="${API_APP_ID_URI}/${ENTRA_SCOPE_NAME}"
  EXPECTED_CALLBACK_URL="https://${API_PUBLIC_HOSTNAME}/.auth/login/aad/callback"
  EXPECTED_ALLOWED_AUDIENCES_JSON="$(csv_to_json_array "$ALLOWED_AUDIENCES")"

  if [[ -n "$AAD_LOGIN_PARAMETERS_JSON" ]]; then
    EXPECTED_LOGIN_PARAMETERS_JSON="$AAD_LOGIN_PARAMETERS_JSON"
  else
    EXPECTED_LOGIN_PARAMETERS_JSON="$(build_default_login_parameters_json "$EXPECTED_SCOPE_VALUE")"
  fi

  echo ">>> Reconciling Entra app registration contract"
  az ad app update \
    --id "$ENTRA_APP_ID" \
    --identifier-uris "$API_APP_ID_URI" \
    --web-home-page-url "https://${API_PUBLIC_HOSTNAME}" \
    --web-redirect-uris "$EXPECTED_CALLBACK_URL" \
    --enable-id-token-issuance true \
    --requested-access-token-version 2 \
    --sign-in-audience AzureADMyOrg \
    --output none

  local current_api_file
  current_api_file="$(mktemp)"
  local patch_file
  patch_file="$(mktemp)"

  az rest \
    --method get \
    --url "https://graph.microsoft.com/v1.0/applications/${entra_object_id}?\$select=api" \
    --output json > "$current_api_file"

  CURRENT_API_FILE="$current_api_file" \
  PATCH_FILE="$patch_file" \
  ENTRA_SCOPE_NAME="$ENTRA_SCOPE_NAME" \
  ENTRA_SCOPE_ADMIN_CONSENT_DISPLAY_NAME="$ENTRA_SCOPE_ADMIN_CONSENT_DISPLAY_NAME" \
  ENTRA_SCOPE_ADMIN_CONSENT_DESCRIPTION="$ENTRA_SCOPE_ADMIN_CONSENT_DESCRIPTION" \
  ENTRA_SCOPE_USER_CONSENT_DISPLAY_NAME="$ENTRA_SCOPE_USER_CONSENT_DISPLAY_NAME" \
  ENTRA_SCOPE_USER_CONSENT_DESCRIPTION="$ENTRA_SCOPE_USER_CONSENT_DESCRIPTION" \
  python - <<'PY'
import json
import os
import uuid

with open(os.environ["CURRENT_API_FILE"], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

api = payload.get("api") or {}
scopes = api.get("oauth2PermissionScopes") or []
value = os.environ["ENTRA_SCOPE_NAME"]
updated = []
found = False

for scope in scopes:
    if scope.get("value") == value:
        merged = dict(scope)
        merged.update(
            {
                "id": scope.get("id") or str(uuid.uuid4()),
                "value": value,
                "type": "User",
                "isEnabled": True,
                "adminConsentDisplayName": os.environ["ENTRA_SCOPE_ADMIN_CONSENT_DISPLAY_NAME"],
                "adminConsentDescription": os.environ["ENTRA_SCOPE_ADMIN_CONSENT_DESCRIPTION"],
                "userConsentDisplayName": os.environ["ENTRA_SCOPE_USER_CONSENT_DISPLAY_NAME"],
                "userConsentDescription": os.environ["ENTRA_SCOPE_USER_CONSENT_DESCRIPTION"],
            }
        )
        updated.append(merged)
        found = True
    else:
        updated.append(scope)

if not found:
    updated.append(
        {
            "id": str(uuid.uuid4()),
            "value": value,
            "type": "User",
            "isEnabled": True,
            "adminConsentDisplayName": os.environ["ENTRA_SCOPE_ADMIN_CONSENT_DISPLAY_NAME"],
            "adminConsentDescription": os.environ["ENTRA_SCOPE_ADMIN_CONSENT_DESCRIPTION"],
            "userConsentDisplayName": os.environ["ENTRA_SCOPE_USER_CONSENT_DISPLAY_NAME"],
            "userConsentDescription": os.environ["ENTRA_SCOPE_USER_CONSENT_DESCRIPTION"],
        }
    )

api["requestedAccessTokenVersion"] = 2
api["oauth2PermissionScopes"] = updated

with open(os.environ["PATCH_FILE"], "w", encoding="utf-8") as handle:
    json.dump({"api": api}, handle)
PY

  az rest \
    --method patch \
    --url "https://graph.microsoft.com/v1.0/applications/${entra_object_id}" \
    --body @"$patch_file" \
    --headers "Content-Type=application/json" \
    --output none

  rm -f "$current_api_file" "$patch_file"

  if ! az ad sp show --id "$ENTRA_APP_ID" --output none 2>/dev/null; then
    echo ">>> Creating Entra service principal for API app"
    az ad sp create --id "$ENTRA_APP_ID" --output none
  fi

  if [[ -z "$ENTRA_TENANT_ID" ]]; then
    ENTRA_TENANT_ID="$ACCOUNT_TENANT_ID"
  fi
  if [[ -z "$ENTRA_ISSUER_URL" ]]; then
    ENTRA_ISSUER_URL="https://login.microsoftonline.com/${ENTRA_TENANT_ID}/v2.0"
  fi

  if [[ -z "$ENTRA_CLIENT_SECRET" ]]; then
    if bool_true "$CREATE_ENTRA_APP" || bool_true "$RESET_ENTRA_CLIENT_SECRET"; then
      echo ">>> Creating or rotating Entra client secret for Easy Auth"
      ENTRA_CLIENT_SECRET="$(
        az ad app credential reset \
          --id "$ENTRA_APP_ID" \
          --append \
          --display-name "$ENTRA_CLIENT_SECRET_DISPLAY_NAME" \
          --query password \
          --output tsv
      )"
    else
      echo "ENTRA_CLIENT_SECRET is required unless CREATE_ENTRA_APP=true or RESET_ENTRA_CLIENT_SECRET=true" >&2
      exit 1
    fi
  fi

  upsert_key_vault_secret "$KEY_VAULT_NAME" "$ENTRA_CLIENT_SECRET_NAME" "$ENTRA_CLIENT_SECRET"
}

configure_easy_auth() {
  if ! container_app_exists "$API_APP_NAME"; then
    echo "API app ${API_APP_NAME} must exist before Easy Auth can be configured" >&2
    exit 1
  fi

  ensure_entra_app_contract

  echo ">>> Storing Easy Auth client secret on API app"
  az containerapp secret set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --secrets "${ENTRA_CLIENT_SECRET_NAME}=${ENTRA_CLIENT_SECRET}" \
    --output none

  echo ">>> Configuring Container Apps authentication"
  az containerapp auth update \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --enabled true \
    --unauthenticated-client-action AllowAnonymous \
    --require-https true \
    --proxy-convention Standard \
    --token-store true \
    --excluded-paths /health /health/readiness \
    --set "identityProviders.azureActiveDirectory.login.loginParameters=${EXPECTED_LOGIN_PARAMETERS_JSON}" \
    --yes \
    --output none

  mapfile -t allowed_audiences < <(json_array_to_lines "$EXPECTED_ALLOWED_AUDIENCES_JSON")
  if [[ ${#allowed_audiences[@]} -eq 0 ]]; then
    echo "At least one allowed audience is required" >&2
    exit 1
  fi

  az containerapp auth microsoft update \
    --resource-group "$RESOURCE_GROUP" \
    --name "$API_APP_NAME" \
    --client-id "$ENTRA_APP_ID" \
    --client-secret-name "$ENTRA_CLIENT_SECRET_NAME" \
    --tenant-id "$ENTRA_TENANT_ID" \
    --issuer "$ENTRA_ISSUER_URL" \
    --allowed-audiences "${allowed_audiences[@]}" \
    --yes \
    --output none
}

verify_phase3_contract() {
  if ! bool_true "$CONFIGURE_EASY_AUTH"; then
    return 0
  fi

  echo ">>> Reading back final auth and ingress state"
  local api_app_json
  local worker_app_json
  local auth_json
  local microsoft_auth_json
  local app_registration_json

  api_app_json="$(
    az containerapp show \
      --resource-group "$RESOURCE_GROUP" \
      --name "$API_APP_NAME" \
      --output json
  )"
  worker_app_json="$(
    az containerapp show \
      --resource-group "$RESOURCE_GROUP" \
      --name "$WORKER_APP_NAME" \
      --output json
  )"
  auth_json="$(
    az containerapp auth show \
      --resource-group "$RESOURCE_GROUP" \
      --name "$API_APP_NAME" \
      --output json
  )"
  microsoft_auth_json="$(
    az containerapp auth microsoft show \
      --resource-group "$RESOURCE_GROUP" \
      --name "$API_APP_NAME" \
      --output json
  )"
  app_registration_json="$(
    az ad app show \
      --id "$ENTRA_APP_ID" \
      --output json
  )"

  API_APP_JSON="$api_app_json" \
  WORKER_APP_JSON="$worker_app_json" \
  AUTH_JSON="$auth_json" \
  MICROSOFT_AUTH_JSON="$microsoft_auth_json" \
  APP_REGISTRATION_JSON="$app_registration_json" \
  EXPECTED_ALLOWED_AUDIENCES_JSON="$EXPECTED_ALLOWED_AUDIENCES_JSON" \
  EXPECTED_LOGIN_PARAMETERS_JSON="$EXPECTED_LOGIN_PARAMETERS_JSON" \
  EXPECTED_CALLBACK_URL="$EXPECTED_CALLBACK_URL" \
  ENTRA_APP_ID="$ENTRA_APP_ID" \
  ENTRA_CLIENT_SECRET_NAME="$ENTRA_CLIENT_SECRET_NAME" \
  ENTRA_ISSUER_URL="$ENTRA_ISSUER_URL" \
  python - <<'PY'
import json
import os
import sys

api_app = json.loads(os.environ["API_APP_JSON"])
worker_app = json.loads(os.environ["WORKER_APP_JSON"])
auth = json.loads(os.environ["AUTH_JSON"])
microsoft_auth = json.loads(os.environ["MICROSOFT_AUTH_JSON"])
app_registration = json.loads(os.environ["APP_REGISTRATION_JSON"])
expected_allowed = json.loads(os.environ["EXPECTED_ALLOWED_AUDIENCES_JSON"])
expected_login_parameters = json.loads(os.environ["EXPECTED_LOGIN_PARAMETERS_JSON"])
expected_callback_url = os.environ["EXPECTED_CALLBACK_URL"]
expected_client_id = os.environ["ENTRA_APP_ID"]
expected_secret_name = os.environ["ENTRA_CLIENT_SECRET_NAME"]
expected_issuer = os.environ["ENTRA_ISSUER_URL"]


def get(obj, *path):
    current = obj
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def ensure(condition, message, errors):
    if not condition:
        errors.append(message)


errors = []

auth_enabled = get(auth, "platform", "enabled")
ensure(auth_enabled is True, "Easy Auth is not enabled on the API app", errors)

unauthenticated_action = get(auth, "globalValidation", "unauthenticatedClientAction")
ensure(
    unauthenticated_action == "AllowAnonymous",
    f"Unexpected unauthenticated action: {unauthenticated_action!r}",
    errors,
)

require_https = get(auth, "httpSettings", "requireHttps")
if require_https is None:
    require_https = get(auth, "httpSettings", "routes", "apiPrefix") is not None
ensure(require_https is True, "Easy Auth HTTPS enforcement is not enabled", errors)

proxy_convention = get(auth, "httpSettings", "forwardProxy", "convention")
if proxy_convention is None:
    proxy_convention = get(auth, "httpSettings", "forwardProxy", "proxyConvention")
ensure(proxy_convention == "Standard", f"Unexpected proxy convention: {proxy_convention!r}", errors)

excluded_paths = get(auth, "globalValidation", "excludedPaths") or []
ensure(
    set(excluded_paths) == {"/health", "/health/readiness"},
    f"Unexpected excluded auth paths: {excluded_paths!r}",
    errors,
)

login_parameters = get(auth, "identityProviders", "azureActiveDirectory", "login", "loginParameters") or []
ensure(
    login_parameters == expected_login_parameters,
    f"Unexpected login parameters: {login_parameters!r}",
    errors,
)

client_id = get(microsoft_auth, "registration", "clientId")
ensure(client_id == expected_client_id, f"Unexpected Entra client ID: {client_id!r}", errors)

client_secret_name = get(microsoft_auth, "registration", "clientSecretSettingName")
ensure(
    client_secret_name == expected_secret_name,
    f"Unexpected Entra client secret setting name: {client_secret_name!r}",
    errors,
)

issuer = get(microsoft_auth, "registration", "openIdIssuer")
ensure(issuer == expected_issuer, f"Unexpected issuer URI: {issuer!r}", errors)

allowed_audiences = get(microsoft_auth, "validation", "allowedAudiences") or []
ensure(
    sorted(allowed_audiences) == sorted(expected_allowed),
    f"Unexpected allowed audiences: {allowed_audiences!r}",
    errors,
)

redirect_uris = get(app_registration, "web", "redirectUris") or []
ensure(
    expected_callback_url in redirect_uris,
    f"Expected callback URL {expected_callback_url!r} not found in app registration redirect URIs: {redirect_uris!r}",
    errors,
)

api_ingress = get(api_app, "properties", "configuration", "ingress") or {}
ensure(api_ingress.get("external") is False, "API app ingress is not internal-only", errors)
ensure(api_ingress.get("targetPort") == 8000, f"Unexpected API target port: {api_ingress.get('targetPort')!r}", errors)

worker_ingress = get(worker_app, "properties", "configuration", "ingress")
ensure(worker_ingress in (None, {}), f"Worker app should not expose ingress: {worker_ingress!r}", errors)

secrets = get(api_app, "properties", "configuration", "secrets") or []
secret_names = sorted(secret.get("name") for secret in secrets if isinstance(secret, dict) and secret.get("name"))
ensure(expected_secret_name in secret_names, f"Missing API secret setting {expected_secret_name!r}", errors)

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    sys.exit(1)
PY

  echo ">>> Phase 3 auth contract verified successfully"
}

echo ">>> Checking Azure login context"
az account show --output none

echo ">>> Ensuring containerapp extension"
az extension add --name containerapp --upgrade --allow-preview true --output none

echo ">>> Setting subscription: ${SUBSCRIPTION}"
az account set --subscription "$SUBSCRIPTION"
ACCOUNT_TENANT_ID="$(az account show --query tenantId --output tsv)"

if [[ "$ROLLOUT_MODE" == "promote" || "$ROLLOUT_MODE" == "rollback" ]]; then
  STABLE_REVISION="$(read_rollout_state_field stable_revision)"
  CANDIDATE_REVISION="$(read_rollout_state_field candidate_revision)"
else
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

  if [[ -n "$USER_ASSIGNED_IDENTITY_NAME" ]]; then
    IDENTITY_RESOURCE_ID="$(
      az identity show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$USER_ASSIGNED_IDENTITY_NAME" \
        --query id \
        --output tsv
    )"
    IDENTITY_PRINCIPAL_ID="$(
      az identity show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$USER_ASSIGNED_IDENTITY_NAME" \
        --query principalId \
        --output tsv
    )"
  fi

  if bool_true "$CREATE_APPS"; then
    if [[ -z "$FRONTEND_IMAGE" ]]; then
      echo "FRONTEND_IMAGE is required when CREATE_APPS=true" >&2
      exit 1
    fi
    if [[ -z "$API_IMAGE" ]]; then
      echo "API_IMAGE is required when CREATE_APPS=true" >&2
      exit 1
    fi
    if [[ -z "$WORKER_IMAGE" ]]; then
      echo "WORKER_IMAGE is required when CREATE_APPS=true" >&2
      exit 1
    fi
    if [[ -z "$APP_PUBLIC_HOSTNAME" ]]; then
      echo "APP_PUBLIC_HOSTNAME is required when CREATE_APPS=true" >&2
      exit 1
    fi
    if [[ -z "$API_PUBLIC_HOSTNAME" ]]; then
      echo "API_PUBLIC_HOSTNAME is required when CREATE_APPS=true" >&2
      exit 1
    fi
    if [[ -z "$TUNNEL_TOKEN" ]]; then
      echo "TUNNEL_TOKEN is required when CREATE_APPS=true" >&2
      exit 1
    fi

    echo ">>> Ensuring frontend app: ${FRONTEND_APP_NAME}"
    if ! container_app_exists "$FRONTEND_APP_NAME"; then
      FRONTEND_ARGS=(
        containerapp create
        --resource-group "$RESOURCE_GROUP"
        --name "$FRONTEND_APP_NAME"
        --environment "$CONTAINER_APP_ENVIRONMENT"
        --image "$FRONTEND_IMAGE"
        --ingress internal
        --target-port 3000
        --transport auto
        --cpu "$FRONTEND_CPU"
        --memory "$FRONTEND_MEMORY"
        --min-replicas "$FRONTEND_MIN_REPLICAS"
        --max-replicas "$FRONTEND_MAX_REPLICAS"
        --env-vars "NEXT_PUBLIC_API_BASE_URL=https://${API_PUBLIC_HOSTNAME}" "CORS_ORIGINS=https://${APP_PUBLIC_HOSTNAME}"
        --output none
      )
      if [[ -n "$IDENTITY_RESOURCE_ID" ]]; then
        FRONTEND_ARGS+=(--user-assigned "$IDENTITY_RESOURCE_ID")
      fi
      az "${FRONTEND_ARGS[@]}"
    fi

    echo ">>> Ensuring API app: ${API_APP_NAME}"
    if ! container_app_exists "$API_APP_NAME"; then
      API_ARGS=(
        containerapp create
        --resource-group "$RESOURCE_GROUP"
        --name "$API_APP_NAME"
        --environment "$CONTAINER_APP_ENVIRONMENT"
        --image "$API_IMAGE"
        --ingress internal
        --target-port 8000
        --transport auto
        --cpu "$API_CPU"
        --memory "$API_MEMORY"
        --min-replicas "$API_MIN_REPLICAS"
        --max-replicas "$API_MAX_REPLICAS"
        --env-vars "APP_ROLE=api" "CORS_ORIGINS=https://${APP_PUBLIC_HOSTNAME}"
        --output none
      )
      if [[ -n "$IDENTITY_RESOURCE_ID" ]]; then
        API_ARGS+=(--user-assigned "$IDENTITY_RESOURCE_ID")
      fi
      az "${API_ARGS[@]}"
    fi

    echo ">>> Ensuring worker app: ${WORKER_APP_NAME}"
    if ! container_app_exists "$WORKER_APP_NAME"; then
      WORKER_ARGS=(
        containerapp create
        --resource-group "$RESOURCE_GROUP"
        --name "$WORKER_APP_NAME"
        --environment "$CONTAINER_APP_ENVIRONMENT"
        --image "$WORKER_IMAGE"
        --cpu "$WORKER_CPU"
        --memory "$WORKER_MEMORY"
        --min-replicas "$WORKER_MIN_REPLICAS"
        --max-replicas "$WORKER_MAX_REPLICAS"
        --env-vars "APP_ROLE=worker"
        --output none
      )
      if [[ -n "$IDENTITY_RESOURCE_ID" ]]; then
        WORKER_ARGS+=(--user-assigned "$IDENTITY_RESOURCE_ID")
      fi
      az "${WORKER_ARGS[@]}"
    fi

    echo ">>> Ensuring tunnel connector app: ${TUNNEL_APP_NAME}"
    if ! container_app_exists "$TUNNEL_APP_NAME"; then
      az containerapp create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$TUNNEL_APP_NAME" \
        --environment "$CONTAINER_APP_ENVIRONMENT" \
        --image "$TUNNEL_IMAGE" \
        --cpu "$TUNNEL_CPU" \
        --memory "$TUNNEL_MEMORY" \
        --min-replicas "$TUNNEL_MIN_REPLICAS" \
        --max-replicas "$TUNNEL_MAX_REPLICAS" \
        --secrets "${TUNNEL_SECRET_REF_NAME}=${TUNNEL_TOKEN}" \
        --env-vars "TUNNEL_TOKEN=secretref:${TUNNEL_SECRET_REF_NAME}" \
        --args tunnel --no-autoupdate run \
        --output none
    fi
  fi

  if container_app_exists "$FRONTEND_APP_NAME"; then
    require_frontend_runtime_contract_hostnames
    echo ">>> Reconciling frontend ingress and runtime contract"
    ensure_container_app_identity "$FRONTEND_APP_NAME"
    ensure_frontend_ingress_contract
  fi

  if container_app_exists "$API_APP_NAME"; then
    require_api_runtime_contract_hostname
    echo ">>> Reconciling API ingress and runtime role"
    ensure_container_app_identity "$API_APP_NAME"
    ensure_api_ingress_contract
  fi

  if container_app_exists "$WORKER_APP_NAME"; then
    echo ">>> Reconciling worker ingress and runtime role"
    ensure_container_app_identity "$WORKER_APP_NAME"
    ensure_worker_ingress_contract
  fi

  if container_app_exists "$TUNNEL_APP_NAME"; then
    echo ">>> Reconciling tunnel connector contract"
    ensure_tunnel_connector_contract
  fi

  if bool_true "$CONFIGURE_EASY_AUTH"; then
    require_easy_auth_hostname
    configure_easy_auth
    verify_phase3_contract
  fi

  CANDIDATE_REVISION="$(latest_revision_name)"
  STABLE_REVISION="$(stable_revision_name "$CANDIDATE_REVISION")"
fi

case "$ROLLOUT_MODE" in
  canary)
    apply_canary_traffic_split "$CANDIDATE_REVISION" "$STABLE_REVISION"
    ;;
  promote)
    promote_full_traffic "$STABLE_REVISION" "$CANDIDATE_REVISION"
    ;;
  rollback)
    rollback_to_stable "$STABLE_REVISION" "$CANDIDATE_REVISION"
    ;;
  reconcile)
    write_rollout_state "reconcile" "$STABLE_REVISION" "$CANDIDATE_REVISION"
    ;;
  *)
    echo "Unsupported ROLLOUT_MODE: $ROLLOUT_MODE" >&2
    exit 1
    ;;
esac

echo

echo "=========================================="
echo "Private-origin ACA provisioning complete."
echo "=========================================="
echo "ACA environment: ${CONTAINER_APP_ENVIRONMENT}"
echo "Default private domain: ${DEFAULT_DOMAIN}"
echo "Private DNS zone: ${PRIVATE_DNS_ZONE}"
if bool_true "$CONFIGURE_EASY_AUTH"; then
  echo "Easy Auth hostname: https://${API_PUBLIC_HOSTNAME}"
  echo "Entra app id: ${ENTRA_APP_ID}"
  echo "Entra audience: ${API_APP_ID_URI}"
fi
echo
if bool_true "$CONFIGURE_EASY_AUTH"; then
  echo "Next validation steps:"
  echo "1. Run scripts/validate-aca-phase3-auth.sh with the same environment inputs."
  echo "2. Verify browser login reaches ${EXPECTED_CALLBACK_URL}."
  echo "3. Verify /.auth/me returns identity after login and /api/* returns 401 when unauthenticated."
  echo "4. Verify wrong-audience and staging tokens are rejected before promotion."
  echo "5. Run docs/runbooks/origin-bypass-verification.md and confirm no FastAPI log entry exists for direct-origin probes."
else
  echo "Next Cloudflare steps:"
  echo "1. Create a remotely managed tunnel for this environment."
  echo "2. Add public hostnames app.<domain> and api.<domain> to the tunnel."
  echo "3. Point app.<domain> to the frontend private origin in ACA."
  echo "4. Point api.<domain> to the API private origin in ACA."
  echo "5. If Easy Auth depends on the public host, set the origin request host header to api.<domain>."
  echo "6. Keep WAF, rate limiting, cache bypass, and optional X-Edge-Secret injection on api.<domain>."
fi
