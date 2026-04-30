#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-gtog-prod}"
SUBSCRIPTION="${SUBSCRIPTION:-1095803e-80bf-47e0-961f-3d74cb4c605c}"
API_APP_NAME="${API_APP_NAME:-ca-gtog-api-prod}"
WORKER_APP_NAME="${WORKER_APP_NAME:-ca-gtog-worker-prod}"
TUNNEL_APP_NAME="${TUNNEL_APP_NAME:-ca-gtog-tunnel-prod}"
TUNNEL_SECRET_REF_NAME="${TUNNEL_SECRET_REF_NAME:-tunnel-token}"
APP_PUBLIC_HOSTNAME="${APP_PUBLIC_HOSTNAME:-}"
API_PUBLIC_HOSTNAME="${API_PUBLIC_HOSTNAME:-}"
API_HEALTH_URL="${API_HEALTH_URL:-}"
PROBE_ORIGIN_URLS="${PROBE_ORIGIN_URLS:-}"
ORIGIN_BYPASS_WORKSPACE="${ORIGIN_BYPASS_WORKSPACE:-}"
ORIGIN_BYPASS_LOG_QUERY="${ORIGIN_BYPASS_LOG_QUERY:-}"

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  export AZURE_CONFIG_DIR="$(pwd)/.azure"
fi
mkdir -p "$AZURE_CONFIG_DIR"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required for this validation step" >&2
    exit 1
  fi
}

print_check() {
  echo "[phase3] $1"
}

require_var RESOURCE_GROUP
require_var API_APP_NAME
require_var WORKER_APP_NAME
require_var TUNNEL_APP_NAME
require_var APP_PUBLIC_HOSTNAME
require_var API_PUBLIC_HOSTNAME

if [[ -z "$API_HEALTH_URL" ]]; then
  API_HEALTH_URL="https://${API_PUBLIC_HOSTNAME}/health"
fi

print_check "Using subscription ${SUBSCRIPTION}"
az account set --subscription "$SUBSCRIPTION"

print_check "Reading current API app, worker app, and tunnel app state"
API_APP_JSON="$(az containerapp show --resource-group "$RESOURCE_GROUP" --name "$API_APP_NAME" --output json)"
WORKER_APP_JSON="$(az containerapp show --resource-group "$RESOURCE_GROUP" --name "$WORKER_APP_NAME" --output json)"
TUNNEL_APP_JSON="$(az containerapp show --resource-group "$RESOURCE_GROUP" --name "$TUNNEL_APP_NAME" --output json)"

API_APP_JSON="$API_APP_JSON" \
WORKER_APP_JSON="$WORKER_APP_JSON" \
TUNNEL_APP_JSON="$TUNNEL_APP_JSON" \
APP_PUBLIC_HOSTNAME="$APP_PUBLIC_HOSTNAME" \
TUNNEL_SECRET_REF_NAME="$TUNNEL_SECRET_REF_NAME" \
python - <<'PY'
import json
import os
import sys

api_app = json.loads(os.environ["API_APP_JSON"])
worker_app = json.loads(os.environ["WORKER_APP_JSON"])
tunnel_app = json.loads(os.environ["TUNNEL_APP_JSON"])
expected_app_origin = f"https://{os.environ['APP_PUBLIC_HOSTNAME']}"
expected_tunnel_secret_ref_name = os.environ["TUNNEL_SECRET_REF_NAME"]

errors = []

api_ingress = api_app.get("properties", {}).get("configuration", {}).get("ingress", {}) or {}
if api_ingress.get("external") is not False:
    errors.append("API app ingress is not internal-only")
if api_ingress.get("targetPort") != 8000:
    errors.append("API app target port is not 8000")
cors_policy = api_ingress.get("corsPolicy") or {}
if sorted(cors_policy.get("allowedOrigins") or []) != [expected_app_origin]:
    errors.append("Ingress CORS allowed origins do not match the expected app origin")
if cors_policy.get("allowCredentials") is not True:
    errors.append("Ingress CORS allowCredentials is not enabled")
if sorted(cors_policy.get("allowedMethods") or []) != ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]:
    errors.append("Ingress CORS allowed methods do not match the expected browser contract")
if sorted(cors_policy.get("allowedHeaders") or []) != ["*"]:
    errors.append("Ingress CORS allowed headers do not match the expected browser contract")
if cors_policy.get("maxAge") != 600:
    errors.append("Ingress CORS maxAge is not 600")

api_containers = api_app.get("properties", {}).get("template", {}).get("containers", []) or []
if len(api_containers) != 1:
    errors.append("API app must define exactly one container")
else:
    api_env = {
        item.get("name"): item.get("secretRef") or item.get("value")
        for item in api_containers[0].get("env") or []
        if isinstance(item, dict) and item.get("name")
    }
    if api_env.get("APP_ROLE") != "api":
        errors.append("API app APP_ROLE must be api")
    if api_env.get("CORS_ORIGINS") != expected_app_origin:
        errors.append("API app CORS_ORIGINS does not match the expected app origin")
    if api_env.get("REQUIRE_EDGE_AUTH") != "true":
        errors.append("API app REQUIRE_EDGE_AUTH must be true")

worker_ingress = worker_app.get("properties", {}).get("configuration", {}).get("ingress")
if worker_ingress not in (None, {}):
    errors.append("Worker app still exposes ingress")
worker_containers = worker_app.get("properties", {}).get("template", {}).get("containers", []) or []
if len(worker_containers) != 1:
    errors.append("Worker app must define exactly one container")
else:
    worker_env = {
        item.get("name"): item.get("secretRef") or item.get("value")
        for item in worker_containers[0].get("env") or []
        if isinstance(item, dict) and item.get("name")
    }
    if worker_env.get("APP_ROLE") != "worker":
        errors.append("Worker app APP_ROLE must be worker")

tunnel_containers = tunnel_app.get("properties", {}).get("template", {}).get("containers", []) or []
if len(tunnel_containers) != 1:
    errors.append("Tunnel app must define exactly one container")
else:
    tunnel_container = tunnel_containers[0]
    if not tunnel_container.get("image"):
        errors.append("Tunnel app image is not configured")
    command = tunnel_container.get("command")
    args = tunnel_container.get("args") or []
    if command not in (None, []):
        errors.append(f"Unexpected tunnel command: {command!r}")
    if args != ["tunnel", "--no-autoupdate", "run"]:
        errors.append(f"Unexpected tunnel args: {args!r}")
    env = {
        item.get("name"): item.get("secretRef") or item.get("value")
        for item in tunnel_container.get("env") or []
        if isinstance(item, dict) and item.get("name")
    }
    if env.get("TUNNEL_TOKEN") != expected_tunnel_secret_ref_name:
        errors.append(
            f"Tunnel app TUNNEL_TOKEN must reference the {expected_tunnel_secret_ref_name!r} secret"
        )

scale = tunnel_app.get("properties", {}).get("template", {}).get("scale") or {}
if scale.get("minReplicas") != 2:
    errors.append(f"Tunnel app minReplicas is not 2: {scale.get('minReplicas')!r}")
if scale.get("maxReplicas") != 2:
    errors.append(f"Tunnel app maxReplicas is not 2: {scale.get('maxReplicas')!r}")
if tunnel_app.get("properties", {}).get("configuration", {}).get("ingress") not in (None, {}):
    errors.append("Tunnel app should not expose ingress")

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    sys.exit(1)
PY

print_check "Checking public health endpoint through the intended route"
curl --fail --silent --show-error "$API_HEALTH_URL" >/dev/null

if [[ -n "$PROBE_ORIGIN_URLS" ]]; then
  print_check "Checking direct-origin public probes"
  while IFS= read -r probe_url; do
    [[ -z "$probe_url" ]] && continue
    probe_status="$(curl --silent --show-error --max-time 10 --output /dev/null --write-out '%{http_code}' "$probe_url" || true)"
    if [[ "$probe_status" =~ ^[2-5][0-9][0-9]$ ]]; then
      echo "Direct-origin probe reached an HTTP handler at $probe_url with status $probe_status" >&2
      exit 1
    fi
  done < <(printf '%s\n' "$PROBE_ORIGIN_URLS" | tr ',' '\n')
fi

if [[ -n "$ORIGIN_BYPASS_WORKSPACE" || -n "$ORIGIN_BYPASS_LOG_QUERY" ]]; then
  require_var ORIGIN_BYPASS_WORKSPACE
  require_var ORIGIN_BYPASS_LOG_QUERY
  print_check "Running backend log query for origin-bypass evidence"
  az monitor log-analytics query \
    --workspace "$ORIGIN_BYPASS_WORKSPACE" \
    --analytics-query "$ORIGIN_BYPASS_LOG_QUERY" \
    --output table
fi

print_check "Phase 3 private-origin validation checks completed successfully"
