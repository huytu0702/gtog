#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-gtog-prod}"
SUBSCRIPTION="${SUBSCRIPTION:-1095803e-80bf-47e0-961f-3d74cb4c605c}"
API_APP_NAME="${API_APP_NAME:-ca-gtog-api-prod}"
WORKER_APP_NAME="${WORKER_APP_NAME:-ca-gtog-worker-prod}"
TUNNEL_APP_NAME="${TUNNEL_APP_NAME:-ca-gtog-tunnel-prod}"
TUNNEL_SECRET_REF_NAME="${TUNNEL_SECRET_REF_NAME:-tunnel-token}"
API_PUBLIC_HOSTNAME="${API_PUBLIC_HOSTNAME:-}"
API_HEALTH_URL="${API_HEALTH_URL:-}"
AUTH_ME_URL="${AUTH_ME_URL:-}"
EXPECTED_CLIENT_ID="${EXPECTED_CLIENT_ID:-}"
EXPECTED_ISSUER_URL="${EXPECTED_ISSUER_URL:-}"
EXPECTED_ALLOWED_AUDIENCES="${EXPECTED_ALLOWED_AUDIENCES:-}"
EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS="${EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS:-https://${APP_PUBLIC_HOSTNAME:-}}"
EXPECTED_LOGIN_PARAMETERS_JSON="${EXPECTED_LOGIN_PARAMETERS_JSON:-}"
PROBE_ORIGIN_URLS="${PROBE_ORIGIN_URLS:-}"
ORIGIN_BYPASS_WORKSPACE="${ORIGIN_BYPASS_WORKSPACE:-}"
ORIGIN_BYPASS_LOG_QUERY="${ORIGIN_BYPASS_LOG_QUERY:-}"
TEST_ACCESS_TOKEN="${TEST_ACCESS_TOKEN:-}"
WRONG_AUDIENCE_TOKEN="${WRONG_AUDIENCE_TOKEN:-}"
PRODUCTION_REJECTION_TOKEN="${PRODUCTION_REJECTION_TOKEN:-}"

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

build_default_login_parameters_json() {
  local audience_csv="$1"
  AUDIENCE_CSV="$audience_csv" python - <<'PY'
import json
import os

first = [item.strip() for item in os.environ["AUDIENCE_CSV"].split(",") if item.strip()][0]
print(json.dumps([f"scope=openid profile email offline_access {first}/access_as_user"]))
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

print_check() {
  echo "[phase3] $1"
}

require_var RESOURCE_GROUP
require_var API_APP_NAME
require_var WORKER_APP_NAME
require_var TUNNEL_APP_NAME
require_var API_PUBLIC_HOSTNAME
require_var EXPECTED_CLIENT_ID
require_var EXPECTED_ISSUER_URL
require_var EXPECTED_ALLOWED_AUDIENCES
require_var EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS

if [[ -z "$API_HEALTH_URL" ]]; then
  API_HEALTH_URL="https://${API_PUBLIC_HOSTNAME}/health"
fi
if [[ -z "$AUTH_ME_URL" ]]; then
  AUTH_ME_URL="https://${API_PUBLIC_HOSTNAME}/.auth/me"
fi
EXPECTED_ALLOWED_AUDIENCES_JSON="$(csv_to_json_array "$EXPECTED_ALLOWED_AUDIENCES")"
EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON="$(csv_to_json_array "$EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS")"
if [[ "$EXPECTED_ALLOWED_AUDIENCES_JSON" == "[]" ]]; then
  echo "EXPECTED_ALLOWED_AUDIENCES must contain at least one non-empty audience" >&2
  exit 1
fi
if [[ "$EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON" == "[]" ]]; then
  echo "EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS must contain at least one non-empty URL" >&2
  exit 1
fi
if [[ -z "$EXPECTED_LOGIN_PARAMETERS_JSON" ]]; then
  EXPECTED_LOGIN_PARAMETERS_JSON="$(build_default_login_parameters_json "$EXPECTED_ALLOWED_AUDIENCES")"
fi

print_check "Using subscription ${SUBSCRIPTION}"
az account set --subscription "$SUBSCRIPTION"

print_check "Reading current API app, worker app, tunnel app, and auth settings"
API_APP_JSON="$(az containerapp show --resource-group "$RESOURCE_GROUP" --name "$API_APP_NAME" --output json)"
WORKER_APP_JSON="$(az containerapp show --resource-group "$RESOURCE_GROUP" --name "$WORKER_APP_NAME" --output json)"
TUNNEL_APP_JSON="$(az containerapp show --resource-group "$RESOURCE_GROUP" --name "$TUNNEL_APP_NAME" --output json)"
AUTH_CONFIG_ID="$(API_APP_JSON="$API_APP_JSON" python - <<'PY'
import json
import os
api_app = json.loads(os.environ["API_APP_JSON"])
print(f"{api_app['id']}/authConfigs/current")
PY
)"
AUTH_JSON="$(az rest --method get --uri "https://management.azure.com${AUTH_CONFIG_ID}?api-version=2025-07-01" --output json)"
MICROSOFT_AUTH_JSON="$(az containerapp auth microsoft show --resource-group "$RESOURCE_GROUP" --name "$API_APP_NAME" --output json)"

API_APP_JSON="$API_APP_JSON" \
WORKER_APP_JSON="$WORKER_APP_JSON" \
TUNNEL_APP_JSON="$TUNNEL_APP_JSON" \
AUTH_JSON="$AUTH_JSON" \
MICROSOFT_AUTH_JSON="$MICROSOFT_AUTH_JSON" \
EXPECTED_ALLOWED_AUDIENCES_JSON="$EXPECTED_ALLOWED_AUDIENCES_JSON" \
EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON="$EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON" \
EXPECTED_LOGIN_PARAMETERS_JSON="$EXPECTED_LOGIN_PARAMETERS_JSON" \
EXPECTED_CLIENT_ID="$EXPECTED_CLIENT_ID" \
EXPECTED_ISSUER_URL="$EXPECTED_ISSUER_URL" \
TUNNEL_SECRET_REF_NAME="$TUNNEL_SECRET_REF_NAME" \
python - <<'PY'
import json
import os
import sys

api_app = json.loads(os.environ["API_APP_JSON"])
worker_app = json.loads(os.environ["WORKER_APP_JSON"])
tunnel_app = json.loads(os.environ["TUNNEL_APP_JSON"])
auth = json.loads(os.environ["AUTH_JSON"])
microsoft_auth = json.loads(os.environ["MICROSOFT_AUTH_JSON"])
expected_allowed = json.loads(os.environ["EXPECTED_ALLOWED_AUDIENCES_JSON"])
expected_allowed_external_redirect_urls = json.loads(
    os.environ["EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON"]
)
expected_login_parameters = json.loads(os.environ["EXPECTED_LOGIN_PARAMETERS_JSON"])
expected_client_id = os.environ["EXPECTED_CLIENT_ID"]
expected_issuer_url = os.environ["EXPECTED_ISSUER_URL"]
expected_app_origin = expected_allowed_external_redirect_urls[0]
expected_tunnel_secret_ref_name = os.environ["TUNNEL_SECRET_REF_NAME"]
auth = auth.get("properties", auth)

errors = []

if not auth.get("platform", {}).get("enabled"):
    errors.append("Easy Auth is not enabled")
if auth.get("globalValidation", {}).get("unauthenticatedClientAction") != "AllowAnonymous":
    errors.append("Unauthenticated action is not AllowAnonymous")
if auth.get("httpSettings", {}).get("requireHttps") is not True:
    errors.append("Easy Auth does not require HTTPS")
if auth.get("httpSettings", {}).get("forwardProxy", {}).get("convention") != "Standard":
    errors.append("Forward proxy convention is not Standard")
if sorted(auth.get("globalValidation", {}).get("excludedPaths") or []) != ["/health", "/health/readiness"]:
    errors.append("Excluded paths are not exactly /health and /health/readiness")
if (auth.get("identityProviders", {})
       .get("azureActiveDirectory", {})
       .get("login", {})
       .get("loginParameters") or []) != expected_login_parameters:
    errors.append("Login parameters do not match the expected environment contract")
login = auth.get("login") or {}
if sorted(login.get("allowedExternalRedirectUrls") or []) != sorted(expected_allowed_external_redirect_urls):
    errors.append("Easy Auth allowed external redirect URLs do not match the expected app origin")
identity_provider_names = sorted((auth.get("identityProviders") or {}).keys())
if identity_provider_names != ["azureActiveDirectory"]:
    errors.append("Identity providers do not match the expected AAD-only contract")
if microsoft_auth.get("registration", {}).get("clientId") != expected_client_id:
    errors.append("Configured Entra clientId does not match the expected app registration")
if microsoft_auth.get("registration", {}).get("openIdIssuer") != expected_issuer_url:
    errors.append("Configured issuer URI does not match the expected tenant issuer")
if sorted(microsoft_auth.get("validation", {}).get("allowedAudiences") or []) != sorted(expected_allowed):
    errors.append("Allowed audiences do not match the expected environment-specific values")
ingress = api_app.get("properties", {}).get("configuration", {}).get("ingress", {})
if ingress.get("external") is not False:
    errors.append("API app ingress is not internal-only")
if ingress.get("targetPort") != 8000:
    errors.append("API app target port is not 8000")
cors_policy = ingress.get("corsPolicy") or {}
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
if worker_app.get("properties", {}).get("configuration", {}).get("ingress") not in (None, {}):
    errors.append("Worker app still exposes ingress")

tunnel_config = tunnel_app.get("properties", {}).get("template", {}).get("containers", [])
if len(tunnel_config) != 1:
    errors.append("Tunnel app must define exactly one container")
else:
    tunnel_container = tunnel_config[0]
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

print_check "Checking /.auth/me without forcing a login redirect"
auth_status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' "$AUTH_ME_URL")"
case "$auth_status" in
  200|401|403|404)
    ;;
  *)
    echo "Unexpected /.auth/me status: $auth_status" >&2
    exit 1
    ;;
esac

print_check "Checking unauthenticated API access returns 401 through Easy Auth"
unauth_status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' "https://${API_PUBLIC_HOSTNAME}/api/collections")"
if [[ "$unauth_status" != "401" ]]; then
  echo "Expected unauthenticated /api/* request to return 401, got $unauth_status" >&2
  exit 1
fi

if [[ -n "$TEST_ACCESS_TOKEN" ]]; then
  print_check "Checking authenticated API access with environment token"
  auth_ok_status="$(
    curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
      -H "Authorization: Bearer ${TEST_ACCESS_TOKEN}" \
      "https://${API_PUBLIC_HOSTNAME}/api/collections"
  )"
  if [[ "$auth_ok_status" != "200" ]]; then
    echo "Expected environment token to return 200, got $auth_ok_status" >&2
    exit 1
  fi
fi

if [[ -n "$WRONG_AUDIENCE_TOKEN" ]]; then
  print_check "Checking wrong-audience token rejection"
  wrong_audience_status="$(
    curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
      -H "Authorization: Bearer ${WRONG_AUDIENCE_TOKEN}" \
      "https://${API_PUBLIC_HOSTNAME}/api/collections"
  )"
  if [[ "$wrong_audience_status" != "401" && "$wrong_audience_status" != "403" ]]; then
    echo "Expected wrong-audience token to be rejected, got $wrong_audience_status" >&2
    exit 1
  fi
fi

if [[ -n "$PRODUCTION_REJECTION_TOKEN" ]]; then
  print_check "Checking cross-environment token rejection"
  cross_env_status="$(
    curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
      -H "Authorization: Bearer ${PRODUCTION_REJECTION_TOKEN}" \
      "https://${API_PUBLIC_HOSTNAME}/api/collections"
  )"
  if [[ "$cross_env_status" != "401" && "$cross_env_status" != "403" ]]; then
    echo "Expected cross-environment token to be rejected, got $cross_env_status" >&2
    exit 1
  fi
fi

if [[ -n "$PROBE_ORIGIN_URLS" ]]; then
  print_check "Checking direct-origin public probes"
  while IFS= read -r probe_url; do
    [[ -z "$probe_url" ]] && continue
    probe_status="$(curl --silent --show-error --max-time 10 --output /dev/null --write-out '%{http_code}' "$probe_url" || true)"
    if [[ "$probe_status" =~ ^[2-5][0-9][0-9]$ ]]; then
      echo "Direct-origin probe reached an HTTP handler at $probe_url with status $probe_status" >&2
      exit 1
    fi
  done < <(printf '%s
' "$PROBE_ORIGIN_URLS" | tr ',' '\n')
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

print_check "Phase 3 validation checks completed successfully"
