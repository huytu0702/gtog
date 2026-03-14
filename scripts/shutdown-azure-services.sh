#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-gtog-prod}"
SUBSCRIPTION="${SUBSCRIPTION:-1095803e-80bf-47e0-961f-3d74cb4c605c}"
SEARCH_SERVICE="${SEARCH_SERVICE:-srch-gtog-prod}"
COSMOS_ACCOUNT="${COSMOS_ACCOUNT:-cdb-gtog-prod}"
COSMOS_DATABASE="${COSMOS_DATABASE:-gtog-control}"
DISABLE_ALERTS="${DISABLE_ALERTS:-true}"
REDUCE_COSMOS_AUTOSCALE="${REDUCE_COSMOS_AUTOSCALE:-true}"
SCALE_CONTAINER_APPS_TO_ZERO="${SCALE_CONTAINER_APPS_TO_ZERO:-true}"

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  export AZURE_CONFIG_DIR="$(pwd)/.azure"
fi
mkdir -p "$AZURE_CONFIG_DIR"

log() {
  printf '>>> %s\n' "$1"
}

warn() {
  printf 'WARN: %s\n' "$1" >&2
}

detect_python() {
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
    return
  fi

  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  warn "Python interpreter not found"
  return 1
}

require_login() {
  az account show --output none
  az account set --subscription "$SUBSCRIPTION" --output none
}

list_container_apps() {
  az containerapp list \
    --resource-group "$RESOURCE_GROUP" \
    --query "[].name" \
    --output tsv
}

scale_container_apps_to_zero() {
  local apps
  apps="$(list_container_apps)"
  if [[ -z "$apps" ]]; then
    log "No Container Apps found in $RESOURCE_GROUP"
    return
  fi

  while IFS= read -r app; do
    app="${app%$'\r'}"
    [[ -z "$app" ]] && continue
    log "Scaling Container App $app to min=0 max=1"
    az containerapp update \
      --name "$app" \
      --resource-group "$RESOURCE_GROUP" \
      --min-replicas 0 \
      --max-replicas 1 \
      --output none
  done <<< "$apps"
}

list_metric_alerts() {
  az monitor metrics alert list \
    --resource-group "$RESOURCE_GROUP" \
    --query "[].name" \
    --output tsv
}

disable_metric_alerts() {
  local alerts
  alerts="$(list_metric_alerts)"
  if [[ -z "$alerts" ]]; then
    log "No metric alerts found in $RESOURCE_GROUP"
    return
  fi

  while IFS= read -r alert; do
    alert="${alert%$'\r'}"
    [[ -z "$alert" ]] && continue
    log "Disabling metric alert $alert"
    az resource update \
      --resource-group "$RESOURCE_GROUP" \
      --resource-type Microsoft.Insights/metricAlerts \
      --name "$alert" \
      --set properties.enabled=false \
      --output none
  done <<< "$alerts"
}

list_cosmos_containers() {
  az cosmosdb sql container list \
    --account-name "$COSMOS_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --database-name "$COSMOS_DATABASE" \
    --query "[].name" \
    --output tsv
}

get_cosmos_throughput_field() {
  local container="$1"
  local query="$2"
  local value

  value="$(az cosmosdb sql container throughput show \
    --account-name "$COSMOS_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --database-name "$COSMOS_DATABASE" \
    --name "$container" \
    --query "$query" \
    --output tsv 2>/dev/null || true)"

  printf '%s\n' "$value"
}

get_cosmos_min_max_throughput() {
  local container="$1"
  get_cosmos_throughput_field "$container" "resource.minimumThroughput"
}

is_cosmos_autoscale_container() {
  local container="$1"
  local autoscale_max
  autoscale_max="$(get_cosmos_throughput_field "$container" "resource.autoscaleSettings.maxThroughput")"

  [[ -n "$autoscale_max" && "$autoscale_max" != "null" ]]
}

reduce_cosmos_autoscale() {
  local containers
  containers="$(list_cosmos_containers)"
  if [[ -z "$containers" ]]; then
    warn "No Cosmos containers found in $COSMOS_DATABASE"
    return
  fi

  while IFS= read -r container; do
    local minimum_max_throughput
    local target_max_throughput

    container="${container%$'\r'}"
    [[ -z "$container" ]] && continue

    if ! is_cosmos_autoscale_container "$container"; then
      warn "Skipping Cosmos container $container because it is not autoscale-enabled"
      continue
    fi

    minimum_max_throughput="$(get_cosmos_min_max_throughput "$container")"
    target_max_throughput=1000
    if [[ -n "$minimum_max_throughput" && "$minimum_max_throughput" != "null" && "$minimum_max_throughput" -gt "$target_max_throughput" ]]; then
      target_max_throughput="$minimum_max_throughput"
    fi

    log "Reducing Cosmos autoscale max throughput for $container to ${target_max_throughput} RU/s"
    az cosmosdb sql container throughput update \
      --account-name "$COSMOS_ACCOUNT" \
      --resource-group "$RESOURCE_GROUP" \
      --database-name "$COSMOS_DATABASE" \
      --name "$container" \
      --max-throughput "$target_max_throughput" \
      --output none
  done <<< "$containers"
}

get_summary() {
  "$PYTHON_BIN" - <<'PY' "$RESOURCE_GROUP" "$SEARCH_SERVICE" "$COSMOS_ACCOUNT" "$COSMOS_DATABASE"
import json
import subprocess
import sys

rg, search_service, cosmos_account, cosmos_db = sys.argv[1:]

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return json.loads(out)
    except Exception:
        return None

summary = {
    "resourceGroup": rg,
    "containerApps": run([
        "az", "containerapp", "list", "--resource-group", rg,
        "--query", "[].{name:name,minReplicas:template.scale.minReplicas,maxReplicas:template.scale.maxReplicas}",
        "--output", "json"
    ]) or [],
    "metricAlerts": run([
        "az", "monitor", "metrics", "alert", "list", "--resource-group", rg,
        "--query", "[].{name:name,enabled:enabled}", "--output", "json"
    ]) or [],
    "search": run([
        "az", "search", "service", "show", "--name", search_service, "--resource-group", rg,
        "--query", "{name:name,sku:sku.name,replicas:replicaCount,partitions:partitionCount,status:status}", "--output", "json"
    ]),
    "cosmosThroughput": run([
        "az", "cosmosdb", "sql", "container", "list", "--account-name", cosmos_account,
        "--resource-group", rg, "--database-name", cosmos_db,
        "--query", "[].{name:name}", "--output", "json"
    ]) or [],
    "unstoppableResources": [
        "Storage account remains provisioned",
        "Key Vault remains provisioned",
        "Managed Identity remains provisioned",
        "Log Analytics remains provisioned",
        "Azure Container Registry remains provisioned",
        "Azure AI Search free SKU cannot scale to zero",
        "Cosmos DB account remains provisioned; only per-container autoscale max throughput is reduced"
    ]
}

containers = summary["cosmosThroughput"]
for container in containers:
    name = container["name"]
    throughput = run([
        "az", "cosmosdb", "sql", "container", "throughput", "show",
        "--account-name", cosmos_account,
        "--resource-group", rg,
        "--database-name", cosmos_db,
        "--name", name,
        "--query", "resource.{name:'%s',throughput:throughput,maxThroughput:autoscaleSettings.maxThroughput,minimumThroughput:minimumThroughput}" % name,
        "--output", "json"
    ])
    container.update(throughput or {})

print(json.dumps(summary, indent=2))
PY
}

main() {
  log "Checking Azure login context"
  detect_python
  require_login

  if [[ "$SCALE_CONTAINER_APPS_TO_ZERO" == "true" ]]; then
    scale_container_apps_to_zero
  fi

  if [[ "$DISABLE_ALERTS" == "true" ]]; then
    disable_metric_alerts
  fi

  if [[ "$REDUCE_COSMOS_AUTOSCALE" == "true" ]]; then
    reduce_cosmos_autoscale
  fi

  log "Capturing shutdown summary"
  get_summary
}

main "$@"
