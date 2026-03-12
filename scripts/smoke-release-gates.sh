#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-}"
API_PUBLIC_HOSTNAME="${API_PUBLIC_HOSTNAME:-}"
AUTH_BEARER_TOKEN="${AUTH_BEARER_TOKEN:-}"
RESOURCE_GROUP="${RESOURCE_GROUP:-}"
SUBSCRIPTION="${SUBSCRIPTION:-}"
API_APP_NAME="${API_APP_NAME:-}"
WORKER_APP_NAME="${WORKER_APP_NAME:-}"
TUNNEL_APP_NAME="${TUNNEL_APP_NAME:-}"
EXPECTED_CLIENT_ID="${EXPECTED_CLIENT_ID:-}"
EXPECTED_ISSUER_URL="${EXPECTED_ISSUER_URL:-}"
EXPECTED_ALLOWED_AUDIENCES="${EXPECTED_ALLOWED_AUDIENCES:-}"
WRONG_AUDIENCE_TOKEN="${WRONG_AUDIENCE_TOKEN:-}"
PRODUCTION_REJECTION_TOKEN="${PRODUCTION_REJECTION_TOKEN:-}"
PROBE_ORIGIN_URLS="${PROBE_ORIGIN_URLS:-}"
SMOKE_ARTIFACT_NAME="${SMOKE_ARTIFACT_NAME:-smoke-staging-report}"
PHASE3_VALIDATION_ARTIFACT_NAME="${PHASE3_VALIDATION_ARTIFACT_NAME:-phase3-auth-origin-validation}"
SMOKE_PHASE_LABEL="${SMOKE_PHASE_LABEL:-staging}"
ROLLOUT_STATE_FILE="${ROLLOUT_STATE_FILE:-}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$(pwd)/artifacts/${SMOKE_ARTIFACT_NAME}}"
COLLECTION_ID="${COLLECTION_ID:-smoke-$(date +%s)}"
SAMPLE_QUERY="${SAMPLE_QUERY:-What does this collection contain?}"
TUNNEL_SECRET_REF_NAME="${TUNNEL_SECRET_REF_NAME:-tunnel-token}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required" >&2
    exit 1
  fi
}

json_escape() {
  python - <<'PY'
import json, sys
print(json.dumps(sys.stdin.read().strip()))
PY
}

record_result() {
  local name="$1"
  local status="$2"
  RESULTS+=("{\"name\":$(printf '%s' "$name" | json_escape),\"status\":$(printf '%s' "$status" | json_escape)}")
}

expect_status() {
  local name="$1"
  local expected="$2"
  shift 2
  local status
  status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' "$@")"
  if [[ "$status" != "$expected" ]]; then
    echo "$name expected status $expected, got $status" >&2
    exit 1
  fi
  record_result "$name" "$status"
}

api_json() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local output_file="${4:-}"
  if [[ -n "$body" ]]; then
    curl --silent --show-error --fail \
      -X "$method" \
      -H "Authorization: Bearer ${AUTH_BEARER_TOKEN}" \
      -H 'Content-Type: application/json' \
      -d "$body" \
      "${API_BASE_URL}${path}"
  else
    curl --silent --show-error --fail \
      -X "$method" \
      -H "Authorization: Bearer ${AUTH_BEARER_TOKEN}" \
      "${API_BASE_URL}${path}"
  fi > "$output_file"
}

api_status() {
  local method="$1"
  local path="$2"
  local expected="$3"
  local body="${4:-}"
  local output_file="${5:-/dev/null}"
  local status
  if [[ -n "$body" ]]; then
    status="$(curl --silent --show-error --output "$output_file" --write-out '%{http_code}' \
      -X "$method" \
      -H "Authorization: Bearer ${AUTH_BEARER_TOKEN}" \
      -H 'Content-Type: application/json' \
      -d "$body" \
      "${API_BASE_URL}${path}")"
  else
    status="$(curl --silent --show-error --output "$output_file" --write-out '%{http_code}' \
      -X "$method" \
      -H "Authorization: Bearer ${AUTH_BEARER_TOKEN}" \
      "${API_BASE_URL}${path}")"
  fi
  if [[ "$status" != "$expected" ]]; then
    echo "$method $path expected status $expected, got $status" >&2
    exit 1
  fi
  printf '%s' "$status"
}

mkdir -p "$EVIDENCE_DIR"
RESULTS=()
PHASE3_OUTPUT_FILE="$EVIDENCE_DIR/${PHASE3_VALIDATION_ARTIFACT_NAME}.txt"
SMOKE_REPORT_FILE="$EVIDENCE_DIR/${SMOKE_ARTIFACT_NAME}.json"
UPLOAD_FILE="$EVIDENCE_DIR/smoke-upload.txt"
SSE_FILE="$EVIDENCE_DIR/sse-output.txt"

require_var API_BASE_URL
require_var API_PUBLIC_HOSTNAME
require_var AUTH_BEARER_TOKEN
require_var RESOURCE_GROUP
require_var SUBSCRIPTION
require_var API_APP_NAME
require_var WORKER_APP_NAME
require_var TUNNEL_APP_NAME
require_var EXPECTED_CLIENT_ID
require_var EXPECTED_ISSUER_URL
require_var EXPECTED_ALLOWED_AUDIENCES
require_var WRONG_AUDIENCE_TOKEN
require_var PRODUCTION_REJECTION_TOKEN
require_var PROBE_ORIGIN_URLS

printf 'smoke document for %s\n' "$COLLECTION_ID" > "$UPLOAD_FILE"

expect_status "health" "200" "${API_BASE_URL}/health"
expect_status "readiness" "200" "${API_BASE_URL}/health/readiness"

auth_me_status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' "${API_BASE_URL}/.auth/me")"
case "$auth_me_status" in
  200|401|403) ;;
  *)
    echo "/.auth/me returned unexpected status $auth_me_status" >&2
    exit 1
    ;;
esac
record_result "auth_me" "$auth_me_status"

collections_status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' "${API_BASE_URL}/api/collections")"
if [[ "$collections_status" != "401" ]]; then
  echo "unauthenticated /api/collections expected 401, got $collections_status" >&2
  exit 1
fi
record_result "unauthenticated_collections" "$collections_status"

create_status="$(api_status POST "/api/collections" "201" "{\"name\":\"${COLLECTION_ID}\",\"description\":\"Phase 5 smoke collection\"}" "$EVIDENCE_DIR/create-collection.json")"
record_result "create_collection" "$create_status"

upload_status="$(curl --silent --show-error --output "$EVIDENCE_DIR/upload-document.json" --write-out '%{http_code}' \
  -X POST \
  -H "Authorization: Bearer ${AUTH_BEARER_TOKEN}" \
  -F "file=@${UPLOAD_FILE};type=text/plain" \
  "${API_BASE_URL}/api/collections/${COLLECTION_ID}/documents")"
if [[ "$upload_status" != "201" ]]; then
  echo "upload document expected status 201, got $upload_status" >&2
  exit 1
fi
record_result "upload_document" "$upload_status"

list_status="$(api_status GET "/api/collections/${COLLECTION_ID}/documents" "200" "" "$EVIDENCE_DIR/list-documents.json")"
record_result "list_documents" "$list_status"

index_status="$(api_status POST "/api/collections/${COLLECTION_ID}/index" "202" "" "$EVIDENCE_DIR/start-indexing.json")"
job_id="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["job_id"])' "$EVIDENCE_DIR/start-indexing.json")"
record_result "start_indexing" "$index_status"

job_status="queued"
for _ in $(seq 1 20); do
  job_poll_status="$(api_status GET "/api/index-jobs/${job_id}" "200" "" "$EVIDENCE_DIR/job-status.json")"
  job_status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$EVIDENCE_DIR/job-status.json")"
  case "$job_status" in
    completed)
      record_result "job_status_polling" "$job_poll_status"
      break
      ;;
    failed|cancelled)
      echo "Index job entered terminal failure state: $job_status" >&2
      exit 1
      ;;
  esac
  sleep 5
done
if [[ "$job_status" != "completed" ]]; then
  echo "Index job did not complete within the polling window" >&2
  exit 1
fi

local_status="$(api_status POST "/api/collections/${COLLECTION_ID}/search/local" "200" "{\"query\":\"${SAMPLE_QUERY}\"}" "$EVIDENCE_DIR/local-search.json")"
record_result "query_local" "$local_status"
global_status="$(api_status POST "/api/collections/${COLLECTION_ID}/search/global" "200" "{\"query\":\"${SAMPLE_QUERY}\"}" "$EVIDENCE_DIR/global-search.json")"
record_result "query_global" "$global_status"
tog_status="$(api_status POST "/api/collections/${COLLECTION_ID}/search/tog" "200" "{\"query\":\"${SAMPLE_QUERY}\"}" "$EVIDENCE_DIR/tog-search.json")"
record_result "query_tog" "$tog_status"

encoded_query="$(python -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$SAMPLE_QUERY")"
sse_status="$(curl --silent --show-error --output "$SSE_FILE" --write-out '%{http_code}' --no-buffer \
  -H "Authorization: Bearer ${AUTH_BEARER_TOKEN}" \
  "${API_BASE_URL}/api/collections/${COLLECTION_ID}/search/agent/stream?query=${encoded_query}" \
  --max-time 35)"
if [[ "$sse_status" != "200" ]]; then
  echo "SSE endpoint expected status 200, got $sse_status" >&2
  exit 1
fi
if ! grep -qi "heartbeat\|data:" "$SSE_FILE"; then
  echo "SSE output did not contain expected event data" >&2
  exit 1
fi
record_result "sse" "200"

RESOURCE_GROUP="$RESOURCE_GROUP" \
SUBSCRIPTION="$SUBSCRIPTION" \
API_APP_NAME="$API_APP_NAME" \
WORKER_APP_NAME="$WORKER_APP_NAME" \
TUNNEL_APP_NAME="$TUNNEL_APP_NAME" \
API_PUBLIC_HOSTNAME="$API_PUBLIC_HOSTNAME" \
EXPECTED_CLIENT_ID="$EXPECTED_CLIENT_ID" \
EXPECTED_ISSUER_URL="$EXPECTED_ISSUER_URL" \
EXPECTED_ALLOWED_AUDIENCES="$EXPECTED_ALLOWED_AUDIENCES" \
TEST_ACCESS_TOKEN="$AUTH_BEARER_TOKEN" \
WRONG_AUDIENCE_TOKEN="$WRONG_AUDIENCE_TOKEN" \
PRODUCTION_REJECTION_TOKEN="$PRODUCTION_REJECTION_TOKEN" \
PROBE_ORIGIN_URLS="$PROBE_ORIGIN_URLS" \
TUNNEL_SECRET_REF_NAME="$TUNNEL_SECRET_REF_NAME" \
bash "$(dirname "$0")/validate-aca-phase3-auth.sh" > "$PHASE3_OUTPUT_FILE" 2>&1

RESULTS_JSON="[$(IFS=,; echo "${RESULTS[*]}")]"
export RESULTS_JSON
python - "$SMOKE_REPORT_FILE" "$SMOKE_ARTIFACT_NAME" "$COLLECTION_ID" "$SMOKE_PHASE_LABEL" "$ROLLOUT_STATE_FILE" <<'PY'
import json
import os
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
artifact_name = sys.argv[2]
collection_id = sys.argv[3]
phase_label = sys.argv[4]
rollout_state_file = sys.argv[5]
results = json.loads(os.environ.get("RESULTS_JSON", "[]"))

payload = {
    "artifact": artifact_name,
    "phase": phase_label,
    "collection_id": collection_id,
    "checks": results,
}
if rollout_state_file:
    payload["rollout_state_file"] = rollout_state_file
    rollout_path = Path(rollout_state_file)
    if rollout_path.exists():
        payload["rollout_state"] = json.loads(rollout_path.read_text(encoding="utf-8"))

report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "Smoke report written to $SMOKE_REPORT_FILE"
echo "Phase 3 validation evidence written to $PHASE3_OUTPUT_FILE"
