#!/usr/bin/env bash
# Start Azurite (Azure Storage emulator) for local GraphRAG backend development.
# Usage: bash scripts/azurite/start-azurite.sh

set -euo pipefail

WAIT_SECONDS="${WAIT_SECONDS:-30}"
BLOB_ENDPOINT="${BLOB_ENDPOINT:-http://127.0.0.1:10000/devstoreaccount1?comp=list}"
CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=<AZURITE_DEFAULT_ACCOUNT_KEY>;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"

echo ">>> Checking Docker availability..."
if ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    echo "Docker is not available. Install Docker (or Docker Desktop) and ensure the engine is running, then retry." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo ">>> Starting Azurite via docker compose..."
docker compose -f docker-compose.azurite.yml up -d

echo ">>> Waiting up to ${WAIT_SECONDS}s for Azurite blob endpoint..."
ready=0
end_ts=$(( $(date +%s) + WAIT_SECONDS ))
while [ "$(date +%s)" -lt "${end_ts}" ]; do
    if command -v curl >/dev/null 2>&1; then
        if curl -fsS -o /dev/null --max-time 3 "${BLOB_ENDPOINT}"; then
            ready=1
            break
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget --spider -q --timeout=3 "${BLOB_ENDPOINT}"; then
            ready=1
            break
        fi
    else
        echo "Neither curl nor wget is installed locally; cannot probe endpoint." >&2
        break
    fi
    sleep 1
done

if [ "${ready}" -ne 1 ]; then
    echo "WARN: Azurite did not respond within ${WAIT_SECONDS} seconds. Check 'docker logs gtog-azurite'." >&2
else
    echo ">>> Azurite is ready on ports 10000 (blob), 10001 (queue), 10002 (table)."
fi

echo ""
echo "Connection string (well-known Azurite credentials):"
echo "${CONNECTION_STRING}"
