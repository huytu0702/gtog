# Phase 5 Backend Validation Report

**Date:** 2026-03-02  
**Environment:** Azure subscription `1095803e-80bf-47e0-961f-3d74cb4c605c`, resource group `rg-gtog-prod`  
**Scope:** Phase 5 security/ops hardening implementation + backend runtime smoke validation

## 1) Implemented Changes

### 1.1 Backend runtime hardening

- Added managed identity + Key Vault bootstrap support:
  - `backend/app/azure_runtime.py`
  - Runtime secret hydration for:
    - `AZURE_STORAGE_CONNECTION_STRING`
    - `AZURE_SEARCH_API_KEY`
    - `AZURE_COSMOS_CONNECTION_STRING`
    - `AZURE_COSMOS_KEY`
    - optional LLM/API keys
- Added Cosmos retry/backoff runtime configuration:
  - `AZURE_COSMOS_CONNECTION_TIMEOUT_SECONDS`
  - `AZURE_COSMOS_RETRY_TOTAL`
  - `AZURE_COSMOS_RETRY_BACKOFF_MAX_SECONDS`
  - `AZURE_COSMOS_RETRY_FIXED_INTERVAL_MS`
  - `AZURE_COSMOS_RETRY_CONNECT`
  - `AZURE_COSMOS_RETRY_READ`
  - `AZURE_COSMOS_RETRY_STATUS`
  - `AZURE_COSMOS_RETRY_ON_STATUS_CODES`
- Updated repositories to support Cosmos auth via:
  - connection string
  - endpoint + key
  - endpoint + managed identity
- Updated storage helper to support blob client creation via:
  - connection string
  - account key fallback
  - managed identity account URL

### 1.2 Config/docs updates

- Updated:
  - `backend/app/config.py`
  - `backend/.env.example`
  - `backend/README.md`
  - `pyproject.toml` (added `azure-keyvault-secrets`)
- Added tests:
  - `backend/tests/unit/test_azure_runtime_phase5.py`

### 1.3 Azure hardening artifacts

- Added script:
  - `scripts/harden-azure-phase5.ps1`
  - `scripts/harden-azure-phase5.sh`
- Script capabilities:
  - managed identity provisioning
  - Key Vault provisioning
  - role assignments
  - baseline security settings (TLS/HTTPS)
  - metric alert provisioning
  - optional private endpoints + network lockdown flags

## 2) Azure Resources Verified

- Managed Identity exists:
  - `mi-gtog-backend`
- Log Analytics workspace exists:
  - `law-gtog-prod`
- Metric alerts exist:
  - `alert-cosmos-ru-high`
  - `alert-cosmos-latency-high`
  - `alert-search-throttle`
  - `alert-search-latency`
  - `alert-storage-availability`
- Storage hardening applied:
  - `minimumTlsVersion = TLS1_2`
  - `enableHttpsTrafficOnly = true`
  - `allowBlobPublicAccess = false`

### Key Vault

- Registered provider `Microsoft.KeyVault` (required for this subscription).
- Key Vault created:
  - `kvgtogp57594`
  - RBAC enabled
  - purge protection enabled
- Roles assigned:
  - `Key Vault Secrets User` for managed identity
  - `Key Vault Secrets Officer` for current user (for secret bootstrap/write)
- Secrets populated:
  - `storage-connection-string`
  - `storage-account-key`
  - `search-api-key`
  - `cosmos-connection-string`
  - `cosmos-key`

## 3) Backend Validation Results

## 3.1 Unit tests

Executed:

```bash
.venv/Scripts/python -m pytest \
  backend/tests/unit/test_azure_runtime_phase5.py \
  backend/tests/unit/services/test_indexing_service_phase1.py \
  backend/tests/unit/services/test_query_service_cosmos_context.py \
  backend/tests/unit/test_search_router.py
```

Result: **11 passed**

## 3.2 Runtime smoke tests (real backend process)

Validated with backend started via:

```bash
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Smoke test outcomes:

- `GET /health` -> `healthy`
- `POST /api/collections` -> created `phase5-smoke-20260302`
- `POST /api/collections/{id}/documents` -> upload success
- `GET /api/collections/{id}/documents` -> returned documents
- `POST /api/collections/{id}/index` -> status `pending`
- `GET /api/collections/{id}/index` after short wait -> status `running`
- cleanup (`DELETE document`, `DELETE collection`) -> success

## 4) Open Items / Constraints

- Azure AI Search is on **Free** SKU:
  - full private endpoint lockdown for Search is intentionally not enforced in this run.
  - for complete private networking posture, upgrade Search SKU then run script flags:
    - `-EnablePrivateEndpoints`
    - `-ApplyNetworkRestrictions`
- Current Key Vault public network access remains `Enabled` to avoid immediate connectivity cutover risk during validation.

## 5) Suggested Next Execution (when ready)

1. Upgrade Azure AI Search from `free` to `basic` or higher.
2. Run:
   - `scripts/harden-azure-phase5.ps1 -EnablePrivateEndpoints -ApplyNetworkRestrictions`
3. Deploy backend in Azure with user-assigned MI:
   - `AZURE_USE_MANAGED_IDENTITY=true`
   - `AZURE_MANAGED_IDENTITY_CLIENT_ID=<mi-client-id>`
   - `AZURE_KEY_VAULT_URL=https://kvgtogp57594.vault.azure.net/`
   - secret-name env vars mapped to existing Key Vault secret names.
