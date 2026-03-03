# docs/2026-03-03-full-deployment-plan.md
# Full Deployment Plan v2 - GToG (ACA + Front Door + WAF)

## Summary

Deploy full stack to Azure using:
- Azure Container Apps for both frontend and backend
- Azure Front Door + WAF for public ingress
- Entra ID (OIDC) for UI + API auth
- Azure DevOps YAML CI/CD

Environment scope: `staging` then `production`

Data layer is kept from completed phases:
- Azure Cosmos DB
- Azure Blob Storage
- Azure AI Search
- Azure Key Vault + Managed Identity

This v2 plan upgrades the original with:
- Backend **origin lock** (anti-bypass WAF)
- `/api/*` **rate limiting**
- Clear auth pattern choice (**BFF-lite recommended**)
- Stronger observability + bypass tests

## Fixed Decisions

1. Runtime platform: Azure Container Apps
2. Environments: Staging + Production
3. CI/CD: Azure DevOps YAML
4. Ingress: Front Door + WAF
5. Auth: Entra ID OIDC for UI + API
6. **Auth pattern: Option A (BFF-lite) is chosen for this release**
7. Traffic profile (first 3 months): small (`<= 50 users/day`)

## Target Architecture (v2)

1. Frontend Container App (Next.js) serves `/*`
2. Backend Container App (FastAPI) serves `/api/*` and health endpoints
3. Azure Front Door + WAF is the only public entry point
4. Front Door enforces:
   - Path routing (`/*` -> FE, `/api/*` -> BE)
   - WAF in blocking mode for production
   - Rate limit policy for `/api/*`
   - **Header injection for backend origin lock**
5. Azure Container Registry stores frontend/backend images
6. Backend reads runtime secrets from Key Vault via Managed Identity
7. Backend connects to Cosmos, Blob, AI Search, and LLM providers
8. Logs/metrics are collected in Log Analytics + Azure Monitor (and App Insights where available)

## Required Application Changes

### Frontend

1. Replace hard-coded API base URL in `frontend/lib/api.ts`.
2. Use `NEXT_PUBLIC_API_BASE_URL` but **point browser calls to the BFF layer** (default `/bff`), not directly to backend `/api`.
3. Implement Entra login on Next.js (recommended: Auth.js / NextAuth.js with Microsoft Entra ID provider).
4. Store session in an HttpOnly + Secure cookie (and/or server-side session store).
5. Add BFF endpoints:
   - `/bff/*` (Next.js route handlers) or server actions that:
     - read the user session,
     - acquire a backend API access token via OBO,
     - call backend `/api/*`,
     - return the response to the browser.
6. Ensure SSR/CSR data fetching uses BFF endpoints only (no direct browser calls to backend).

### Backend

1. Replace open CORS (`*`) with env-based allowlist (`CORS_ORIGINS`)
2. Add readiness endpoint (example: `/health/readiness`) with dependency checks
3. Keep `/health` as liveness-only
4. Add JWT validation for `/api/*` using Entra issuer/audience/scope
5. **NEW (v2): Backend Origin Lock**
   - Reject requests that do not include a Front Door injected header:
     - `X-AFD-Secret: <value>`
   - This must be checked before auth validation.
6. **NEW (v2): Rate-limit fallback**
   - Add a basic per-IP or per-user limiter (defense in depth).

### Auth Pattern Decision (v2)

✅ **Chosen: Option A (recommended): BFF-lite**

- Browser authenticates to **Next.js frontend** using Entra ID (OIDC auth code flow).
- Browser keeps only an **HttpOnly + Secure session cookie** (no bearer token stored/exposed to browser JS).
- Next.js server (BFF-lite) calls the backend API **server-to-server** and attaches a **user access token**.
  - Recommended implementation: **On-Behalf-Of (OBO) flow** to obtain an access token for the backend API scope.
  - Store any required confidential credentials (e.g., frontend client secret/cert for OBO) in **Key Vault** and load via Managed Identity (or via pipeline to Key Vault).

Implications for routing:
- The public `/api/*` still routes to the backend, but the **app should not call it directly from browser code**.
- Frontend should expose BFF endpoints (e.g. `/bff/*` or Next.js server actions) and proxy to backend internally.

### Runtime Config Contract

- Frontend (public):
  - `NEXT_PUBLIC_BFF_BASE_URL=/bff` (recommended)
  - `NEXT_PUBLIC_AUTH_TENANT_ID`
  - `NEXT_PUBLIC_AUTH_CLIENT_ID`

- Frontend (server-only / secret):
  - `AUTH_CLIENT_SECRET` **or** client certificate reference (for OBO)
  - `AUTH_BACKEND_SCOPE=api://<backend-app-id>/access_as_user`

- Backend:
  - `CORS_ORIGINS=https://<frontdoor-domain>`
  - `AUTH_ENABLED=true`
  - `AUTH_TENANT_ID`
  - `AUTH_AUDIENCE`
  - `AUTH_ISSUER`
  - **NEW (v2): `AFD_ORIGIN_SECRET=<value>`** (or equivalent)
  - Existing `AZURE_*`, `GRAPHRAG_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`

## Containerization

1. Add `backend/Dockerfile`:
   - Base: `python:3.11-slim`
   - Entrypoint: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
2. Add `frontend/Dockerfile`:
   - Multi-stage Node build
   - Runtime command: `npm run start`
3. Add `.dockerignore` in both backend/frontend
4. Image tags:
   - `backend:<git-sha>`
   - `frontend:<git-sha>`
   - rolling aliases: `staging-latest`, `prod-latest`

## App-Layer Infrastructure Provisioning

### Naming Baseline

- Region: `southeastasia`
- Resource groups:
  - `rg-gtog-stg`
  - `rg-gtog-prod`
- ACA environments:
  - `cae-gtog-stg`
  - `cae-gtog-prod`
- Apps:
  - `ca-gtog-frontend-stg`
  - `ca-gtog-backend-stg`
  - `ca-gtog-frontend-prod`
  - `ca-gtog-backend-prod`
- ACR: one shared registry (for example `acrgtogshared`)

### Front Door + WAF (v2)

1. Two host routes (staging/prod)
2. Route rules:
   - `/api/*` -> backend origin group
   - `/*` -> frontend origin group
3. Enable managed WAF rules in blocking mode for production
4. **NEW (v2): Rate limiting for `/api/*`**
   - Configure a conservative rate limit for small traffic profile.
5. **NEW (v2): Backend Origin Lock**
   - Add a rule to inject a secret header to backend origin:
     - `X-AFD-Secret: <value>`
   - Backend validates this header for all `/api/*`.

### Identity and Secrets

1. One user-assigned Managed Identity per backend environment
2. Grant `Key Vault Secrets User` role to each backend MI
3. Keep secrets in Key Vault only (no plaintext pipeline vars)
4. Enable diagnostic settings (Key Vault audit logs) to Log Analytics

### Networking / Data Access (v2)

- Minimum required for go-live:
  - Cosmos/Blob/Search: firewall locked down (no open public access by default)
  - Key Vault: Managed Identity only; restrict network where possible
- If Azure AI Search stays on Free SKU, full private endpoint lockdown may be deferred.
- Document egress domains/timeouts for external LLM providers and Tavily.

### Initial Scale (small traffic profile)

1. Backend staging: min 1 / max 1, 1 vCPU, 2 GiB
2. Backend prod: min 1 / max 1, 2 vCPU, 4 GiB
3. Frontend staging: min 1 / max 1
4. Frontend prod: min 1 / max 2

### Important Runtime Constraint

- Current indexing execution is in-process, so backend replicas stay at `max=1` to avoid job concurrency/race risk in this release.
- **NEW (v2): Guardrail**
  - Add a runtime safety check / distributed lock so accidental scale-up does not run indexing concurrently.

## Entra ID OIDC Plan

1. App registrations per environment:
   - `gtog-frontend-{env}`
   - `gtog-backend-api-{env}`
2. Backend exposes API scope:
   - `api://<backend-app-id>/access_as_user`
3. Frontend auth integration per chosen pattern (**Option A BFF-lite**)
   - Configure OBO (frontend as confidential client) to request backend API scope
4. Backend enforces issuer, audience, and scope checks

## Azure DevOps CI/CD Plan

### Pipeline Stages

1. `Validate`
   - Backend test set + static checks
   - Frontend `npm ci && npm run build`
2. `BuildImages`
   - Build and push backend/frontend images to ACR
3. `DeployStaging`
   - Deploy new revisions to staging ACA apps
4. `SmokeStaging`
   - Health, auth, collection CRUD, upload, index, query method checks
   - **NEW (v2): bypass test**
     - Direct call to backend origin (not via Front Door) must fail
5. `ManualApproval`
6. `DeployProduction`
   - Canary revision (10%), then full traffic on pass

### Rollback

1. Roll traffic back to previous ACA revision (no rebuild required)
2. Rollback target SLA: under 10 minutes

## Test Cases and Acceptance Scenarios (v2)

### Functional

1. `GET /health` returns healthy
2. `GET /health/readiness` returns ready
3. Create collection, upload document, start indexing, poll status
4. Query all methods: global/local/tog/drift
5. Verify SSE endpoint over Front Door path routing

### Security

1. Browser direct call to `/api/*` without bearer token -> `401/403` (expected under Option A)
2. No token -> `401/403`
2. Wrong audience token -> denied
3. CORS allows only configured Front Door origins
4. ToG debug endpoint remains disabled by default
5. **NEW (v2): backend origin lock**
   - Requests without `X-AFD-Secret` are denied (even if token is valid)

### Reliability

1. Restart backend during indexing and verify Cosmos job recovery behavior
2. Delete collection and verify metadata/artifact cleanup path

### Performance Baseline

1. p95 `/api/collections` < 500 ms
2. p95 readiness endpoint < 1 s
3. No sustained 429/retry storm from Cosmos/Search in smoke run

### Deployment Behavior

1. Each deploy creates a new revision
2. Staging rollback test is executed before production go-live

## Rollout Phases

1. Phase A - Code readiness + containerization
2. Phase B - App-layer infra + Entra setup
3. Phase C - CI/CD with staging gate and prod canary
4. Phase D - Go-live and post-release validation
5. **Phase E (post go-live) - Hardening and scale**
   - Optional: move indexing out-of-process (worker/job)
   - Expand private endpoints / VNet integration where applicable

## Public Interface / API Changes

1. Frontend public runtime config:
   - `NEXT_PUBLIC_API_BASE_URL` becomes required
2. Backend API surface:
   - add `GET /health/readiness`
   - enforce OIDC on `/api/*` endpoints
3. Backend settings:
   - add auth env fields and `CORS_ORIGINS`
   - **NEW (v2): `AFD_ORIGIN_SECRET`**

## Assumptions and Defaults

1. Azure subscription remains `1095803e-80bf-47e0-961f-3d74cb4c605c`
2. Region remains `southeastasia`
3. Existing database layer + phase5 hardening remain valid
4. If Azure AI Search stays on Free SKU, full private endpoint lockdown may be deferred
5. Worker separation for indexing is out of this deployment scope (Phase E)
6. Front Door default domain is acceptable for first go-live

## Topology Reference

- See: `docs/topo_v2.md`