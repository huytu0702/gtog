# Full Deployment Plan v2 - GToG (ACA + Front Door + WAF)

## Summary

Deploy full stack to Azure using:

- Azure Container Apps for frontend and backend
- Azure Front Door + WAF as the only public ingress
- Entra ID (OIDC) via ACA Easy Auth for UI + API auth
- Azure DevOps YAML CI/CD

Environment scope: `staging` then `production`

Data layer is already deployed and reused in this plan:

- Azure Cosmos DB
- Azure Blob Storage
- Azure AI Search
- Azure Key Vault + backend Managed Identity

This v2 plan focuses on:

- Backend origin lock (`X-AFD-Secret`) to prevent direct bypass
- `/api/*` rate limiting at Front Door + fallback limiter in backend
- Clear auth pattern choice (ACA Easy Auth, no custom JWT validation code)
- Strong observability + bypass tests
- Per-environment secrets (staging/prod)
- CSP headers on frontend
- Distributed lock guardrail for indexing (Cosmos lease)
- Frontend health probe route
- WAF managed + custom rules
- Alert thresholds in Azure Monitor / Log Analytics

## Fixed Decisions

1. Runtime platform: Azure Container Apps
2. Environments: Staging + Production
3. CI/CD: Azure DevOps YAML
4. Ingress: Front Door + WAF
5. Auth: Entra ID OIDC for UI + API
6. Auth pattern: ACA Managed Authentication (Easy Auth)
7. Traffic profile (first 3 months): small (`<= 50 users/day`)
8. API routing model: browser calls backend via Front Door (`/api/*`)
9. Frontend managed identity: not required

## Current State (Already Done)

Based on completed phases and validation report (`docs/plans/2026-03-02-phase5-backend-validation-report.md`):

1. Data layer exists: Cosmos, Blob, Search, Key Vault
2. Backend managed identity + Key Vault secret bootstrap is implemented
3. Baseline hardening/alerts scripts for data layer already exist (`scripts/harden-azure-phase5.ps1/.sh`)

Out of scope for this doc: re-provision data layer from scratch.

## Target Architecture (v2)

1. Frontend Container App (Next.js) serves `/*`
2. Backend Container App (FastAPI) serves `/api/*` and health endpoints
3. Azure Front Door + WAF is the public entry point
4. Front Door enforces:
  - Path routing (`/*` -> frontend, `/api/*` -> backend)
  - WAF blocking mode in production
  - Rate limit policy for `/api/*`
  - Header injection for backend origin lock (`X-AFD-Secret`)
5. Azure Container Registry stores frontend/backend images
6. Backend reads runtime secrets from Key Vault via Managed Identity
7. Backend connects to Cosmos, Blob, AI Search, and external LLM providers
8. Logs/metrics are collected in Log Analytics + Azure Monitor

## Required Application Changes

### Frontend

1. Replace hard-coded API base URL in `frontend/lib/api.ts`
2. Use `NEXT_PUBLIC_API_BASE_URL=https://<frontdoor-domain>`
3. Add Easy Auth token flow:
  - Call `GET /.auth/me` to get access token
  - Redirect to `/.auth/login/aad` when unauthenticated
  - Attach `Authorization: Bearer <access_token>` to `/api/*` calls
4. Add login/logout UI (`/.auth/login/aad`, `/.auth/logout`)
5. Add health route `app/api/health/route.ts` returning `{ status: "ok" }`
6. Add CSP headers in `next.config.ts`

### Backend

1. Replace open CORS (`*`) with env-based allowlist (`CORS_ORIGINS`)
2. Add readiness endpoint `/health/readiness` with dependency checks:
  - Cosmos DB reachable
  - Key Vault reachable
  - AI Search reachable
  - Exclude external LLM/Tavily checks
3. Keep `/health` as liveness-only
4. Easy Auth identity guard on `/api/*`:
  - Require `X-MS-CLIENT-PRINCIPAL`
  - Return `401` if missing
5. Backend origin lock:
  - Require `X-AFD-Secret: <value>`
  - Validate before identity checks
6. Add basic backend rate-limit fallback (defense in depth)

## Auth Pattern Decision

Chosen: ACA Managed Authentication (Easy Auth)

How it works:

- Frontend ACA: unauthenticated -> redirect to login
- Backend ACA: unauthenticated -> `401`
- Easy Auth validates token before request reaches FastAPI
- Frontend gets token from `/.auth/me` and sends Bearer token to `/api/*` via Front Door

What Easy Auth handles:

- OIDC redirect/callback
- Session cookie
- Token validation and refresh
- Logout flow

What app code still handles:

- Frontend token retrieval + API header attachment
- Backend header presence guards (`X-MS-CLIENT-PRINCIPAL`, `X-AFD-Secret`)

## Runtime Config Contract

- Frontend:
  - `NEXT_PUBLIC_API_BASE_URL=https://<frontdoor-domain>`
- Backend:
  - `CORS_ORIGINS=https://<frontdoor-domain>`
  - `AFD_ORIGIN_SECRET=<value>`
  - Existing `AZURE_*`, `GRAPHRAG_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`

## Containerization

1. Add `backend/Dockerfile`
2. Add `frontend/Dockerfile`
3. Add `.dockerignore` for backend/frontend
4. Image tags:
  - `backend:<git-sha>`
  - `frontend:<git-sha>`
  - aliases `staging-latest`, `prod-latest`

## App-Layer Infrastructure Provisioning

### Naming Baseline

- Region: `southeastasia`
- Resource groups: `rg-gtog-stg`, `rg-gtog-prod`
- ACA environments: `cae-gtog-stg`, `cae-gtog-prod`
- Apps: `ca-gtog-frontend-{env}`, `ca-gtog-backend-{env}`
- ACR: shared (example `acrgtogshared`)

### Front Door + WAF

1. Two hosts (staging/prod)
2. Route rules:
  - `/api/*` -> backend origin group
  - `/*` -> frontend origin group
3. WAF managed rules in blocking mode for production
4. Custom rules:
  - Block missing `User-Agent`
  - Optional geo filter
5. Rate limiting for `/api/*`
6. Inject `X-AFD-Secret` to backend origin
7. Use separate secret values per environment

### Identity and Secrets

1. One user-assigned Managed Identity per backend environment
2. Frontend does not need Managed Identity
3. Grant `Key Vault Secrets User` to backend MI
4. Keep runtime secrets in Key Vault
5. Enable Key Vault diagnostics to Log Analytics
6. Configure Key Vault expiry alerts (30 days)

### Networking / Data Access

- Reuse existing deployed data resources
- Keep Cosmos/Blob/Search firewall restrictions aligned with current hardening phase
- Key Vault access via Managed Identity
- AI Search Free SKU limitation: no private endpoint on Free (known gap if still on Free)

### Initial Scale (small traffic)

1. Backend staging: min 1 / max 1, 1 vCPU, 2 GiB
2. Backend prod: min 1 / max 1, 2 vCPU, 4 GiB
3. Frontend staging: min 1 / max 1
4. Frontend prod: min 1 / max 2

### Runtime Constraint

- Current indexing is in-process; keep backend max replicas = 1 initially
- Add distributed lock guardrail using Cosmos lease document (`indexing-lock`)

## Entra ID / Easy Auth Setup

1. App registrations per environment:
  - `gtog-frontend-{env}`
  - `gtog-backend-api-{env}`
2. Backend exposes scope `api://<backend-app-id>/access_as_user`
3. Grant backend scope to frontend app (admin consent)
4. Easy Auth per ACA:
  - Frontend: `RedirectToLoginPage`
  - Backend: `Return401`
5. Redirect URIs include Front Door domain callback
6. Token isolation smoke test: staging token must fail on prod backend

## Azure DevOps CI/CD Plan

### Pipeline Stages

1. `Validate` (backend tests + static checks, frontend build)
2. `BuildImages` (build/push frontend/backend to ACR)
3. `DeployStaging` (new revisions)
4. `SmokeStaging`:
  - health/auth/CRUD/upload/index/query/SSE checks
  - direct backend bypass test must fail
  - token isolation test
5. `ManualApproval`
6. `DeployProduction` (canary 10% -> full)

### Rollback

1. Roll traffic to previous ACA revision
2. Rollback target SLA: under 10 minutes

## Test Cases and Acceptance

### Functional

1. `GET /health` returns healthy
2. `GET /health/readiness` returns ready
3. Collection CRUD + upload + indexing + status polling
4. Query methods: global/local/tog/drift
5. SSE endpoint works through Front Door

### Security

1. Unauthenticated frontend request -> redirect to Entra login
2. Unauthenticated backend `/api/*` request -> `401`
3. Wrong audience token -> denied
4. CORS allows only configured Front Door origin
5. ToG debug endpoint disabled by default
6. Missing `X-AFD-Secret` -> denied
7. Staging token rejected by prod backend

### Reliability

1. Restart backend during indexing; verify recovery behavior
2. Delete collection; verify metadata/artifact cleanup

### Performance Baseline

1. p95 `/api/collections` < 500 ms
2. p95 `/health/readiness` < 1 s
3. No sustained Cosmos/Search 429 storm in smoke run

### Observability Alerts

1. 5xx error rate > 1% over 5 min -> Sev2
2. p95 `/api/*` latency > 2 s over 10 min -> Sev3
3. Cosmos/Search 429 > 5/min -> Sev3
4. Key Vault secret expiry < 30 days -> Sev2
5. ACA replica restarts > 2 in 15 min -> Sev2

## Rollout Phases

1. Phase A - Code readiness + containerization
2. Phase B - App-layer infra + Entra setup
3. Phase C - CI/CD with staging gate + prod canary
4. Phase D - Go-live + post-release validation
5. Phase E - hardening/scale improvements (optional)

## Public Interface / API Changes

1. Frontend runtime requires `NEXT_PUBLIC_API_BASE_URL`
2. Backend adds `GET /health/readiness`
3. Backend settings add `CORS_ORIGINS` and `AFD_ORIGIN_SECRET`

## Assumptions and Defaults

1. Subscription: `1095803e-80bf-47e0-961f-3d74cb4c605c`
2. Region: `southeastasia`
3. Existing data layer and phase5 hardening remain valid
4. Worker split for indexing is out of this release scope
5. Front Door default domain is acceptable for initial go-live

## Topology Reference

- See `docs/topo_v2.md`
