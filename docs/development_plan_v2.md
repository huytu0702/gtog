# Full Deployment Plan v2 - GToG (ACA + Cloudflare Edge)

## Summary

Deploy full stack to Azure using:

- Azure Container Apps for frontend and backend
- Cloudflare proxied DNS as the public edge
- Microsoft Entra ID (OIDC) via ACA Easy Auth on the backend
- Azure DevOps YAML CI/CD

Environment scope: `staging` then `production`

Data layer is already deployed and reused in this plan:

- Azure Cosmos DB
- Azure Blob Storage
- Azure AI Search
- Azure Key Vault + backend Managed Identity

This v2 plan focuses on:

- Replacing Azure Front Door with a student-compatible edge design
- Dual public hostnames instead of single-host path routing
- Backend origin lock with a Cloudflare-injected header
- Cloudflare rate limiting on `api.<domain>` + fallback limiter in backend
- Clear auth pattern choice: public frontend, protected backend
- Strong observability + bypass tests + request correlation
- Per-environment secrets (staging/prod)
- CSP headers on frontend
- Distributed lock guardrail for indexing (Cosmos lease)
- Frontend health probe route
- Cloudflare WAF or custom rules depending selected plan
- Alert thresholds in Azure Monitor / Log Analytics
- SSE compatibility through Cloudflare with heartbeat events
- Frontend image built per environment (build-time env injection)

## Why the Architecture Changes

Azure Front Door is not available in the target Azure Student setup. Cloudflare is used as the edge replacement.

To keep the design low-cost and avoid Cloudflare Enterprise or Worker-based reverse proxying, the public routing model changes from one hostname with path routing to two hostnames:

- `app.<domain>` -> frontend ACA
- `api.<domain>` -> backend ACA

This keeps the platform simple and preserves backend protection, rate limiting, and auth without introducing a custom proxy layer.

## Fixed Decisions

1. Runtime platform: Azure Container Apps
2. Environments: Staging + Production
3. CI/CD: Azure DevOps YAML
4. Public edge: Cloudflare proxied DNS
5. Public host model: dual subdomains (`app.<domain>` and `api.<domain>`)
6. Auth: Entra ID OIDC for backend API
7. Auth pattern: ACA Managed Authentication (Easy Auth) on backend only
8. Traffic profile (first 3 months): small (`<= 50 users/day`)
9. API routing model: browser calls backend via `https://api.<domain>/api/*`
10. Frontend managed identity: not required
11. Cloudflare Worker and Enterprise-only routing features: out of scope

## Current State (Already Done)

Based on completed phases and validation report (`docs/plans/2026-03-02-phase5-backend-validation-report.md`):

1. Data layer exists: Cosmos, Blob, Search, Key Vault
2. Backend managed identity + Key Vault secret bootstrap is implemented
3. Baseline hardening/alerts scripts for data layer already exist (`scripts/harden-azure-phase5.ps1/.sh`)

Out of scope for this doc: re-provision data layer from scratch.

## Target Architecture (v2)

1. Frontend Container App (Next.js) serves the UI on `https://app.<domain>`
2. Backend Container App (FastAPI) serves the API on `https://api.<domain>/api/*`
3. Cloudflare is the public edge for both hostnames
4. Cloudflare enforces:
  - Proxied DNS for `app.<domain>` and `api.<domain>`
  - Rate limiting on `api.<domain>`
  - WAF or custom edge rules based on the chosen Cloudflare plan
  - Header injection for backend origin lock (`X-Edge-Secret`)
  - Cache bypass for `/api/*`, `/.auth/*`, and SSE routes
5. Azure Container Registry stores frontend/backend images
6. Backend reads runtime secrets from Key Vault via Managed Identity
7. Backend connects to Cosmos, Blob, AI Search, and external LLM providers
8. Logs/metrics are collected in Log Analytics + Azure Monitor

## Required Application Changes

### Frontend

1. Replace any cloud deployment assumption that the frontend and API share one hostname
2. Use `NEXT_PUBLIC_API_BASE_URL=https://api.<domain>`
3. Keep Easy Auth token flow against the backend host:
  - Call `GET https://api.<domain>/.auth/me` with credentials
  - Redirect to `https://api.<domain>/.auth/login/aad?post_login_redirect_uri=https://app.<domain>/` when unauthenticated
  - Redirect logout to `https://api.<domain>/.auth/logout?post_logout_redirect_uri=https://app.<domain>/`
  - Attach `Authorization: Bearer <access_token>` to `/api/*` calls when token retrieval succeeds
4. SSE (EventSource) auth:
  - Continue to use cookie-based auth via Easy Auth session cookie for SSE endpoints
  - Keep `withCredentials: true` on `EventSource`
  - Backend SSE endpoints accept Easy Auth session cookie as alternative to Bearer token
5. Add login/logout UI that targets the backend auth host
6. Keep health route `app/api/health/route.ts` returning `{ status: "ok" }`
7. Add CSP headers in `next.config.ts`

### Backend

1. Replace open CORS (`*`) with env-based allowlist (`CORS_ORIGINS`)
2. Add readiness endpoint `/health/readiness` with dependency checks:
  - Cosmos DB reachable
  - Key Vault reachable
  - AI Search reachable
  - Blob Storage reachable
  - Exclude external LLM/Tavily checks
3. Keep `/health` as liveness-only
4. Easy Auth identity guard on `/api/*`:
  - Require `X-MS-CLIENT-PRINCIPAL`
  - Return `401` if missing
5. Backend origin lock:
  - Require `X-Edge-Secret: <value>`
  - Validate before identity checks
6. Add basic backend rate-limit fallback (defense in depth)
7. SSE response handling:
  - Set `Cache-Control: no-cache`
  - Keep no-buffering headers for streaming responses
  - Emit heartbeat events every 25-30 seconds to avoid idle proxy timeout
8. Structured JSON logging with `Cf-Ray` and `CF-Connecting-IP` capture for request correlation across Cloudflare -> Backend

## Auth Pattern Decision

Chosen: ACA Managed Authentication (Easy Auth) on backend only

How it works:

- Frontend ACA is public and serves the UI
- Backend ACA requires authentication and returns `401` for unauthenticated API requests
- Frontend starts login by redirecting users to `https://api.<domain>/.auth/login/aad`
- Easy Auth validates the session before request reaches FastAPI
- Frontend gets token from `https://api.<domain>/.auth/me` and sends Bearer token to `/api/*`

What Easy Auth handles:

- OIDC redirect/callback
- Session cookie on the backend host
- Token validation and refresh
- Logout flow

What app code still handles:

- Frontend token retrieval + API header attachment
- Frontend login/logout redirect URLs back to `app.<domain>`
- Frontend SSE: use Easy Auth session cookie (`EventSource` cannot set custom `Authorization` headers)
- Backend header presence guards (`X-MS-CLIENT-PRINCIPAL`, `X-Edge-Secret`)

SSE auth note: `EventSource` does not support custom `Authorization` headers. SSE endpoints rely on the Easy Auth session cookie stored for `api.<domain>`. Backend must accept both Bearer token and session cookie for `/api/*` routes (Easy Auth handles both transparently).

## Runtime Config Contract

- Frontend (build-time):
  - `NEXT_PUBLIC_API_BASE_URL=https://api.<domain>`
  - Note: `NEXT_PUBLIC_` vars are inlined at build time. Frontend image must be built separately per environment.
- Backend (runtime):
  - `CORS_ORIGINS=https://app.<domain>`
  - `EDGE_ORIGIN_SECRET=<value>`
  - Existing `AZURE_*`, `GRAPHRAG_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`

## Containerization

### Dockerfile Requirements

1. `backend/Dockerfile`:
  - Multi-stage build: builder stage (install deps) + runtime stage (copy artifacts) + include GraphRAG folder in this repo
  - Base: `python:3.11-slim`
  - Non-root user (`appuser`)
  - `HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1`
2. `frontend/Dockerfile`:
  - Multi-stage build: deps -> build -> runtime
  - Base: `node:20-alpine`
  - Non-root user (`nextjs`)
  - Build args: `NEXT_PUBLIC_API_BASE_URL` (injected at build time per environment)
  - `HEALTHCHECK CMD curl -f http://localhost:3000/api/health || exit 1`
3. Add `.dockerignore` for backend (`__pycache__`, `.venv`, `.env`, `tests/`) and frontend (`node_modules`, `.next`, `.env`)

### Image Tags

- `backend:<git-sha>` - single image works for all environments (runtime env vars)
- `frontend:<git-sha>-<env>` - separate build per environment due to `NEXT_PUBLIC_` build-time injection
- Aliases: `staging-latest`, `prod-latest`

### Local Validation

- Add `docker-compose.dev.yml` for local testing of both containers before pushing to ACR

## App-Layer Infrastructure Provisioning

### Naming Baseline

- Region: `southeastasia`
- Resource groups: `rg-gtog-stg`, `rg-gtog-prod`
- ACA environments: `cae-gtog-stg`, `cae-gtog-prod`
- Apps: `ca-gtog-frontend-{env}`, `ca-gtog-backend-{env}`
- ACR: shared (example `acrgtogshared`)

### Cloudflare Edge

1. Two public hosts per environment:
  - `app.<domain>` -> frontend ACA
  - `api.<domain>` -> backend ACA
2. Cloudflare proxied DNS enabled for both hosts
3. Edge protections:
  - Rate limiting on `api.<domain>`
  - WAF managed rules if plan supports them
  - Custom rules for low-cost plans when managed rules are unavailable
  - Optional rule to block missing `User-Agent`
4. Add Request Header Transform Rule on `api.<domain>`:
  - Inject `X-Edge-Secret: <value>` to origin requests
  - Use separate secret values per environment
5. Disable caching for:
  - `/api/*`
  - `/.auth/*`
  - SSE routes
6. Do not use Cloudflare Worker or single-host path routing in v2

### Custom Domains and Certificates

1. Bind `app.<domain>` to the frontend ACA
2. Bind `api.<domain>` to the backend ACA
3. Do not use ACA managed certificates in this topology
4. Use uploaded certificates for ACA custom domains
5. If validation requires direct DNS resolution, temporarily disable Cloudflare proxy during domain validation
6. Cloudflare SSL mode: `Full (strict)`

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

1. One backend app registration per environment:
  - `gtog-backend-api-stg`
  - `gtog-backend-api-prod`
2. Backend exposes scope `api://<backend-app-id>/access_as_user`
3. Backend Easy Auth `allowedAudiences` must include `api://<backend-app-id>` to accept tokens issued for the backend scope
4. Backend ACA Easy Auth action: `Return401`
5. Redirect URIs include the backend auth host callback for each environment
6. Token isolation smoke test: staging token must fail on prod backend
7. Validation step: test token flow with `az rest` or Postman before wiring frontend login UI

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
2. Keep Cloudflare DNS/rules unchanged unless the incident is edge-specific
3. Rollback target SLA: under 10 minutes

## Test Cases and Acceptance

### Functional

1. `GET /health` returns healthy
2. `GET /health/readiness` returns ready
3. Collection CRUD + upload + indexing + status polling
4. Query methods: global/local/tog/drift
5. SSE endpoint works through Cloudflare (including long-running streams > 60s) and sends heartbeat events before idle timeout

### Security

1. Frontend is reachable at `app.<domain>`
2. Unauthenticated backend `/api/*` request -> `401`
3. Wrong audience token -> denied
4. CORS allows only configured frontend origin
5. ToG debug endpoint disabled by default
6. Missing `X-Edge-Secret` -> denied
7. Staging token rejected by prod backend

### Reliability

1. Restart backend during indexing; verify recovery behavior
2. Delete collection; verify metadata/artifact cleanup

### Performance Baseline

1. p95 `/api/collections` < 500 ms
2. p95 `/health/readiness` < 1 s
3. No sustained Cosmos/Search 429 storm in smoke run

### Observability and Logging

1. Structured JSON logging (backend): use Python `logging` with JSON formatter
2. Log `Cf-Ray` on every backend request for Cloudflare correlation
3. Log `CF-Connecting-IP` as the original client IP when present
4. Log retention: 30 days in Log Analytics (staging), 90 days (prod)
5. Frontend: Next.js server logs forwarded to Log Analytics via ACA stdout

### Observability Alerts

1. 5xx error rate > 1% over 5 min -> Sev2
2. p95 `/api/*` latency > 2 s over 10 min -> Sev3
3. Cosmos/Search 429 > 5/min -> Sev3
4. Key Vault secret expiry < 30 days -> Sev2
5. ACA replica restarts > 2 in 15 min -> Sev2

## Rollout Phases

1. Phase A - Code readiness + containerization
  - Implement all Required Application Changes (frontend + backend)
  - Write Dockerfiles and `.dockerignore`
  - Local validation: `docker-compose.dev.yml` - build and run both containers locally, verify health/CORS/basic flow
2. Phase B - App-layer infra + Entra setup
  - Provision ACA environments, ACR, Cloudflare DNS/rules, and custom domain bindings
  - Upload certificates for ACA custom domains
  - Entra app registrations + backend Easy Auth config
  - Validate token flow with `az rest` / Postman before connecting frontend login UI
3. Phase C - CI/CD with staging gate + prod canary
  - Extend `.vsts-ci.yml` with Build, Deploy, Smoke stages
  - Run full smoke suite on staging including SSE long-running test
  - Execute rollback drill on staging to verify procedure
4. Phase D - Go-live + post-release validation
5. Phase E - hardening/scale improvements (optional)
  - Secret rotation procedures documentation
  - Worker split for indexing (separate container)

## Public Interface / API Changes

1. Frontend runtime requires `NEXT_PUBLIC_API_BASE_URL` pointing to `https://api.<domain>`
2. Backend adds `GET /health/readiness`
3. Backend settings add `CORS_ORIGINS` and `EDGE_ORIGIN_SECRET`

## Known Limitations

1. Frontend ACA default domain may remain reachable outside Cloudflare unless additional ingress controls are introduced
2. This is acceptable for student/dev scope because the backend remains protected by Easy Auth + `X-Edge-Secret`
3. If strict edge-only frontend ingress becomes a hard requirement later, re-evaluate Azure Front Door, Application Gateway, or a Worker-based proxy design

## Assumptions and Defaults

1. Subscription: `1095803e-80bf-47e0-961f-3d74cb4c605c`
2. Region: `southeastasia`
3. Existing data layer and phase5 hardening remain valid
4. Worker split for indexing is out of this release scope
5. Cloudflare-managed custom domains are available for initial go-live
6. Cloudflare Enterprise-only origin routing features are intentionally not used

## Topology Reference

- See `docs/topo_v2.md`
