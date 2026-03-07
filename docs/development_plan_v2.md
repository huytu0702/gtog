# Full Deployment Plan v2 - GToG (ACA + Cloudflare Edge)

## Summary

Deploy full stack to Azure using:

- Azure Container Apps for frontend, API, worker, and tunnel connector
- Cloudflare proxied DNS as the public edge
- Cloudflare Tunnel for the API hostname private-origin path
- Microsoft Entra ID (OIDC) via ACA Easy Auth on the backend
- Azure DevOps YAML CI/CD

Environment scope: `staging` then `production`

Data layer is already deployed and reused in this plan:

- Azure Cosmos DB
- Azure Blob Storage
- Azure AI Search
- Azure Key Vault + backend Managed Identity

This v2 plan focuses on:

- Replacing Azure Front Door with a Cloudflare-based edge that does not leave the API origin public
- Dual public hostnames instead of single-host path routing
- Private API origin through Cloudflare Tunnel into a private ACA environment
- Cloudflare rate limiting on `api.<domain>` + fallback limiter in backend
- Clear auth pattern choice: public frontend, protected backend
- Strong observability + network-layer bypass tests + request correlation
- Per-environment secrets and tunnel tokens
- CSP headers on frontend
- Distributed lock guardrail for indexing (Cosmos lease)
- Frontend health probe route
- Cloudflare WAF or custom rules depending selected plan
- Alert thresholds in Azure Monitor and Log Analytics
- SSE compatibility through Cloudflare with heartbeat events
- Frontend image built per environment (build-time env injection)

## Why the Architecture Changes

Azure Front Door is not available in the target Azure Student setup. Cloudflare remains the public edge replacement.

The earlier public-origin design kept the API behind Cloudflare controls and an injected shared secret, but it still left the ACA API publicly reachable. This revision removes that gap by publishing `api.<domain>` through Cloudflare Tunnel into a private ACA environment, so origin isolation no longer depends on a shared header alone.

The public routing model remains:

- `app.<domain>` -> frontend ACA
- `api.<domain>` -> Cloudflare Tunnel -> private ACA API

This keeps the browser contract simple while moving origin trust to the network path instead of only the application layer.

## Fixed Decisions

1. Runtime platform: Azure Container Apps
2. Environments: Staging + Production
3. CI/CD: Azure DevOps YAML
4. Public edge: Cloudflare proxied DNS
5. Public host model: dual subdomains (`app.<domain>` and `api.<domain>`)
6. API origin model: Cloudflare Tunnel into a private ACA environment
7. Auth: Entra ID OIDC for backend API
8. Auth pattern: ACA Managed Authentication (Easy Auth) on backend only
9. Traffic profile (first 3 months): small (`<= 50 users/day`)
10. API routing model: browser calls backend via `https://api.<domain>/api/*`
11. Frontend managed identity: not required
12. Cloudflare Worker and Enterprise-only routing features: out of scope

## Current State (Already Done)

Based on completed phases and validation report (`docs/plans/2026-03-02-phase5-backend-validation-report.md`):

1. Data layer exists: Cosmos, Blob, Search, Key Vault
2. Backend managed identity + Key Vault secret bootstrap is implemented
3. Baseline hardening and alerts scripts for the data layer already exist (`scripts/harden-azure-phase5.ps1/.sh`)

Out of scope for this doc: re-provision data layer from scratch.

## Target Architecture (v2)

1. Frontend Container App (Next.js) serves the UI on `https://app.<domain>`
2. API Container App (FastAPI) serves the API on internal ingress only
3. Worker Container App handles indexing and long-running graph jobs with no public ingress
4. Tunnel Connector Container App (`cloudflared`) publishes `https://api.<domain>` through Cloudflare Tunnel to the private API origin
5. Cloudflare is the public edge for both hostnames
6. Cloudflare enforces:
  - proxied DNS for `app.<domain>`
  - Tunnel public hostname for `api.<domain>`
  - rate limiting on `api.<domain>`
  - WAF or custom edge rules based on the chosen Cloudflare plan
  - optional header injection for a secondary backend guard (`X-Edge-Secret`)
  - cache bypass for `/api/*`, `/.auth/*`, and SSE routes
7. Backend plane runs inside a private ACA environment:
  - delegated subnet
  - private endpoint + private DNS
  - public network access disabled
8. Azure Container Registry stores frontend, API, and worker images
9. Backend reads runtime secrets from Key Vault via Managed Identity
10. Logs and metrics are collected in Log Analytics + Azure Monitor

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
  - Exclude external LLM and Tavily checks
3. Keep `/health` as liveness-only
4. Easy Auth identity guard on `/api/*`:
  - Require `X-MS-CLIENT-PRINCIPAL`
  - Return `401` if missing
5. Secondary backend edge guard:
  - Optionally require `X-Edge-Secret: <value>`
  - Validate before identity checks when configured
  - Do not treat this header as the primary origin lock
6. Add basic backend rate-limit fallback (defense in depth)
7. SSE response handling:
  - Set `Cache-Control: no-cache`
  - Keep no-buffering headers for streaming responses
  - Emit heartbeat events every 25-30 seconds to avoid idle proxy timeout
8. Structured JSON logging with `Cf-Ray` and `CF-Connecting-IP` capture for request correlation across Cloudflare -> Tunnel -> Backend

## Auth Pattern Decision

Chosen: ACA Managed Authentication (Easy Auth) on backend only

How it works:

- Frontend ACA is public and serves the UI
- API ACA requires authentication and returns `401` for unauthenticated API requests
- Frontend starts login by redirecting users to `https://api.<domain>/.auth/login/aad`
- Easy Auth validates the session before request reaches FastAPI
- Frontend gets token from `https://api.<domain>/.auth/me` and sends Bearer token to `/api/*`

What Easy Auth handles:

- OIDC redirect and callback
- Session cookie on the backend host
- Token validation and refresh
- Logout flow

What app code still handles:

- Frontend token retrieval + API header attachment
- Frontend login and logout redirect URLs back to `app.<domain>`
- Frontend SSE: use Easy Auth session cookie (`EventSource` cannot set custom `Authorization` headers)
- Backend header presence guards (`X-MS-CLIENT-PRINCIPAL` and optional `X-Edge-Secret`)

SSE auth note: `EventSource` does not support custom `Authorization` headers. SSE endpoints rely on the Easy Auth session cookie stored for `api.<domain>`. Backend must accept both Bearer token and session cookie for `/api/*` routes (Easy Auth handles both transparently).

## Runtime Config Contract

- Frontend (build-time):
  - `NEXT_PUBLIC_API_BASE_URL=https://api.<domain>`
  - Note: `NEXT_PUBLIC_` vars are inlined at build time. Frontend image must be built separately per environment.
- Backend (runtime):
  - `CORS_ORIGINS=https://app.<domain>`
  - `EDGE_ORIGIN_SECRET=<value>` only if the secondary backend guard is enabled
  - Existing `AZURE_*`, `GRAPHRAG_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`
- Tunnel connector (runtime):
  - `CLOUDFLARE_TUNNEL_TOKEN=<value>`
  - Optional `X-Edge-Secret` injection remains configured on the Cloudflare side if the backend secondary guard is enabled

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
3. Tunnel image:
  - `cloudflare/cloudflared`
  - command: `cloudflared tunnel --no-autoupdate run --token <token>`
  - min replicas: `2`
4. Add `.dockerignore` for backend (`__pycache__`, `.venv`, `.env`, `tests/`) and frontend (`node_modules`, `.next`, `.env`)

### Image Tags

- `backend:<git-sha>` - single image works for all environments (runtime env vars)
- `frontend:<git-sha>-<env>` - separate build per environment due to `NEXT_PUBLIC_` build-time injection
- `worker:<git-sha>` - single image for the background job worker
- `tunnel:managed` - Cloudflare provided `cloudflared` image pinned in deployment manifests or scripts
- Aliases: `staging-latest`, `prod-latest`

### Local Validation

- Add `docker-compose.dev.yml` for local testing of frontend and backend before pushing to ACR
- Tunnel connector is not required for local development

## App-Layer Infrastructure Provisioning

### Naming Baseline

- Region: `southeastasia`
- Resource groups: `rg-gtog-stg`, `rg-gtog-prod`
- ACA environments: `cae-gtog-stg`, `cae-gtog-prod`
- Apps:
  - `ca-gtog-frontend-{env}`
  - `ca-gtog-api-{env}`
  - `ca-gtog-worker-{env}`
  - `ca-gtog-tunnel-{env}`
- ACR: shared (example `acrgtogshared`)

### Cloudflare Edge

1. Two public hosts per environment:
  - `app.<domain>` -> frontend ACA
  - `api.<domain>` -> Cloudflare Tunnel -> private ACA API
2. Cloudflare proxied DNS enabled for `app.<domain>`
3. Cloudflare Tunnel public hostname enabled for `api.<domain>`
4. Edge protections:
  - Rate limiting on `api.<domain>`
  - WAF managed rules if plan supports them
  - Custom rules for low-cost plans when managed rules are unavailable
  - Optional rule to block missing `User-Agent`
5. Optional Request Header Transform Rule on `api.<domain>`:
  - Inject `X-Edge-Secret: <value>` to origin requests
  - Use separate secret values per environment
6. Disable caching for:
  - `/api/*`
  - `/.auth/*`
  - SSE routes
7. Do not use Cloudflare Worker or single-host path routing in v2

### Custom Domains and Certificates

1. Bind `app.<domain>` to the frontend ACA
2. Keep `api.<domain>` stable at the browser layer, but do not depend on direct public ACA DNS routing for the API
3. Retain API custom domain binding on ACA only if required for Easy Auth redirect, cookie, or host-header behavior
4. Do not rely on ACA managed certificates in this topology
5. Use uploaded certificates where ACA custom domains are retained
6. If validation requires direct DNS resolution, temporarily disable Cloudflare proxy during domain validation
7. Cloudflare SSL mode: `Full (strict)`

### Identity and Secrets

1. One user-assigned Managed Identity per backend environment
2. Frontend does not need Managed Identity
3. Grant `Key Vault Secrets User` to backend MI
4. Keep runtime secrets in Key Vault
5. Add `CLOUDFLARE_TUNNEL_TOKEN` secret per environment
6. Enable Key Vault diagnostics to Log Analytics
7. Configure Key Vault expiry alerts (30 days)

### Networking / Data Access

- Reuse existing deployed data resources
- Keep Cosmos, Blob, and Search firewall restrictions aligned with current hardening phase
- Key Vault access via Managed Identity
- Provision a workload-profile ACA environment inside a dedicated VNet and delegated subnet
- Create a private endpoint and private DNS zone for the ACA environment
- Disable public network access on the ACA backend environment before production sign-off
- Configure API app ingress as internal only
- Configure worker app with no public ingress
- AI Search Free SKU limitation: no private endpoint on Free (known gap if still on Free)

### Initial Scale (small traffic)

1. API staging: min 1 / max 2, 1 vCPU, 2 GiB
2. API prod: min 1 / max 2, 2 vCPU, 4 GiB
3. Worker staging: min 1 / max 1, 1 vCPU, 2 GiB
4. Worker prod: min 1 / max 2, 2 vCPU, 4 GiB
5. Tunnel staging: min 2 / max 2, 0.5 vCPU, 1 GiB
6. Tunnel prod: min 2 / max 2, 0.5 vCPU, 1 GiB
7. Frontend staging: min 1 / max 1
8. Frontend prod: min 1 / max 2

### Runtime Constraint

- Current indexing is in-process; keep worker concurrency conservative initially
- Keep distributed lock guardrail using Cosmos lease document (`indexing-lock`)

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
2. `BuildImages` (build and push frontend, API, and worker images to ACR)
3. `DeployStaging` (new ACA revisions + tunnel connector)
4. `SmokeStaging`:
  - health, auth, CRUD, upload, index, query, and SSE checks
  - public direct-origin probe must fail at the network layer
  - tunnel connector failover test
  - token isolation test
5. `ManualApproval`
6. `DeployProduction` (canary 10% -> full)

### Rollback

1. Roll traffic to previous ACA revision
2. Keep Cloudflare DNS and tunnel public hostname unchanged unless the incident is tunnel-specific
3. Rollback target SLA: under 10 minutes
4. Break-glass path must not re-enable public network access unless a documented incident override is approved

## Test Cases and Acceptance

### Functional

1. `GET /health` returns healthy
2. `GET /health/readiness` returns ready
3. Collection CRUD + upload + indexing + status polling
4. Query methods: global, local, tog, drift
5. SSE endpoint works through Cloudflare (including long-running streams > 60s) and sends heartbeat events before idle timeout

### Security

1. Frontend is reachable at `app.<domain>`
2. Unauthenticated backend `/api/*` request -> `401`
3. Wrong audience token -> denied
4. CORS allows only configured frontend origin
5. ToG debug endpoint disabled by default
6. Public direct-origin probes to the ACA API fail before reaching the app
7. If `EDGE_ORIGIN_SECRET` is enabled, missing `X-Edge-Secret` -> denied
8. Staging token rejected by prod backend

### Reliability

1. Restart backend during indexing; verify recovery behavior
2. Delete collection; verify metadata and artifact cleanup
3. Stop one tunnel connector replica; verify API traffic still succeeds

### Performance Baseline

1. p95 `/api/collections` < 500 ms
2. p95 `/health/readiness` < 1 s
3. No sustained Cosmos and Search 429 storm in smoke run

### Observability and Logging

1. Structured JSON logging (backend): use Python `logging` with JSON formatter
2. Log `Cf-Ray` on every backend request for Cloudflare correlation
3. Log `CF-Connecting-IP` as the original client IP when present
4. Log tunnel connector health and reconnect events
5. Log retention: 30 days in Log Analytics (staging), 90 days (prod)
6. Frontend: Next.js server logs forwarded to Log Analytics via ACA stdout

### Observability Alerts

1. 5xx error rate > 1% over 5 min -> Sev2
2. p95 `/api/*` latency > 2 s over 10 min -> Sev3
3. Cosmos or Search 429 > 5/min -> Sev3
4. Key Vault secret expiry < 30 days -> Sev2
5. ACA replica restarts > 2 in 15 min -> Sev2
6. Tunnel connector healthy replicas < 2 -> Sev2

## Rollout Phases

1. Phase A - Application split + local readiness
  - Implement all Required Application Changes for frontend and backend API
  - Split backend responsibilities into `API` and `Worker` paths
  - Introduce queue-based job dispatch for indexing and other long-running graph jobs
  - Persist job state and distributed lock ownership in Cosmos
  - Keep synchronous API path stateless and reserve SSE for read-side streaming only
  - Update Dockerfiles and `.dockerignore` for frontend, API, and worker images
  - Local validation with `docker-compose.dev.yml`: verify health, CORS, auth flow shape, queue dispatch, and basic worker execution
2. Phase B - Private backend plane + identity
  - Provision ACA private environment, API app, worker app, tunnel app, ACR, queue resource, private endpoint, private DNS, and Cloudflare Tunnel public hostname
  - Upload certificates for retained ACA custom domains where required
  - Configure Entra app registrations and backend Easy Auth
  - Validate token flow with `az rest` or Postman before wiring frontend login UI
  - Validate public direct-origin probes fail after private-origin cutover
3. Phase C - Data protection + operational guardrails
  - Finalize Cosmos lease and job containers and retention rules
  - Enable Key Vault diagnostics, soft delete, purge protection, and expiry alerts
  - Enable Blob soft delete and versioning where cost permits
  - Define Cosmos backup policy and document restore procedure
  - Document AI Search rebuild procedure from Blob and Cosmos source-of-truth data
  - Define target `RPO` and `RTO` for metadata, uploaded documents, and serving indexes
4. Phase D - CI/CD + release gates
  - Extend `.vsts-ci.yml` with Validate, BuildImages, DeployStaging, SmokeStaging, ManualApproval, and DeployProduction stages
  - Add IaC deployment and validation for ACA, Cloudflare, Entra, and alerting resources
  - Add SBOM generation, image vulnerability scan, and image signing before publish
  - Run full staging smoke suite including public direct-origin denial, audience isolation, queue-to-worker flow, tunnel failover, and SSE long-running test
  - Execute rollback drill on staging and verify ACA revision rollback under the target SLA
5. Phase E - Go-live + controlled production rollout
  - Deploy production with canary traffic split before full promotion
  - Monitor 5xx rate, latency, replica restarts, Cosmos and Search 429s, auth failures, and tunnel connector health during rollout
  - Validate end-to-end logging correlation with `Cf-Ray`, client IP, tunnel events, and app request IDs
  - Confirm backup and restore runbooks, on-call ownership, and secret rotation runbooks are in place before full cutover
6. Phase F - Post-release hardening and scale improvements
  - Rotate tunnel token and optional `X-Edge-Secret` on a documented schedule
  - Add autoscaling policy for worker throughput once job patterns are stable
  - Revisit search and index separation, cost controls, and provider fallback policies as traffic grows

## Public Interface / API Changes

1. Frontend runtime requires `NEXT_PUBLIC_API_BASE_URL` pointing to `https://api.<domain>`
2. Backend adds `GET /health/readiness`
3. Backend settings add `CORS_ORIGINS` and optionally `EDGE_ORIGIN_SECRET`
4. Tunnel deployment adds `CLOUDFLARE_TUNNEL_TOKEN`

## Known Limitations

1. Frontend ACA default domain may remain reachable outside Cloudflare unless additional ingress controls are introduced
2. Azure AI Search Free SKU still prevents full private endpoint alignment on the data plane
3. Cloudflare Tunnel public hostname and routing policy must be maintained outside Azure, so tunnel misconfiguration becomes a production dependency

## Assumptions and Defaults

1. Subscription: `1095803e-80bf-47e0-961f-3d74cb4c605c`
2. Region: `southeastasia`
3. Existing data layer and phase 5 hardening remain valid
4. Cloudflare Tunnel is remotely managed and uses per-environment tokens stored in Key Vault
5. Cloudflare Enterprise-only origin routing features are intentionally not used

## Topology Reference

- Baseline topology: `docs/topo_v2.md`
- Production-refined topology: `docs/topo_v3.md`
- Private-origin runbook: `docs/runbooks/api-private-origin-cloudflare-tunnel.md`
