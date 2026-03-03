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
- Clear auth pattern choice (**ACA Easy Auth — no app code required**)
- Stronger observability + bypass tests
- Frontend Managed Identity for Key Vault access
- ACA internal ingress for frontend-to-backend calls (no double Front Door hop)
- Per-environment AFD secrets
- CSP headers on frontend
- Distributed lock implementation detail (Cosmos DB lease)
- Explicit AI Search SKU decision
- Frontend health probe + route
- WAF custom rules
- Log Analytics alert thresholds

## Fixed Decisions

1. Runtime platform: Azure Container Apps
2. Environments: Staging + Production
3. CI/CD: Azure DevOps YAML
4. Ingress: Front Door + WAF
5. Auth: Entra ID OIDC for UI + API
6. **Auth pattern: ACA Managed Authentication (Easy Auth) — no app auth code required**
7. Traffic profile (first 3 months): small (`<= 50 users/day`)

## Target Architecture (v2)

1. Frontend Container App (Next.js) serves `/*`
2. Backend Container App (FastAPI) serves `/api/*` and health endpoints; **internal ingress only** (not publicly reachable — only reachable from within the ACA environment or via Front Door)
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
2. Update API calls to include `Authorization: Bearer <token>`:
  - Call `GET /.auth/me` (provided by Easy Auth at runtime) to retrieve the current user's access token.
  - If `/.auth/me` returns no token or `401`, redirect to `/.auth/login`.
  - Attach the token as `Authorization: Bearer <access_token>` on all calls to `/api/*`.
  - Easy Auth handles token refresh automatically — no refresh logic needed in app code.
3. Add login/logout UI using Easy Auth platform endpoints:
  - Login: link to `/.auth/login/aad`
  - Logout: link to `/.auth/logout`
  - Current user info: `GET /.auth/me`
4. Add a health route at `app/api/health/route.ts` returning `{ status: "ok" }` for ACA liveness probe.
5. Add Content Security Policy (CSP) headers in `next.config.js`:
  - `default-src 'self'`
  - `script-src 'self'` (adjust for any required inline scripts)
  - `connect-src 'self' https://<frontdoor-domain>`

### Backend

1. Replace open CORS (`*`) with env-based allowlist (`CORS_ORIGINS`)
2. Add readiness endpoint (example: `/health/readiness`) with dependency checks:
  - Checks: Cosmos DB reachable, Key Vault reachable, AI Search reachable
  - Excludes: external LLM providers and Tavily (external; false negatives risk)
3. Keep `/health` as liveness-only
4. Read user identity from Easy Auth headers injected by ACA:
  - `X-MS-CLIENT-PRINCIPAL` (base64-encoded JSON with claims)
  - `X-MS-CLIENT-PRINCIPAL-ID` (user object ID)
  - `X-MS-CLIENT-PRINCIPAL-NAME` (user display name)
  - No JWT validation code needed — Easy Auth validates the token at the platform level before the request reaches FastAPI.
  - For defense-in-depth: assert that `X-MS-CLIENT-PRINCIPAL` is present on all `/api/*` routes; return `401` if missing.
5. **NEW (v2): Backend Origin Lock**
  - Reject requests that do not include the Front Door injected header:
    - `X-AFD-Secret: <value>`
  - This must be checked before identity checks.
6. **NEW (v2): Rate-limit fallback**
  - Add a basic per-IP or per-user limiter (defense in depth).

### Auth Pattern Decision (v2)

✅ **Chosen: ACA Managed Authentication (Easy Auth)**

Auth is handled entirely at the **Azure Container Apps platform level** — no auth code required in the Next.js or FastAPI application.

**How it works:**

- **Frontend ACA**: Easy Auth is enabled with action = `RedirectToLoginPage`. Unauthenticated requests are automatically redirected to Entra ID login. After login, ACA manages the session cookie and token refresh transparently.
- **Backend ACA**: Easy Auth is enabled with action = `Return401`. Requests without a valid token are rejected at the platform level before reaching FastAPI. ACA injects user identity headers into every authenticated request.
- **Token flow**: The frontend calls `GET /.auth/me` to retrieve the current access token, then attaches it as `Authorization: Bearer` when calling the backend `/api/*` via Front Door.

**What Easy Auth handles automatically (no app code):**

- OIDC redirect and callback
- Session cookie (HttpOnly, Secure, SameSite)
- Token validation (signature, expiry, issuer, audience)
- Silent token refresh via stored refresh token
- Logout via `/.auth/logout`

**What the app still does:**

- Frontend reads `/.auth/me` to get the token for API calls
- Frontend shows login/logout links using `/.auth/login/aad` and `/.auth/logout`
- Backend checks for `X-MS-CLIENT-PRINCIPAL` header presence as a guard

Implications for routing:

- Browser calls frontend at `/*`, frontend calls backend at `/api/*` via Front Door with Bearer token.
- Backend has internal ingress only; it is only reachable via Front Door (public) or ACA internal network.
- The origin lock (`X-AFD-Secret`) still applies to all backend requests via Front Door.

### Runtime Config Contract

- Frontend (public):
  - `NEXT_PUBLIC_API_BASE_URL=https://<frontdoor-domain>` (used to call `/api/*`)
  - Easy Auth configuration is set at the ACA level, not via app env vars
- Backend:
  - `CORS_ORIGINS=https://<frontdoor-domain>`
  - **NEW (v2): `AFD_ORIGIN_SECRET=<value>`** (or equivalent)
  - Existing `AZURE_*`, `GRAPHRAG_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`
  - No `AUTH_*` env vars needed — token validation is handled by Easy Auth at platform level

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
4. **NEW (v2): WAF custom rules**
  - Block requests with no `User-Agent` header
  - Add geo-filtering rule if app is region-specific (optional)
5. **NEW (v2): Rate limiting for `/api/*`**
  - Configure a conservative rate limit for small traffic profile.
6. **NEW (v2): Backend Origin Lock**
  - Add a rule to inject a secret header to backend origin:
    - `X-AFD-Secret: <value>`
  - Backend validates this header for all `/api/*`.
  - Use **separate secret values per environment** (staging vs prod), stored in separate Key Vault secrets.

### Identity and Secrets

1. One user-assigned Managed Identity per **backend** environment (for Key Vault access)
2. Frontend does **not** need a Managed Identity — Easy Auth uses the ACA app registration directly
3. Grant `Key Vault Secrets User` role to each backend MI
4. Keep secrets in Key Vault only (no plaintext pipeline vars)
5. Enable diagnostic settings (Key Vault audit logs) to Log Analytics
6. Configure Key Vault secret **expiry alerts** in Azure Monitor to fire 30 days before expiration; rotate secrets on alert

### Networking / Data Access (v2)

- Minimum required for go-live:
  - Cosmos/Blob/Search: firewall locked down (no open public access by default)
  - Key Vault: Managed Identity only; restrict network where possible
- **Azure AI Search SKU decision (explicit):**
  - Free SKU does **not** support private endpoints. Choose one:
    - Option 2: Accept the risk and remain on Free SKU with firewall-only restriction; document as a known gap and schedule upgrade in Phase E
  - This must be decided before Phase B infra provisioning.
- Document egress domains/timeouts for external LLM providers and Tavily.

### Initial Scale (small traffic profile)

1. Backend staging: min 1 / max 1, 1 vCPU, 2 GiB
2. Backend prod: min 1 / max 1, 2 vCPU, 4 GiB
3. Frontend staging: min 1 / max 1
4. Frontend prod: min 1 / max 2
  - Easy Auth session is managed by ACA platform, not the app — no shared session store required for scaling.

### Important Runtime Constraint

- Current indexing execution is in-process, so backend replicas stay at `max=1` to avoid job concurrency/race risk in this release.
- **NEW (v2): Guardrail — distributed lock**
  - Implement using a **Cosmos DB lease document** in a `system_state` container:
    - On indexing start: write `{ id: "indexing-lock", status: "running", started_at: <timestamp> }` with optimistic concurrency (ETag check).
    - On indexing end: update status to `"idle"`.
    - On startup: check lock status; refuse to start indexing if `status == "running"` and `started_at` is recent (configurable threshold, e.g. 2 hours).
  - This prevents accidental concurrent indexing if max replicas is ever increased.
- **Risk note**: in-process indexing means a backend OOM crash during indexing may leave the index in a partial state. The Cosmos job recovery behavior (tested in Reliability scenarios) is the primary mitigation for this release.

## Entra ID / Easy Auth Setup

1. App registrations per environment:
  - `gtog-frontend-{env}` — used by Easy Auth on the frontend ACA
  - `gtog-backend-api-{env}` — used by Easy Auth on the backend ACA
2. Backend app registration exposes API scope:
  - `api://<backend-app-id>/access_as_user`
  - Grant this scope to the frontend app registration (admin consent required)
3. Easy Auth configuration per ACA:
  - **Frontend ACA**: provider = Microsoft (Entra), client ID = `gtog-frontend-{env}`, action = `RedirectToLoginPage`, allowed token audiences = frontend client ID
  - **Backend ACA**: provider = Microsoft (Entra), client ID = `gtog-backend-api-{env}`, action = `Return401`, allowed token audiences = backend client ID
  - Easy Auth client secrets are stored as ACA secrets (not Key Vault) — managed by Easy Auth internally
4. Redirect URIs for frontend app registration:
  - `https://<frontdoor-domain>/.auth/login/aad/callback`
  - `https://<aca-frontend-fqdn>/.auth/login/aad/callback` (for staging direct access if needed)
5. **Token isolation smoke test:** verify that a valid staging token is rejected by the prod backend Easy Auth (audience mismatch)

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
  - **NEW (v2): token isolation test**
    - Staging-issued token sent to prod backend must be rejected (audience mismatch)
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

1. Unauthenticated request to frontend → Easy Auth redirects to Entra login
2. Unauthenticated request to backend `/api/*` → Easy Auth returns `401` (no redirect)
3. Wrong audience token → Easy Auth on backend denies
4. CORS allows only configured Front Door origins
5. ToG debug endpoint remains disabled by default
6. **NEW (v2): backend origin lock**
  - Requests without `X-AFD-Secret` are denied (even if token is valid)
7. **NEW (v2): token isolation**
  - Valid staging token rejected by prod backend Easy Auth (audience mismatch)

### Reliability

1. Restart backend during indexing and verify Cosmos job recovery behavior
2. Delete collection and verify metadata/artifact cleanup path

### Performance Baseline

1. p95 `/api/collections` < 500 ms
2. p95 readiness endpoint < 1 s
3. No sustained 429/retry storm from Cosmos/Search in smoke run

### Observability Alerts

Configure the following Azure Monitor alerts against Log Analytics:

1. **5xx error rate** > 1% over any 5-minute window → severity 2
2. **p95 latency** on `/api/*` > 2 s over 10-minute window → severity 3
3. **Cosmos/Search 429 rate** > 5 per minute → severity 3
4. **Key Vault secret expiry** < 30 days remaining → severity 2
5. **ACA replica restart** count > 2 in 15 minutes → severity 2

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
  - user identity available via Easy Auth headers (no JWT middleware to add)
3. Backend settings:
  - add `CORS_ORIGINS`
  - **NEW (v2): `AFD_ORIGIN_SECRET`**

## Assumptions and Defaults

1. Azure subscription remains `1095803e-80bf-47e0-961f-3d74cb4c605c`
2. Region remains `southeastasia`
3. Existing database layer + phase5 hardening remain valid
4. **Azure AI Search SKU must be explicitly decided before Phase B** — see Networking / Data Access section
5. Worker separation for indexing is out of this deployment scope (Phase E)
6. Front Door default domain is acceptable for first go-live

## Topology Reference

- See: `docs/topo_v2.md`

