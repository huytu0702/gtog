# GToG Production Implementation Plan

> This document translates the deployment architecture into an execution-ready production delivery plan. It freezes key decisions, defines system contracts, sets validation requirements, and establishes release and runbook requirements for staging and production rollout.

## 1. Scope

This document turns the target deployment architecture into an execution-ready production plan.

### In scope

- Frontend, API, worker, and tunnel deployment model
- Auth and hostname model
- Queue and worker orchestration
- Private backend networking
- Cloudflare edge policy
- CI/CD promotion gates
- Validation, rollback, and runbooks

### Out of scope

- Re-provisioning the existing data layer from scratch
- Replacing Azure AI Search Free SKU in this release
- Adopting Cloudflare paid-only features in the initial rollout

### Referenced docs

- `docs/development_plan_v2.md`
- `docs/topo_v3.md`

---

## 2. Frozen Production Decisions

### Queue choice

Use **Azure Storage Queue** for background dispatch.

- Cosmos DB is the source of truth for job state.
- Queue messages are dispatch signals only.
- Retry/backoff and poison handling are implemented in worker logic.

### API hostname and domain binding

Use `https://api.<domain>` as the public API hostname, owned at the **Cloudflare edge**.

- The ACA API remains **internal-only**.
- Do not expose the ACA API as a separate public custom-domain origin.
- Browser traffic always targets `api.<domain>`.

### Cloudflare Tunnel ownership model

Use **one dedicated tunnel per environment**, managed as infrastructure.

- Staging tunnel: `gtog-stg-api`
- Production tunnel: `gtog-prod-api`
- Run at least **2 tunnel connector replicas** per environment.
- Store tunnel tokens in Key Vault.

### Easy Auth callback model

Use **backend-only Easy Auth** on `api.<domain>`.

- Canonical callback:
  - `https://api.<domain>/.auth/login/aad/callback`
- Frontend initiates login and logout through the backend host.
- Frontend reads session state from `https://api.<domain>/.auth/me`.

### AI Search private-network stance

Keep **Azure AI Search Free SKU** for the initial rollout.

- This is an **accepted temporary production risk**.
- Full private-network alignment for Search is deferred to a later hardening phase.
- This exception must appear in release records and sign-off artifacts.

### `EDGE_ORIGIN_SECRET` production stance

Enable `EDGE_ORIGIN_SECRET` in production as a secondary defense-in-depth control.

- Use a different value per environment.
- Store it in Key Vault.
- Never log the secret value.
- Rotate it on schedule and after incidents.

### Cloudflare plan stance

Use **Cloudflare Free** for the initial rollout.

- Supported in this design:
  - proxied DNS for `app.<domain>`
  - Tunnel for `api.<domain>`
  - Free Managed Ruleset
  - limited custom WAF rules
  - one rate-limiting rule
- Backend fallback protections remain required because Free-tier edge policy is limited.

---

## 3. System Contracts

### Frontend ↔ API contract

- Frontend UI is served from `https://app.<domain>`.
- API and auth endpoints are served from `https://api.<domain>`.
- Frontend login redirect:
  - `https://api.<domain>/.auth/login/aad?post_login_redirect_uri=https://app.<domain>/`
- Frontend logout redirect:
  - `https://api.<domain>/.auth/logout?post_logout_redirect_uri=https://app.<domain>/`
- Session inspection:
  - `GET https://api.<domain>/.auth/me`
- Standard API calls target:
  - `https://api.<domain>/api/*`
- Bearer token is used for normal API calls when available.
- SSE uses Easy Auth cookie on `api.<domain>`.
- CORS allows only:
  - `https://app.<domain>`

### API ↔ Worker contract

- The API stays synchronous and stateless for request handling.
- Long-running work is dispatched asynchronously.
- API flow:
  1. validate request
  2. create job record in Cosmos DB
  3. enqueue dispatch message to Azure Storage Queue
  4. return `job_id` immediately

### Worker ↔ Queue contract

- Queue messages contain only minimal routing metadata:
  - `job_id`
  - `job_type`
  - `attempt`
- Worker loads authoritative state from Cosmos DB.
- Worker updates job state before deleting the queue message.

### Worker ↔ Cosmos contract

Cosmos DB stores:

- job metadata
- lifecycle state
- retry count
- lease ownership
- timestamps
- sanitized error summary
- resumability state

Required job states:

- `queued`
- `running`
- `retrying`
- `failed`
- `completed`
- `cancelled`

### Lease and recovery contract

- Only one worker may own an active lease at a time.
- Lease record must include:
  - `lease_owner_id`
  - `lease_acquired_at`
  - `lease_expires_at`
  - `heartbeat_at`
- Expired leases may be reclaimed.
- Recovery after worker crash must rely on Cosmos state, not memory.
- Reprocessing must be idempotent.

### Health and readiness contract

- `GET /health` is liveness-only.
- `GET /health/readiness` checks:
  - Cosmos DB
  - Key Vault
  - Blob Storage
  - Azure AI Search
- External LLM providers and Tavily are excluded from readiness.
- Frontend exposes:
  - `app/api/health/route.ts`
  - returns `{ "status": "ok" }`

### Security boundary contract

- `api.<domain>` is reachable only through Cloudflare Tunnel.
- ACA API ingress is internal-only.
- Direct public-origin probes must fail at the network layer.
- Backend Easy Auth protects authenticated API access.
- `EDGE_ORIGIN_SECRET` is secondary control only.
- Staging and production are isolated for:
  - tunnels
  - secrets
  - Entra registrations
  - app settings

### Logging and correlation contract

All backend and worker logs must be structured JSON.

Required fields where available:

- application request ID
- `Cf-Ray`
- `CF-Connecting-IP`
- principal ID
- `job_id`

Never log:

- secrets
- tokens
- raw credential material

---

## 4. Delivery Phases

### Phase 1: Application boundary changes

**Goal:** Align frontend and backend behavior with the dual-host production model.

**Deliverables**

- Frontend uses `NEXT_PUBLIC_API_BASE_URL=https://api.<domain>`
- Frontend login/logout/session flows target backend auth host
- Backend replaces wildcard CORS with `CORS_ORIGINS`
- Backend adds `/health/readiness`
- Backend logs `Cf-Ray` and `CF-Connecting-IP`
- SSE emits heartbeat events and disables caching/buffering

**Exit criteria**

- No remaining same-host assumption between UI and API
- Auth flow shape matches production host model
- CORS and health behavior are validated

**Implementation plan**

1. **Confirm frontend dual-host boundaries**
   - Verify `frontend/lib/api.ts` uses `NEXT_PUBLIC_API_BASE_URL` for all API and auth-facing paths.
   - Verify login, logout, and session inspection target `https://api.<domain>/.auth/*`.
   - Verify `frontend/components/ui/AuthLinks.tsx` and `frontend/components/ui/NBLayout.tsx` do not assume same-host API routing.
   - Verify `frontend/next.config.ts` allows the backend origin in the deployed CSP/connect policy.

2. **Lock backend boundary controls**
   - Verify `backend/app/config.py` exposes `CORS_ORIGINS` and `backend/app/main.py` parses it into an explicit allowlist.
   - Verify `backend/app/main.py` applies `CORSMiddleware` with the configured origins only.
   - Verify `/health` remains liveness-only and `/health/readiness` checks Cosmos DB, Blob Storage, Key Vault, and Azure AI Search.
   - Verify request logging captures the `Cf-Ray` and `CF-Connecting-IP` headers as structured fields such as `cf_ray` and `cf_connecting_ip` without logging secrets or tokens.

3. **Close the SSE production gap**
   - Update `backend/app/routers/search.py` SSE responses to emit heartbeat events on a fixed cadence during long-lived streams.
   - Preserve anti-buffering and anti-caching headers for Cloudflare and proxy compatibility, including `Cache-Control: no-cache` and `X-Accel-Buffering: no`.
   - Verify SSE endpoints continue to support the backend-auth-host cookie model.

4. **Add validation coverage before phase sign-off**
   - Extend backend tests to cover allowed-origin and rejected-origin CORS behavior.
   - Extend readiness tests to preserve the 200/503 dependency contract.
   - Add or update SSE tests to verify heartbeat events and buffering/cache headers.
   - Add request logging assertions for `Cf-Ray` and `CF-Connecting-IP` when those headers are present.

5. **Align deployment-facing configuration and docs**
   - Verify frontend build/runtime configuration sets `NEXT_PUBLIC_API_BASE_URL` per environment.
   - Verify backend environment examples and deployment manifests document `CORS_ORIGINS` and `EDGE_ORIGIN_SECRET` consistently.
   - Record Phase 1 evidence in the release checklist and validation bundle.

**Recommended execution order**

1. Implement SSE heartbeat support and associated tests.
2. Complete CORS, readiness, and logging validation coverage.
3. Reconcile deployment configuration examples and documentation.
4. Run Phase 1 smoke validation against the staging host split.

**Phase 1 validation evidence**

- Frontend login, logout, and `/.auth/me` flow evidence against `api.<domain>`
- Backend CORS allowlist test evidence
- `/health` and `/health/readiness` verification output
- Structured log sample showing `Cf-Ray` and `CF-Connecting-IP`
- SSE stream evidence showing heartbeat cadence and no-buffering headers

**Phase 1 risks and focus points**

- The primary remaining implementation risk is SSE idle-stream stability through Cloudflare without heartbeat traffic.
- Configuration drift between local compose, environment examples, and deployed settings can reintroduce same-host assumptions.
- Phase sign-off should be blocked if any auth path, CORS rule, or readiness dependency still depends on local-only defaults.

---

### Phase 2: Worker and job orchestration

**Goal:** Move indexing and long-running graph work off the synchronous API path.

**Deliverables**

- Job records persisted in Cosmos DB
- Dispatch messages sent to Azure Storage Queue
- Worker consumes queued jobs
- Durable lifecycle states and lease ownership implemented
- Retry and poison/failure handling implemented
- Job status endpoint available

**Exit criteria**

- API returns immediately after job submission
- Job state survives worker restart
- Duplicate dispatch does not create duplicate final artifacts

---

### Phase 3: Private networking and identity

**Goal:** Deploy backend components into a private ACA environment with backend-only Easy Auth.

**Deliverables**

- Private ACA backend environment
- API with internal ingress only
- Worker with no public ingress
- Entra app registration and backend Easy Auth config
- Allowed audience configuration enforced

**Exit criteria**

- Direct-origin probes fail before app layer
- Easy Auth callback works on `api.<domain>`
- Staging token is rejected by production

---

### Phase 4: Cloudflare edge and tunnel hardening

**Goal:** Make Cloudflare the stable public edge for both application hosts.

**Deliverables**

- `app.<domain>` proxied through Cloudflare
- `api.<domain>` published through Tunnel
- Two tunnel connector replicas
- Free Managed Ruleset enabled
- Cache bypass configured for:
  - `/api/*`
  - `/.auth/*`
  - SSE routes
- Initial rate-limiting rule configured
- `EDGE_ORIGIN_SECRET` header injection configured if enabled

**Exit criteria**

- API is reachable only through Tunnel
- Tunnel failover works
- SSE remains stable through Cloudflare

---

### Phase 5: CI/CD and security release gates

**Goal:** Automate validation, deployment, smoke testing, and promotion gates.

**Deliverables**

- Azure DevOps stages:
  - `Validate`
  - `BuildImages`
  - `DeployStaging`
  - `SmokeStaging`
  - `ManualApproval`
  - `DeployProduction`
- Smoke suite covers:
  - health
  - auth
  - CRUD
  - upload
  - indexing
  - status polling
  - query methods
  - SSE
  - direct-origin denial
  - audience isolation
  - tunnel failover
- Rollback drill completed in staging

**Exit criteria**

- Staging deployment is repeatable through pipeline only
- Production requires explicit manual approval
- Rollback path is tested and documented

---

### Phase 6: Production validation and controlled rollout

**Goal:** Promote to production with canary rollout, evidence-based sign-off, and immediate rollback readiness.

**Deliverables**

- Production deployment through approved pipeline path
- Canary traffic split before full promotion
- Active monitoring during rollout
- Final evidence bundle retained with release record

**Exit criteria**

- Canary is healthy
- Full promotion succeeds
- Accepted risks are documented
- Rollback remains available during rollout

---

## 5. Validation Matrix

Each critical requirement must define:

- control
- validation method
- expected result
- retained evidence artifact

### Mandatory validation areas

- frontend reachability at `app.<domain>`
- API reachability at `api.<domain>`
- direct-origin denial for ACA API
- unauthenticated `/api/`* returns `401`
- wrong-audience token rejection
- staging token rejection in production
- CORS restriction to `app.<domain>`
- `EDGE_ORIGIN_SECRET` enforcement when enabled
- Easy Auth callback success
- `/.auth/me` session correctness
- SSE stability for long-running streams
- tunnel failover behavior
- tunnel health observability
- backend log correlation fields
- readiness/liveness correctness
- async job dispatch behavior
- durable job state across worker restart
- idempotency and duplicate protection
- retry/failure handling
- CRUD/upload/index/query flow
- ToG debug endpoint disabled by default
- rollback drill success
- restore drill success
- alert configuration and routing
- AI Search Free SKU exception documented

### Minimum evidence bundle

- staging smoke report
- production smoke report
- direct-origin denial evidence
- auth audience isolation result
- SSE stream evidence
- tunnel failover evidence
- rollback drill evidence
- restore drill evidence
- alert evidence
- accepted-risk record for AI Search Free SKU

---

## 6. Release Checklist

### Configuration and secrets

- All required environment variables exist for frontend, API, worker, and tunnel
- Staging and production use separate tunnel tokens, Entra registrations, and secrets
- Key Vault references resolve successfully
- No secrets are hardcoded in code, images, or pipeline YAML

### Networking and origin isolation

- `app.<domain>` is proxied through Cloudflare
- `api.<domain>` is served through Cloudflare Tunnel
- ACA API ingress is internal-only
- Public-origin probes fail at the network layer
- No public-origin fallback path remains enabled
- At least two tunnel replicas are running

### Authentication and authorization

- Backend Easy Auth is enabled
- Callback works at `https://api.<domain>/.auth/login/aad/callback`
- `/.auth/me` works after login
- Unauthenticated `/api/*` requests return `401`
- Wrong-audience tokens are rejected
- Staging tokens are rejected by production

### Application behavior

- Frontend uses `https://api.<domain>`
- CORS allows only `https://app.<domain>`
- `/health` works as liveness
- `/health/readiness` checks required dependencies
- SSE works through Cloudflare with heartbeat events
- Long-running jobs run through worker path only

### Background job durability

- API writes job metadata before queue dispatch
- Worker processes queue messages successfully
- Job lifecycle transitions are correct
- Lease expiry and recovery are validated
- Duplicate dispatch does not duplicate artifacts
- Poison/failure path is tested

### Cloudflare edge policy

- Free Managed Ruleset is enabled
- Cache bypass is configured
- Initial rate-limiting rule is active
- Required custom WAF rules fit within Free-tier limits
- `EDGE_ORIGIN_SECRET` enforcement is active when enabled

### CI/CD and release controls

- Validate stage passes
- BuildImages stage passes
- DeployStaging stage passes
- SmokeStaging stage passes
- Manual approval is recorded before production
- Canary rollout is used before full promotion
- Rollback has been tested in staging

### Observability and alerting

- Backend logs are structured JSON
- Logs include Cloudflare correlation fields
- Worker logs include lifecycle state transitions
- Tunnel health and reconnect events are visible
- Alert rules exist and route correctly

### Recovery and operations

- Rollback runbook exists and is current
- Backup and restore runbook exists and is current
- Tunnel rotation runbook exists and is current
- Origin-bypass verification runbook exists and is current
- Release promotion checklist exists and is current

### Accepted risks

- Azure AI Search Free SKU exception is documented
- No undocumented production exceptions remain

---

## 7. Required Runbook Set

### `docs/runbooks/rollback.md`

Must cover:

- rollback triggers
- ACA revision rollback steps
- post-rollback verification
- approval rules for break-glass actions

### `docs/runbooks/backup-restore.md`

Must cover:

- Cosmos restore
- Blob recovery
- AI Search rebuild
- restore verification
- target RPO and RTO
- drill cadence

### `docs/runbooks/cloudflare-tunnel-rotation.md`

Must cover:

- token ownership and storage location
- planned rotation steps
- rollback steps
- emergency rotation process

### `docs/runbooks/origin-bypass-verification.md`

Must cover:

- public probe procedure
- expected failure modes
- evidence capture format
- pass/fail criteria

### `docs/runbooks/release-promotion-checklist.md`

Must cover:

- promotion prerequisites
- mandatory evidence bundle
- canary approval steps
- full promotion criteria
- rollback decision points

### Optional: `docs/runbooks/incident-triage.md`

Recommended contents:

- first logs and dashboards to inspect
- common failure patterns
- immediate containment actions
- escalation path

---

## 8. Accepted Risks and Deferred Improvements

### Accepted risk

The initial production rollout keeps **Azure AI Search Free SKU** as a temporary exception.

This means:

- full private-network alignment is not yet achieved for Search
- the exception must be documented in release records
- the exception must be revisited in a later hardening phase

### Deferred improvements

- Upgrade AI Search to a SKU that supports stronger private-network alignment
- Expand Cloudflare protections if Free-tier limits become insufficient
- Revisit worker autoscaling after real production traffic is observed

---

## 9. Production Sign-off Rule

Production is ready only when:

1. mandatory checklist items are complete,
2. validation evidence exists for critical controls,
3. required runbooks exist and are current,
4. accepted risks are explicitly documented,
5. rollback has been tested and remains available.

