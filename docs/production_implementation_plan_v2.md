# GToG Production Implementation Plan v2

> This version simplifies production delivery around one rule: every request enters through Cloudflare, and no Azure application origin is publicly exposed.

## 1. Scope

This document translates the simplified deployment architecture into an execution-ready production plan.

### In scope

- Cloudflare-only ingress for frontend and API
- One private Azure Container Apps environment per environment
- Frontend, API, worker, and tunnel connector deployment
- Backend auth, CORS, health, and SSE behavior
- Async job execution with API + worker split
- Minimal CI/CD and smoke validation
- Simple rollback to the last known good revision

### Out of scope

- Canary rollout in the initial production release
- Full private-endpoint rollout for all Azure dependencies
- Cloudflare paid-only features in the initial rollout
- Full disaster-recovery automation

### Referenced docs

- `docs/topo_v4.md`
- `docs/development_plan_v2.md`

---

## 2. Simplification Principles

1. **One public edge:** Cloudflare is the only internet-facing entrypoint.
2. **One private compute layer:** frontend, API, and worker run inside a private ACA environment.
3. **One tunnel per environment:** the same tunnel publishes both `app.<domain>` and `api.<domain>`.
4. **One deploy path:** `Validate -> BuildImages -> Deploy -> Smoke`.
5. **Defer non-essential hardening:** ship the private ingress model first, then add deeper network controls later.

---

## 3. Frozen Production Decisions

### Compute layout

Use one ACA environment per environment and deploy four workloads:

- frontend
- API
- worker
- cloudflared connector

Rules:

- the ACA environment must not expose a public application origin for frontend or API traffic
- frontend uses **internal ingress only**
- API uses **internal ingress only**
- worker has **no ingress**
- connector publishes the internal services through Cloudflare Tunnel

### Hostname model

Use two public hostnames, both owned at the Cloudflare edge:

- `https://app.<domain>` -> frontend through Cloudflare Tunnel
- `https://api.<domain>` -> API through Cloudflare Tunnel

Do not expose ACA custom domains directly.

### Auth model

Keep backend-only Easy Auth on `api.<domain>`.

- frontend login redirects:
  - `https://api.<domain>/.auth/login/aad?post_login_redirect_uri=https://app.<domain>/`
  - `https://api.<domain>/.auth/login/google?post_login_redirect_uri=https://app.<domain>/`
- frontend logout redirect:
  - `https://api.<domain>/.auth/logout?post_logout_redirect_uri=https://app.<domain>/`
- session inspection:
  - `GET https://api.<domain>/.auth/me`
- Microsoft callback:
  - `https://api.<domain>/.auth/login/aad/callback`
- Google callback:
  - `https://api.<domain>/.auth/login/google/callback`

Optional:

- add Cloudflare Access to `app.<domain>` if the application should be visible only to approved users
- keep Easy Auth on `api.<domain>` as the primary API authentication boundary; Access on `api.<domain>` is optional defense in depth, not a required control

### Background job model

Use the current API + worker split as the production execution boundary:

- API validates the request
- API creates the job record in Cosmos DB
- API enqueues a minimal dispatch message to Azure Storage Queue
- API returns immediately with `job_id`
- worker processes the long-running job asynchronously

Rules:

- Cosmos DB is the source of truth for job state
- queue messages are dispatch signals only
- worker recovery must rely on Cosmos state, not process memory

### Data-service stance

Keep the managed Azure data services simple for v1:

- Azure Cosmos DB for job state and control-plane metadata
- Azure Storage Queue for dispatch
- Azure Blob Storage for uploaded documents and generated artifacts
- Azure Key Vault for secrets
- Azure AI Search for serving indexes

Accepted simplification:

- full private-endpoint rollout for Cosmos, Blob, and Key Vault is deferred
- Azure AI Search Free SKU remains an accepted production exception

### Cloudflare stance

Use Cloudflare Free for the initial rollout.

Required controls:

- proxied DNS and tunnel routing
- managed ruleset
- one rate-limiting rule on the API host
- cache bypass for `/api/`*, `/.auth/*`, and all SSE routes

Optional control:

- `EDGE_ORIGIN_SECRET` as a secondary defense-in-depth signal only

---

## 4. System Contracts

### Frontend ↔ API contract

- frontend UI is served from `https://app.<domain>`
- API and auth endpoints are served from `https://api.<domain>`
- `NEXT_PUBLIC_API_BASE_URL` must be `https://api.<domain>`
- standard API calls target `https://api.<domain>/api/*`
- session inspection uses `GET https://api.<domain>/.auth/me`
- `CORS_ORIGINS` allows only `https://app.<domain>`
- SSE responses emit heartbeat events every 25-30 seconds and disable caching/buffering

### API ↔ Worker contract

- API remains synchronous and stateless for request handling
- worker handles indexing and any long-running graph work
- queue messages contain only minimal routing metadata such as:
  - `job_id`
  - `job_type`
  - `attempt`

### Worker ↔ Cosmos contract

Cosmos DB stores:

- job metadata
- lifecycle state
- retry count
- lease ownership
- timestamps
- sanitized error summary
- resumability state

Required lifecycle states:

- `queued`
- `running`
- `retrying`
- `failed`
- `completed`
- `cancelled`

### Security boundary contract

- Cloudflare is the only public ingress for frontend and API
- frontend and API ACA origins are internal-only
- direct-origin probes must fail at the network layer
- backend Easy Auth protects authenticated API access
- `EDGE_ORIGIN_SECRET` is optional and secondary only
- staging and production must use separate tunnel tokens, secrets, and Entra configuration

### Logging and observability contract

All backend, worker, connector, and frontend logs should be structured where possible.

Required fields where available:

- request ID
- `Cf-Ray`
- principal ID
- `job_id`
- service role

Never log:

- secrets
- tokens
- raw credential material

---

## 5. Delivery Phases

### Phase 1: Private ACA foundation

**Goal:** Provision the private runtime foundation.

**Deliverables**

- one ACA environment
- frontend app with internal ingress
- API app with internal ingress
- worker app with no ingress
- cloudflared connector app with at least 2 replicas
- Key Vault-backed secret configuration

**Exit criteria**

- frontend is reachable from the connector only
- API is reachable from the connector only
- worker is not reachable over HTTP
- connector can route to both frontend and API services

**Implementation notes**

- use the existing ACA provisioning scripts as the deployment entrypoint where possible
- keep managed identity and Key Vault references in the first release
- do not introduce a second ingress path outside the tunnel

---

### Phase 2: Application deployment and runtime configuration

**Goal:** Deploy the application images and lock the runtime contract.

**Deliverables**

- frontend image configured with `NEXT_PUBLIC_API_BASE_URL`
- backend image reused for both API and worker roles
- queue, Cosmos, storage, and secret settings configured
- readiness endpoint, CORS, and SSE behavior enabled
- worker capable of processing a test job

**Exit criteria**

- frontend loads successfully behind the internal route
- API health and readiness pass
- job submission returns immediately
- worker completes a test job and updates durable state

**Implementation notes**

- keep API request handling stateless
- keep long-running work out of the API process
- reuse the same backend image for API and worker to keep build and deploy simpler

---

### Phase 3: Cloudflare configuration

**Goal:** Make Cloudflare the only public entrypoint.

**Deliverables**

- one tunnel per environment
- route for `app.<domain>`
- route for `api.<domain>`
- managed ruleset enabled
- API rate-limiting rule enabled
- cache bypass rules for `/api/`*, `/.auth/*`, and SSE
- optional Cloudflare Access policy for `app.<domain>` if required

**Exit criteria**

- app and API are reachable through Cloudflare hostnames
- direct ACA origin access is not possible
- tunnel failover across connector replicas works

**Implementation notes**

- keep both public hostnames on the same tunnel unless scale or ownership later requires a split
- prefer the minimal Cloudflare rule set needed to support auth, API traffic, and SSE

---

### Phase 4: Smoke validation and go-live

**Goal:** Validate the critical production flow and release.

**Required smoke checks**

- `https://app.<domain>` loads
- `https://api.<domain>/health` returns success
- `https://api.<domain>/health/readiness` returns success
- login flow succeeds
- `GET https://api.<domain>/.auth/me` returns the expected session shape
- one authenticated API request succeeds
- one indexing job is submitted successfully
- job status polling works
- one SSE stream stays alive with heartbeat events
- direct-origin probe to ACA fails

**Exit criteria**

- all critical smoke checks pass
- the last known good ACA revision is identified
- accepted risks are recorded in the release notes

---

## 6. Minimal CI/CD Plan

Use a small release pipeline with four stages.

### Stage 1: `Validate`

Run before any deployment:

- dependency install or sync
- backend and library tests
- frontend build validation
- linting and formatting checks
- static analysis that is already part of the repository workflow

### Stage 2: `BuildImages`

Build and publish:

- frontend image
- backend image

Retain:

- image tags
- image digests
- commit SHA

### Stage 3: `Deploy`

Deploy or update:

- frontend app
- API app
- worker app
- cloudflared connector
- environment variables and Key Vault references

### Stage 4: `Smoke`

Run the critical smoke checks through Cloudflare hostnames only.

If staging exists:

- run `Deploy` + `Smoke` in staging first
- promote the same built images to production

For the initial release:

- no canary rollout is required
- manual approval is recommended before production, but the deployment flow itself stays the same

---

## 7. Validation Matrix


| Control                 | Validation method                                                 | Expected result                                             |
| ----------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------- |
| Cloudflare-only ingress | Access `app.<domain>` and `api.<domain>` through public hostnames | Both hosts work through Cloudflare                          |
| No public ACA origin    | Probe ACA frontend/API origin directly                            | Probe fails at network layer                                |
| Frontend/API split      | Inspect frontend runtime config                                   | `NEXT_PUBLIC_API_BASE_URL` points to `https://api.<domain>` |
| Auth flow               | Login and call `/.auth/me`                                        | Session is returned correctly                               |
| CORS restriction        | Test allowed and disallowed origins                               | Only `https://app.<domain>` is allowed                      |
| Health/readiness        | Call `/health` and `/health/readiness`                            | Both behave as documented                                   |
| Async job flow          | Submit indexing request and poll status                           | Request returns quickly and job completes asynchronously    |
| Worker durability       | Restart worker during test job                                    | Job state remains durable and recoverable                   |
| SSE stability           | Hold an SSE connection through Cloudflare                         | Heartbeats continue and stream stays usable                 |
| Connector redundancy    | Stop one connector replica                                        | Service remains available                                   |


---

## 8. Release Checklist

### Configuration and secrets

- all required environment variables exist for frontend, API, worker, and connector
- staging and production use separate tunnel tokens, secrets, and Entra values
- Key Vault references resolve successfully
- no secrets are hardcoded in source or deployment config

### Networking and ingress

- `app.<domain>` is served through Cloudflare Tunnel
- `api.<domain>` is served through Cloudflare Tunnel
- frontend and API use internal ingress only
- worker has no public ingress
- at least two connector replicas are running
- no direct ACA public path remains enabled

### Authentication and browser behavior

- backend Easy Auth is enabled on `api.<domain>`
- login callback works
- `/.auth/me` works after login
- unauthenticated protected API requests are rejected
- CORS allows only `https://app.<domain>`

### Application behavior

- `/health` works
- `/health/readiness` works
- SSE works with heartbeat events
- job submission and status polling work
- worker performs long-running work instead of the API process

### Release evidence

- validate output retained
- image metadata retained
- deploy logs retained
- smoke results retained
- direct-origin denial evidence retained
- accepted-risk note retained for Azure AI Search Free SKU

### Rollback readiness

- the last known good ACA revision is identified before rollout
- rollback steps are documented and tested in a lower environment when possible

---

## 9. Accepted Risks and Deferred Work

### Accepted risks

- Azure AI Search Free SKU remains a temporary production exception
- full private-endpoint rollout for Cosmos DB, Blob Storage, and Key Vault is deferred
- Cloudflare Free-tier protections are limited to the basic controls listed in this plan

### Deferred improvements

- add private endpoints and tighter network isolation for managed data services
- add canary rollout after the simple deploy path is stable
- expand Cloudflare policy if Free-tier limits become insufficient
- add Cloudflare Access if the application needs a stricter private-user gate

---

## 10. Production Sign-off Rule

Production is ready only when:

1. `app.<domain>` and `api.<domain>` are reachable through Cloudflare,
2. frontend and API ACA origins are private,
3. critical smoke validation passes,
4. a rollback target exists,
5. accepted risks are documented.

