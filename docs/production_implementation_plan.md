# Production Implementation Plan

## Core Decisions

### Cloudflare ingress model

Use Cloudflare as the only public edge for both frontend and API.

- `https://app.<domain>` goes through Cloudflare to the frontend private origin
- `https://api.<domain>` goes through Cloudflare Tunnel to the API private origin
- no ACA application origin is exposed publicly

### Edge secret stance

Use `EDGE_ORIGIN_SECRET` only as secondary defense in depth.

- keep a different value per environment
- store it in Key Vault
- do not treat it as the primary origin lock

### Data-service stance

- reuse Cosmos DB, Blob Storage, Key Vault, and Azure AI Search
- Azure AI Search Free SKU remains an accepted temporary production risk

## System Contracts

### Frontend to API

- frontend UI is served from `https://app.<domain>`
- API traffic is served from `https://api.<domain>/api/*`
- frontend build config must set `NEXT_PUBLIC_API_BASE_URL=https://api.<domain>`
- CORS allows only `https://app.<domain>`
- SSE stays on the API host and must emit heartbeat events

### API to Worker

- API remains synchronous and stateless for request handling
- long-running work is dispatched asynchronously
- API writes job metadata first, then enqueues queue messages
- worker owns indexing and other long-running execution

### Security boundary

- `api.<domain>` is reachable only through Cloudflare Tunnel
- ACA API ingress is internal-only
- direct-origin probes must fail at the network layer or before application handling
- `REQUIRE_EDGE_AUTH=true` is the runtime default for deployed API apps
- staging and production stay isolated for tunnels, secrets, and app settings

### Logging and correlation

Required fields where available:

- request ID
- `Cf-Ray`
- `CF-Connecting-IP`
- `job_id`
- service role

Never log:

- secrets
- tokens
- raw credential material

## Delivery Phases

### Phase 1: Boundary alignment

Goal:

- align frontend and backend with the dual-host production model

Deliverables:

- `NEXT_PUBLIC_API_BASE_URL` is environment-specific
- backend CORS allowlist is explicit
- readiness endpoint is available
- SSE heartbeats and anti-buffering headers are present
- request logs capture Cloudflare correlation headers

### Phase 2: Worker durability

Goal:

- move long-running work off the API process

Deliverables:

- durable job records in Cosmos DB
- queue-backed dispatch
- worker runtime for indexing
- lease ownership and retry behavior

### Phase 3: Private ACA topology

Goal:

- enforce private-origin deployment for frontend, API, and worker

Deliverables:

- private ACA environment
- frontend and API with internal ingress only
- worker with no ingress
- tunnel connector with at least two replicas

Exit criteria:

- direct-origin probes fail before app handling
- API and frontend work only through Cloudflare
- tunnel failover succeeds

### Phase 4: Cloudflare hardening

Goal:

- make Cloudflare the stable ingress boundary

Deliverables:

- proxied DNS and tunnel routes
- managed rules
- API rate limiting
- cache bypass for API and SSE
- optional `X-Edge-Secret` injection

### Phase 5: CI/CD and release gates

Goal:

- automate validation, deployment, smoke testing, and promotion gates

Smoke coverage must include:

- health and readiness
- CRUD flow
- upload flow
- indexing submission and polling
- query methods
- SSE behavior
- direct-origin denial
- tunnel failover

Required evidence:

- validate report
- image metadata
- deploy logs
- smoke report
- private-origin validation helper output
- rollback drill evidence

## Validation Matrix

Mandatory validation areas:

- frontend reachability at `app.<domain>`
- API reachability at `api.<domain>`
- direct-origin denial for ACA frontend and API
- CORS restriction to `app.<domain>`
- `EDGE_ORIGIN_SECRET` enforcement when enabled
- SSE stability through Cloudflare
- tunnel failover behavior
- backend log correlation fields
- readiness and liveness behavior
- async job dispatch behavior
- durable job state across worker restart
- CRUD, upload, indexing, query, and streaming flows

## Release Checklist

### Configuration and secrets

- all required environment variables exist for frontend, API, worker, and tunnel
- staging and production use separate tunnel tokens and secrets
- Key Vault references resolve successfully

### Networking and origin isolation

- `app.<domain>` is served through Cloudflare
- `api.<domain>` is served through Cloudflare Tunnel
- frontend and API ACA ingress are internal-only
- worker has no ingress
- direct-origin probes fail at the network layer
- at least two tunnel replicas are running

### Application behavior

- frontend uses `https://api.<domain>`
- CORS allows only `https://app.<domain>`
- `/health` works as liveness
- `/health/readiness` checks required dependencies
- SSE works through Cloudflare with heartbeat events
- long-running jobs run through the worker path only

### Release controls

- validate stage passes
- build stage passes
- deploy staging passes
- smoke staging passes
- manual approval is recorded before production
- canary or staged production rollout retains rollback state
