# Production Implementation Plan v2

## Hostname Model

Use two public hostnames, both owned at the Cloudflare edge:

- `https://app.<domain>` -> frontend through Cloudflare Tunnel
- `https://api.<domain>` -> API through Cloudflare Tunnel

Do not expose ACA custom domains directly to the public Internet.

## Runtime Model

- frontend and API use internal ACA ingress only
- worker has no ingress
- queue and Cosmos provide the durable execution boundary
- `EDGE_ORIGIN_SECRET` remains optional secondary defense in depth
- the Azure origin stays private, but `app.<domain>` and `api.<domain>` remain publicly reachable through Cloudflare

## System Contracts

### Frontend ↔ API

- frontend UI is served from `https://app.<domain>`
- API is served from `https://api.<domain>/api/*`
- `NEXT_PUBLIC_API_BASE_URL` must point to `https://api.<domain>`
- `CORS_ORIGINS` allows only `https://app.<domain>`
- SSE emits heartbeat events every 25-30 seconds

### Security boundary

- Cloudflare is the only public ingress for frontend and API
- frontend and API ACA origins are internal-only
- direct-origin probes must fail before app handling
- platform-level authentication is not part of the deployed topology contract
- staging and production use separate tunnel tokens and secrets

### Logging and observability

Required fields where available:

- request ID
- `Cf-Ray`
- `CF-Connecting-IP`
- `job_id`
- service role

## Delivery Phases

### Phase 1: Private ACA foundation

- one ACA environment
- frontend app with internal ingress
- API app with internal ingress
- worker app with no ingress
- cloudflared connector app with at least two replicas

### Phase 2: Application deployment and runtime configuration

- frontend image configured with `NEXT_PUBLIC_API_BASE_URL`
- backend image reused for API and worker roles
- queue, Cosmos, storage, and secret settings configured
- readiness endpoint, CORS, and SSE behavior enabled

### Phase 3: Cloudflare configuration

- one tunnel per environment
- route for `app.<domain>`
- route for `api.<domain>`
- managed ruleset enabled
- API rate-limiting rule enabled
- cache bypass rules for API and SSE

### Phase 4: Smoke validation and go-live

Required smoke checks:

- `https://app.<domain>` loads
- `https://api.<domain>/health` succeeds
- `https://api.<domain>/health/readiness` succeeds
- CRUD, upload, indexing, polling, and query flows succeed
- one SSE stream stays alive with heartbeat events
- direct-origin probe to ACA fails

## Validation Matrix

| Control | Validation method | Expected result |
| --- | --- | --- |
| Cloudflare-only ingress | Access `app.<domain>` and `api.<domain>` | Both hosts work only through Cloudflare |
| No public ACA origin | Probe ACA origin directly | Probe fails before app handling |
| Public app, private origin | Access `app.<domain>` and `api.<domain>` from the Internet | Cloudflare serves the public hosts while ACA origin remains private |
| Frontend/API split | Inspect frontend runtime config | `NEXT_PUBLIC_API_BASE_URL` points to `https://api.<domain>` |
| CORS restriction | Test allowed and disallowed origins | Only `https://app.<domain>` is allowed |
| Health/readiness | Call `/health` and `/health/readiness` | Both behave as documented |
| Async job flow | Submit indexing request and poll status | Request returns quickly and job completes asynchronously |
| SSE stability | Hold an SSE connection through Cloudflare | Heartbeats continue and stream stays usable |
| Connector redundancy | Stop one connector replica | Service remains available |

## Release Checklist

- all required environment variables exist for frontend, API, worker, and connector
- staging and production use separate tunnel tokens and secrets
- frontend and API use internal ingress only
- worker has no public ingress
- at least two connector replicas are running
- no direct ACA public path remains enabled
- smoke results and private-origin evidence are retained
