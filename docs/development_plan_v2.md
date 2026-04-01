# Full Deployment Plan v2 - GToG (ACA + Cloudflare Edge)

## Summary

Deploy the full stack to Azure using:

- Azure Container Apps for frontend, API, worker, and tunnel connector
- Cloudflare proxied DNS and Cloudflare Tunnel as the only public ingress
- Azure DevOps YAML CI/CD
- Azure Cosmos DB, Blob Storage, Azure AI Search, and Key Vault as reused data services

Environment scope: `staging` then `production`

This revision removes Azure Easy Auth and Microsoft Entra from the deployment contract. The platform boundary is now:

- Cloudflare-only public ingress
- private ACA origin for frontend and API
- optional `X-Edge-Secret` as defense in depth
- application/runtime auth decisions handled inside the app where required

## Fixed Decisions

1. Runtime platform: Azure Container Apps
2. Environments: staging and production
3. CI/CD: Azure DevOps YAML
4. Public edge: Cloudflare
5. Public host model: `app.<domain>` and `api.<domain>`
6. API origin model: Cloudflare Tunnel into a private ACA environment
7. API runtime contract: `CORS_ORIGINS=https://app.<domain>` and `REQUIRE_EDGE_AUTH=true`
8. Worker model: queue-backed background execution, never public ingress

## Target Architecture

1. Frontend Container App serves the UI on `https://app.<domain>` through Cloudflare.
2. API Container App serves `https://api.<domain>/api/*` through Cloudflare Tunnel.
3. Worker Container App handles indexing and long-running graph jobs with no ingress.
4. Tunnel Connector Container App publishes the private ACA frontend and API origins.
5. Cloudflare enforces WAF, rate limiting, cache bypass, and optional `X-Edge-Secret` injection.
6. ACA backend plane uses:
   - delegated subnet
   - private endpoint and private DNS
   - public network access disabled

## Required Application Changes

### Frontend

1. Use `NEXT_PUBLIC_API_BASE_URL=https://api.<domain>`.
2. Remove any assumption that auth state comes from `/.auth/*` on the backend host.
3. Keep browser requests pointed at `api.<domain>` for API and SSE traffic.
4. Keep the frontend health route at `app/api/health/route.ts`.
5. Keep CSP headers and explicit host allowlists.

### Backend

1. Replace open CORS with env-based allowlist via `CORS_ORIGINS`.
2. Keep `/health` as liveness-only.
3. Keep `/health/readiness` for dependency checks.
4. Require `X-Edge-Secret` on `/api/*` when `REQUIRE_EDGE_AUTH=true`.
5. Keep application rate limiting as fallback defense in depth.
6. Keep structured logging with `Cf-Ray` and `CF-Connecting-IP`.
7. Keep SSE heartbeat events and anti-buffering headers.

## Runtime Config Contract

- Frontend:
  - `NEXT_PUBLIC_API_BASE_URL=https://api.<domain>`
- Backend:
  - `CORS_ORIGINS=https://app.<domain>`
  - `REQUIRE_EDGE_AUTH=true`
  - `EDGE_ORIGIN_SECRET=<value>` when enabled
- Tunnel connector:
  - `CLOUDFLARE_TUNNEL_TOKEN=<value>`

## Networking Contract

- Cloudflare is the only public ingress for frontend and API.
- Frontend and API ACA apps use internal ingress only.
- Worker uses no ingress.
- Direct-origin probes must fail before application handling.
- `X-Edge-Secret` is not enough on its own to claim origin isolation.

## CI/CD Direction

The delivery path remains:

1. `Validate`
2. `BuildImages`
3. `DeployStaging`
4. `SmokeStaging`
5. `ManualApproval`
6. `DeployProduction`

Release gates must retain evidence for:

- health and readiness
- CRUD, upload, indexing, polling, and SSE
- direct-origin denial
- tunnel failover
- rollback readiness

## Notes

- Azure AI Search Free SKU remains an accepted initial production exception.
- The docs and scripts should use one contract only: private origin plus Cloudflare ingress.
