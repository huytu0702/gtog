# API Private-Origin Runbook (Cloudflare Tunnel + ACA)

## Goal

Publish `https://api.<domain>` through Cloudflare without exposing the ACA API origin directly to the public Internet.

Target state:

- `api.<domain>` is served through Cloudflare Tunnel
- API and worker run in a private ACA environment
- API ingress is internal only
- ACA public network access is disabled
- `X-Edge-Secret` is optional defense in depth, not the primary origin lock

## Azure Provisioning

Use one of:

- `scripts/provision-aca-private-origin.ps1`
- `scripts/provision-aca-private-origin.sh`

Recommended flow:

1. Provision or reuse a Log Analytics workspace.
2. Provision a VNet with a delegated ACA infrastructure subnet and a private endpoint subnet.
3. Create a workload-profile ACA environment with `internal-only` enabled.
4. Disable public network access on the ACA environment.
5. Create the ACA private endpoint and private DNS zone link.
6. Deploy or reconcile:
   - `ca-gtog-frontend-{env}`
   - `ca-gtog-api-{env}`
   - `ca-gtog-worker-{env}`
   - `ca-gtog-tunnel-{env}`

The provisioning scripts now reconcile these guarantees on every run:

- frontend ingress stays internal-only
- API ingress stays internal-only
- worker ingress stays disabled
- tunnel connector keeps the expected `cloudflared tunnel --no-autoupdate run` contract
- API runtime keeps `CORS_ORIGINS=https://app.<domain>` and `REQUIRE_EDGE_AUTH=true`

## Cloudflare Configuration

1. Create a remotely managed tunnel for the environment.
2. Store the tunnel token in Key Vault and pass it to the tunnel connector app.
3. Add public hostnames:
   - `app.<domain>`
   - `api.<domain>`
4. Point the tunnel routes to the ACA private origins.
5. Keep WAF, rate limiting, and cache bypass rules on the public hosts.
6. If the secondary backend guard is enabled, inject `X-Edge-Secret` on the Cloudflare side.

## Validation

1. Run `scripts/validate-aca-phase3-auth.sh` or `.ps1` with:
   - `APP_PUBLIC_HOSTNAME`
   - `API_PUBLIC_HOSTNAME`
   - `API_APP_NAME`
   - `WORKER_APP_NAME`
   - `TUNNEL_APP_NAME`
   - optional `PROBE_ORIGIN_URLS`
2. Confirm `https://api.<domain>/health` succeeds through Cloudflare.
3. Confirm direct-origin probes fail before reaching an HTTP handler.
4. Confirm CRUD, upload, indexing, query, and SSE flows succeed through the public hosts.
5. Stop one tunnel replica and confirm traffic still succeeds.
6. Confirm logs include request correlation and tunnel reconnect evidence.

## Rotation and Break-Glass

1. Rotate `CLOUDFLARE_TUNNEL_TOKEN` in staging first, then production.
2. Rotate `EDGE_ORIGIN_SECRET` separately if the backend secondary guard is enabled.
3. Do not re-enable ACA public network access as a routine rollback step.
4. Break-glass should prefer:
   - tunnel route rollback
   - tunnel token rollback
   - ACA revision rollback
