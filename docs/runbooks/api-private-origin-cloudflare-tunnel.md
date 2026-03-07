# API Private-Origin Runbook (Cloudflare Tunnel + ACA)

## Goal

Publish `https://api.<domain>` through Cloudflare without leaving the ACA API origin publicly reachable on the Internet.

Target state:

- `api.<domain>` is a Cloudflare Tunnel public hostname
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
2. Provision a VNet with:
   - delegated ACA infrastructure subnet
   - private endpoint subnet
3. Create a workload-profile ACA environment with `internal-only` enabled.
4. Disable public network access on the ACA environment.
5. Create the ACA private endpoint and private DNS zone link.
6. Deploy:
   - `ca-gtog-api-{env}`
   - `ca-gtog-worker-{env}`
   - `ca-gtog-tunnel-{env}`

## Cloudflare Configuration

1. Create a remotely managed tunnel for the environment.
2. Store the tunnel token in Key Vault and pass it to the tunnel connector app.
3. Add public hostname:
   - `api.<domain>` -> the tunnel
4. Point the tunnel service to the API private origin:
   - internal ACA URL or private ACA environment FQDN
5. If Easy Auth or cookie behavior depends on the public host, set the origin request host header to `api.<domain>`.
6. Keep WAF, rate limits, and cache bypass rules on `api.<domain>`.
7. If the backend secondary guard is enabled, inject `X-Edge-Secret` on the Cloudflare side.

## Validation

1. From the public Internet, direct probes to the ACA API origin must fail at the network layer.
2. `https://api.<domain>/.auth/me` must still work through Cloudflare.
3. Browser login, logout, and token retrieval must remain on `api.<domain>`.
4. CRUD, upload, indexing, query, and SSE flows must pass through the tunnel path.
5. Stop one tunnel replica and confirm traffic still succeeds.
6. Confirm logs include:
   - `Cf-Ray`
   - app `request_id`
   - tunnel connector health and reconnect events

## Rotation and Break-Glass

1. Rotate `CLOUDFLARE_TUNNEL_TOKEN` in staging first, then production.
2. Rotate `EDGE_ORIGIN_SECRET` separately if the backend secondary guard is enabled.
3. Do not re-enable ACA public network access as a routine rollback step.
4. Break-glass should prefer:
   - tunnel route rollback
   - tunnel token rollback
   - ACA revision rollback
5. Any temporary public-origin re-enable must be time-bounded, approved, and followed by a post-incident cleanup.
