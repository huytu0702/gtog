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
7. Configure backend-only Easy Auth on the API app using the same script path, not a separate deployment path.

### Phase 3 inputs

Phase 3 freezes the identity contract per environment:

- one Entra app registration per environment
- Easy Auth callback host fixed to `https://api.<domain>`
- Microsoft callback path: `https://api.<domain>/.auth/login/aad/callback`
- Google callback path: `https://api.<domain>/.auth/login/google/callback`
- Microsoft `allowedAudiences` fixed to the environment-specific API App ID URI (`api://<backend-app-id>`)
- Google `allowedAudiences` fixed to the configured Google client ID unless explicitly overridden
- staging and production must not share Entra audiences or Google client registrations

The provisioning scripts support both pre-created and script-created Entra registrations while reconciling Google provider settings in the same Easy Auth payload.

### Phase 3 script flags and environment variables

Use the provisioning script with:

- `CONFIGURE_EASY_AUTH=true` / `-ConfigureEasyAuth`
- `API_PUBLIC_HOSTNAME=api.<domain>` / `-ApiPublicHostname`
- either:
  - `ENTRA_APP_ID`, `ENTRA_TENANT_ID`, `ENTRA_CLIENT_SECRET`, `API_APP_ID_URI`, `ALLOWED_AUDIENCES`
  - or `CREATE_ENTRA_APP=true` / `-CreateEntraApp` with `ENTRA_APP_DISPLAY_NAME`
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`

Optional but recommended:

- `GOOGLE_CLIENT_SECRET_NAME` / `-GoogleClientSecretName` if the API app should use a non-default secret reference name
- `GOOGLE_ALLOWED_AUDIENCES` / `-GoogleAllowedAudiences` when Google `allowedAudiences` should differ from the client ID
- `GOOGLE_LOGIN_SCOPES_JSON` / `-GoogleLoginScopesJson` when you need to override the default Google scope list (`["openid", "profile", "email"]`)
- `RESET_ENTRA_CLIENT_SECRET=true` / `-ResetEntraClientSecret` when rotating the Entra Easy Auth secret
- `AAD_LOGIN_PARAMETERS_JSON` / `-AadLoginParametersJson` when you need to override the default Microsoft login parameter payload

The scripts reconcile these platform guarantees on every run:

- API ingress stays internal-only
- worker ingress stays disabled
- Easy Auth stays enabled on the API app only
- unauthenticated `api.<domain>/api/*` requests return `401`
- the configured `allowedAudiences` exactly match the environment contract

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

1. Run `scripts/validate-aca-phase3-auth.sh` or `.ps1` with the environment inputs, including `TUNNEL_APP_NAME`, to verify the API, worker, and tunnel connector contracts together. If the tunnel app uses a non-default secret reference name, also pass `TUNNEL_SECRET_REF_NAME`.
2. From the public Internet, direct probes to the ACA API origin must fail at the network layer.
3. `https://api.<domain>/.auth/me` must still work through Cloudflare.
4. Browser login, logout, and token retrieval must remain on `api.<domain>`.
5. CRUD, upload, indexing, query, and SSE flows must pass through the tunnel path.
6. Stop one tunnel replica and confirm traffic still succeeds.
7. Confirm logs include:
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
