# Cloudflare Tunnel Rotation Runbook

## Purpose

Rotate Cloudflare Tunnel credentials safely for the GToG API path without exposing the ACA API publicly or causing unnecessary downtime.

## Scope

This runbook covers:
- token ownership and storage location
- planned tunnel token rotation
- rollback steps if rotation fails
- emergency rotation after suspected credential exposure or tunnel compromise

## Ownership and Storage

### Ownership
Tunnel credentials are infrastructure-owned, not application-owned.

### Environment isolation
Use separate tunnels and separate tokens for:
- staging
- production

### Storage location
- Store tunnel tokens in Azure Key Vault
- Reference tokens from the tunnel connector app configuration
- Do not store tokens in source control or static deployment files

## Rotation Triggers

Rotate tunnel credentials when:
- scheduled credential hygiene requires it
- a token is suspected to be exposed
- tunnel administration access changed hands
- Cloudflare configuration was changed during an incident and confidence in token secrecy is reduced

## Planned Rotation Procedure

### 1. Prepare
- Confirm current tunnel name and environment
- Confirm Key Vault secret name and current revision or versioning approach
- Confirm at least one stable connector replica is healthy before changing anything
- Schedule rotation during a controlled maintenance window if production sensitivity requires it

### 2. Create or obtain replacement token
- Generate or retrieve the new token through the approved Cloudflare administration path
- Confirm it belongs to the correct environment tunnel

### 3. Update secret storage
- Write the new token into the correct Key Vault secret path or version
- Keep auditability of the change

### 4. Roll out connector update
- Update the tunnel connector app to consume the new token
- Restart or roll the connector revision using the standard deployment path
- Keep at least one healthy path available if the platform rollout model permits it

### 5. Verify post-rotation behavior
Verify:
- `api.<domain>` still routes successfully through Tunnel
- connector replicas become healthy
- auth flow still works on `api.<domain>`
- direct-origin denial still holds
- no unexpected reconnect storm persists

## Rollback Procedure

Rollback if:
- the connector cannot establish a stable session
- API traffic fails after token rotation
- replica health degrades and does not recover quickly

Rollback steps:
1. Restore the previous token reference in Key Vault or the previous valid secret version.
2. Roll the connector back to the previous known-good configuration.
3. Re-verify API availability and connector health.
4. Preserve logs and evidence before retrying rotation.

## Emergency Rotation Procedure

Use emergency rotation when a token may be compromised.

Steps:
1. Generate a replacement token immediately.
2. Update Key Vault with the replacement token.
3. Roll connector configuration as quickly as possible.
4. Revoke or invalidate the previous token through the approved Cloudflare administration path.
5. Verify API traffic, connector health, and auth flow after the change.
6. Record the incident and attach evidence.

## Post-Rotation Verification Checklist

After planned or emergency rotation, verify:
- API is reachable at `https://api.<domain>`
- connector replicas are healthy
- tunnel reconnect behavior is normal
- authentication still works
- release or incident records include the rotation event

## Evidence to Retain

Keep:
- date and environment
- approver or operator
- Key Vault secret version or update record
- deployment or restart record
- post-rotation verification evidence
- rollback evidence if rollback was required
