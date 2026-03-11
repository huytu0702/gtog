# Incident Triage Runbook

## Purpose

Provide a fast starting point for diagnosing production issues involving frontend reachability, backend auth, tunnel health, worker execution, and core platform dependencies.

## First Places to Check

### Frontend symptoms
Check:
- frontend ACA revision health
- browser console and network traces
- frontend server logs in ACA / Log Analytics
- Cloudflare edge status for `app.<domain>`

### API symptoms
Check:
- `https://api.<domain>/health`
- backend ACA revision health
- Easy Auth behavior on `/.auth/me`
- backend structured logs
- Cloudflare edge and tunnel status

### Worker symptoms
Check:
- worker ACA revision health
- queue depth or queue processing activity
- Cosmos job state transitions
- lease ownership and retry behavior
- worker logs for repeated failures or duplicate handling

### Tunnel symptoms
Check:
- tunnel connector replica health
- reconnect or disconnect events
- Cloudflare route configuration
- recent token rotation or config changes

## Common Failure Patterns

### Auth flow failure
Possible indicators:
- `/.auth/me` returns unexpected state
- login redirects loop or fail
- production rejects expected tokens

### Origin isolation failure
Possible indicators:
- direct-origin probes succeed
- requests hit app logic from an unintended origin path
- release gate fails on network-layer denial test

### Worker durability failure
Possible indicators:
- jobs stuck in `queued` or `running`
- repeated retries without progress
- duplicate final artifacts
- stale leases not expiring cleanly

### Tunnel degradation
Possible indicators:
- intermittent API availability
- repeated reconnect logs
- one connector failure causes visible impact

## Immediate Containment Actions

Depending on the incident:
- stop release progression
- pause risky config changes
- rollback to the last known-good ACA revision
- rotate tunnel token if compromise is suspected
- disable only the specific changed edge rule if an edge policy caused the regression

Do not expose the ACA API publicly as an emergency shortcut.

## Escalation Path

Escalate when:
- rollback fails
- the previous revision is also unhealthy
- auth audience isolation breaks
- direct-origin denial fails
- data recovery or rebuild is required

Use the specialized runbooks as needed:
- `docs/runbooks/rollback.md`
- `docs/runbooks/backup-restore.md`
- `docs/runbooks/cloudflare-tunnel-rotation.md`
- `docs/runbooks/origin-bypass-verification.md`
- `docs/runbooks/release-promotion-checklist.md`
