# Incident Triage Runbook

## Purpose

Provide a fast starting point for diagnosing production issues involving frontend reachability, API traffic through Cloudflare, tunnel health, worker execution, and core platform dependencies.

## First Places to Check

### Frontend symptoms

- frontend ACA revision health
- browser console and network traces
- frontend server logs
- Cloudflare edge status for `app.<domain>`

### API symptoms

- `https://api.<domain>/health`
- `https://api.<domain>/health/readiness`
- backend ACA revision health
- backend structured logs
- Cloudflare edge and tunnel status

### Worker symptoms

- worker ACA revision health
- queue depth or processing activity
- Cosmos job state transitions
- lease ownership and retry behavior
- worker logs for repeated failures

### Tunnel symptoms

- tunnel connector replica health
- reconnect or disconnect events
- Cloudflare route configuration
- recent token rotation or config changes

## Common Failure Patterns

### Origin isolation failure

Possible indicators:

- direct-origin probes succeed
- requests hit app logic from an unintended origin path
- release gate fails on network-layer denial tests

### Tunnel degradation

Possible indicators:

- intermittent API availability
- repeated reconnect logs
- one connector failure causes visible impact

### Worker durability failure

Possible indicators:

- jobs stuck in `queued` or `running`
- repeated retries without progress
- duplicate final artifacts

## Immediate Containment Actions

- stop release progression
- pause risky config changes
- rollback to the last known-good ACA revision
- rotate the tunnel token if compromise is suspected
- disable only the specific changed edge rule if an edge policy caused the regression

Do not expose the ACA API publicly as an emergency shortcut.
