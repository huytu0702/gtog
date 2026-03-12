# Release Promotion Checklist

## Purpose

Standardize promotion from staging to production using evidence-based gates, controlled canary rollout, and immediate rollback readiness.

## Promotion Prerequisites

Before requesting production promotion, confirm:
- staging deployment completed through the approved pipeline path
- required smoke tests passed
- rollback drill evidence exists and is still relevant
- Phase 3 validation helper passed in staging, including tunnel connector contract checks
- direct-origin denial was verified
- auth audience isolation was verified, including cross-environment token rejection
- `/.auth/me` and unauthenticated `/api/*` behavior were verified on `api.<domain>`
- tunnel failover was verified
- accepted risks are documented
- required runbooks are current

## Mandatory Evidence Bundle

Attach or reference the following before promotion:
- staging smoke report
- Phase 3 validation helper output (`scripts/validate-aca-phase3-auth.sh` or `.ps1`), including tunnel connector evidence
- direct-origin denial evidence
- auth audience isolation result, including staging-token rejection by production
- `/.auth/me` and unauthenticated `/api/*` verification result
- SSE long-running stream evidence
- tunnel failover evidence
- rollback drill evidence
- restore drill evidence where required by release policy
- alert configuration or alert-routing evidence
- accepted-risk record for Azure AI Search Free SKU

## Canary Approval Steps

### 1. Confirm readiness to promote
- pipeline stages are green
- no open blocker remains from staging validation
- approver has reviewed the evidence bundle

### 2. Start canary rollout
- deploy production through the approved deployment path
- route only canary traffic initially
- preserve the ability to return to the last known-good revision immediately

### 3. Observe canary behavior
Watch:
- 5xx rate
- latency
- auth failures
- tunnel health
- replica restarts
- Cosmos and Search throttling
- SSE behavior where possible

### 4. Approve or reject canary
Approve full promotion only if canary remains healthy and no security or reliability gate regresses.

## Full Promotion Criteria

Proceed from canary to full production only when:
- canary metrics are stable
- no public-origin bypass path is present
- auth works on `api.<domain>`
- `/.auth/me` behavior is consistent with the environment contract
- unauthenticated `/api/*` requests still return `401`
- required alerts are active
- rollback remains immediately available
- release owner or approver explicitly approves full promotion

## Rollback Decision Points

Rollback immediately if:
- canary metrics degrade materially
- direct-origin denial check fails
- auth audience isolation fails
- SSE or tunnel behavior regresses badly
- a deployment introduces queue, lease, or duplicate-processing faults
- alerting indicates severe instability after promotion

When rollback is triggered, use:
- `docs/runbooks/rollback.md`

## Final Sign-Off Checklist

Before marking the release complete:
- production rollout reached intended traffic level
- production smoke checks succeeded
- monitoring stabilized after cutover
- evidence bundle is attached to the release record
- accepted risks remain documented
- no unresolved production exception is left undocumented
