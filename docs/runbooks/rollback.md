# Rollback Runbook

## Purpose

Restore service safely after a failed deployment, unhealthy canary, or production degradation while preserving the private-origin API design.

## Scope

This runbook applies to:
- ACA frontend revisions
- ACA API revisions
- ACA worker revisions
- ACA tunnel connector revisions
- Cloudflare edge changes that must be reverted to the last known-good configuration

This runbook does not authorize re-enabling a public ACA API origin except through an approved break-glass process.

## Rollback Triggers

Start rollback when any of the following occurs after deployment:
- Canary metrics degrade beyond alert thresholds
- Smoke tests fail in a release gate
- Direct-origin denial checks fail
- Authentication flow fails on `api.<domain>`
- SSE becomes unstable through Cloudflare
- Tunnel failover does not preserve API availability
- Worker release introduces job loss, duplicate artifact creation, or lease corruption
- Production error rate or latency exceeds agreed thresholds and does not recover quickly

## Preconditions

Before rollback:
- Confirm the last known-good ACA revision IDs for frontend, API, worker, and tunnel
- Confirm whether the problem is application, infrastructure, tunnel, or edge-policy related
- Preserve current evidence: failing smoke output, logs, screenshots, alert timestamps, and pipeline run ID
- Notify the release approver or on-call owner

## Rollback Strategy

### Default strategy
Rollback by shifting traffic to the previous ACA revision.

### Principles
- Prefer revision rollback over ad hoc patching during an incident
- Keep `app.<domain>` and `api.<domain>` stable for users
- Keep Cloudflare Tunnel and DNS hostnames stable unless the incident is tunnel-specific
- Do not re-enable public network access on the ACA API during normal rollback

## ACA Revision Rollback Steps

### 1. Identify last known-good revisions
Record the previous healthy revisions for:
- frontend
- API
- worker
- tunnel connector

### 2. Roll back frontend if required
Rollback frontend if the issue is UI-only or if frontend changes are part of the incident scope.

### 3. Roll back API if required
Rollback API if auth, request handling, SSE, or request validation regressed.

### 4. Roll back worker if required
Rollback worker if queue processing, retries, lease ownership, or indexing behavior regressed.

### 5. Roll back tunnel connector if required
Rollback tunnel connector only when the incident is caused by connector image, configuration, or token rotation issues.

### 6. Re-run critical smoke checks
After rollback, verify:
- `app.<domain>` is reachable
- `api.<domain>/health` responds correctly
- protected `/api/*` requests still require auth
- direct-origin denial still holds
- SSE remains stable
- worker queue processing resumes normally

## Cloudflare Rollback Rules

Only roll back Cloudflare changes when the incident is caused by edge policy or tunnel routing.

Possible rollback targets:
- rate-limiting rule changes
- custom WAF rule changes
- header transform rule changes for `EDGE_ORIGIN_SECRET`
- tunnel route or hostname routing changes

Do not remove Cloudflare protection or expose the ACA API publicly as a shortcut.

## Post-Rollback Verification

A rollback is considered successful only when:
- critical smoke checks pass
- alerts return to normal state or are clearly recovering
- auth flow works on `api.<domain>`
- direct-origin denial still fails at the network layer
- job processing is healthy or safely paused
- rollback evidence is attached to the release or incident record

## Evidence to Capture

Retain the following:
- pipeline run ID
- deployed and restored revision IDs
- rollback command output or deployment logs
- smoke test results after rollback
- alert timestamps and screenshots if available
- brief incident summary and decision log

## Break-Glass Approval Rules

Break-glass actions require explicit approval from the release owner or incident owner.

Break-glass actions include:
- temporarily disabling a critical edge rule
- rotating a production tunnel token under time pressure
- bypassing normal canary progression
- changing auth configuration outside the standard release path
- any action that could expose the ACA API publicly

The following remains prohibited without explicit incident approval:
- enabling a direct public ACA API origin
- disabling private-origin protections to restore service faster
- skipping evidence capture for rollback decisions

## Escalation

Escalate immediately if:
- rollback fails to restore service
- the previous revision is also unhealthy
- tunnel health remains degraded after rollback
- auth audience isolation fails
- the rollback would require public-origin exposure to restore service
