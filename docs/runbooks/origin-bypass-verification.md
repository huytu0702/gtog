# Origin Bypass Verification Runbook

## Purpose

Prove that the ACA API origin is not publicly reachable and that `api.<domain>` is served only through Cloudflare Tunnel.

## Scope

This runbook covers:
- public probe procedure
- expected failure modes
- evidence capture format
- pass/fail criteria
- recommended validation cadence

## Verification Goal

The production claim is not merely that the API requires auth or a secret header. The stronger claim is that the ACA API origin itself is not publicly reachable from the internet.

## Test Targets

Probe any known or candidate direct-origin paths that should not be publicly reachable, including:
- ACA API default endpoints if known
- direct ACA ingress endpoints if known
- any previously used public API hostname or origin path

Do not use normal `api.<domain>` traffic as the proof of origin isolation. That hostname is expected to work through Cloudflare Tunnel.

## Public Probe Procedure

### 1. Confirm normal public path still works
As a control, verify that the expected public path works:
- `https://api.<domain>/health`

This confirms the service is available through the intended route.

### 2. Attempt direct-origin access from a public network
From a public network path outside the private Azure boundary, attempt to reach the ACA API origin using known origin endpoints.

### 3. Record the failure mode
Capture whether the probe fails due to:
- DNS resolution failure
- connection timeout
- connection refused
- platform-level inaccessibility before app handling

### 4. Confirm the request did not reach the app layer
Check backend logs to confirm the direct-origin probe did not generate application request handling for the attempted origin path.

## Expected Failure Modes

Acceptable pass outcomes include:
- cannot resolve public endpoint
- cannot establish network connection
- connection refused or timed out before application handling
- no application-layer request evidence for the probe

The following are not sufficient to claim success:
- the request reaches the app and receives `401`
- the request reaches the app and fails due to missing `EDGE_ORIGIN_SECRET`
- the request reaches the app and returns any application-level response

Those outcomes mean origin isolation is not complete.

## Pass / Fail Criteria

### Pass
Origin bypass verification passes only when:
- direct public-origin probes fail at the network layer or before application handling
- no application request path is opened by the probe
- the intended public API hostname still works through Cloudflare Tunnel

### Fail
Origin bypass verification fails when:
- a public direct-origin path is reachable
- the probe reaches FastAPI, Easy Auth, or header validation logic
- the proof depends only on application-layer rejection

## Evidence Capture Format

For each verification run, capture:
- date and environment
- operator name
- tested origin identifiers
- exact probe commands or tool output
- observed failure mode
- backend log check result
- final pass/fail conclusion

Preferred evidence artifacts:
- terminal output
- screenshots if useful
- log query output showing absence of app handling for the probe
- pipeline artifact if run as part of release validation

## Recommended Validation Cadence

Run this verification:
- before production sign-off
- after networking or ACA ingress changes
- after tunnel topology changes
- after major auth or domain changes
- after incident remediation that touched ingress or routing

## Escalation

If this verification fails:
- stop release progression immediately
- do not rely on `EDGE_ORIGIN_SECRET` or `401` responses as a substitute
- fix the networking or ingress path first
- rerun verification before promotion continues
