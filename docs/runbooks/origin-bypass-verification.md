# Origin Bypass Verification Runbook

## Purpose

Prove that the ACA API origin is not publicly reachable and that `api.<domain>` is served only through Cloudflare Tunnel.

## Verification Goal

The production claim is stronger than “the backend rejects unauthorized requests.” The claim is that the ACA API origin itself is not reachable from the public Internet outside the Cloudflare path.

## Public Probe Procedure

1. Confirm the intended public path still works:
   - `https://api.<domain>/health`
   - `https://api.<domain>/health/readiness`
2. Run `scripts/validate-aca-phase3-auth.sh` or `.ps1` first to confirm the private-origin contract and optional direct-origin probes.
3. From a public network path, attempt to reach known ACA origin endpoints directly.
4. Record whether the probe fails due to DNS failure, timeout, refusal, or another network-layer denial.
5. Confirm the probe did not generate an application request in backend logs.

## Pass Criteria

Origin bypass verification passes only when:

- direct-origin probes fail before application handling
- no backend request evidence exists for the probe
- the intended public hostname still works through Cloudflare Tunnel

The following are not sufficient:

- the request reaches FastAPI and returns `401`
- the request reaches FastAPI and fails due to missing `X-Edge-Secret`
- the request reaches any application handler at all

## Evidence to Capture

For each run, retain:

- date and environment
- tested origin identifiers
- exact probe commands or validation-helper output
- observed failure mode
- backend log query output
- final pass/fail conclusion

Preferred artifacts:

- terminal output
- `scripts/validate-aca-phase3-auth.sh` or `.ps1` output
- pipeline artifact when this is part of release validation
