# Release Promotion Checklist

## Promotion Prerequisites

Before requesting production promotion, confirm:

- staging deployment completed through the approved pipeline path
- required smoke tests passed
- private-origin validation helper passed in staging
- direct-origin denial was verified
- tunnel failover was verified
- rollback drill evidence exists and is still relevant
- accepted risks are documented

## Mandatory Evidence Bundle

Attach or reference:

- staging smoke report
- private-origin validation helper output
- direct-origin denial evidence
- tunnel failover evidence
- SSE long-running stream evidence
- rollback drill evidence
- alert configuration or routing evidence
- accepted-risk record for Azure AI Search Free SKU

## Full Promotion Criteria

Proceed from canary to full production only when:

- canary metrics are stable
- no public-origin bypass path is present
- health and readiness succeed through the public host
- required alerts are active
- rollback remains immediately available
- release approver explicitly approves promotion

## Rollback Decision Points

Rollback immediately if:

- canary metrics degrade materially
- direct-origin denial check fails
- tunnel failover or connector health regresses badly
- SSE or queue-backed execution regresses badly
- alerting indicates severe instability after promotion
