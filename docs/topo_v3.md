# Production Topology v3 - GToG

```mermaid
flowchart TB
  U[User Browser]

  U -->|https://app.<domain>| CFAPP[Cloudflare Edge - app host]
  U -->|https://api.<domain>| CFAPI[Cloudflare Edge - api host]

  subgraph EDGE[Cloudflare Edge Controls]
    E1[Proxied DNS]
    E2[Managed rules]
    E3[Rate limit on api host]
    E4[Cache bypass for API and SSE]
    E5[Optional header transform injects X-Edge-Secret]
    E6[Cloudflare Tunnel public hostnames]
    E1 --- E2 --- E3 --- E4 --- E5 --- E6
  end

  CFAPP --> TUNNEL[Cloudflare Tunnel]
  CFAPI --> TUNNEL

  subgraph PRIVATE[Private ACA Environment]
    FE[ACA Frontend - internal ingress]
    API[ACA API - internal ingress]
    WORKER[ACA Worker - no ingress]
    QUEUE[Azure Storage Queue]
    LEASE[Cosmos job state + leases]
    TUNNEL --> FE
    TUNNEL --> API
    API --> QUEUE
    API --> LEASE
    QUEUE --> WORKER
    WORKER --> LEASE
  end

  API --> KV[Key Vault]
  API --> SEARCH[Azure AI Search]
  API --> BLOB[Blob Storage]
  WORKER --> BLOB
  WORKER --> SEARCH
```

## Notes

- Cloudflare Tunnel replaces any public ACA API path.
- `X-Edge-Secret` is optional defense in depth, not the primary origin lock.
- Frontend and API use separate public hostnames, but both resolve through Cloudflare-only ingress.
- Worker execution remains off the synchronous API path.
