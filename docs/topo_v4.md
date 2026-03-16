# Production Topology v4 - GToG

> Simplified production topology for a Cloudflare-only ingress model. Every browser request enters through Cloudflare, and no ACA origin is exposed directly to the public Internet.

```mermaid
flowchart TB
  U[User Browser]

  U -->|https://app.<domain>| CFAPP[Cloudflare Edge - app host]
  U -->|https://api.<domain>| CFAPI[Cloudflare Edge - api host]

  subgraph EDGE[Cloudflare Edge]
    DNS[Proxied DNS]
    WAF[Managed rules]
    RL[Rate limit on api host]
    CACHE[Cache bypass for /api/* and SSE routes]
    ACCESS[Optional Cloudflare Access]
    DNS --- WAF --- RL --- CACHE --- ACCESS
  end

  CFAPP --> TUNNEL[Cloudflare Tunnel]
  CFAPI --> TUNNEL

  subgraph ACA[Private Azure Container Apps Environment]
    CONN[cloudflared connectors x2]
    FE[Frontend - internal ingress]
    API[API - internal ingress]
    WORKER[Worker - no ingress]
    CONN --> FE
    CONN --> API
  end
```

## Design Principles

- Cloudflare is the only public edge for both frontend and API traffic.
- Frontend and API use internal ACA ingress only.
- The worker has no ingress.
- `X-Edge-Secret` can exist as a secondary guard, but origin isolation comes from topology.
- Direct-origin probes must fail because there is no public ACA ingress path.
