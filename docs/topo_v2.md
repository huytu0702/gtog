```mermaid
flowchart TB
  U[User Browser]

  U -->|Visit https://app.<domain>| CFAPP[Cloudflare Edge - app host]
  U -->|API calls to https://api.<domain>| CFAPI[Cloudflare Edge - api host]

  CFAPP --> FE[ACA Frontend - Next.js]
  CFAPI -->|Cloudflare Tunnel + optional X-Edge-Secret| BE[ACA Backend - FastAPI]

  subgraph EDGE[Cloudflare Edge Controls]
    E1[Proxied DNS]
    E2[Rate limit on api host]
    E3[WAF or custom edge rules]
    E4[Cache bypass for API and SSE]
    E5[Optional request transform injects X-Edge-Secret]
    E1 --- E2 --- E3 --- E4 --- E5
  end

  CFAPP --- EDGE
  CFAPI --- EDGE

  subgraph PRIVATE[Private ACA Environment]
    FE
    BE
    WORKER[ACA Worker]
    TUNNEL[cloudflared connectors x2]
    TUNNEL --> FE
    TUNNEL --> BE
  end

  BE --> LOCK[Secondary edge guard + rate limiting]
  BE --> COSMOS[Azure Cosmos DB]
  BE --> BLOB[Azure Blob Storage]
  BE --> SEARCH[Azure AI Search]
  BE --> KV[Azure Key Vault]
```

**Notes:**

- Cloudflare is the only public edge.
- Frontend and API stay split by hostname: `app.<domain>` and `api.<domain>`.
- Backend protection comes from private ingress, Cloudflare Tunnel, optional `X-Edge-Secret`, and application rate limiting.
- `CORS_ORIGINS` must allow only `https://app.<domain>`.
- SSE traffic stays on the API host and must emit heartbeat events.
